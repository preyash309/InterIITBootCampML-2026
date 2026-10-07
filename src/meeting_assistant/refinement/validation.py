"""Strict structure and conservative deterministic vetoes, independent of the LLM."""

import json
import re
import unicodedata

from .exceptions import RefinementSchemaError, RefinementValidationError
from .models import RefinementDecision, RefinementRequest, RefinementResponse

PROTECTED = frozenset(
    """not no never cannot won't don't doesn't didn't isn't aren't
wasn't weren't shouldn't couldn't wouldn't mustn't might may could should must will
would shall agreed approved rejected confirmed all some none one two three four five
six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred
thousand million billion half most percent percentage monday tuesday wednesday
thursday friday saturday sunday january february march april may june july august
september october november december today tomorrow yesterday am pm noon midnight
second seconds minute minutes hour hours day days week weeks month months year years
can can't deploy deadline jan feb mar apr jun jul aug sep oct nov dec before after
kg g mg km m cm mm ft inch inches lb lbs l ml hz khz mhz ghz watt watts kw volt volts
celsius fahrenheit utc gmt ist plus minus twice fewer more every each any""".split()
)
NEGATIONS = frozenset(
    "not no never cannot won't don't doesn't didn't isn't aren't wasn't weren't shouldn't couldn't wouldn't mustn't can't".split()
)
MODALITY = frozenset(
    "might may could should must will would shall can agreed approved rejected confirmed".split()
)
DATES = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november december jan feb mar apr jun jul aug sep oct nov dec today tomorrow yesterday am pm noon midnight utc gmt ist".split()
)


def tokens(text: str) -> list[str]:
    return re.findall(
        r"\d+(?:[.,:/-]\d+)*(?:%|[a-z]+)?|[a-z]+(?:'[a-z]+)?|[%$€£¥]",
        unicodedata.normalize("NFKC", text).casefold().replace("’", "'"),
    )


def protected_signature(text: str) -> tuple[str, ...]:
    # Preserve order as well as content (swapping two numbers changes meaning).
    return tuple(
        t for t in tokens(text) if t in PROTECTED or any(c.isdigit() for c in t) or t in "%$€£¥"
    )


def safety_violations(original: str, refined: str) -> dict[str, int]:
    """Observed category changes, not a claim to measure all semantic corruption."""
    before, after = tokens(original), tokens(refined)

    def changed(predicate):
        return int(
            tuple(t for t in before if predicate(t)) != tuple(t for t in after if predicate(t))
        )

    return {
        "number_changes": changed(lambda t: any(c.isdigit() for c in t)),
        "negation_changes": changed(lambda t: t in NEGATIONS),
        "date_time_changes": changed(
            lambda t: t in DATES or bool(re.fullmatch(r"\d+[:/-]\d+(?:[:/-]\d+)?", t))
        ),
        "modality_changes": changed(lambda t: t in MODALITY),
        "protected_content_changes": int(
            protected_signature(original) != protected_signature(refined)
        ),
    }


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RefinementSchemaError("Duplicate JSON object key.")
        result[key] = value
    return result


def parse_decisions(payload: str, request: RefinementRequest) -> tuple[RefinementDecision, ...]:
    try:
        data = json.loads(payload, object_pairs_hook=_pairs)
        if (
            not isinstance(data, dict)
            or set(data) != {"utterance_id", "decisions"}
            or data["utterance_id"] != request.target.utterance_id
            or not isinstance(data["decisions"], list)
        ):
            raise RefinementSchemaError("Response must cover the exact target utterance.")
        decisions = []
        for item in data["decisions"]:
            if not isinstance(item, dict) or set(item) != {
                "grounding_record_id",
                "action",
                "candidate_entry_id",
                "reason_code",
            }:
                raise RefinementSchemaError("Decision fields must match the schema exactly.")
            decisions.append(RefinementDecision(**item))
        result = tuple(decisions)
        validate_decisions(request, result)
        return result
    except (ValueError, TypeError, KeyError, RecursionError, RefinementValidationError) as exc:
        raise RefinementSchemaError(
            "Malformed refinement decisions or evidence references."
        ) from exc


def validate_decisions(
    request: RefinementRequest, decisions: tuple[RefinementDecision, ...]
) -> None:
    records = {r.id: r for r in request.records}
    if (
        not isinstance(decisions, tuple)
        or any(not isinstance(d, RefinementDecision) for d in decisions)
        or len(decisions) != len(records)
        or {d.grounding_record_id for d in decisions} != set(records)
    ):
        raise RefinementSchemaError("Decisions must cover every record exactly once.")
    for d in decisions:
        if d.action == "REPLACE" and d.candidate_entry_id not in {
            c.entry_id for c in records[d.grounding_record_id].candidates
        }:
            raise RefinementSchemaError("Replacement candidate does not belong to this record.")


def validate_response(request: RefinementRequest, response: RefinementResponse) -> None:
    if (
        not isinstance(response, RefinementResponse)
        or response.utterance_id != request.target.utterance_id
    ):
        raise RefinementSchemaError("Backend response does not belong to target.")
    validate_decisions(request, response.decisions)


def comparable(text: str) -> str:
    return re.sub(r"[\s\-–—]+", "", unicodedata.normalize("NFKC", text).casefold())


def replacement_rejection(record, candidate, utterance_text: str) -> str | None:
    source, replacement = record.observed_text, candidate.canonical
    # Defense in depth: no punctuation/control-bearing prose masquerading as a term.
    if (
        len(replacement) > 160
        or len(replacement.split()) > 6
        or any(c in replacement for c in "\n\r\t")
    ):
        return "invalid_term_shape"
    if protected_signature(source) != protected_signature(replacement):
        return "protected_meaning"
    edited = utterance_text[: record.char_start] + replacement + utterance_text[record.char_end :]
    if protected_signature(utterance_text) != protected_signature(edited):
        return "protected_meaning"
    if comparable(source) == comparable(replacement):
        return "formatting_only"
    # A supported original must not be replaced by a secondary candidate (GPU -> CPU).
    if any(comparable(source) == comparable(c.canonical) for c in record.candidates):
        return "original_supported"

    # No expansion/shortening of acronyms. Explicit phonetic ASR aliases are different.
    def acronym(s):
        return bool(re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", s))

    if candidate.match_type not in ("asr_alias_exact",) and (
        (acronym(replacement) and len(source.split()) > 1)
        or (acronym(source) and len(replacement.split()) > 1)
    ):
        return "acronym_expansion"
    name_category = candidate.domain == "organizations" or any(
        s in candidate.category.lower() for s in ("person", "participant", "company", "name")
    )
    if name_category and (
        candidate.scope not in ("project", "meeting")
        or candidate.match_type not in ("asr_alias_exact", "alias_exact", "normalized_exact")
    ):
        return "unsupported_name"
    # Exact ordinary aliases frequently denote equivalent wording, not recognition errors.
    if candidate.match_type == "alias_exact":
        return "equivalent_wording"
    if candidate.match_type == "fuzzy" and not (
        candidate.lexical_score >= 0.85
        and candidate.phonetic_score >= 0.88
        and candidate.semantic_score >= 0.30
    ):
        return "weak_candidate"
    return None
