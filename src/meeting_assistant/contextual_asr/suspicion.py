"""Agreement of noncanonical acoustic resemblance and supplied context; not probability."""

from meeting_assistant.grounding.normalization import normalize
from meeting_assistant.grounding.retrieval import STOP
from meeting_assistant.refinement.validation import comparable, protected_signature

from .models import SuspiciousSpan


def detect_suspicious_spans(grounding, context, config):
    context_ids = {t.id for t in context.terms} if context else set()
    results = []
    for record in grounding.records:
        if protected_signature(record.observed_text) or record.observed_text.casefold() in STOP:
            continue
        if any(
            comparable(record.observed_text) == comparable(c.canonical) for c in record.candidates
        ):
            continue  # Correct GPU/ordinary homophones must not be replaced by context bias.
        evidence = []
        for candidate in record.candidates:
            supplied = candidate.entry_id in context_ids
            if not supplied and not config.use_global_glossary:
                continue
            if protected_signature(candidate.canonical) or candidate.match_type == "alias_exact":
                continue
            phonetic = candidate.phonetic_score >= 0.88
            lexical = candidate.lexical_score >= (0.70 if supplied else 0.78)
            explicit = candidate.match_type == "asr_alias_exact"
            # Ordinary inflections and phrases containing function words are not
            # acoustic errors. Context cannot turn these into automatic targets.
            observed = comparable(record.observed_text)
            canonical = comparable(candidate.canonical)
            inflection = any(
                observed == canonical + suffix or canonical == observed + suffix
                for suffix in ("s", "es")
            )
            if inflection or (
                not explicit
                and any(token in STOP for token in normalize(record.observed_text).split())
            ):
                continue
            if record.observed_text.strip().isupper():
                continue  # Already explicit acronyms (GPU/CPU) need stronger future evidence.
            if not explicit and not (phonetic and lexical):
                continue
            priority = min(
                1.0, candidate.score + (0.10 if supplied else 0) + (0.05 if explicit else 0)
            )
            if priority >= config.min_priority:
                signals = tuple(
                    name
                    for name, value in (
                        ("supplied_context", supplied),
                        ("phonetic_match", phonetic),
                        ("lexical_match", lexical),
                        ("registered_asr_alias", explicit),
                    )
                    if value
                )
                evidence.append((priority, signals))
        if evidence:
            priority, signals = max(evidence)
            results.append(SuspiciousSpan(record.id, priority, signals))
    return tuple(sorted(results, key=lambda span: (-span.priority, span.grounding_id)))
