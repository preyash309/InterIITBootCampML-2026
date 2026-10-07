"""Structural evidence checks; semantic support remains the independent Phase VII role."""

import json
import re
from dataclasses import asdict, replace

from .exceptions import (
    ConsolidationError,
    IntelligenceSchemaError,
    IntelligenceValidationError,
    UnknownEvidenceError,
)
from .models import (
    KINDS,
    SECTIONS,
    ActionItem,
    ActionOwner,
    Decision,
    IntelligenceRequest,
    MeetingContent,
    MinuteItem,
    SummaryPoint,
)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntelligenceSchemaError("Duplicate JSON keys are not allowed.")
        result[key] = value
    return result


def strict_json(payload, max_chars=250000):
    try:
        if not isinstance(payload, str) or len(payload) > max_chars:
            raise ValueError
        return json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise IntelligenceSchemaError("Expected bounded, finite strict JSON.") from exc


def fields(data, names):
    if not isinstance(data, dict) or set(data) != set(names):
        raise IntelligenceSchemaError("Response contains missing or unknown fields.")


def _lexical(phrase, source):
    # Apostrophe/whitespace normalization only; no inferred identity or date conversion.
    def normalize(s):
        return " ".join(re.findall(r"\w+", s.casefold()))

    return " " + normalize(phrase) + " " in " " + normalize(source) + " "


def validate_content(content: MeetingContent, utterances):
    sources = {u.utterance_id: u for u in utterances}
    for item in content.items:
        if any(uid not in sources for uid in item.evidence_utterance_ids):
            raise UnknownEvidenceError("An item references an unknown source utterance.")
        evidence = tuple(sources[uid] for uid in item.evidence_utterance_ids)
        if isinstance(item, ActionItem):
            # Each value must be supported in a single cited utterance, not accidentally
            # assembled by concatenating unrelated utterances.
            if item.owner is not None:
                if item.owner.kind == "speaker":
                    if item.owner.speaker_id not in {u.speaker_id for u in evidence}:
                        raise IntelligenceSchemaError(
                            "Owner speaker is absent from cited evidence."
                        )
                elif not any(_lexical(item.owner.display_text, u.refined_text) for u in evidence):
                    raise IntelligenceSchemaError("Named owner lacks literal cited support.")
            if item.deadline_text is not None and not any(
                _lexical(item.deadline_text, u.refined_text) for u in evidence
            ):
                raise IntelligenceSchemaError("Deadline lacks literal cited support.")
    return content


def parse_extraction(payload: str, request: IntelligenceRequest, *, max_items=128):
    data = strict_json(payload)
    fields(data, SECTIONS)
    sections = {}
    try:
        for section, prefix, kind in zip(
            SECTIONS,
            ("sum", "min", "dec", "act"),
            (SummaryPoint, MinuteItem, Decision, ActionItem),
        ):
            items = data[section]
            if not isinstance(items, list) or len(items) > max_items:
                raise IntelligenceSchemaError("Invalid section length.")
            parsed = []
            for index, item in enumerate(items, 1):
                names = ("text", "evidence_utterance_ids")
                if section == "minutes":
                    names += ("topic", "kind")
                if section == "action_items":
                    names = ("task", "owner", "deadline_text", "evidence_utterance_ids")
                fields(item, names)
                if not isinstance(item["evidence_utterance_ids"], list):
                    raise IntelligenceSchemaError("Evidence must be an array.")
                values = {
                    **item,
                    "id": f"{prefix}_{index:04d}",
                    "evidence_utterance_ids": tuple(item["evidence_utterance_ids"]),
                }
                if section == "action_items" and values["owner"] is not None:
                    fields(values["owner"], ("kind", "speaker_id", "display_text"))
                    values["owner"] = ActionOwner(**values["owner"])
                parsed.append(kind(**values))
            sections[section] = tuple(parsed)
        return validate_content(MeetingContent(**sections), request.utterances)
    except (IntelligenceValidationError, TypeError, ValueError) as exc:
        if isinstance(exc, IntelligenceSchemaError):
            raise
        raise IntelligenceSchemaError("Invalid typed extraction fields.") from exc


def content_identity(item):
    data = asdict(item)
    data.pop("id")
    data.pop("evidence_utterance_ids")
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def parse_consolidation(payload, request, *, max_items=128):
    """Selection-only consolidation cannot create new claims, owners or deadlines."""
    data = strict_json(payload)
    fields(data, SECTIONS)
    result = {}
    all_seen = set()
    for section in SECTIONS:
        available = {i.id: i for i in getattr(request.partial, section)}
        groups = data[section]
        if not isinstance(groups, list) or len(groups) > max_items:
            raise ConsolidationError("Invalid consolidation length.")
        merged = []
        for group in groups:
            fields(group, ("primary_item_id", "merge_item_ids"))
            primary = group["primary_item_id"]
            other = group["merge_item_ids"]
            if not isinstance(other, list) or any(not isinstance(i, str) for i in other):
                raise ConsolidationError("Invalid merge references.")
            ids = [primary, *other]
            if any(not isinstance(i, str) or i not in available for i in ids):
                raise ConsolidationError("Consolidation references an unknown partial item.")
            if len(set(ids)) != len(ids) or any(i in all_seen for i in ids):
                raise ConsolidationError("Consolidation reuses a partial item.")
            all_seen.update(ids)
            items = [available[i] for i in ids]
            if section == "action_items" and any(
                (i.owner, i.deadline_text) != (items[0].owner, items[0].deadline_text)
                for i in items
            ):
                raise ConsolidationError("Cannot merge conflicting owners/deadlines.")
            if section == "minutes" and any(i.kind != items[0].kind for i in items):
                raise ConsolidationError("Cannot merge conflicting minute classifications.")
            refs = tuple(dict.fromkeys(uid for i in items for uid in i.evidence_utterance_ids))
            merged.append(replace(items[0], evidence_utterance_ids=refs))
        result[section] = tuple(merged)
    return MeetingContent(**result)


def parse_response(payload, request, *, max_items=128):
    if request.stage == "consolidation":
        return parse_consolidation(payload, request, max_items=max_items)
    return parse_extraction(payload, request, max_items=max_items)


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def response_schema(request):
    if request.stage == "consolidation":
        return object_schema(
            {
                name: {
                    "type": "array",
                    "items": object_schema(
                        {
                            "primary_item_id": {
                                "type": "string",
                                "enum": [i.id for i in getattr(request.partial, name)],
                            }
                            if getattr(request.partial, name)
                            else {"type": "string"},
                            "merge_item_ids": {"type": "array", "items": {"type": "string"}},
                        }
                    ),
                }
                for name in SECTIONS
            }
        )
    evidence = {
        "type": "array",
        "items": {"type": "string", "enum": [u.utterance_id for u in request.utterances]},
    }
    base = {"text": {"type": "string"}, "evidence_utterance_ids": evidence}
    speakers = sorted({u.speaker_id for u in request.utterances if u.speaker_id})
    branches = [
        object_schema(
            {
                "kind": {"type": "string", "enum": ["named_entity"]},
                "speaker_id": {"type": "null"},
                "display_text": {"type": "string"},
            }
        ),
        {"type": "null"},
    ]
    if speakers:
        branches.insert(
            0,
            object_schema(
                {
                    "kind": {"type": "string", "enum": ["speaker"]},
                    "speaker_id": {"type": "string", "enum": speakers},
                    "display_text": {"type": "null"},
                }
            ),
        )
    owner = {"anyOf": branches}
    item_schemas = (
        object_schema(base),
        object_schema(
            {**base, "topic": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)}}
        ),
        object_schema(base),
        object_schema(
            {
                "task": {"type": "string"},
                "owner": owner,
                "deadline_text": {"type": ["string", "null"]},
                "evidence_utterance_ids": evidence,
            }
        ),
    )
    return object_schema(
        {name: {"type": "array", "items": schema} for name, schema in zip(SECTIONS, item_schemas)}
    )
