"""A single canonical record drives JSON, Markdown and its bound evidence manifest."""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from uuid import UUID, uuid4

from meeting_assistant.refinement.serialization import _timestamp

from .evidence import evidence_manifest, resolve_meeting_record_evidence
from .exceptions import IntelligenceValidationError, IntelligenceWriteError
from .models import (
    SECTIONS,
    ActionItem,
    ActionOwner,
    AudioReference,
    Decision,
    IntelligenceCall,
    IntelligenceModelInfo,
    IntelligenceProcessingInfo,
    MeetingArtifacts,
    MeetingContent,
    MeetingRecord,
    MinuteItem,
    SummaryPoint,
)
from .validation import fields, strict_json

logger = logging.getLogger(__name__)


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def meeting_record_to_json(record):
    return _json(asdict(record))


def meeting_record_from_json(payload):
    try:
        data = strict_json(payload, max_chars=128 * 1024**2)
        fields(data, MeetingRecord.__dataclass_fields__)
        content = data["content"]
        fields(content, SECTIONS)
        values = {}
        for section, kind in zip(SECTIONS, (SummaryPoint, MinuteItem, Decision, ActionItem)):
            if not isinstance(content[section], list):
                raise ValueError
            parsed = []
            for item in content[section]:
                fields(item, kind.__dataclass_fields__)
                if not isinstance(item["evidence_utterance_ids"], list):
                    raise ValueError
                item = {**item, "evidence_utterance_ids": tuple(item["evidence_utterance_ids"])}
                if section == "action_items" and item["owner"] is not None:
                    fields(item["owner"], ActionOwner.__dataclass_fields__)
                    item["owner"] = ActionOwner(**item["owner"])
                parsed.append(kind(**item))
            values[section] = tuple(parsed)
        data["content"] = MeetingContent(**values)
        data["model_info"] = IntelligenceModelInfo(**data["model_info"])
        info = data["processing_info"]
        data["processing_info"] = IntelligenceProcessingInfo(
            **{**info, "calls": tuple(IntelligenceCall(**call) for call in info["calls"])}
        )
        data["audio"] = AudioReference(**data["audio"]) if data["audio"] is not None else None
        return MeetingRecord(**data)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise IntelligenceValidationError("Malformed meeting record JSON.") from exc


def _escape(value):
    # Untrusted speech/claims are rendered as text, never HTML or active Markdown links.
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for char in "\\`*_[]()#!|":
        value = value.replace(char, "\\" + char)
    return value.replace("\r", " ").replace("\n", " ")


def render_meeting_markdown(record, refined):
    resolved = resolve_meeting_record_evidence(record, refined)
    lines = [
        "# Meeting record",
        "",
        "Raw-derived meeting intelligence; semantic verification pending.",
        "",
    ]
    for section, heading in zip(
        SECTIONS, ("Meeting summary", "Meeting minutes", "Decisions", "Action items")
    ):
        lines.extend(("## " + heading, ""))
        if not getattr(record.content, section):
            lines.extend(("None recorded.", ""))
        for item in getattr(record.content, section):
            label = f"[{item.topic} / {item.kind}] " if isinstance(item, MinuteItem) else ""
            lines.append(f"- {item.id}: {_escape(label + item.text)}")
            if isinstance(item, ActionItem):
                owner = (
                    "unspecified"
                    if item.owner is None
                    else (
                        item.owner.speaker_id
                        if item.owner.kind == "speaker"
                        else item.owner.display_text
                    )
                )
                lines.append(
                    f"  Owner: {_escape(owner)}; deadline: {_escape(item.deadline_text or 'unspecified')}"
                )
            for span in resolved[item.id]:
                lines.append(
                    f"  Evidence: {span.utterance_id} / {span.speaker_id or 'UNKNOWN_SPEAKER'} / "
                    f"{_timestamp(span.start)}–{_timestamp(span.end)}"
                )
                lines.append(f"  Source: {_escape(span.refined_text)}")
            lines.append("")
    return "\n".join(lines) + "\n"


def save_meeting_record(record, refined_transcript, output_dir="transcripts"):
    # Resolve all evidence before creating staging artifacts.
    payloads = (
        meeting_record_to_json(record),
        render_meeting_markdown(record, refined_transcript),
        _json(evidence_manifest(record, refined_transcript)),
    )
    names = ("meeting_record.json", "meeting_record.md", "evidence_manifest.json")
    try:
        root = Path(output_dir)
        if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
            raise IntelligenceWriteError("Output directory cannot be a link/junction.")
        root.mkdir(parents=True, exist_ok=True)
        destination = root.resolve() / str(UUID(record.id))
        if destination.exists():
            raise IntelligenceWriteError(
                "Meeting record already exists; overwrites are prohibited."
            )
        staging = root.resolve() / (".pending-" + uuid4().hex)
        staging.mkdir()
        try:
            for name, payload in zip(names, payloads):
                with (staging / name).open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.rename(staging, destination)
        finally:
            for name in names:
                (staging / name).unlink(missing_ok=True)
            if staging.exists():
                staging.rmdir()
        logger.info("meeting_record_published", extra={"meeting_record_id": record.id})
        return MeetingArtifacts(*(destination / name for name in names))
    except OSError as exc:
        raise IntelligenceWriteError(
            "Cannot publish meeting record; check access and disk space."
        ) from exc
