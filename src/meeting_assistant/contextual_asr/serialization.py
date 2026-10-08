"""Versioned typed JSON and atomic new UUID bundles, never overwrite raw ASR."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from meeting_assistant.asr.models import TranscriptSegment, TranscriptWord

from .config import ContextualASRConfig
from .exceptions import ContextualASRError, InvalidContext, InvalidContextualEvidence
from .models import (
    AudioWindow,
    CandidateLink,
    ContextSource,
    ContextTerm,
    ContextualASRHypothesis,
    ContextualASRResult,
    MeetingContextPack,
    RetrievedTerm,
    SkippedSpan,
    SuspiciousSpan,
)


def to_json(value):
    return (
        json.dumps(asdict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )


context_to_json = to_json
contextual_asr_to_json = to_json


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidContextualEvidence("Duplicate JSON key.")
        result[key] = value
    return result


def _load(payload):
    if not isinstance(payload, str) or len(payload) > 2_000_000:
        raise InvalidContextualEvidence("Saved context/evidence exceeds safe bounds.")
    return json.loads(payload, object_pairs_hook=_pairs)


def context_from_json(payload):
    try:
        data = _load(payload)
        data["sources"] = tuple(ContextSource(**s) for s in data["sources"])
        for name in ("explicit_terms", "extracted_terms"):
            data[name] = tuple(
                ContextTerm(
                    **{**t, "aliases": tuple(t["aliases"]), "source_ids": tuple(t["source_ids"])}
                )
                for t in data[name]
            )
        for name in ("agenda", "participant_names"):
            data[name] = tuple(data[name])
        return MeetingContextPack(**data)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise InvalidContext("Malformed meeting context JSON.") from exc


def contextual_asr_from_json(payload):
    try:
        data = _load(payload)
        data["config"] = ContextualASRConfig(**data["config"])
        data["suspicious_spans"] = tuple(
            SuspiciousSpan(**{**s, "signals": tuple(s["signals"])})
            for s in data["suspicious_spans"]
        )
        data["skipped"] = tuple(SkippedSpan(**s) for s in data["skipped"])
        hypotheses = []
        for item in data["hypotheses"]:
            item["window"] = AudioWindow(
                **{**item["window"], "grounding_ids": tuple(item["window"]["grounding_ids"])}
            )
            item["terms"] = tuple(
                RetrievedTerm(**{**t, "context_source_ids": tuple(t["context_source_ids"])})
                for t in item["terms"]
            )
            item["segments"] = tuple(
                TranscriptSegment(**{**s, "words": tuple(TranscriptWord(**w) for w in s["words"])})
                for s in item["segments"]
            )
            item["candidate_links"] = tuple(
                CandidateLink(**link) for link in item["candidate_links"]
            )
            item["conflicts"] = tuple(item["conflicts"])
            hypotheses.append(ContextualASRHypothesis(**item))
        data["hypotheses"] = tuple(hypotheses)
        return ContextualASRResult(**data)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise InvalidContextualEvidence("Malformed contextual ASR JSON.") from exc


def read_context(path):
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            return context_from_json(stream.read(2_000_001))
    except OSError as exc:
        raise InvalidContext("Cannot read meeting context pack.") from exc


def render_contextual_text(result):
    lines = [
        "CONTEXTUAL ASR HYPOTHESES — not transcript corrections",
        f"Calls: {result.provider_calls}; audio: {result.audio_seconds:.3f} s",
    ]
    for hypothesis in result.hypotheses:
        lines.extend(
            [
                f"{hypothesis.id} [{hypothesis.window.start:.3f} --> {hypothesis.window.end:.3f}] {hypothesis.status}",
                "Pass 1: " + hypothesis.pass1_text,
                "Pass 2: " + hypothesis.pass2_text,
                "Vocabulary: " + ", ".join(t.canonical for t in hypothesis.terms),
                "Candidate IDs: "
                + ", ".join(link.candidate_entry_id for link in hypothesis.candidate_links),
                "Conflicts: " + ", ".join(hypothesis.conflicts),
            ]
        )
    lines.extend(f"Skipped {s.grounding_id}: {s.reason}" for s in result.skipped)
    return "\n".join(lines) + "\n"


def save_contextual_asr(result, output_dir="transcripts", *, context=None, refined=None):
    """One atomic bundle; optional Phase V decision links are separate audit metadata."""
    payloads = {
        "contextual_asr.json": to_json(result),
        "contextual_asr.txt": render_contextual_text(result),
    }
    if context is not None:
        from .context import sha256

        if sha256(to_json(context)) != result.context_sha256:
            raise InvalidContextualEvidence("Context pack does not match ASR evidence.")
        payloads["meeting_context.json"] = to_json(context)
    if refined is not None:
        if (
            refined.grounding_result_id != result.source_grounding_id
            or refined.grounding_sha256 != result.source_grounding_sha256
        ):
            raise InvalidContextualEvidence("Refinement does not match contextual grounding.")
        payloads["contextual_refinement_links.json"] = (
            json.dumps(
                {
                    "contextual_result_id": result.result_id,
                    "refined_transcript_id": refined.id,
                    "links": [
                        {
                            "hypothesis_id": h.id,
                            "edit_id": e.id,
                            "grounding_record_id": e.grounding_record_id,
                            "candidate_entry_id": link.candidate_entry_id,
                            "decision_candidate_entry_id": e.candidate_entry_id,
                            "action": e.action,
                            "validation_status": e.validation_status,
                            "rejection_reason": e.rejection_reason,
                        }
                        for h in result.hypotheses
                        for link in h.candidate_links
                        for e in refined.edit_log
                        if e.grounding_record_id == link.grounding_id
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
    root = Path(output_dir) / "contextual_asr"
    try:
        if any(
            p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
            for p in (root, *root.parents)
        ):
            raise ContextualASRError("Contextual output cannot traverse links/junctions.")
        root.mkdir(parents=True, exist_ok=True)
        destination = root / result.result_id
        if destination.exists():
            raise ContextualASRError("Contextual bundle already exists; cannot overwrite.")
        staging = root / (".pending-" + uuid4().hex)
        staging.mkdir()
        try:
            for name, payload in payloads.items():
                with (staging / name).open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.rename(staging, destination)
        finally:
            for name in payloads:
                (staging / name).unlink(missing_ok=True)
            if staging.exists():
                staging.rmdir()
        return destination
    except OSError as exc:
        raise ContextualASRError(
            "Cannot publish contextual artifacts; check access/disk space."
        ) from exc
