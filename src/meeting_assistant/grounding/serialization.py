"""Separate versioned evidence artifacts; never modify Phase II/III output."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import UUID, uuid4

from meeting_assistant.diarization.models import WordReference

from .exceptions import GroundingWriteError, InvalidGroundingResult
from .models import (
    GroundingArtifacts,
    GroundingCandidate,
    GroundingProcessingInfo,
    GroundingRecord,
    GroundingResult,
    RetrievalPolicy,
)


def grounding_to_json(result: GroundingResult) -> str:
    return (
        json.dumps(asdict(result), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def grounding_from_json(payload: str) -> GroundingResult:
    try:
        data = json.loads(payload)
        records = []
        for record in data["records"]:
            candidates = tuple(
                GroundingCandidate(**{**c, "reasons": tuple(c["reasons"])})
                for c in record["candidates"]
            )
            records.append(
                GroundingRecord(
                    **{
                        **record,
                        "candidates": candidates,
                        "source_word_references": tuple(
                            WordReference(**r) for r in record["source_word_references"]
                        ),
                        "source_segment_ids": tuple(record["source_segment_ids"]),
                    }
                )
            )
        return GroundingResult(
            **{
                **data,
                "records": tuple(records),
                "processing_info": GroundingProcessingInfo(**data["processing_info"]),
                "retrieval_policy": RetrievalPolicy(**data["retrieval_policy"]),
            }
        )
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise InvalidGroundingResult("Malformed grounding JSON.") from exc


def render_grounding_text(result: GroundingResult) -> str:
    lines = [
        "DOMAIN GROUNDING CANDIDATES — no transcript edits",
        f"Glossary: {result.glossary_version}",
        "",
    ]
    for record in result.records:
        lines.extend(
            [
                f"{record.id} {record.utterance_id} {record.speaker_id or 'UNKNOWN_SPEAKER'} [{record.start:.3f}–{record.end:.3f}]",
                f"Observed: {record.observed_text}",
                f"Context: {record.context}",
            ]
        )
        for index, candidate in enumerate(record.candidates, 1):
            lines.append(
                f"  {index}. {candidate.canonical} ({candidate.scope}; {candidate.entry_id}) score={candidate.score:.3f} lex={candidate.lexical_score:.3f} phon={candidate.phonetic_score:.3f} semantic={candidate.semantic_score:.3f} {candidate.match_type}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def save_grounding(
    result: GroundingResult, output_dir: Path | str = "transcripts"
) -> GroundingArtifacts:
    names = ("grounding.json", "grounding.txt")
    try:
        root = Path(output_dir)
        if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
            raise GroundingWriteError("Output directory cannot be a link or junction.")
        root.mkdir(parents=True, exist_ok=True)
        destination = root.resolve() / str(UUID(result.grounding_id))
        if destination.exists():
            raise GroundingWriteError("Grounding evidence already exists; cannot overwrite.")
        staging = root.resolve() / (".pending-" + uuid4().hex)
        staging.mkdir()
        try:
            for name, text in zip(
                names, (grounding_to_json(result), render_grounding_text(result))
            ):
                with (staging / name).open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.rename(staging, destination)
        finally:
            for name in names:
                (staging / name).unlink(missing_ok=True)
            if staging.exists():
                staging.rmdir()
        return GroundingArtifacts(*(destination / name for name in names))
    except OSError as exc:
        raise GroundingWriteError(
            "Cannot publish grounding evidence; check access and disk space."
        ) from exc
