"""Atomically publish a new immutable three-file bundle; never overwrite evidence."""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from uuid import UUID, uuid4

from meeting_assistant.diarization.models import WordReference

from .exceptions import RefinementValidationError, RefinementWriteError
from .models import (
    ProviderCall,
    RefinedTranscript,
    RefinedUtterance,
    RefinementArtifacts,
    RefinementProcessingInfo,
    RefinerModelInfo,
    TranscriptEdit,
)

logger = logging.getLogger(__name__)


def refined_to_json(result: RefinedTranscript) -> str:
    return (
        json.dumps(asdict(result), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def refined_from_json(payload: str) -> RefinedTranscript:
    try:
        data = json.loads(payload)
        utterances = tuple(
            RefinedUtterance(
                **{
                    **u,
                    "source_word_refs": tuple(WordReference(**r) for r in u["source_word_refs"]),
                    **{
                        key: tuple(u[key])
                        for key in (
                            "applied_edit_ids",
                            "uncertain_record_ids",
                            "source_segment_ids",
                        )
                    },
                }
            )
            for u in data["utterances"]
        )
        edits = tuple(
            TranscriptEdit(
                **{
                    **e,
                    "source_word_refs": tuple(WordReference(**r) for r in e["source_word_refs"]),
                }
            )
            for e in data["edit_log"]
        )
        info = data["processing_info"]
        return RefinedTranscript(
            **{
                **data,
                "utterances": utterances,
                "edit_log": edits,
                "model_info": RefinerModelInfo(**data["model_info"]),
                "processing_info": RefinementProcessingInfo(
                    **{**info, "calls": tuple(ProviderCall(**c) for c in info["calls"])}
                ),
            }
        )
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise RefinementValidationError("Malformed refined transcript JSON.") from exc


def _timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def render_refined_text(result: RefinedTranscript) -> str:
    return "".join(
        f"[{_timestamp(u.start)} --> {_timestamp(u.end)}] {u.speaker_id or 'UNKNOWN_SPEAKER'}: {u.refined_text}\n"
        for u in result.utterances
    )


def edit_log_to_json(result: RefinedTranscript) -> str:
    data = {
        "schema_version": result.schema_version,
        "refined_transcript_id": result.id,
        "source_speaker_sha256": result.source_speaker_sha256,
        "grounding_sha256": result.grounding_sha256,
        "policy_version": result.policy_version,
        "edits": [asdict(e) for e in result.edit_log],
    }
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def save_refined_transcript(
    result: RefinedTranscript, output_dir: Path | str = "transcripts"
) -> RefinementArtifacts:
    names = ("refined_transcript.json", "refined_transcript.txt", "edit_log.json")
    try:
        root = Path(output_dir)
        if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
            raise RefinementWriteError("Output directory cannot be a link or junction.")
        root.mkdir(parents=True, exist_ok=True)
        destination = root.resolve() / str(UUID(result.id))
        if destination.exists():
            raise RefinementWriteError("Refinement already exists; cannot overwrite.")
        staging = root.resolve() / (".pending-" + uuid4().hex)
        staging.mkdir()
        try:
            for name, text in zip(
                names,
                (refined_to_json(result), render_refined_text(result), edit_log_to_json(result)),
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
        logger.info(
            "refinement_published",
            extra={"event": "refinement_published", "refined_transcript_id": result.id},
        )
        return RefinementArtifacts(*(destination / name for name in names))
    except OSError as exc:
        raise RefinementWriteError(
            "Cannot publish refinement; check access and disk space."
        ) from exc
