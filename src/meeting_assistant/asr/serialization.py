"""Versioned UTF-8 JSON and raw timestamped text, published together atomically."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import UUID, uuid4

from .exceptions import InvalidTranscript, TranscriptWriteError
from .models import (
    ModelInfo,
    ProcessingInfo,
    TranscriptArtifacts,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)


def transcript_to_json(result: TranscriptResult) -> str:
    return (
        json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, fraction = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{fraction:03d}"


def render_transcript_text(result: TranscriptResult) -> str:
    # Segment text is preserved verbatim, including leading whitespace and punctuation.
    return "".join(
        f"[{_timestamp(segment.start)} --> {_timestamp(segment.end)}] {segment.text}\n"
        for segment in result.segments
    )


def transcript_from_json(content: str) -> TranscriptResult:
    try:
        raw = json.loads(content)
        if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
            raise InvalidTranscript("Unsupported transcript schema version.")
        raw["segments"] = tuple(
            TranscriptSegment(
                **{**segment, "words": tuple(TranscriptWord(**word) for word in segment["words"])}
            )
            for segment in raw["segments"]
        )
        raw["model_info"] = ModelInfo(**raw["model_info"])
        raw["processing_info"] = ProcessingInfo(**raw["processing_info"])
        return TranscriptResult(**raw)
    except (TypeError, ValueError, KeyError) as exc:
        raise InvalidTranscript("Saved transcript JSON is malformed.") from exc


def save_transcript(
    result: TranscriptResult, output_dir: Path | str = "transcripts"
) -> TranscriptArtifacts:
    """Publish a new UUID directory; never overwrite an earlier raw transcript."""
    try:
        name = str(UUID(result.transcript_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise InvalidTranscript("Transcript publication requires a UUID transcript ID.") from exc
    try:
        root = Path(output_dir)
        if root.is_symlink():
            raise TranscriptWriteError("Transcript output directory must not be a symbolic link.")
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
        destination = root / name
        if destination.exists():
            raise TranscriptWriteError(
                "Transcript already exists; raw outputs cannot be overwritten."
            )
        # A normal directory inherits workspace permissions. Python 3.14 Windows
        # TemporaryDirectory uses a restrictive ACL that would survive publication.
        staging = root / f".pending-{uuid4().hex}"
        staging.mkdir()
        try:
            for filename, content in (
                ("raw_transcript.json", transcript_to_json(result)),
                ("raw_transcript.txt", render_transcript_text(result)),
            ):
                with (staging / filename).open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.rename(staging, destination)
        finally:
            # Only these two files can be created here; no recursive deletion.
            for filename in ("raw_transcript.json", "raw_transcript.txt"):
                (staging / filename).unlink(missing_ok=True)
            if staging.exists():
                staging.rmdir()
        return TranscriptArtifacts(
            destination / "raw_transcript.json", destination / "raw_transcript.txt"
        )
    except OSError as exc:
        raise TranscriptWriteError(
            "Cannot publish transcript files; check directory access and disk space."
        ) from exc
