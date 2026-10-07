"""Versioned speaker evidence; atomic publication without touching raw transcripts."""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from uuid import UUID, uuid4

from meeting_assistant.asr.exceptions import InvalidTranscript
from meeting_assistant.asr.models import TranscriptWord

from .config import DiarizationOptions, ReconciliationConfig
from .exceptions import InvalidDiarizationResult, SpeakerTranscriptWriteError
from .models import (
    DiarizationModelInfo,
    DiarizationProcessingInfo,
    DiarizationResult,
    OverlapRegion,
    ReconciliationInfo,
    SpeakerAwareTranscript,
    SpeakerTranscriptArtifacts,
    SpeakerTurn,
    SpeakerUtterance,
    SpeakerWord,
    WordReference,
)

logger = logging.getLogger(__name__)


def diarization_to_json(result: DiarizationResult) -> str:
    return (
        json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def speaker_transcript_to_json(result: SpeakerAwareTranscript) -> str:
    return (
        json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def diarization_from_json(content: str) -> DiarizationResult:
    try:
        data = json.loads(content)
        data["speakers"] = tuple(data["speakers"])
        for key in ("regular_turns", "exclusive_turns"):
            data[key] = tuple(SpeakerTurn(**turn) for turn in data[key])
        data["model_info"] = DiarizationModelInfo(**data["model_info"])
        data["processing_info"] = DiarizationProcessingInfo(**data["processing_info"])
        data["options"] = DiarizationOptions(**data["options"])
        return DiarizationResult(**data)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise InvalidDiarizationResult("Saved diarization JSON is malformed.") from exc


def speaker_transcript_from_json(content: str) -> SpeakerAwareTranscript:
    try:
        data = json.loads(content)
        data["speakers"] = tuple(data["speakers"])
        data["overlap_regions"] = tuple(
            OverlapRegion(**{**region, "speakers": tuple(region["speakers"])})
            for region in data["overlap_regions"]
        )
        utterances = []
        for utterance in data["utterances"]:
            words = tuple(
                SpeakerWord(
                    **{
                        **word,
                        "word": TranscriptWord(**word["word"]),
                        "source_word_reference": WordReference(**word["source_word_reference"]),
                        "source_turn_ids": tuple(word["source_turn_ids"]),
                    }
                )
                for word in utterance["words"]
            )
            utterances.append(
                SpeakerUtterance(
                    **{
                        **utterance,
                        "words": words,
                        "source_segment_ids": tuple(utterance["source_segment_ids"]),
                    }
                )
            )
        data["utterances"] = tuple(utterances)
        info = data["reconciliation_info"]
        data["reconciliation_info"] = ReconciliationInfo(
            **{**info, "config": ReconciliationConfig(**info["config"])}
        )
        return SpeakerAwareTranscript(**data)
    except (ValueError, TypeError, KeyError, AttributeError, InvalidTranscript) as exc:
        raise InvalidDiarizationResult("Saved speaker transcript JSON is malformed.") from exc


def _timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, fraction = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{fraction:03d}"


def render_speaker_transcript_text(result: SpeakerAwareTranscript) -> str:
    return "".join(
        f"[{_timestamp(u.start)} --> {_timestamp(u.end)}] "
        f"{u.speaker_id or 'UNKNOWN_SPEAKER'}:\n{u.text}\n\n"
        for u in result.utterances
    )


def save_speaker_transcript(
    result: SpeakerAwareTranscript,
    diarization: DiarizationResult,
    output_dir: Path | str = "transcripts",
) -> SpeakerTranscriptArtifacts:
    """Publish a new UUID bundle of three files. Existing bundles cannot be replaced."""
    if (
        result.diarization_id != diarization.diarization_id
        or result.speakers != diarization.speakers
    ):
        raise InvalidDiarizationResult("Speaker transcript must refer to the supplied diarization.")
    names = ("diarization.json", "speaker_transcript.json", "speaker_transcript.txt")
    try:
        root = Path(output_dir)
        if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
            raise SpeakerTranscriptWriteError("Output directory cannot be a link or junction.")
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
        destination = root / str(UUID(result.transcript_id))
        if destination.exists():
            raise SpeakerTranscriptWriteError("Speaker evidence already exists; cannot overwrite.")
        staging = root / f".pending-{uuid4().hex}"
        staging.mkdir()
        try:
            contents = (
                diarization_to_json(diarization),
                speaker_transcript_to_json(result),
                render_speaker_transcript_text(result),
            )
            for name, content in zip(names, contents):
                with (staging / name).open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.rename(staging, destination)
        finally:
            for name in names:
                (staging / name).unlink(missing_ok=True)
            if staging.exists():
                staging.rmdir()
        logger.info(
            "speaker_transcript_published",
            extra={"event": "speaker_transcript_published", "transcript_id": result.transcript_id},
        )
        return SpeakerTranscriptArtifacts(*(destination / name for name in names))
    except OSError as exc:
        raise SpeakerTranscriptWriteError(
            "Cannot publish speaker evidence; check directory access and disk space."
        ) from exc
