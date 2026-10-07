"""Deterministic source resolution and optional frame-exact PCM clips; no API calls."""

import hashlib
import math
import os
import wave
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from meeting_assistant.asr.audio import inspect_audio
from meeting_assistant.asr.config import ASRConfig

from .exceptions import EvidenceAudioError, IntelligenceSourceMismatch, UnknownEvidenceError
from .models import AudioReference, EvidenceClip, EvidenceSpan
from .service import refined_digest
from .validation import validate_content


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind_audio(path, speaker, *, diarization=None):
    try:
        audio = inspect_audio(Path(path), ASRConfig())
        sha = _sha(audio.path)
        if abs(audio.duration_seconds - speaker.duration_seconds) > 1 / 16000:
            raise EvidenceAudioError(
                "Canonical audio duration does not match transcript provenance."
            )
        if diarization is not None and (
            speaker.diarization_id != diarization.diarization_id or sha != diarization.audio_sha256
        ):
            raise EvidenceAudioError(
                "Audio digest/diarization identity does not match the transcript."
            )
        return AudioReference(sha, audio.duration_seconds, audio.frames)
    except EvidenceAudioError:
        raise
    except Exception as exc:
        raise EvidenceAudioError("Expected readable canonical 16-kHz mono PCM-16 WAV.") from exc


def validate_record_source(record, refined):
    if (
        record.source_refined_id,
        record.source_refined_sha256,
        record.source_speaker_id,
        record.source_speaker_sha256,
        record.source_grounding_id,
        record.source_grounding_sha256,
        record.source_raw_id,
        record.source_raw_sha256,
    ) != (
        refined.id,
        refined_digest(refined),
        refined.source_speaker_transcript_id,
        refined.source_speaker_sha256,
        refined.grounding_result_id,
        refined.grounding_sha256,
        refined.source_raw_transcript_id,
        refined.source_raw_sha256,
    ):
        raise IntelligenceSourceMismatch("Meeting record belongs to different refined evidence.")
    validate_content(record.content, refined.utterances)


def resolve_meeting_record_evidence(record, refined_transcript):
    validate_record_source(record, refined_transcript)
    by_id = {u.utterance_id: u for u in refined_transcript.utterances}
    return {
        item.id: tuple(
            EvidenceSpan(
                u.utterance_id,
                u.speaker_id,
                u.start,
                u.end,
                u.raw_text,
                u.refined_text,
                u.source_word_refs,
                u.applied_edit_ids,
                u.source_segment_ids,
            )
            for u in (by_id[uid] for uid in item.evidence_utterance_ids)
        )
        for item in record.content.items
    }


def get_item_evidence(record, item_id, refined_transcript):
    resolved = resolve_meeting_record_evidence(record, refined_transcript)
    if item_id not in resolved:
        raise UnknownEvidenceError("Unknown meeting item ID.")
    return resolved[item_id]


def evidence_manifest(record, refined):
    return {
        "schema_version": record.schema_version,
        "evidence_policy": record.evidence_policy_version,
        "meeting_record_id": record.id,
        "source_refined_id": record.source_refined_id,
        "source_refined_sha256": record.source_refined_sha256,
        "source_raw_id": record.source_raw_id,
        "source_raw_sha256": record.source_raw_sha256,
        "audio": asdict(record.audio) if record.audio else None,
        "items": {
            key: [asdict(span) for span in spans]
            for key, spans in resolve_meeting_record_evidence(record, refined).items()
        },
    }


def extract_evidence_clip(
    canonical_audio_path,
    evidence_span,
    destination_path,
    *,
    padding_seconds=0.0,
    audio_reference=None,
):
    """Floor/ceil at 16 kHz; stream frames, never overwrite source or existing destination."""
    temporary = None
    try:
        audio = inspect_audio(Path(canonical_audio_path), ASRConfig())
        for value in (evidence_span.start, evidence_span.end, padding_seconds):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not math.isfinite(value) or value < 0)
            ):
                raise EvidenceAudioError("Evidence/padding must be finite nonnegative seconds.")
        if (
            evidence_span.end <= evidence_span.start
            or evidence_span.start >= audio.duration_seconds
        ):
            raise EvidenceAudioError("Evidence requires a positive interval inside source audio.")
        sha = _sha(audio.path)
        if audio_reference is not None and (
            sha != audio_reference.sha256 or audio.frames != audio_reference.frame_count
        ):
            raise EvidenceAudioError("Canonical audio does not match record audio binding.")
        start = max(0, math.floor((evidence_span.start - padding_seconds) * 16000))
        end = min(audio.frames, math.ceil((evidence_span.end + padding_seconds) * 16000))
        if end <= start:
            raise EvidenceAudioError("Empty evidence interval after clamping.")
        destination = Path(destination_path)
        if destination.suffix.lower() != ".wav" or destination.exists() or destination.is_symlink():
            raise EvidenceAudioError("Choose a new WAV destination; overwrites are prohibited.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink() or (
            hasattr(destination.parent, "is_junction") and destination.parent.is_junction()
        ):
            raise EvidenceAudioError("Clip directory cannot be a link/junction.")
        destination = destination.resolve()
        if destination == audio.path:
            raise EvidenceAudioError("Cannot overwrite the canonical source.")
        temporary = destination.parent / (".pending-" + uuid4().hex + ".wav")
        # Exclusive creation; only clean this invocation's owned temporary artifact.
        with temporary.open("xb") as stream:
            with wave.open(str(audio.path), "rb") as source, wave.open(stream, "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16000)
                source.setpos(start)
                remaining = end - start
                while remaining:
                    amount = min(remaining, 32000)
                    block = source.readframes(amount)
                    if len(block) != amount * 2:
                        raise EvidenceAudioError("Canonical audio changed during clip extraction.")
                    target.writeframesraw(block)
                    remaining -= amount
            stream.flush()
            os.fsync(stream.fileno())
        checked = inspect_audio(temporary, ASRConfig())
        if checked.frames != end - start or _sha(audio.path) != sha:
            raise EvidenceAudioError("Clip verification/source preservation failed.")
        # Hard-link publication refuses races/overwrites on all supported desktop filesystems.
        os.link(temporary, destination)
        return EvidenceClip(
            destination,
            sha,
            evidence_span.start,
            evidence_span.end,
            start / 16000,
            end / 16000,
            start,
            end,
        )
    except EvidenceAudioError:
        raise
    except Exception as exc:
        raise EvidenceAudioError(
            "Cannot extract evidence; check canonical audio, path and disk access."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
