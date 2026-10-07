"""Phase III models; local inference implementation is loaded lazily."""

from .base import DiarizationBackend
from .cache import prepare_model
from .config import DiarizationConfig, DiarizationOptions, ReconciliationConfig
from .exceptions import DiarizationError
from .models import DiarizationResult, SpeakerAwareTranscript, SpeakerUtterance, SpeakerWord
from .pyannote_backend import PyannoteBackend
from .reconciliation import reconcile_transcript
from .serialization import (
    diarization_from_json,
    diarization_to_json,
    render_speaker_transcript_text,
    save_speaker_transcript,
    speaker_transcript_from_json,
    speaker_transcript_to_json,
)
from .service import diarize_audio, diarize_transcript

__all__ = [
    "DiarizationBackend",
    "DiarizationConfig",
    "DiarizationOptions",
    "ReconciliationConfig",
    "DiarizationError",
    "DiarizationResult",
    "SpeakerAwareTranscript",
    "SpeakerUtterance",
    "SpeakerWord",
    "PyannoteBackend",
    "prepare_model",
    "diarize_audio",
    "diarize_transcript",
    "reconcile_transcript",
    "diarization_from_json",
    "diarization_to_json",
    "render_speaker_transcript_text",
    "save_speaker_transcript",
    "speaker_transcript_from_json",
    "speaker_transcript_to_json",
]
