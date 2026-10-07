"""API-backed raw ASR with a replaceable backend; no downstream intelligence."""

import logging

from .api_backend import WhisperAPIBackend
from .base import ASRBackend
from .config import ASRConfig, TranscriptionOptions
from .exceptions import ASRError
from .models import (
    ModelInfo,
    ProcessingInfo,
    TranscriptArtifacts,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)
from .serialization import (
    render_transcript_text,
    save_transcript,
    transcript_from_json,
    transcript_to_json,
)
from .service import transcribe_audio

__all__ = [
    "ASRBackend",
    "ASRConfig",
    "ASRError",
    "ModelInfo",
    "ProcessingInfo",
    "TranscriptArtifacts",
    "TranscriptResult",
    "TranscriptSegment",
    "TranscriptWord",
    "TranscriptionOptions",
    "WhisperAPIBackend",
    "render_transcript_text",
    "save_transcript",
    "transcribe_audio",
    "transcript_from_json",
    "transcript_to_json",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
