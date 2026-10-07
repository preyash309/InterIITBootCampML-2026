"""Public transcription entry point. Inject a future local backend unchanged."""

from pathlib import Path

from .api_backend import WhisperAPIBackend
from .base import ASRBackend
from .config import ASRConfig, TranscriptionOptions
from .exceptions import ASRConfigurationError, InvalidTranscript
from .models import TranscriptResult


def transcribe_audio(
    audio_path: Path | str,
    *,
    backend: ASRBackend | None = None,
    config: ASRConfig | None = None,
    options: TranscriptionOptions | None = None,
) -> TranscriptResult:
    if backend is not None and config is not None:
        raise ASRConfigurationError("Supply backend or config, not both.")
    active = backend if backend is not None else WhisperAPIBackend(config)
    result = active.transcribe(Path(audio_path), options)
    if not isinstance(result, TranscriptResult):
        raise InvalidTranscript("ASR backend must return a typed TranscriptResult.")
    return result
