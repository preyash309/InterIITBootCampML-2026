"""Public orchestration; backend injection supports explicit reuse and tests."""

from functools import lru_cache
from pathlib import Path
from threading import Lock

from meeting_assistant.asr.models import TranscriptResult

from .base import DiarizationBackend
from .config import DiarizationConfig, DiarizationOptions, ReconciliationConfig
from .exceptions import InvalidDiarizationAudio, InvalidDiarizationResult, ReconciliationError
from .models import DiarizationResult, SpeakerAwareTranscript
from .pyannote_backend import PyannoteBackend
from .reconciliation import reconcile_transcript

_backend_factory_lock = Lock()


@lru_cache(maxsize=1)
def _default_backend(config: DiarizationConfig) -> PyannoteBackend:
    return PyannoteBackend(config)


def diarize_audio(
    audio_path: Path | str,
    *,
    backend: DiarizationBackend | None = None,
    config: DiarizationConfig | None = None,
    options: DiarizationOptions | None = None,
) -> DiarizationResult:
    if backend is not None and config is not None:
        raise ReconciliationError("Supply backend or configuration, not both.")
    if backend is None:
        with _backend_factory_lock:
            backend = _default_backend(
                config if config is not None else DiarizationConfig.from_env()
            )
    try:
        path = Path(audio_path)
    except (TypeError, ValueError) as exc:
        raise InvalidDiarizationAudio(
            "Diarization requires a canonical audio filesystem path."
        ) from exc
    result = backend.diarize(path, options)
    if not isinstance(result, DiarizationResult):
        raise InvalidDiarizationResult("Diarization backend returned an untyped result.")
    return result


def diarize_transcript(
    audio_path: Path | str,
    transcript: TranscriptResult,
    *,
    backend: DiarizationBackend | None = None,
    config: DiarizationConfig | None = None,
    options: DiarizationOptions | None = None,
    reconciliation_config: ReconciliationConfig | None = None,
) -> SpeakerAwareTranscript:
    if not isinstance(transcript, TranscriptResult):
        raise ReconciliationError("Expected the immutable Phase II TranscriptResult.")
    config = config if config is not None or backend is not None else DiarizationConfig.from_env()
    result = diarize_audio(audio_path, backend=backend, config=config, options=options)
    return reconcile_transcript(
        transcript,
        result,
        config=reconciliation_config
        if reconciliation_config is not None
        else (
            config.reconciliation
            if config is not None
            else getattr(getattr(backend, "config", None), "reconciliation", None)
        ),
    )
