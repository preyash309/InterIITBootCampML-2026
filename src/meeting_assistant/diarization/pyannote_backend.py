"""Reusable, serialized local Community-1 inference; no silent device fallback."""

import importlib.metadata
import logging
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audio import ensure_unchanged, inspect_audio, load_waveform
from .cache import validate_model_cache
from .config import DiarizationConfig, DiarizationOptions
from .exceptions import (
    DiarizationDeviceError,
    DiarizationError,
    DiarizationInferenceError,
    DiarizationModelLoadError,
    DiarizationOutOfMemory,
    InvalidDiarizationResult,
)
from .models import (
    DiarizationModelInfo,
    DiarizationProcessingInfo,
    DiarizationResult,
    SpeakerTurn,
)

logger = logging.getLogger(__name__)


def _load_runtime():
    os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import constants as hub_constants

    hub_constants.HF_HUB_OFFLINE = True
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="\ntorchcodec is not installed correctly.*")
        from pyannote.audio import Pipeline
    return torch, Pipeline, importlib.metadata.version("pyannote.audio")


def translate_error(exc: Exception, *, loading: bool) -> DiarizationError:
    message = str(exc).lower()
    if "out of memory" in message or isinstance(exc, MemoryError):
        return DiarizationOutOfMemory(
            "Insufficient memory for diarization; close other GPU jobs or choose explicit CPU."
        )
    if any(token in message for token in ("cuda", "cudnn", "cublas", "device ordinal")):
        return DiarizationDeviceError(
            "CUDA diarization failed; check the GPU/runtime or explicitly choose cpu."
        )
    if loading:
        return DiarizationModelLoadError(
            "Cannot initialize local Community-1; check dependencies and model cache."
        )
    return DiarizationInferenceError(
        "Local diarization failed; check canonical audio and runtime compatibility."
    )


def convert_turns(
    output: Any,
) -> tuple[tuple[str, ...], tuple[SpeakerTurn, ...], tuple[SpeakerTurn, ...]]:
    try:

        def collect(annotation):
            return sorted(
                (
                    (float(turn.start), float(turn.end), str(label))
                    for turn, _, label in annotation.itertracks(yield_label=True)
                ),
                key=lambda item: (item[0], item[1], item[2]),
            )

        regular = collect(output.speaker_diarization)
        exclusive = collect(output.exclusive_speaker_diarization)
        names = dict.fromkeys(label for _, _, label in sorted((*regular, *exclusive)))
        labels = {name: f"SPEAKER_{index:02d}" for index, name in enumerate(names)}

        def typed(turns, kind):
            return tuple(
                SpeakerTurn(f"{kind}_turn_{index:06d}", labels[label], start, end)
                for index, (start, end, label) in enumerate(turns, 1)
            )

        return tuple(labels.values()), typed(regular, "regular"), typed(exclusive, "exclusive")
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        raise InvalidDiarizationResult(
            "Community-1 returned malformed regular/exclusive turns."
        ) from exc


class PyannoteBackend:
    def __init__(self, config: DiarizationConfig | None = None):
        self.config = config if config is not None else DiarizationConfig.from_env()
        if not isinstance(self.config, DiarizationConfig):
            raise DiarizationModelLoadError("Backend requires typed diarization configuration.")
        self._lock = threading.RLock()
        self._pipeline = None
        self._torch = None
        self._model_info = None

    @property
    def model_info(self) -> DiarizationModelInfo | None:
        with self._lock:
            return self._model_info

    def load_model(self) -> float:
        with self._lock:
            if self._pipeline is not None:
                return 0.0
            cache = validate_model_cache(self.config)
            start = time.perf_counter()
            try:
                torch, pipeline_class, version = _load_runtime()
                if self.config.device == "cuda" and not torch.cuda.is_available():
                    raise DiarizationDeviceError(
                        "CUDA is unavailable; select cpu explicitly to use the fallback."
                    )
                pipeline = pipeline_class.from_pretrained(cache, token=False)
                if pipeline is None:
                    raise DiarizationModelLoadError(
                        "Local Community-1 pipeline did not initialize."
                    )
                pipeline.to(torch.device(self.config.device))
                if self.config.device == "cuda":
                    torch.cuda.synchronize()
                info = DiarizationModelInfo(
                    self.config.model, self.config.revision, version, self.config.device
                )
            except DiarizationError:
                raise
            except Exception as exc:
                logger.error(
                    "model_initialization_failed",
                    extra={
                        "event": "model_initialization_failed",
                        "exception_type": type(exc).__name__,
                        "device": self.config.device,
                    },
                )
                raise translate_error(exc, loading=True) from exc
            self._pipeline, self._torch, self._model_info = pipeline, torch, info
            elapsed = time.perf_counter() - start
            logger.info(
                "model_loaded",
                extra={
                    "event": "model_loaded",
                    "device": info.device,
                    "model_load_seconds": elapsed,
                },
            )
            return elapsed

    def diarize(
        self, audio_path: Path | str, options: DiarizationOptions | None = None
    ) -> DiarizationResult:
        options = options if options is not None else self.config.options
        if not isinstance(options, DiarizationOptions):
            raise InvalidDiarizationResult("Diarization options must be typed.")
        with self._lock:
            started = time.perf_counter()
            audio = inspect_audio(audio_path, self.config.max_duration_seconds)
            logger.info(
                "diarization_started",
                extra={"event": "diarization_started", "duration_seconds": audio.duration_seconds},
            )
            load_seconds = self.load_model()
            try:
                torch = self._torch
                if self.config.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                waveform = load_waveform(audio, torch)
                inference_start = time.perf_counter()
                with torch.inference_mode():
                    output = self._pipeline(
                        {"waveform": waveform, "sample_rate": 16000}, **options.pipeline_arguments()
                    )
                if self.config.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - inference_start
                ensure_unchanged(audio)
                speakers, regular, exclusive = convert_turns(output)
                peak = (
                    int(torch.cuda.max_memory_allocated()) if self.config.device == "cuda" else None
                )
                result = DiarizationResult(
                    str(uuid4()),
                    audio.duration_seconds,
                    audio.sha256,
                    speakers,
                    regular,
                    exclusive,
                    self._model_info,
                    DiarizationProcessingInfo(
                        load_seconds, elapsed, time.perf_counter() - started, peak
                    ),
                    options,
                )
            except DiarizationError:
                raise
            except Exception as exc:
                logger.error(
                    "diarization_failed",
                    extra={"event": "diarization_failed", "exception_type": type(exc).__name__},
                )
                raise translate_error(exc, loading=False) from exc
            logger.info(
                "diarization_completed",
                extra={
                    "event": "diarization_completed",
                    "speaker_count": len(speakers),
                    "regular_turn_count": len(regular),
                    "exclusive_turn_count": len(exclusive),
                    "runtime": elapsed,
                },
            )
            return result
