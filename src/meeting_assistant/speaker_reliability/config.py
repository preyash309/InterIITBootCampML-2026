"""Explicit shadow-mode policy; no effect on canonical diarization."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

from meeting_assistant.asr.config import read_environment

from .exceptions import InvalidReliability

SORTFORMER_MODEL = "nvidia/diar_sortformer_4spk-v1"
SORTFORMER_REVISION = "2617bffbd820aa29d8f1fb6ab6f9ed7f0adbc996"


@dataclass(frozen=True)
class SpeakerReliabilityConfig:
    enabled: bool = False
    backend: str = "sortformer"
    device: str = "cuda"
    python_path: Path = Path(".venv-secondary") / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    model_path: Path = Path(".models/sortformer/diar_sortformer_4spk-v1.nemo")
    timeout_seconds: float = 600
    max_duration_seconds: float = 120
    max_speakers: int = 4
    reliable_threshold: float = 0.90
    mixed_threshold: float = 0.65
    low_overlap_threshold: float = 0.10
    split_merge_min_fraction: float = 0.20
    schema_version: str = "1.0"
    alignment_policy: str = "speaker_alignment_v1"
    reliability_policy: str = "speaker_reliability_v1"

    def __post_init__(self):
        if type(self.enabled) is not bool or self.backend != "sortformer":
            raise InvalidReliability("Unsupported reliability configuration.")
        if self.device not in ("cuda", "cpu"):
            raise InvalidReliability("Secondary device must be cuda or cpu; no implicit fallback.")
        if type(self.max_speakers) is not int or not 1 <= self.max_speakers <= 4:
            raise InvalidReliability("This Sortformer checkpoint supports at most four speakers.")
        for value in (self.timeout_seconds, self.max_duration_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise InvalidReliability("Secondary runtime limits must be finite and positive.")
        for value in (
            self.reliable_threshold,
            self.mixed_threshold,
            self.low_overlap_threshold,
            self.split_merge_min_fraction,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise InvalidReliability("Reliability thresholds must be fractions.")
        if self.mixed_threshold > self.reliable_threshold:
            raise InvalidReliability("Mixed threshold cannot exceed reliable threshold.")
        if not isinstance(self.python_path, Path) or not isinstance(self.model_path, Path):
            raise InvalidReliability("Secondary executable/cache must be pathlib paths.")

    @classmethod
    def from_env(cls, *, environ=None):
        values = read_environment() if environ is None else environ

        def boolean(key, default):
            value = values.get(key, str(default)).lower().strip()
            if value not in ("true", "false", "1", "0"):
                raise InvalidReliability(f"{key} must be true or false.")
            return value in ("true", "1")

        try:
            return cls(
                enabled=boolean("SPEAKER_RELIABILITY_ENABLED", False),
                backend=values.get("SECONDARY_DIARIZER_BACKEND", "sortformer"),
                device=values.get("SECONDARY_DIARIZER_DEVICE", "cuda"),
                python_path=Path(values.get("SECONDARY_DIARIZER_PYTHON", str(cls.python_path))),
                model_path=Path(values.get("SECONDARY_DIARIZER_MODEL_PATH", str(cls.model_path))),
                timeout_seconds=float(values.get("SPEAKER_RELIABILITY_TIMEOUT_SECONDS", 600)),
                max_duration_seconds=float(
                    values.get("SECONDARY_DIARIZER_MAX_DURATION_SECONDS", 120)
                ),
                max_speakers=int(values.get("SPEAKER_RELIABILITY_MAX_SPEAKERS", 4)),
                reliable_threshold=float(
                    values.get("SPEAKER_RELIABILITY_RELIABLE_THRESHOLD", 0.90)
                ),
                mixed_threshold=float(values.get("SPEAKER_RELIABILITY_MIXED_THRESHOLD", 0.65)),
            )
        except (ValueError, TypeError, AttributeError) as exc:
            raise InvalidReliability(
                "Invalid speaker reliability environment configuration."
            ) from exc
