"""Typed, isolated model and reconciliation configuration."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from meeting_assistant.asr.config import read_environment
from meeting_assistant.asr.exceptions import ASRConfigurationError

from .exceptions import DiarizationConfigurationError

COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"
COMMUNITY_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"


def _positive_count(value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise DiarizationConfigurationError("Speaker counts must be positive integers or None.")


@dataclass(frozen=True)
class DiarizationOptions:
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    def __post_init__(self) -> None:
        for value in (self.num_speakers, self.min_speakers, self.max_speakers):
            _positive_count(value)
        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise DiarizationConfigurationError("Use exact speaker count or bounds, not both.")
        if self.min_speakers is not None and self.max_speakers is not None:
            if self.min_speakers > self.max_speakers:
                raise DiarizationConfigurationError("Minimum speakers cannot exceed maximum.")

    def pipeline_arguments(self) -> dict[str, int]:
        return {
            name: value
            for name in ("num_speakers", "min_speakers", "max_speakers")
            if (value := getattr(self, name)) is not None
        }


@dataclass(frozen=True)
class ReconciliationConfig:
    alignment_tolerance_seconds: float = 0.1
    nearest_turn_max_gap_seconds: float = 0.2
    utterance_max_gap_seconds: float = 1.0
    duration_tolerance_seconds: float = 0.5

    def __post_init__(self) -> None:
        for value in (
            self.alignment_tolerance_seconds,
            self.nearest_turn_max_gap_seconds,
            self.utterance_max_gap_seconds,
            self.duration_tolerance_seconds,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise DiarizationConfigurationError(
                    "Reconciliation tolerances must be finite and nonnegative."
                )
        if self.alignment_tolerance_seconds > 1 or self.nearest_turn_max_gap_seconds > 1:
            raise DiarizationConfigurationError(
                "Attribution gap limits must not exceed one second."
            )


@dataclass(frozen=True)
class DiarizationConfig:
    model: str = COMMUNITY_MODEL
    revision: str = COMMUNITY_REVISION
    model_cache: Path = field(default_factory=lambda: Path(".models/diarization/community-1"))
    device: str = "cuda"
    offline_only: bool = True
    hf_token: str | None = field(default=None, repr=False)
    options: DiarizationOptions = field(default_factory=DiarizationOptions)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    max_duration_seconds: float = 8 * 3600

    def __post_init__(self) -> None:
        if self.model != COMMUNITY_MODEL:
            raise DiarizationConfigurationError("Phase III supports Community-1 only.")
        if self.device not in ("cuda", "cpu") or not isinstance(self.offline_only, bool):
            raise DiarizationConfigurationError(
                "Choose cuda or explicit cpu, and a boolean offline flag."
            )
        if (
            not isinstance(self.revision, str)
            or len(self.revision) != 40
            or any(c not in "0123456789abcdef" for c in self.revision)
        ):
            raise DiarizationConfigurationError("Model revision must be a full commit SHA.")
        try:
            object.__setattr__(self, "model_cache", Path(self.model_cache))
        except (TypeError, ValueError) as exc:
            raise DiarizationConfigurationError("Model cache must be a filesystem path.") from exc
        if not isinstance(self.options, DiarizationOptions) or not isinstance(
            self.reconciliation, ReconciliationConfig
        ):
            raise DiarizationConfigurationError(
                "Diarization options and reconciliation config must be typed."
            )
        if self.hf_token is not None and (
            not isinstance(self.hf_token, str)
            or not self.hf_token
            or any(ord(c) < 33 or ord(c) > 126 for c in self.hf_token)
        ):
            raise DiarizationConfigurationError("HF_TOKEN must be a printable token.")
        if (
            isinstance(self.max_duration_seconds, bool)
            or not isinstance(self.max_duration_seconds, (int, float))
            or not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise DiarizationConfigurationError(
                "Maximum diarization duration must be finite and positive."
            )

    @classmethod
    def from_env(
        cls, *, env_file: Path | None = Path(".env"), environ: Mapping[str, str] | None = None
    ) -> "DiarizationConfig":
        try:
            values = read_environment(env_file) if environ is None else environ
        except ASRConfigurationError as exc:
            raise DiarizationConfigurationError(
                "Cannot read the diarization environment file."
            ) from exc

        def count(name):
            return int(values[name]) if values.get(name) else None

        offline = values.get("DIARIZATION_OFFLINE_ONLY", "true").lower()
        if offline not in ("true", "false", "1", "0"):
            raise DiarizationConfigurationError(
                "DIARIZATION_OFFLINE_ONLY requires true/false or 1/0."
            )
        try:
            return cls(
                model=values.get("DIARIZATION_MODEL", COMMUNITY_MODEL),
                revision=values.get("DIARIZATION_MODEL_REVISION", COMMUNITY_REVISION),
                model_cache=Path(
                    values.get("DIARIZATION_MODEL_CACHE", ".models/diarization/community-1")
                ),
                device=values.get("DIARIZATION_DEVICE", "cuda"),
                offline_only=offline in ("true", "1"),
                hf_token=values.get("HF_TOKEN") or None,
                options=DiarizationOptions(
                    count("DIARIZATION_NUM_SPEAKERS"),
                    count("DIARIZATION_MIN_SPEAKERS"),
                    count("DIARIZATION_MAX_SPEAKERS"),
                ),
                reconciliation=ReconciliationConfig(
                    alignment_tolerance_seconds=float(
                        values.get("DIARIZATION_ALIGNMENT_TOLERANCE_SECONDS", "0.1")
                    ),
                    nearest_turn_max_gap_seconds=float(
                        values.get("DIARIZATION_NEAREST_TURN_MAX_GAP_SECONDS", "0.2")
                    ),
                    utterance_max_gap_seconds=float(
                        values.get("DIARIZATION_UTTERANCE_MAX_GAP_SECONDS", "1")
                    ),
                ),
                max_duration_seconds=float(
                    values.get("DIARIZATION_MAX_DURATION_SECONDS", str(8 * 3600))
                ),
            )
        except ValueError as exc:
            raise DiarizationConfigurationError(
                "Invalid numeric diarization configuration."
            ) from exc
