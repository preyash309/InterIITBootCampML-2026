"""Fixed canonical contract and configurable resource limits."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

from .exceptions import ConfigurationError

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_s16le"
_Number = TypeVar("_Number", int, float)


def _env_number(name: str, parser: Callable[[str], _Number], default: _Number) -> _Number:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return parser(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must contain a valid numeric limit.") from exc


@dataclass(frozen=True)
class AudioIngestionConfig:
    work_dir: Path = field(default_factory=lambda: Path("jobs"))
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    ffmpeg_timeout_seconds: float = 3600.0
    ffprobe_timeout_seconds: float = 60.0
    max_input_size_bytes: int = 4 * 1024**3
    min_audio_duration_seconds: float = 0.1
    duration_tolerance_seconds: float = 1.0
    duration_tolerance_ratio: float = 0.02

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "work_dir", Path(self.work_dir))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("work_dir must be a filesystem path.") from exc
        positive = (
            self.ffmpeg_timeout_seconds,
            self.ffprobe_timeout_seconds,
            self.min_audio_duration_seconds,
        )
        nonnegative = (self.duration_tolerance_seconds, self.duration_tolerance_ratio)
        if (
            any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
                for v in positive
            )
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v < 0
                for v in nonnegative
            )
            or not isinstance(self.max_input_size_bytes, int)
            or isinstance(self.max_input_size_bytes, bool)
            or self.max_input_size_bytes <= 0
        ):
            raise ConfigurationError(
                "Audio limits must be finite and positive (tolerances may be zero)."
            )

    @classmethod
    def from_env(cls) -> "AudioIngestionConfig":
        """Read environment on each call, without mutable global configuration."""
        defaults = cls()
        return cls(
            work_dir=Path(os.environ.get("AUDIO_WORK_DIR", "jobs")),
            ffmpeg_path=os.environ.get("FFMPEG_PATH") or None,
            ffprobe_path=os.environ.get("FFPROBE_PATH") or None,
            ffmpeg_timeout_seconds=_env_number(
                "AUDIO_FFMPEG_TIMEOUT", float, defaults.ffmpeg_timeout_seconds
            ),
            ffprobe_timeout_seconds=_env_number(
                "AUDIO_FFPROBE_TIMEOUT", float, defaults.ffprobe_timeout_seconds
            ),
            max_input_size_bytes=_env_number(
                "AUDIO_MAX_INPUT_SIZE", int, defaults.max_input_size_bytes
            ),
            min_audio_duration_seconds=_env_number(
                "AUDIO_MIN_DURATION", float, defaults.min_audio_duration_seconds
            ),
            duration_tolerance_seconds=_env_number(
                "AUDIO_DURATION_TOLERANCE_SECONDS", float, defaults.duration_tolerance_seconds
            ),
            duration_tolerance_ratio=_env_number(
                "AUDIO_DURATION_TOLERANCE_RATIO", float, defaults.duration_tolerance_ratio
            ),
        )
