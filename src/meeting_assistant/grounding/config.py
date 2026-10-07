"""Reproducible CPU retrieval defaults, independent of ASR and diarization."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from meeting_assistant.asr.config import read_environment

from .exceptions import GroundingConfigurationError

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class GroundingConfig:
    model: str = MODEL
    revision: str = REVISION
    model_cache: Path = field(default_factory=lambda: Path(".models/grounding/minilm"))
    index_cache: Path = field(default_factory=lambda: Path(".cache/grounding"))
    offline_only: bool = True
    top_k: int = 5
    min_score: float = 0.64
    lexical_floor: float = 0.78
    phonetic_floor: float = 0.88
    context_floor: float = 0.30
    max_span_words: int = 4
    context_chars: int = 600
    neighbor_gap_seconds: float = 3.0
    batch_size: int = 32
    cpu_threads: int = 4
    lexical_weight: float = 0.45
    phonetic_weight: float = 0.25
    semantic_weight: float = 0.25
    scope_weight: float = 0.05

    def __post_init__(self) -> None:
        for name in ("model_cache", "index_cache"):
            try:
                object.__setattr__(self, name, Path(getattr(self, name)))
            except (TypeError, ValueError) as exc:
                raise GroundingConfigurationError("Cache locations must be paths.") from exc
        if self.model != MODEL or self.revision != REVISION:
            raise GroundingConfigurationError(
                "This baseline supports the pinned MiniLM model only."
            )
        for name, upper in (
            ("top_k", 20),
            ("max_span_words", 4),
            ("context_chars", 4000),
            ("batch_size", 256),
            ("cpu_threads", 64),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise GroundingConfigurationError(f"Invalid {name}.")
        for name in (
            "min_score",
            "lexical_floor",
            "phonetic_floor",
            "context_floor",
            "lexical_weight",
            "phonetic_weight",
            "semantic_weight",
            "scope_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise GroundingConfigurationError(f"Invalid {name}.")
        if (
            abs(
                sum(
                    (
                        self.lexical_weight,
                        self.phonetic_weight,
                        self.semantic_weight,
                        self.scope_weight,
                    )
                )
                - 1
            )
            > 1e-9
        ):
            raise GroundingConfigurationError("Score weights must sum to one.")
        if (
            not isinstance(self.offline_only, bool)
            or isinstance(self.neighbor_gap_seconds, bool)
            or not isinstance(self.neighbor_gap_seconds, (int, float))
            or not math.isfinite(self.neighbor_gap_seconds)
            or self.neighbor_gap_seconds < 0
        ):
            raise GroundingConfigurationError("Invalid offline/context policy.")

    @classmethod
    def from_env(
        cls, env_file: Path | None = Path(".env"), *, environ: Mapping[str, str] | None = None
    ) -> "GroundingConfig":
        values = dict(environ) if environ is not None else read_environment(env_file)
        options = {}
        defaults = cls()
        for name in cls.__dataclass_fields__:
            key = "GROUNDING_" + name.upper()
            if key not in values:
                continue
            default = getattr(defaults, name)
            try:
                if isinstance(default, bool):
                    if values[key].lower() not in ("true", "false", "1", "0"):
                        raise ValueError
                    options[name] = values[key].lower() in ("true", "1")
                elif isinstance(default, int):
                    options[name] = int(values[key])
                elif isinstance(default, float):
                    options[name] = float(values[key])
                else:
                    options[name] = values[key]
            except (ValueError, TypeError) as exc:
                raise GroundingConfigurationError(f"Invalid {key}.") from exc
        return cls(**options)
