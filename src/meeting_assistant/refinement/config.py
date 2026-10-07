"""Immutable configuration, using the existing optional dotenv convention."""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from meeting_assistant.asr.config import read_environment

from .exceptions import RefinementConfigurationError


@dataclass(frozen=True)
class RefinementConfig:
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    api_key: str | None = field(default=None, repr=False, compare=False)
    temperature: float = 0.0
    structured_mode: str = "json_schema"
    request_timeout_seconds: float = 90.0
    transport_retries: int = 1
    max_backoff_seconds: float = 60.0
    schema_repair_retries: int = 1
    max_completion_tokens: int = 4096
    max_target_chars: int = 8000
    max_request_chars: int = 24000
    context_chars: int = 1000
    description_chars: int = 300
    top_k: int = 3
    max_records: int = 64
    neighbor_gap_seconds: float = 5.0

    def __post_init__(self) -> None:
        if (
            self.provider != "groq"
            or not isinstance(self.model, str)
            or not re.fullmatch(r"[a-zA-Z0-9_./-]{1,120}", self.model)
        ):
            raise RefinementConfigurationError("Choose groq and an explicit valid model ID.")
        if self.structured_mode not in ("json_schema", "json_object"):
            raise RefinementConfigurationError(
                "Structured mode must be json_schema or json_object."
            )
        if isinstance(self.temperature, bool) or self.temperature != 0:
            raise RefinementConfigurationError("The conservative baseline requires temperature 0.")
        for name, maximum in (
            ("transport_retries", 2),
            ("schema_repair_retries", 1),
            ("max_completion_tokens", 16000),
            ("max_target_chars", 32000),
            ("max_request_chars", 128000),
            ("context_chars", 4000),
            ("description_chars", 1000),
            ("top_k", 20),
            ("max_records", 128),
        ):
            value = getattr(self, name)
            minimum = 0 if name.endswith("retries") else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise RefinementConfigurationError(f"Invalid {name}.")
        for name in ("request_timeout_seconds", "neighbor_gap_seconds", "max_backoff_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise RefinementConfigurationError(f"Invalid {name}.")
        if not 1 <= self.request_timeout_seconds <= 600:
            raise RefinementConfigurationError("Request timeout must be between 1 and 600 seconds.")
        if not 0 <= self.max_backoff_seconds <= 60:
            raise RefinementConfigurationError(
                "Retry backoff must be bounded to at most 60 seconds."
            )
        if self.api_key is not None and (
            not isinstance(self.api_key, str)
            or not self.api_key
            or any(ord(c) < 33 or ord(c) > 126 for c in self.api_key)
        ):
            raise RefinementConfigurationError(
                "GROQ_API_KEY must be a nonempty header-safe secret."
            )

    @classmethod
    def from_env(
        cls, env_file: Path | None = Path(".env"), *, environ: Mapping[str, str] | None = None
    ):
        values = dict(environ) if environ is not None else read_environment(env_file)
        options = {"api_key": values.get("GROQ_API_KEY")}
        defaults = cls()
        for name in cls.__dataclass_fields__:
            if name == "api_key":
                continue
            key = "REFINER_" + name.upper()
            if key not in values:
                continue
            try:
                default = getattr(defaults, name)
                options[name] = (
                    int(values[key])
                    if isinstance(default, int)
                    else float(values[key])
                    if isinstance(default, float)
                    else values[key]
                )
            except (ValueError, TypeError) as exc:
                raise RefinementConfigurationError(f"Invalid {key}.") from exc
        return cls(**options)
