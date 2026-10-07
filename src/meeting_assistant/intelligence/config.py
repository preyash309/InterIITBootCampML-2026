"""Project-local environment convention; no credential-bearing repr."""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from meeting_assistant.asr.config import read_environment

from .exceptions import IntelligenceConfigurationError


@dataclass(frozen=True)
class IntelligenceConfig:
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    api_key: str | None = field(default=None, repr=False, compare=False)
    temperature: float = 0.0
    structured_mode: str = "json_schema"
    request_timeout_seconds: float = 90.0
    transport_retries: int = 1
    max_backoff_seconds: float = 60.0
    schema_repair_retries: int = 1
    max_completion_tokens: int = 8192
    max_chunk_chars: int = 18000
    overlap_utterances: int = 2
    max_chunks: int = 32
    max_consolidation_chars: int = 90000
    max_items_per_section: int = 128

    def __post_init__(self):
        if (
            self.provider != "groq"
            or not isinstance(self.model, str)
            or not re.fullmatch(r"[a-zA-Z0-9_./-]{1,120}", self.model)
        ):
            raise IntelligenceConfigurationError("Choose groq and an explicit valid model ID.")
        if isinstance(self.temperature, bool) or self.temperature != 0:
            raise IntelligenceConfigurationError("The reproducible baseline uses temperature 0.")
        if self.structured_mode != "json_schema":
            raise IntelligenceConfigurationError("Phase VI requires strict json_schema mode.")
        for name, low, high in (
            ("transport_retries", 0, 2),
            ("schema_repair_retries", 0, 1),
            ("max_completion_tokens", 256, 16000),
            ("max_chunk_chars", 256, 64000),
            ("overlap_utterances", 0, 4),
            ("max_chunks", 1, 128),
            ("max_consolidation_chars", 256, 128000),
            ("max_items_per_section", 1, 256),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise IntelligenceConfigurationError(f"Invalid {name}.")
        for name, low, high in (
            ("request_timeout_seconds", 1, 600),
            ("max_backoff_seconds", 0, 60),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not (math.isfinite(value) and low <= value <= high)
            ):
                raise IntelligenceConfigurationError(f"Invalid {name}.")
        if self.api_key is not None and (
            not isinstance(self.api_key, str)
            or not self.api_key
            or any(ord(c) < 33 or ord(c) > 126 for c in self.api_key)
        ):
            raise IntelligenceConfigurationError("GROQ_API_KEY must be a header-safe secret.")

    @classmethod
    def from_env(cls, env_file: Path | None = Path(".env"), *, environ: Mapping | None = None):
        values = dict(environ) if environ is not None else read_environment(env_file)
        options = {"api_key": values.get("GROQ_API_KEY")}
        defaults = cls()
        for name in cls.__dataclass_fields__:
            key = "INTELLIGENCE_" + name.upper()
            if name == "request_timeout_seconds":
                key = "INTELLIGENCE_TIMEOUT_SECONDS"
            if name == "api_key" or key not in values:
                continue
            try:
                default = getattr(defaults, name)
                options[name] = (
                    int(values[key])
                    if isinstance(default, int)
                    else (float(values[key]) if isinstance(default, float) else values[key])
                )
            except (ValueError, TypeError) as exc:
                raise IntelligenceConfigurationError(f"Invalid {key}.") from exc
        return cls(**options)
