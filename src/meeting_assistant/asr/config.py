"""Small provider configuration; no CUDA, local models, or SDK dependencies."""

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .exceptions import ASRConfigurationError

PROVIDERS = {
    "groq": ("api.groq.com", "/openai/v1/audio/transcriptions", "whisper-large-v3"),
    "openai": ("api.openai.com", "/v1/audio/transcriptions", "whisper-1"),
}
MODELS = {"groq": {"whisper-large-v3", "whisper-large-v3-turbo"}, "openai": {"whisper-1"}}


def read_environment(env_file: Path | None = Path(".env")) -> dict[str, str]:
    """Read a simple optional dotenv file. Process variables win; no global mutation.

    No interpolation, execution, multiline values, or ancestor-directory searches.
    """
    values: dict[str, str] = {}
    if env_file is not None:
        try:
            with Path(env_file).open(encoding="utf-8-sig") as stream:
                content = stream.read(65_537)
        except FileNotFoundError:
            content = ""
        except (OSError, ValueError) as exc:
            raise ASRConfigurationError("Cannot read the ASR environment file.") from exc
        if len(content) > 65_536:
            raise ASRConfigurationError("The ASR environment file is too large.")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            name, value = name.strip(), value.strip()
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ASRConfigurationError("Environment file requires NAME=value lines.")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[name] = value
    return {**values, **os.environ}


@dataclass(frozen=True)
class TranscriptionOptions:
    language: str | None = "en"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.language is not None and (
            not isinstance(self.language, str) or not re.fullmatch(r"[a-z]{2,3}", self.language)
        ):
            raise ASRConfigurationError("Language must be an ISO language code or None for auto.")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (float, int))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 1
        ):
            raise ASRConfigurationError("Temperature must be finite and between zero and one.")


@dataclass(frozen=True)
class ASRConfig:
    provider: str = "groq"
    model: str | None = None
    api_key: str | None = field(default=None, repr=False)
    options: TranscriptionOptions = field(default_factory=TranscriptionOptions)
    request_timeout_seconds: float = 300.0
    total_timeout_seconds: float = 7200.0
    chunk_duration_seconds: float = 600.0
    max_upload_size_bytes: int = 20_000_000
    max_input_size_bytes: int = 4 * 1024**3

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or self.provider not in PROVIDERS:
            raise ASRConfigurationError("ASR_PROVIDER must be groq or openai.")
        model = self.model if self.model is not None else PROVIDERS[self.provider][2]
        if not isinstance(model, str) or model not in MODELS[self.provider]:
            raise ASRConfigurationError("Choose a supported Whisper model for this provider.")
        object.__setattr__(self, "model", model)
        if not isinstance(self.options, TranscriptionOptions):
            raise ASRConfigurationError("ASR options must be TranscriptionOptions.")
        for value in (
            self.request_timeout_seconds,
            self.total_timeout_seconds,
            self.chunk_duration_seconds,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ASRConfigurationError(
                    "ASR time and chunk limits must be finite and positive."
                )
        if not 0.1 <= self.chunk_duration_seconds <= 3600:
            raise ASRConfigurationError("Chunk duration must be between 0.1 and 3600 seconds.")
        if (
            isinstance(self.max_upload_size_bytes, bool)
            or not isinstance(self.max_upload_size_bytes, int)
            or not 4096 <= self.max_upload_size_bytes <= 24_000_000
            or isinstance(self.max_input_size_bytes, bool)
            or not isinstance(self.max_input_size_bytes, int)
            or self.max_input_size_bytes <= 0
        ):
            raise ASRConfigurationError(
                "ASR file limits are invalid (uploads must stay below 25 MB)."
            )
        if self.api_key is not None and (
            not isinstance(self.api_key, str)
            or not self.api_key
            or any(ord(char) < 33 or ord(char) > 126 for char in self.api_key)
        ):
            raise ASRConfigurationError("API key must be a nonempty printable token.")

    @classmethod
    def from_env(
        cls, *, env_file: Path | None = Path(".env"), environ: Mapping[str, str] | None = None
    ) -> "ASRConfig":
        values = read_environment(env_file) if environ is None else environ
        provider = values.get("ASR_PROVIDER", "groq")
        key_name = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
        language = values.get("ASR_LANGUAGE", "en")
        try:
            return cls(
                provider=provider,
                model=values.get("ASR_MODEL") or None,
                api_key=values.get(key_name) or None,
                options=TranscriptionOptions(
                    language=None if language == "auto" else language,
                    temperature=float(values.get("ASR_TEMPERATURE", "0")),
                ),
                request_timeout_seconds=float(values.get("ASR_REQUEST_TIMEOUT", "300")),
                total_timeout_seconds=float(values.get("ASR_TOTAL_TIMEOUT", "7200")),
                chunk_duration_seconds=float(values.get("ASR_CHUNK_SECONDS", "600")),
                max_upload_size_bytes=int(values.get("ASR_MAX_UPLOAD_SIZE", "20000000")),
                max_input_size_bytes=int(values.get("ASR_MAX_INPUT_SIZE", str(4 * 1024**3))),
            )
        except ValueError as exc:
            raise ASRConfigurationError(
                "ASR numeric configuration contains an invalid value."
            ) from exc
