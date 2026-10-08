"""Opt-in costs. No context means no new provider calls by default."""

import math
from dataclasses import dataclass, fields

from meeting_assistant.asr.config import read_environment

from .exceptions import ContextualConfigurationError


@dataclass(frozen=True)
class ContextualASRConfig:
    enabled: bool = False
    use_global_glossary: bool = False
    max_windows: int = 6
    max_window_seconds: float = 15.0
    padding_seconds: float = 2.0
    merge_gap_seconds: float = 0.5
    top_k_terms: int = 6
    min_priority: float = 0.70
    max_total_audio_seconds: float = 60.0
    total_timeout_seconds: float = 180.0
    policy_version: str = "context_asr_v1"

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.use_global_glossary) is not bool:
            raise ContextualConfigurationError("Feature flags must be booleans.")
        for value, low, high in ((self.max_windows, 1, 12), (self.top_k_terms, 1, 8)):
            if type(value) is not int or not low <= value <= high:
                raise ContextualConfigurationError("Contextual count budget is out of bounds.")
        for value, low, high in (
            (self.max_window_seconds, 1, 30),
            (self.padding_seconds, 0, 3),
            (self.merge_gap_seconds, 0, 2),
            (self.min_priority, 0, 1),
            (self.max_total_audio_seconds, 1, 180),
            (self.total_timeout_seconds, 1, 600),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not (math.isfinite(value) and low <= value <= high)
            ):
                raise ContextualConfigurationError("Contextual numeric budget is out of bounds.")
        if self.policy_version != "context_asr_v1":
            raise ContextualConfigurationError("Unknown contextual ASR policy.")

    @classmethod
    def from_env(cls, *, environ=None):
        values = read_environment() if environ is None else environ
        options = {}
        defaults = cls()
        try:
            for field in fields(cls):
                if field.name == "policy_version":
                    continue
                key = "CONTEXT_ASR_" + field.name.upper()
                if key not in values:
                    continue
                default = getattr(defaults, field.name)
                if isinstance(default, bool):
                    if values[key].lower() not in ("true", "false", "1", "0"):
                        raise ValueError
                    options[field.name] = values[key].lower() in ("true", "1")
                else:
                    options[field.name] = type(default)(values[key])
            return cls(**options)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContextualConfigurationError("Invalid CONTEXT_ASR environment setting.") from exc
