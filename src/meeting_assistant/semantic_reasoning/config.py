"""Bounded shadow-mode configuration. Secrets belong only to provider transport."""

import os
from dataclasses import dataclass

from .models import Validated, require


@dataclass(frozen=True)
class SemanticConfig(Validated):
    enabled: bool = False
    provider: str = "julia"
    model: str = "SupersonicLabs/Julia-1"
    model_cache: str = ".models/semantic/julia-1"
    local_python: str = ".venv-semantic/" + (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    local_device: str = "cuda"
    local_max_length: int = 4096
    local_head_length: int = 768
    acceptance_probability: float = 0.85
    review_probability: float = 0.60
    max_event_calls: int = 80
    max_relation_calls: int = 100
    max_verification_calls: int = 50
    max_coverage_calls: int = 70
    max_total_calls: int = 250
    total_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 60.0
    max_retries: int = 1
    max_retry_delay_seconds: float = 5.0
    relation_window: int = 12
    relation_gap_seconds: float = 120.0
    max_state_chars: int = 14000
    max_coverage_candidates: int = 3

    def __post_init__(self):
        super().__post_init__()
        require(self.provider in ("julia", "gliner", "typesafe"))
        require(
            self.model
            == {
                "julia": "SupersonicLabs/Julia-1",
                "gliner": "fastino/GLiNER2.5-Decide",
                "typesafe": "jev-1.13.0",
            }[self.provider],
            "Choose the verified model for the selected semantic provider.",
        )
        require(self.local_device in ("cuda", "cpu"))
        require(
            1024 <= self.local_max_length <= 8192
            and 256 <= self.local_head_length < self.local_max_length - 4
        )
        require(0 <= self.review_probability < self.acceptance_probability <= 1)
        for name in (
            "max_event_calls",
            "max_relation_calls",
            "max_verification_calls",
            "max_coverage_calls",
            "max_total_calls",
            "max_state_chars",
            "max_coverage_candidates",
        ):
            require(0 < getattr(self, name) <= 100000)
        require(1 <= self.relation_window <= 15 and 0 <= self.max_retries <= 3)
        require(self.total_timeout_seconds > 0 and self.request_timeout_seconds > 0)
        require(self.relation_gap_seconds > 0)

    @classmethod
    def from_env(cls, *, environ=None):
        from meeting_assistant.asr.config import read_environment

        values = read_environment() if environ is None else environ
        defaults = cls()
        kwargs = {}
        for name, default in vars(defaults).items():
            value = values.get("SEMANTIC_" + name.upper(), values.get("JEV_" + name.upper()))
            if value is None:
                continue
            try:
                if isinstance(default, bool):
                    require(
                        value.lower() in ("true", "false", "1", "0"), "Invalid semantic boolean."
                    )
                    kwargs[name] = value.lower() in ("true", "1")
                else:
                    kwargs[name] = type(default)(value)
            except (ValueError, TypeError) as exc:
                from .exceptions import InvalidSemantics

                raise InvalidSemantics("Invalid semantic configuration value.") from exc
        if "model" not in kwargs:
            kwargs["model"] = {
                "julia": "SupersonicLabs/Julia-1",
                "gliner": "fastino/GLiNER2.5-Decide",
                "typesafe": "jev-1.13.0",
            }.get(kwargs.get("provider", defaults.provider), defaults.model)
        if kwargs.get("provider") == "gliner":
            kwargs.setdefault("model_cache", ".models/semantic/gliner-decide")
            kwargs.setdefault(
                "local_python",
                ".venv-decision/" + ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
            )
        return cls(**kwargs)
