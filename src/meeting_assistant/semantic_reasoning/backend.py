"""Provider-independent bounded choice interface; no provider prose escapes."""

from typing import Protocol

from .models import ProviderCall, ProviderDecision, Question


class SemanticDecisionBackend(Protocol):
    provider: str
    model: str
    calls: list[ProviderCall]

    def decide(
        self,
        state: dict,
        questions: tuple[Question, ...],
        *,
        purpose: str,
        policy: str,
        timeout_seconds: float,
        max_attempts: int,
    ) -> tuple[ProviderDecision, ...]: ...
