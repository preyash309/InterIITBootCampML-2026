"""A later local backend can satisfy this protocol without changing downstream claims."""

from typing import Protocol

from .models import IntelligenceModelInfo, IntelligenceRequest, IntelligenceResponse


class MeetingIntelligenceBackend(Protocol):
    @property
    def model_info(self) -> IntelligenceModelInfo: ...

    def extract(self, request: IntelligenceRequest) -> IntelligenceResponse: ...
