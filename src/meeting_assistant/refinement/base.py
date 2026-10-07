"""Only this protocol connects provider code to deterministic refinement."""

from typing import Protocol

from .models import RefinementRequest, RefinementResponse, RefinerModelInfo


class TranscriptRefinerBackend(Protocol):
    @property
    def model_info(self) -> RefinerModelInfo: ...

    def refine(self, request: RefinementRequest) -> RefinementResponse: ...
