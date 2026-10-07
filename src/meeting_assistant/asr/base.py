"""Only this contract is needed by downstream processing or a future local engine."""

from pathlib import Path
from typing import Protocol

from .config import TranscriptionOptions
from .models import TranscriptResult


class ASRBackend(Protocol):
    def transcribe(
        self, audio_path: Path, options: TranscriptionOptions | None = None
    ) -> TranscriptResult: ...
