"""Replaceable local diarization boundary."""

from pathlib import Path
from typing import Protocol

from .config import DiarizationOptions
from .models import DiarizationResult


class DiarizationBackend(Protocol):
    def diarize(
        self, audio_path: Path, options: DiarizationOptions | None = None
    ) -> DiarizationResult: ...
