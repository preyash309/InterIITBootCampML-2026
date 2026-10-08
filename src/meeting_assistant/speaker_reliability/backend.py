"""Independent secondary inference contract."""

from pathlib import Path
from typing import Protocol

from .models import SecondaryDiarizationResult


class SecondaryDiarizationBackend(Protocol):
    def diarize(self, audio_path: Path) -> SecondaryDiarizationResult: ...
