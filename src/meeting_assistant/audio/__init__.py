"""Phase 2 consumes only ingest_audio(...).canonical_audio_path."""

import logging

from .config import AudioIngestionConfig
from .exceptions import AudioIngestionError
from .ingestion import ingest_audio
from .models import AudioIngestionResult, AudioMetadata

__all__ = [
    "AudioIngestionConfig",
    "AudioIngestionError",
    "AudioIngestionResult",
    "AudioMetadata",
    "ingest_audio",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
