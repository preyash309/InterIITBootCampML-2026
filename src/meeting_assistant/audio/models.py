"""Immutable, typed Phase 1 outputs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AudioMetadata:
    container_format: str
    codec: str
    audio_stream_index: int
    duration_seconds: float | None
    duration_source: Literal["stream", "container", "unknown"]
    sample_rate: int | None
    channels: int | None
    bits_per_sample: int | None
    file_size_bytes: int


@dataclass(frozen=True)
class AudioIngestionResult:
    job_id: str
    source_path: Path
    canonical_audio_path: Path
    source_metadata: AudioMetadata
    canonical_metadata: AudioMetadata
    status: Literal["success"] = "success"
