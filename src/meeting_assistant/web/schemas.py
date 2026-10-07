"""Only application metadata lives in SQLite; phase payloads retain their own schemas."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StageName(str, Enum):
    INGESTING = "INGESTING"
    TRANSCRIBING = "TRANSCRIBING"
    DIARIZING = "DIARIZING"
    GROUNDING = "GROUNDING"
    REFINING = "REFINING"
    EXTRACTING_INTELLIGENCE = "EXTRACTING_INTELLIGENCE"


STAGES = tuple(StageName)


class SafeError(BaseModel):
    code: str
    message: str


class StageState(BaseModel):
    name: StageName
    status: Literal["pending", "running", "complete", "failed"] = "pending"
    duration_seconds: float | None = None


class MeetingJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    original_filename: str
    size_bytes: int
    created_at: str
    updated_at: str
    status: str = "QUEUED"
    current_stage: StageName | None = None
    stages: list[StageState] = Field(default_factory=lambda: [StageState(name=s) for s in STAGES])
    error: SafeError | None = None


class JobResponse(MeetingJob):
    completed_stages: int
    total_stages: int = 6
    status_url: str
    workspace_url: str


class DownloadInfo(BaseModel):
    artifact: str
    filename: str
    media_type: str
    advanced: bool
    download_url: str
