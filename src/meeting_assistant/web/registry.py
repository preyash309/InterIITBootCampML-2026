"""Allowlisted artifacts and strict paths below one server-owned UUID workspace."""

import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from .errors import WebError


@dataclass(frozen=True)
class ArtifactSpec:
    filename: str
    media_type: str
    advanced: bool = False


ARTIFACTS = {
    "canonical_audio": ArtifactSpec("canonical.wav", "audio/wav"),
    "raw_json": ArtifactSpec("raw_transcript.json", "application/json"),
    "raw_txt": ArtifactSpec("raw_transcript.txt", "text/plain"),
    "speaker_json": ArtifactSpec("speaker_transcript.json", "application/json"),
    "speaker_txt": ArtifactSpec("speaker_transcript.txt", "text/plain"),
    "refined_json": ArtifactSpec("refined_transcript.json", "application/json"),
    "refined_txt": ArtifactSpec("refined_transcript.txt", "text/plain"),
    "meeting_json": ArtifactSpec("meeting_record.json", "application/json"),
    "meeting_md": ArtifactSpec("meeting_record.md", "text/markdown"),
    "evidence_json": ArtifactSpec("evidence_manifest.json", "application/json"),
    "diarization_json": ArtifactSpec("diarization.json", "application/json", True),
    "grounding_json": ArtifactSpec("grounding.json", "application/json", True),
    "grounding_txt": ArtifactSpec("grounding.txt", "text/plain", True),
    "edit_log_json": ArtifactSpec("edit_log.json", "application/json", True),
}


def job_id(value):
    try:
        if str(UUID(str(value))) != str(value):
            raise ValueError
        return str(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise WebError("meeting_not_found", "Meeting not found.", 404) from exc


def safe_path(root: Path, relative: str, *, must_exist=True):
    """Reject traversal and every symlink/reparse point, including Windows junctions."""
    rel = Path(relative)
    if rel.is_absolute() or rel.drive or not rel.parts or any(p in ("..", ".") for p in rel.parts):
        raise WebError("unsafe_artifact", "Artifact is unavailable.", 404)
    root = root.absolute()
    current = root.anchor and Path(root.anchor)
    for part in root.parts[1:] + rel.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
                stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
            ):
                raise WebError("unsafe_artifact", "Artifact is unavailable.", 404)
    path = root / rel
    if not path.resolve().is_relative_to(root.resolve()):
        raise WebError("unsafe_artifact", "Artifact is unavailable.", 404)
    if must_exist and not path.is_file():
        raise WebError("artifact_missing", "Artifact is unavailable on this server.", 404)
    return path
