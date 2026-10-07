"""Public ingestion entry point and safe per-job publication."""

import logging
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

from .config import AudioIngestionConfig
from .exceptions import (
    AudioIngestionError,
    CorruptMedia,
    ExternalCommandFailed,
    FileAccessError,
    InvalidCanonicalAudio,
    InvalidInputFile,
    InvalidJobId,
    JobBusy,
    MetadataExtractionFailed,
    NoAudioStream,
    UnsupportedMedia,
)
from .ffmpeg import FFmpegTools
from .metadata import extract_metadata
from .models import AudioIngestionResult
from .validation import validate_canonical, validate_input

logger = logging.getLogger(__name__)
# Extensions only help label a failed probe, never admit or reject valid content.
RECOGNIZED_SUFFIXES = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".webm",
        ".mp4",
        ".mov",
        ".mkv",
        ".aif",
        ".aiff",
        ".wma",
        ".opus",
    }
)


def _event(event: str, job_id: str) -> None:
    logger.info(event, extra={"event": event, "job_id": job_id})


def _job_id(value: str | UUID | None) -> str:
    if value is None:
        return str(uuid4())
    try:
        return str(UUID(str(value)))
    except (ValueError, AttributeError) as exc:
        raise InvalidJobId(
            "The job ID must be a UUID; paths and filenames are not valid job IDs."
        ) from exc


def _is_link(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    # lstat/reparse attributes also cover Windows junctions on Python 3.11,
    # before pathlib introduced is_junction().
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _directory(path: Path, root: Path) -> None:
    if _is_link(path) or not path.resolve().is_relative_to(root):
        raise FileAccessError("The job workspace contains an unsafe directory link.")
    path.mkdir(exist_ok=True)
    if not path.is_dir():
        raise FileAccessError("A job workspace path is not a directory.")


@contextmanager
def _job_workspace(config: AudioIngestionConfig, job_id: str) -> Iterator[Path]:
    root = config.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    job = root / job_id
    _directory(job, root)
    lock = job / ".ingestion.lock"
    try:
        with lock.open("xb"):
            pass
    except FileExistsError as exc:
        raise JobBusy(
            "This job is already being processed. Retry after its current ingestion finishes."
        ) from exc
    try:
        audio = job / "audio"
        _directory(audio, root)
        yield audio
    finally:
        try:
            lock.unlink()
        except OSError as exc:
            logger.exception(
                "job_lock_cleanup_failed",
                extra={"event": "job_lock_cleanup_failed", "job_id": job_id},
            )
            raise FileAccessError(
                "Could not remove the job lock. Check workspace permissions before retrying."
            ) from exc


def ingest_audio(
    input_path: str | Path,
    *,
    config: AudioIngestionConfig | None = None,
    job_id: str | UUID | None = None,
) -> AudioIngestionResult:
    """Validate and normalize one local file; failures raise AudioIngestionError.

    Reusing a job ID intentionally regenerates its output. Only a fully validated
    temporary WAV can replace the previous result. Input media is never deleted.
    """
    current_job = _job_id(job_id)
    _event("ingestion_started", current_job)
    temporary_path = None
    try:
        settings = config if config is not None else AudioIngestionConfig.from_env()
        source_path = Path(input_path).absolute()
        source_size = validate_input(source_path, settings)
        source_path = source_path.resolve()
        _event("source_validated", current_job)
        tools = FFmpegTools(settings)
        try:
            source_probe = tools.probe(source_path)
        except ExternalCommandFailed as exc:
            error = (
                CorruptMedia
                if source_path.suffix.lower() in RECOGNIZED_SUFFIXES
                else UnsupportedMedia
            )
            raise error(
                "The recording is damaged or is not a supported media format.",
                stderr=exc.stderr,
                returncode=exc.returncode,
            ) from exc
        source_metadata = extract_metadata(source_probe, source_size)
        _event("metadata_extracted", current_job)
        with _job_workspace(settings, current_job) as audio_dir:
            canonical_path = audio_dir / "canonical.wav"
            if _is_link(canonical_path):
                raise FileAccessError("The canonical destination must not be a filesystem link.")
            if source_path == canonical_path.resolve() or (
                canonical_path.exists() and source_path.samefile(canonical_path)
            ):
                raise InvalidInputFile(
                    "The source recording must be separate from the job's canonical output."
                )
            with tempfile.NamedTemporaryFile(
                prefix=".canonical-", suffix=".wav", dir=audio_dir, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                _event("conversion_started", current_job)
                tools.convert(source_path, temporary_path, source_metadata.audio_stream_index)
                _event("conversion_completed", current_job)
                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    raise InvalidCanonicalAudio(
                        "Conversion did not generate a non-empty audio file."
                    )
                try:
                    canonical_metadata = extract_metadata(
                        tools.probe(temporary_path), temporary_path.stat().st_size
                    )
                except (
                    ExternalCommandFailed,
                    MetadataExtractionFailed,
                    NoAudioStream,
                    UnsupportedMedia,
                ) as exc:
                    raise InvalidCanonicalAudio(
                        "The generated WAV failed audio stream inspection.",
                        stderr=exc.stderr,
                        returncode=exc.returncode,
                    ) from exc
                validate_canonical(temporary_path, canonical_metadata, source_metadata, settings)
                _event("canonical_validation_passed", current_job)
                os.replace(temporary_path, canonical_path)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            return AudioIngestionResult(
                job_id=current_job,
                source_path=source_path,
                canonical_audio_path=canonical_path,
                source_metadata=source_metadata,
                canonical_metadata=canonical_metadata,
            )
    except AudioIngestionError as exc:
        logger.warning(
            "ingestion_failed",
            extra={"event": "ingestion_failed", "job_id": current_job, "error_code": exc.code},
        )
        raise
    except (OSError, ValueError) as exc:
        logger.exception(
            "ingestion_failed",
            extra={
                "event": "ingestion_failed",
                "job_id": current_job,
                "error_code": "file_access_error",
            },
        )
        raise FileAccessError(
            "Could not access the recording or workspace. Check paths, permissions, and available disk space."
        ) from exc
