"""Bounded, shell-free FFmpeg/ffprobe execution and command construction."""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import TARGET_CHANNELS, TARGET_CODEC, TARGET_SAMPLE_RATE, AudioIngestionConfig
from .exceptions import (
    AudioConversionFailed,
    ExternalCommandFailed,
    ExternalCommandTimeout,
    FFmpegUnavailable,
    FFprobeUnavailable,
    FileAccessError,
    MetadataExtractionFailed,
)

logger = logging.getLogger(__name__)
# Exclude playlists, network protocols, and reference/sequence demuxers. Their
# contents can cause reads outside an uploaded file even with a local-file input.
MEDIA_DEMUXERS = "wav,mp3,mov,matroska,webm,aac,flac,ogg,aiff,asf,avi,au,mpeg,mpegts"


def resolve_executable(name: str, configured: str | None) -> str:
    """An explicit override is authoritative; never silently fall back from it."""
    candidate = shutil.which(configured if configured is not None else name)
    if candidate is None:
        error = FFmpegUnavailable if name == "ffmpeg" else FFprobeUnavailable
        variable = "FFMPEG_PATH" if name == "ffmpeg" else "FFPROBE_PATH"
        raise error(f"{name} is unavailable. Install it on PATH or configure {variable}.")
    return str(Path(candidate).resolve())


def run_command(
    command: list[str], *, timeout: float, tool: str
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **(
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else {}
            ),
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        logger.error(
            "external_command_timeout",
            extra={"event": "external_command_timeout", "tool": tool, "stderr": stderr},
        )
        raise ExternalCommandTimeout(
            f"{tool} exceeded its time limit. Check the recording or increase the configured timeout.",
            stderr=stderr,
        ) from exc
    except FileNotFoundError as exc:
        error = FFmpegUnavailable if tool == "ffmpeg" else FFprobeUnavailable
        raise error(f"{tool} could not be started. Check its executable configuration.") from exc
    except OSError as exc:
        raise FileAccessError(
            f"Could not start {tool}. Check executable permissions and configuration."
        ) from exc
    logger.debug(
        "external_command_completed",
        extra={
            "event": "external_command_completed",
            "tool": tool,
            "returncode": completed.returncode,
            "stderr": completed.stderr,
        },
    )
    if completed.returncode != 0:
        logger.error(
            "external_command_failed",
            extra={
                "event": "external_command_failed",
                "tool": tool,
                "returncode": completed.returncode,
                "stderr": completed.stderr,
            },
        )
        raise ExternalCommandFailed(
            f"{tool} rejected the media.", stderr=completed.stderr, returncode=completed.returncode
        )
    return completed


class FFmpegTools:
    def __init__(self, config: AudioIngestionConfig) -> None:
        self.config = config
        self.ffmpeg = resolve_executable("ffmpeg", config.ffmpeg_path)
        self.ffprobe = resolve_executable("ffprobe", config.ffprobe_path)

    def probe(self, path: Path) -> dict[str, Any]:
        completed = run_command(
            [
                self.ffprobe,
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                MEDIA_DEMUXERS,
                "-show_entries",
                "format=format_name,duration:stream=index,codec_type,codec_name,sample_rate,channels,bits_per_sample,duration,duration_ts,time_base:stream_disposition=default",
                "-of",
                "json",
                str(path),
            ],
            timeout=self.config.ffprobe_timeout_seconds,
            tool="ffprobe",
        )
        try:
            data = json.loads(completed.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise MetadataExtractionFailed("ffprobe returned malformed JSON metadata.") from exc
        if not isinstance(data, dict):
            raise MetadataExtractionFailed("ffprobe metadata must be a JSON object.")
        return data

    def convert(self, source: Path, destination: Path, stream_index: int) -> None:
        try:
            run_command(
                [
                    self.ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-xerror",
                    "-err_detect",
                    "explode",
                    "-protocol_whitelist",
                    "file",
                    "-format_whitelist",
                    MEDIA_DEMUXERS,
                    "-i",
                    str(source),
                    "-map",
                    f"0:{stream_index}",
                    "-vn",
                    "-sn",
                    "-dn",
                    "-map_metadata",
                    "-1",
                    "-ac",
                    str(TARGET_CHANNELS),
                    "-ar",
                    str(TARGET_SAMPLE_RATE),
                    "-c:a",
                    TARGET_CODEC,
                    "-f",
                    "wav",
                    "-y",
                    str(destination),
                ],
                timeout=self.config.ffmpeg_timeout_seconds,
                tool="ffmpeg",
            )
        except ExternalCommandFailed as exc:
            raise AudioConversionFailed(
                "Audio conversion failed. The recording may be damaged or use an unsupported codec.",
                stderr=exc.stderr,
                returncode=exc.returncode,
            ) from exc
