"""Defensive parsing of ffprobe JSON; never parse console text."""

import math
from fractions import Fraction
from typing import Any, Literal

from .exceptions import MetadataExtractionFailed, NoAudioStream, UnsupportedMedia
from .models import AudioMetadata


def optional_number(value: Any, name: str, *, integer: bool = False) -> int | float | None:
    if value is None or value == "N/A" or value == "":
        return None
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or number < 0:
            raise ValueError
        if integer:
            if not number.is_integer():
                raise ValueError
            return int(number)
        return number
    except (ValueError, TypeError, OverflowError) as exc:
        raise MetadataExtractionFailed(f"ffprobe returned an invalid {name}.") from exc


def extract_metadata(data: dict[str, Any], file_size_bytes: int) -> AudioMetadata:
    streams = data.get("streams")
    container = data.get("format")
    if (
        not isinstance(streams, list)
        or not all(isinstance(s, dict) for s in streams)
        or not isinstance(container, dict)
    ):
        raise MetadataExtractionFailed(
            "ffprobe metadata is missing valid streams or format fields."
        )
    format_name = container.get("format_name")
    if not isinstance(format_name, str) or not format_name:
        raise MetadataExtractionFailed("ffprobe did not identify a container format.")
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise NoAudioStream(
            "The recording contains no audio stream. Supply audio or a video with audio."
        )
    # Prefer the container's designated default audio track, otherwise the first.
    stream = next(
        (
            s
            for s in audio_streams
            if isinstance(s.get("disposition"), dict) and s["disposition"].get("default") == 1
        ),
        audio_streams[0],
    )
    index = optional_number(stream.get("index"), "stream index", integer=True)
    if index is None:
        raise MetadataExtractionFailed("ffprobe did not provide an audio stream index.")
    codec = stream.get("codec_name")
    if not isinstance(codec, str) or not codec or codec == "unknown":
        raise UnsupportedMedia("The audio stream uses an unidentified or unsupported codec.")
    sample_rate = optional_number(stream.get("sample_rate"), "sample rate", integer=True)
    channels = optional_number(stream.get("channels"), "channel count", integer=True)
    bits = optional_number(stream.get("bits_per_sample"), "sample width", integer=True)
    if sample_rate == 0 or channels == 0:
        raise MetadataExtractionFailed(
            "The audio stream has an invalid sample rate or channel count."
        )
    duration = optional_number(stream.get("duration"), "stream duration")
    duration_source: Literal["stream", "container", "unknown"] = (
        "stream" if duration is not None else "unknown"
    )
    if (
        duration is None
        and stream.get("duration_ts") not in (None, "N/A")
        and stream.get("time_base") not in (None, "N/A")
    ):
        try:
            ticks = optional_number(stream["duration_ts"], "duration ticks")
            time_base = Fraction(str(stream["time_base"]))
            if ticks is None or time_base <= 0:
                raise ValueError
            duration = optional_number(float(ticks * time_base), "stream duration")
            duration_source = "stream"
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise MetadataExtractionFailed("ffprobe returned an invalid stream time base.") from exc
    if duration is None:
        duration = optional_number(container.get("duration"), "container duration")
        duration_source = "container" if duration is not None else "unknown"
    return AudioMetadata(
        container_format=format_name,
        codec=codec,
        audio_stream_index=int(index),
        duration_seconds=duration,
        duration_source=duration_source,
        sample_rate=int(sample_rate) if sample_rate is not None else None,
        channels=int(channels) if channels is not None else None,
        bits_per_sample=int(bits) if bits else None,
        file_size_bytes=file_size_bytes,
    )
