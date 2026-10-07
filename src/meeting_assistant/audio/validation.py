"""File checks and independent, streaming PCM WAV verification."""

import math
import stat
import wave
from pathlib import Path

from .config import TARGET_CHANNELS, TARGET_CODEC, TARGET_SAMPLE_RATE, AudioIngestionConfig
from .exceptions import (
    EmptyInputFile,
    InputFileNotFound,
    InputTooLarge,
    InvalidCanonicalAudio,
    InvalidInputFile,
)
from .models import AudioMetadata


def validate_input(path: Path, config: AudioIngestionConfig) -> int:
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        raise InputFileNotFound(
            "The input recording does not exist. Check the supplied path."
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise InvalidInputFile("The input must be a regular file, not a directory or device.")
    if info.st_size == 0:
        raise EmptyInputFile("The input recording is empty. Supply a non-empty media file.")
    if info.st_size > config.max_input_size_bytes:
        raise InputTooLarge(
            "The recording exceeds AUDIO_MAX_INPUT_SIZE. Increase the limit or use a smaller input."
        )
    # Check read access without buffering the recording into memory.
    with path.open("rb") as source:
        source.read(1)
    return info.st_size


def validate_canonical(
    path: Path, metadata: AudioMetadata, source: AudioMetadata, config: AudioIngestionConfig
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise InvalidCanonicalAudio("Conversion did not produce a non-empty WAV file.")
    if (
        metadata.container_format != "wav"
        or metadata.codec != TARGET_CODEC
        or metadata.sample_rate != TARGET_SAMPLE_RATE
        or metadata.channels != TARGET_CHANNELS
        or metadata.bits_per_sample != 16
    ):
        raise InvalidCanonicalAudio(
            "Converted audio does not match mono 16-kHz signed 16-bit PCM WAV."
        )
    duration = metadata.duration_seconds
    if (
        duration is None
        or not math.isfinite(duration)
        or duration < config.min_audio_duration_seconds
    ):
        raise InvalidCanonicalAudio("Converted audio is too short or has no usable duration.")
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (
                TARGET_SAMPLE_RATE,
                TARGET_CHANNELS,
                2,
                "NONE",
            ):
                raise InvalidCanonicalAudio("The WAV header violates the canonical audio contract.")
            expected_bytes = wav.getnframes() * 2
            read_bytes = 0
            signal_present = False
            while chunk := wav.readframes(32_768):
                read_bytes += len(chunk)
                signal_present = signal_present or bool(chunk.strip(b"\x00"))
            if read_bytes != expected_bytes or expected_bytes == 0:
                raise InvalidCanonicalAudio("The canonical WAV is empty or truncated.")
            if not signal_present:
                raise InvalidCanonicalAudio("The canonical audio contains only digital silence.")
            header_duration = wav.getnframes() / wav.getframerate()
            if abs(header_duration - duration) > 1 / TARGET_SAMPLE_RATE + 0.001:
                raise InvalidCanonicalAudio("The WAV header and probed duration disagree.")
    except (wave.Error, EOFError) as exc:
        raise InvalidCanonicalAudio("The canonical WAV could not be decoded.") from exc
    # A video's container duration may include a longer video track; only compare
    # fallback container duration when the container is audio-only by definition.
    audio_only_formats = {"wav", "mp3", "flac", "aac", "aiff", "au"}
    reliable_duration = (
        source.duration_source == "stream" or source.container_format in audio_only_formats
    )
    if source.duration_seconds is not None and reliable_duration:
        tolerance = max(
            config.duration_tolerance_seconds,
            source.duration_seconds * config.duration_tolerance_ratio,
        )
        if abs(duration - source.duration_seconds) > tolerance:
            raise InvalidCanonicalAudio(
                "Source and canonical audio durations differ beyond the allowed tolerance."
            )
