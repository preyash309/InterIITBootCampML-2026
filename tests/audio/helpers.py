"""Tiny generated fixtures; no committed binary recordings."""

import math
import struct
import wave
from pathlib import Path


def write_wav(
    path: Path,
    *,
    rate: int = 16_000,
    channels: int = 1,
    duration: float = 0.5,
    silent: bool = False,
) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        samples = (
            0 if silent else int(5000 * math.sin(2 * math.pi * 440 * frame / rate))
            for frame in range(int(rate * duration))
        )
        output.writeframes(b"".join(struct.pack("<h", sample) * channels for sample in samples))
    return path


def probe_data(
    *,
    rate: int = 16_000,
    channels: int = 1,
    duration: str = "0.5",
    codec: str = "pcm_s16le",
    container: str = "wav",
) -> dict:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": codec,
                "sample_rate": str(rate),
                "channels": channels,
                "bits_per_sample": 16,
                "duration": duration,
            }
        ],
        "format": {"format_name": container, "duration": duration},
    }
