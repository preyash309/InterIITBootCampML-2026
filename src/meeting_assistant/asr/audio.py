"""ASR boundary checks and bounded-memory, lossless PCM chunking."""

import stat
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import ASRConfig
from .exceptions import InvalidASRAudio

RATE = 16_000
FRAME_BYTES = 2


@dataclass(frozen=True)
class CanonicalAudio:
    path: Path
    frames: int
    size_bytes: int
    mtime_ns: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / RATE


def inspect_audio(path: Path, config: ASRConfig) -> CanonicalAudio:
    """Validate PCM headers AND actual frame length before making any billed request.

    Unlike Phase I, this accepts digital silence to permit explicit no-speech tests.
    """
    try:
        path = Path(path)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
            raise InvalidASRAudio("ASR requires a nonempty regular canonical WAV file.")
        if info.st_size > config.max_input_size_bytes:
            raise InvalidASRAudio("ASR input exceeds the configured file-size limit.")
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getframerate() != RATE
                or audio.getnchannels() != 1
                or audio.getsampwidth() != FRAME_BYTES
                or audio.getcomptype() != "NONE"
                or audio.getnframes() == 0
            ):
                raise InvalidASRAudio("ASR requires 16-kHz mono PCM 16-bit WAV; run Phase I first.")
            expected = audio.getnframes()
            received = 0
            while block := audio.readframes(RATE * 2):
                received += len(block)
            if received != expected * FRAME_BYTES:
                raise InvalidASRAudio("Canonical WAV contains truncated or invalid PCM frames.")
        return CanonicalAudio(path.resolve(), expected, info.st_size, info.st_mtime_ns)
    except (OSError, ValueError, TypeError, EOFError, wave.Error) as exc:
        raise InvalidASRAudio(
            "Cannot read canonical audio; run Phase I on a valid recording."
        ) from exc


def iter_audio_chunks(
    audio: CanonicalAudio, config: ASRConfig, directory: Path
) -> Iterator[tuple[Path, float, float]]:
    frames_per_chunk = min(
        int(config.chunk_duration_seconds * RATE),
        (config.max_upload_size_bytes - 44) // FRAME_BYTES,
    )
    # Directly upload a short canonical file without a redundant copy.
    if audio.frames <= frames_per_chunk and audio.size_bytes <= config.max_upload_size_bytes:
        yield audio.path, 0.0, audio.duration_seconds
        return
    try:
        with wave.open(str(audio.path), "rb") as source:
            offset = 0
            while offset < audio.frames:
                count = min(frames_per_chunk, audio.frames - offset)
                chunk = directory / "chunk.wav"
                with wave.open(str(chunk), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(FRAME_BYTES)
                    output.setframerate(RATE)
                    remaining = count
                    while remaining:
                        block = source.readframes(min(remaining, RATE * 2))
                        if not block or len(block) % FRAME_BYTES:
                            raise InvalidASRAudio("Canonical audio changed or became truncated.")
                        output.writeframesraw(block)
                        remaining -= len(block) // FRAME_BYTES
                yield chunk, offset / RATE, count / RATE
                chunk.unlink()
                offset += count
    except (OSError, ValueError, EOFError, wave.Error) as exc:
        raise InvalidASRAudio("Cannot prepare temporary ASR audio chunks.") from exc


def ensure_unchanged(audio: CanonicalAudio) -> None:
    try:
        current = audio.path.stat()
        if current.st_size != audio.size_bytes or current.st_mtime_ns != audio.mtime_ns:
            raise InvalidASRAudio("Canonical audio changed during transcription; retry safely.")
    except OSError as exc:
        raise InvalidASRAudio("Canonical audio became inaccessible during transcription.") from exc
