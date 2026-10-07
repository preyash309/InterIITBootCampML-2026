"""Validate canonical PCM independently of ML dependencies; supply a CPU tensor."""

import hashlib
import stat
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import InvalidDiarizationAudio

RATE = 16_000


@dataclass(frozen=True)
class CanonicalAudio:
    path: Path
    frames: int
    size_bytes: int
    mtime_ns: int
    sha256: str

    @property
    def duration_seconds(self) -> float:
        return self.frames / RATE


def inspect_audio(path: Path | str, max_duration_seconds: float) -> CanonicalAudio:
    try:
        path = Path(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
            raise InvalidDiarizationAudio("Diarization requires a nonempty regular canonical WAV.")
        if before.st_size > max_duration_seconds * RATE * 2 + 1_048_576:
            raise InvalidDiarizationAudio(
                "Audio exceeds the configured diarization duration limit."
            )
        with wave.open(str(path), "rb") as source:
            frames = source.getnframes()
            if (
                source.getframerate() != RATE
                or source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
                or frames == 0
            ):
                raise InvalidDiarizationAudio(
                    "Diarization requires 16-kHz mono PCM16 WAV; run Phase I."
                )
            if frames / RATE > max_duration_seconds:
                raise InvalidDiarizationAudio(
                    "Audio exceeds the configured diarization duration limit."
                )
            received = 0
            while block := source.readframes(RATE * 2):
                received += len(block)
            if received != frames * 2:
                raise InvalidDiarizationAudio("Canonical WAV contains truncated PCM frames.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        audio = CanonicalAudio(
            path.resolve(), frames, before.st_size, before.st_mtime_ns, digest.hexdigest()
        )
        ensure_unchanged(audio)
        return audio
    except (OSError, ValueError, TypeError, EOFError, wave.Error) as exc:
        raise InvalidDiarizationAudio(
            "Cannot read canonical WAV; run Phase I on a valid recording."
        ) from exc


def ensure_unchanged(audio: CanonicalAudio) -> None:
    try:
        current = audio.path.stat()
        if current.st_size != audio.size_bytes or current.st_mtime_ns != audio.mtime_ns:
            raise InvalidDiarizationAudio(
                "Canonical audio changed during diarization; retry safely."
            )
    except OSError as exc:
        raise InvalidDiarizationAudio(
            "Canonical audio became inaccessible during diarization."
        ) from exc


def load_waveform(audio: CanonicalAudio, torch: Any) -> Any:
    # One full float32 CPU allocation: pyannote needs the waveform, but not a
    # second full byte buffer or NumPy copy. PCM blocks stay bounded in size.
    import numpy as np

    try:
        waveform = torch.empty((1, audio.frames), dtype=torch.float32, device="cpu")
        offset = 0
        with wave.open(str(audio.path), "rb") as source:
            while block := source.readframes(RATE * 2):
                pcm = np.frombuffer(block, dtype="<i2").astype(np.float32)
                count = len(pcm)
                if offset + count > audio.frames:
                    raise InvalidDiarizationAudio(
                        "Canonical WAV changed while reading its waveform."
                    )
                waveform[0, offset : offset + count] = torch.from_numpy(pcm / 32768.0)
                offset += count
        if offset != audio.frames:
            raise InvalidDiarizationAudio(
                "Canonical WAV became truncated while reading its waveform."
            )
        ensure_unchanged(audio)
        return waveform
    except (OSError, ValueError, EOFError, wave.Error) as exc:
        raise InvalidDiarizationAudio(
            "Cannot prepare the canonical waveform for diarization."
        ) from exc
