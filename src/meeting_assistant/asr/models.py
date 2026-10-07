"""Immutable raw transcription contract, independent of provider or local engine."""

import math
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from .exceptions import InvalidTranscript


def finite_number(value: object, *, minimum: float | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise InvalidTranscript("Transcript contains an invalid numeric value.")


def probability(value: float | None) -> None:
    if value is not None:
        finite_number(value, minimum=0)
        if value > 1:
            raise InvalidTranscript("Transcript probability must be between zero and one.")


def timestamps(start: float, end: float) -> None:
    finite_number(start, minimum=0)
    finite_number(end, minimum=start)


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start: float
    end: float
    probability: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise InvalidTranscript("Word text must be a nonempty string.")
        timestamps(self.start, self.end)
        probability(self.probability)


@dataclass(frozen=True)
class TranscriptSegment:
    id: str
    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...] = ()
    avg_logprob: float | None = None
    no_speech_probability: float | None = None
    compression_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not isinstance(self.text, str):
            raise InvalidTranscript("Segment ID and text must be strings.")
        timestamps(self.start, self.end)
        probability(self.no_speech_probability)
        if self.avg_logprob is not None:
            finite_number(self.avg_logprob)
        if self.compression_ratio is not None:
            finite_number(self.compression_ratio, minimum=0)
        if not isinstance(self.words, tuple) or any(
            not isinstance(word, TranscriptWord) for word in self.words
        ):
            raise InvalidTranscript("Segment words must be an immutable tuple of TranscriptWord.")
        previous_start = self.start - 0.5
        for word in self.words:
            if word.start < previous_start or word.start < self.start - 0.5:
                raise InvalidTranscript("Words must be ordered and stay within segment bounds.")
            if word.end > self.end + 0.5:
                raise InvalidTranscript("Word end exceeds its segment bounds.")
            previous_start = word.start


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    device: str = "remote"
    compute_type: str | None = None
    requested_language: str | None = "en"
    temperature: float = 0.0
    word_timestamps: bool = True
    vad_enabled: bool | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.provider, self.model, self.device)
        ):
            raise InvalidTranscript(
                "Provider, model, and device metadata must be nonempty strings."
            )
        if self.compute_type is not None and not isinstance(self.compute_type, str):
            raise InvalidTranscript("Compute type must be a string or None.")
        if self.requested_language is not None and not isinstance(self.requested_language, str):
            raise InvalidTranscript("Requested language must be a string or None.")
        if not isinstance(self.word_timestamps, bool) or (
            self.vad_enabled is not None and not isinstance(self.vad_enabled, bool)
        ):
            raise InvalidTranscript("Timestamp and VAD settings must have valid boolean types.")
        finite_number(self.temperature, minimum=0)
        if self.temperature > 1:
            raise InvalidTranscript("Decoding temperature must be between zero and one.")


@dataclass(frozen=True)
class ProcessingInfo:
    model_load_seconds: float
    transcription_seconds: float
    total_seconds: float
    chunk_count: int

    def __post_init__(self) -> None:
        for value in (self.model_load_seconds, self.transcription_seconds, self.total_seconds):
            finite_number(value, minimum=0)
        if self.total_seconds < self.transcription_seconds:
            raise InvalidTranscript("Total processing time cannot be shorter than transcription.")
        if isinstance(self.chunk_count, bool) or not isinstance(self.chunk_count, int):
            raise InvalidTranscript("Chunk count must be an integer.")
        if self.chunk_count < 1:
            raise InvalidTranscript("Chunk count must be positive.")


@dataclass(frozen=True)
class TranscriptResult:
    transcript_id: str
    language: str | None
    language_probability: float | None
    duration_seconds: float
    text: str
    segments: tuple[TranscriptSegment, ...]
    model_info: ModelInfo
    processing_info: ProcessingInfo
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        finite_number(self.duration_seconds, minimum=0.000001)
        probability(self.language_probability)
        if not isinstance(self.text, str) or not isinstance(self.transcript_id, str):
            raise InvalidTranscript("Transcript text and ID must be strings.")
        try:
            UUID(self.transcript_id)
        except ValueError as exc:
            raise InvalidTranscript("Transcript ID must be a UUID.") from exc
        if self.schema_version != "1.0":
            raise InvalidTranscript("Unsupported transcript schema version.")
        if self.language is not None and not isinstance(self.language, str):
            raise InvalidTranscript("Transcript language must be a string or None.")
        if not isinstance(self.model_info, ModelInfo) or not isinstance(
            self.processing_info, ProcessingInfo
        ):
            raise InvalidTranscript("Transcript model and processing metadata must be typed.")
        if not isinstance(self.segments, tuple) or any(
            not isinstance(segment, TranscriptSegment) for segment in self.segments
        ):
            raise InvalidTranscript("Transcript segments must be an immutable typed tuple.")
        if self.text.strip() and not self.segments:
            raise InvalidTranscript("Nonempty transcript must include segment timestamps.")
        previous_start = 0.0
        previous_word_start = 0.0
        for index, segment in enumerate(self.segments, 1):
            if segment.id != f"seg_{index:06d}":
                raise InvalidTranscript(
                    "Segment IDs must be sequential seg_000001, seg_000002, ..."
                )
            if segment.start < previous_start or segment.end > self.duration_seconds + 1.0:
                raise InvalidTranscript("Segments must be ordered and within audio duration.")
            previous_start = segment.start
            for word in segment.words:
                if word.start < previous_word_start or word.end > self.duration_seconds + 1.0:
                    raise InvalidTranscript(
                        "Global word timestamps must be ordered and within audio."
                    )
                previous_word_start = word.start

    @property
    def real_time_factor(self) -> float:
        return self.processing_info.transcription_seconds / self.duration_seconds

    @property
    def word_count(self) -> int:
        return sum(len(segment.words) for segment in self.segments)


@dataclass(frozen=True)
class TranscriptArtifacts:
    json_path: Path
    text_path: Path
