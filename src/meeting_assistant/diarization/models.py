"""Frozen diarization and derived speaker evidence, separate from raw ASR models."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from meeting_assistant.asr.models import TranscriptWord

from .config import DiarizationOptions, ReconciliationConfig
from .exceptions import InvalidDiarizationResult


def finite(value: object, minimum: float = 0) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise InvalidDiarizationResult("Diarization contains an invalid numeric value.")


def interval(start: float, end: float) -> None:
    finite(start)
    finite(end, start)


def uuid(value: str) -> None:
    try:
        UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise InvalidDiarizationResult("Record identifiers must be UUIDs.") from exc


def immutable_tuple(value: object, kind: type) -> None:
    if not isinstance(value, tuple) or any(not isinstance(item, kind) for item in value):
        raise InvalidDiarizationResult("Evidence sequences must be immutable typed tuples.")


@dataclass(frozen=True)
class SpeakerTurn:
    id: str
    speaker_id: str
    start: float
    end: float

    def __post_init__(self) -> None:
        interval(self.start, self.end)
        if (
            self.end == self.start
            or not isinstance(self.id, str)
            or not isinstance(self.speaker_id, str)
        ):
            raise InvalidDiarizationResult(
                "Speaker turns require positive duration and string labels."
            )


@dataclass(frozen=True)
class DiarizationModelInfo:
    model: str
    revision: str
    package_version: str
    device: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.model, self.revision, self.package_version)
        ) or self.device not in ("cuda", "cpu"):
            raise InvalidDiarizationResult("Invalid diarization model metadata.")


@dataclass(frozen=True)
class DiarizationProcessingInfo:
    model_load_seconds: float
    diarization_seconds: float
    total_seconds: float
    peak_gpu_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for value in (self.model_load_seconds, self.diarization_seconds, self.total_seconds):
            finite(value)
        if self.total_seconds < self.diarization_seconds:
            raise InvalidDiarizationResult("Total time must include inference time.")
        if self.peak_gpu_memory_bytes is not None:
            finite(self.peak_gpu_memory_bytes)
            if not isinstance(self.peak_gpu_memory_bytes, int):
                raise InvalidDiarizationResult("GPU memory must be measured in integer bytes.")


@dataclass(frozen=True)
class DiarizationResult:
    diarization_id: str
    duration_seconds: float
    audio_sha256: str
    speakers: tuple[str, ...]
    regular_turns: tuple[SpeakerTurn, ...]
    exclusive_turns: tuple[SpeakerTurn, ...]
    model_info: DiarizationModelInfo
    processing_info: DiarizationProcessingInfo
    options: DiarizationOptions
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        uuid(self.diarization_id)
        finite(self.duration_seconds, 0.000001)
        immutable_tuple(self.speakers, str)
        if self.speakers != tuple(f"SPEAKER_{index:02d}" for index in range(len(self.speakers))):
            raise InvalidDiarizationResult("Speaker labels must be sequential anonymous IDs.")
        if (
            not isinstance(self.audio_sha256, str)
            or len(self.audio_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.audio_sha256)
        ):
            raise InvalidDiarizationResult("Canonical audio must have a SHA-256 provenance digest.")
        if (
            not isinstance(self.model_info, DiarizationModelInfo)
            or not isinstance(self.processing_info, DiarizationProcessingInfo)
            or not isinstance(self.options, DiarizationOptions)
        ):
            raise InvalidDiarizationResult("Diarization metadata must use typed models.")
        if self.schema_version != "1.0":
            raise InvalidDiarizationResult("Unsupported diarization schema version.")
        seen = set()
        for name, turns in (("regular", self.regular_turns), ("exclusive", self.exclusive_turns)):
            immutable_tuple(turns, SpeakerTurn)
            previous_start = previous_end = 0.0
            for index, turn in enumerate(turns, 1):
                if turn.id != f"{name}_turn_{index:06d}" or turn.speaker_id not in self.speakers:
                    raise InvalidDiarizationResult("Turn IDs or speaker references are invalid.")
                if turn.start < previous_start or turn.end > self.duration_seconds + 0.5:
                    raise InvalidDiarizationResult(
                        "Speaker turns must be ordered and within audio duration."
                    )
                if name == "exclusive" and turn.start < previous_end - 1e-9:
                    raise InvalidDiarizationResult("Exclusive speaker turns cannot overlap.")
                previous_start, previous_end = turn.start, turn.end
                seen.add(turn.speaker_id)
        if seen != set(self.speakers):
            raise InvalidDiarizationResult(
                "Every declared speaker must occur in diarization turns."
            )

    @property
    def real_time_factor(self) -> float:
        return self.processing_info.diarization_seconds / self.duration_seconds

    @property
    def detected_speaker_count(self) -> int:
        return len(self.speakers)


@dataclass(frozen=True)
class WordReference:
    segment_id: str
    word_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.segment_id, str)
            or re.fullmatch(r"seg_\d{6,}", self.segment_id) is None
            or isinstance(self.word_index, bool)
            or not isinstance(self.word_index, int)
            or self.word_index < 0
        ):
            raise InvalidDiarizationResult(
                "Word references require a segment ID and zero-based index."
            )


AttributionMethod = Literal["overlap", "tolerance", "nearest", "unknown"]


@dataclass(frozen=True)
class SpeakerWord:
    word: TranscriptWord
    source_word_reference: WordReference
    speaker_id: str | None
    attribution_method: AttributionMethod
    overlap_fraction: float | None
    source_turn_ids: tuple[str, ...]
    overlap_present: bool

    def __post_init__(self) -> None:
        if not isinstance(self.word, TranscriptWord) or not isinstance(
            self.source_word_reference, WordReference
        ):
            raise InvalidDiarizationResult("Speaker words must reference typed raw ASR evidence.")
        immutable_tuple(self.source_turn_ids, str)
        if self.overlap_fraction is not None:
            finite(self.overlap_fraction)
            if self.overlap_fraction > 1:
                raise InvalidDiarizationResult("Overlap fraction cannot exceed one.")
        if self.attribution_method not in (
            "overlap",
            "tolerance",
            "nearest",
            "unknown",
        ) or not isinstance(self.overlap_present, bool):
            raise InvalidDiarizationResult("Invalid word attribution metadata.")
        if (self.speaker_id is None) != (self.attribution_method == "unknown"):
            raise InvalidDiarizationResult("Unknown attribution must have no speaker label.")
        if self.speaker_id is not None and not isinstance(self.speaker_id, str):
            raise InvalidDiarizationResult("Speaker label must be a string or None.")
        if (self.speaker_id is None and self.source_turn_ids) or (
            self.speaker_id is not None and not self.source_turn_ids
        ):
            raise InvalidDiarizationResult("Assigned words must retain their source turns.")

    @property
    def text(self) -> str:
        return self.word.text

    @property
    def start(self) -> float:
        return self.word.start

    @property
    def end(self) -> float:
        return self.word.end

    @property
    def asr_probability(self) -> float | None:
        return self.word.probability


@dataclass(frozen=True)
class SpeakerUtterance:
    id: str
    speaker_id: str | None
    start: float
    end: float
    text: str
    words: tuple[SpeakerWord, ...]
    source_segment_ids: tuple[str, ...]
    overlap_present: bool
    attribution_method: Literal[
        "words", "segment_overlap", "segment_tolerance", "segment_nearest", "unknown"
    ] = "words"

    def __post_init__(self) -> None:
        interval(self.start, self.end)
        immutable_tuple(self.words, SpeakerWord)
        immutable_tuple(self.source_segment_ids, str)
        if (
            not isinstance(self.text, str)
            or not isinstance(self.overlap_present, bool)
            or not self.source_segment_ids
        ):
            raise InvalidDiarizationResult(
                "Utterances require text and original segment references."
            )
        if any(word.speaker_id != self.speaker_id for word in self.words):
            raise InvalidDiarizationResult("Utterance cannot merge different speaker assignments.")
        if self.attribution_method not in (
            "words",
            "segment_overlap",
            "segment_tolerance",
            "segment_nearest",
            "unknown",
        ) or bool(self.words) != (self.attribution_method == "words"):
            raise InvalidDiarizationResult("Utterance attribution must describe its evidence.")
        if len(set(self.source_segment_ids)) != len(self.source_segment_ids):
            raise InvalidDiarizationResult("Source segment references must be unique.")
        if self.words and (
            self.start != self.words[0].start or self.end != max(word.end for word in self.words)
        ):
            raise InvalidDiarizationResult("Utterance bounds must retain original word times.")
        if self.words and self.source_segment_ids != tuple(
            dict.fromkeys(word.source_word_reference.segment_id for word in self.words)
        ):
            raise InvalidDiarizationResult("Utterance references must match its raw word evidence.")
        previous_start = self.start
        for word in self.words:
            if word.start < previous_start:
                raise InvalidDiarizationResult("Utterance word evidence must be ordered.")
            previous_start = word.start


@dataclass(frozen=True)
class OverlapRegion:
    start: float
    end: float
    speakers: tuple[str, ...]

    def __post_init__(self) -> None:
        interval(self.start, self.end)
        immutable_tuple(self.speakers, str)
        if self.start == self.end or len(set(self.speakers)) < 2:
            raise InvalidDiarizationResult(
                "Overlap regions require simultaneous distinct speakers."
            )


@dataclass(frozen=True)
class ReconciliationInfo:
    total_words: int
    direct_assignments: int
    tolerance_assignments: int
    nearest_assignments: int
    unassigned_words: int
    segment_fallback_count: int
    speaker_switches: int
    config: ReconciliationConfig

    def __post_init__(self) -> None:
        for value in (
            self.total_words,
            self.direct_assignments,
            self.tolerance_assignments,
            self.nearest_assignments,
            self.unassigned_words,
            self.segment_fallback_count,
            self.speaker_switches,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidDiarizationResult(
                    "Reconciliation counts must be nonnegative integers."
                )
        if (
            self.direct_assignments
            + self.tolerance_assignments
            + self.nearest_assignments
            + self.unassigned_words
            != self.total_words
        ):
            raise InvalidDiarizationResult("Word attribution counts must sum to total words.")
        if not isinstance(self.config, ReconciliationConfig):
            raise InvalidDiarizationResult("Reconciliation configuration must be typed.")

    @property
    def assignment_coverage(self) -> float | None:
        return (
            (self.total_words - self.unassigned_words) / self.total_words
            if self.total_words
            else None
        )


@dataclass(frozen=True)
class SpeakerAwareTranscript:
    transcript_id: str
    source_raw_transcript_id: str
    source_raw_sha256: str
    diarization_id: str
    duration_seconds: float
    speakers: tuple[str, ...]
    utterances: tuple[SpeakerUtterance, ...]
    overlap_regions: tuple[OverlapRegion, ...]
    reconciliation_info: ReconciliationInfo
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        for value in (self.transcript_id, self.source_raw_transcript_id, self.diarization_id):
            uuid(value)
        finite(self.duration_seconds, 0.000001)
        immutable_tuple(self.speakers, str)
        immutable_tuple(self.utterances, SpeakerUtterance)
        immutable_tuple(self.overlap_regions, OverlapRegion)
        if self.speakers != tuple(f"SPEAKER_{i:02d}" for i in range(len(self.speakers))):
            raise InvalidDiarizationResult("Speaker labels must be sequential anonymous IDs.")
        previous_end = 0.0
        for region in self.overlap_regions:
            if (
                region.start < previous_end
                or region.end > self.duration_seconds + 0.5
                or any(name not in self.speakers for name in region.speakers)
            ):
                raise InvalidDiarizationResult(
                    "Overlap regions must be ordered and reference speakers."
                )
            previous_end = region.end
        if (
            not isinstance(self.reconciliation_info, ReconciliationInfo)
            or self.schema_version != "1.0"
        ):
            raise InvalidDiarizationResult("Invalid reconciliation metadata or schema version.")
        if (
            not isinstance(self.source_raw_sha256, str)
            or len(self.source_raw_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_raw_sha256)
        ):
            raise InvalidDiarizationResult("Raw transcript provenance digest is invalid.")
        references = []
        for index, utterance in enumerate(self.utterances, 1):
            if utterance.id != f"utt_{index:06d}" or (
                utterance.speaker_id is not None and utterance.speaker_id not in self.speakers
            ):
                raise InvalidDiarizationResult("Utterance ID or speaker reference is invalid.")
            if utterance.end > self.duration_seconds + 1:
                raise InvalidDiarizationResult("Utterance extends beyond canonical audio.")
            for word in utterance.words:
                reference = (
                    word.source_word_reference.segment_id,
                    word.source_word_reference.word_index,
                )
                if reference[0] not in utterance.source_segment_ids:
                    raise InvalidDiarizationResult(
                        "Word reference is absent from source segment IDs."
                    )
                references.append(reference)
        if (
            len(references) != len(set(references))
            or len(references) != self.reconciliation_info.total_words
        ):
            raise InvalidDiarizationResult(
                "Speaker transcript must retain each timestamped word exactly once."
            )

    @property
    def detected_speaker_count(self) -> int:
        return len(self.speakers)


@dataclass(frozen=True)
class SpeakerTranscriptArtifacts:
    diarization_path: Path
    json_path: Path
    text_path: Path
