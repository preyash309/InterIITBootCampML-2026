"""Immutable additive observations. Temporal fractions are not correctness probabilities."""

import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Literal

from meeting_assistant.diarization.models import WordReference

from .config import SpeakerReliabilityConfig
from .exceptions import InvalidReliability

State = Literal[
    "AGREE",
    "DISAGREE",
    "PRIMARY_ONLY",
    "SECONDARY_ONLY",
    "OVERLAP_AGREE",
    "OVERLAP_DISAGREE",
    "UNMAPPED_SECONDARY",
    "SILENCE",
]
Status = Literal["RELIABLE", "MIXED", "UNCERTAIN"]


class Observation:
    def __post_init__(self):
        validate(self)
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("_fraction") or field.name.endswith("_coverage"):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not 0 <= value <= 1
                ):
                    raise InvalidReliability("Temporal fractions must lie in [0, 1].")
        if hasattr(self, "status") and self.status not in (
            "RELIABLE",
            "MIXED",
            "UNCERTAIN",
            "mapped",
            "unmapped",
        ):
            raise InvalidReliability("Unknown reliability status.")


def validate(value):
    """Reject mutable containers and nonfinite observations at the public boundary."""
    if isinstance(value, (list, dict, set)):
        raise InvalidReliability("Evidence collections must be immutable tuples.")
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidReliability("Observations must be finite.")
    if isinstance(value, tuple):
        for item in value:
            validate(item)
    if is_dataclass(value):
        for field in fields(value):
            validate(getattr(value, field.name))


def interval(start, end):
    if (
        any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
            for x in (start, end)
        )
        or start < 0
        or end <= start
    ):
        raise InvalidReliability("Segments must have finite positive duration.")


@dataclass(frozen=True)
class SecondarySpeakerSegment:
    speaker_id: str
    start: float
    end: float

    def __post_init__(self):
        interval(self.start, self.end)
        if not isinstance(self.speaker_id, str) or not self.speaker_id:
            raise InvalidReliability("Secondary labels must be nonempty strings.")


@dataclass(frozen=True)
class SecondaryDiarizationResult:
    segments: tuple[SecondarySpeakerSegment, ...]
    backend: str
    model: str
    revision: str
    package_version: str
    device: str
    audio_sha256: str
    duration_seconds: float
    model_sha256: str
    model_load_seconds: float
    inference_seconds: float
    total_seconds: float
    peak_gpu_memory_bytes: int | None = None

    def __post_init__(self):
        validate(self)
        if not isinstance(self.segments, tuple) or any(
            not isinstance(x, SecondarySpeakerSegment) for x in self.segments
        ):
            raise InvalidReliability("Invalid secondary segments.")
        if self.duration_seconds <= 0 or self.device not in ("cuda", "cpu"):
            raise InvalidReliability("Invalid secondary recording/device.")
        if any(
            len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
            for x in (self.audio_sha256, self.model_sha256)
        ):
            raise InvalidReliability("Invalid secondary source/checkpoint fingerprint.")
        if any(not x for x in (self.backend, self.model, self.revision, self.package_version)):
            raise InvalidReliability("Secondary provenance is required.")
        if any(
            x < 0 for x in (self.model_load_seconds, self.inference_seconds, self.total_seconds)
        ):
            raise InvalidReliability("Secondary timings cannot be negative.")
        if any(x.end > self.duration_seconds + 1e-6 for x in self.segments):
            raise InvalidReliability("Secondary segments extend beyond the recording.")
        if self.segments != tuple(
            sorted(self.segments, key=lambda x: (x.start, x.end, x.speaker_id))
        ):
            raise InvalidReliability("Secondary segments must be deterministically ordered.")

    @property
    def speakers(self):
        return tuple(sorted({x.speaker_id for x in self.segments}))

    @property
    def speaker_count(self):
        return len(self.speakers)


@dataclass(frozen=True)
class SpeakerMapping(Observation):
    secondary_id: str
    primary_id: str | None
    overlap_seconds: float
    secondary_coverage: float
    primary_coverage: float
    status: Literal["mapped", "unmapped"]


@dataclass(frozen=True)
class SpeakerAlignment(Observation):
    mappings: tuple[SpeakerMapping, ...]
    primary_count: int
    secondary_count: int
    count_mismatch: bool
    unmapped_primary: tuple[str, ...]
    unmapped_secondary: tuple[str, ...]
    possible_splits: tuple[tuple[str, tuple[str, ...]], ...]
    possible_merges: tuple[tuple[str, tuple[str, ...]], ...]
    overlap_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ComparisonInterval:
    start: float
    end: float
    primary_speakers: tuple[str, ...]
    primary_exclusive_speaker: str | None
    secondary_speakers: tuple[str, ...]
    mapped_secondary_speakers: tuple[str, ...]
    state: State

    def __post_init__(self):
        interval(self.start, self.end)
        validate(self)
        if self.state not in (
            "AGREE",
            "DISAGREE",
            "PRIMARY_ONLY",
            "SECONDARY_ONLY",
            "OVERLAP_AGREE",
            "OVERLAP_DISAGREE",
            "UNMAPPED_SECONDARY",
            "SILENCE",
        ):
            raise InvalidReliability("Unknown comparison state.")


@dataclass(frozen=True)
class SpeakerWordReliability(Observation):
    source_word_reference: WordReference
    primary_speaker: str | None
    secondary_mapped_speaker: str | None
    agreement_fraction: float
    disagreement_fraction: float
    primary_only_fraction: float
    secondary_only_fraction: float
    silence_fraction: float
    primary_overlap_fraction: float
    secondary_overlap_fraction: float
    overlap_agreement_fraction: float
    unmapped_fraction: float
    status: Status
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class UtteranceSpeakerReliability(Observation):
    utterance_id: str
    primary_speaker: str | None
    secondary_mapped_speaker: str | None
    agreement_fraction: float
    disagreement_fraction: float
    primary_only_fraction: float
    secondary_only_fraction: float
    silence_fraction: float
    primary_overlap_fraction: float
    secondary_overlap_fraction: float
    overlap_agreement_fraction: float
    unmapped_fraction: float
    start_delta_seconds: float | None
    end_delta_seconds: float | None
    boundary_disagreement_seconds: float | None
    status: Status
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Provenance:
    canonical_audio_sha256: str
    primary_diarization_sha256: str
    speaker_transcript_sha256: str
    secondary_result_sha256: str | None
    primary_model: str
    primary_revision: str
    primary_package_version: str
    primary_device: str
    primary_inference_seconds: float
    configuration: SpeakerReliabilityConfig


@dataclass(frozen=True)
class DiarizationComparisonResult(Observation):
    alignment: SpeakerAlignment
    intervals: tuple[ComparisonInterval, ...]
    agreement_fraction: float
    speech_union_seconds: float
    silence_seconds: float
    state_seconds: tuple[tuple[str, float], ...]
    comparison_seconds: float


@dataclass(frozen=True)
class SpeakerReliabilityResult:
    availability: Literal["available", "unavailable"]
    provenance: Provenance
    secondary: SecondaryDiarizationResult | None
    comparison: DiarizationComparisonResult | None
    words: tuple[SpeakerWordReliability, ...]
    utterances: tuple[UtteranceSpeakerReliability, ...]
    warnings: tuple[str, ...]
    total_seconds: float
    schema_version: str = "1.0"

    def __post_init__(self):
        validate(self)
        if self.availability not in ("available", "unavailable"):
            raise InvalidReliability("Invalid reliability availability.")
        if (self.availability == "available") != (
            self.secondary is not None and self.comparison is not None
        ):
            raise InvalidReliability("Unavailable observations cannot claim completed comparison.")
        refs = tuple(
            (x.source_word_reference.segment_id, x.source_word_reference.word_index)
            for x in self.words
        )
        if len(set(refs)) != len(refs):
            raise InvalidReliability(
                "Each canonical word may have exactly one sidecar observation."
            )
        if self.availability == "unavailable" and (self.words or self.utterances):
            raise InvalidReliability(
                "Unavailable results cannot contain fabricated reliability scores."
            )

    def get_speaker_reliability(self, utterance_id: str):
        return next((x for x in self.utterances if x.utterance_id == utterance_id), None)
