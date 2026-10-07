"""Frozen candidate evidence; scores are heuristics, never truth probabilities."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from meeting_assistant.diarization.models import WordReference

from .exceptions import GroundingConfigurationError, InvalidGlossary, InvalidGroundingResult

DOMAINS = frozenset(
    (
        "ai_ml",
        "ml_systems",
        "software",
        "data",
        "retrieval",
        "cloud",
        "organizations",
        "business",
        "chemical",
        "mechanical",
        "electrical",
        "civil",
        "mathematics",
        "statistics",
        "computer_science",
        "academic",
        "meeting",
    )
)
SCOPES = ("global", "project", "meeting")


def _number(value: float, low: float = 0, high: float = math.inf) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise InvalidGroundingResult("Invalid finite numeric evidence.")


def _tuple(value: object, kind: type) -> None:
    if not isinstance(value, tuple) or any(not isinstance(item, kind) for item in value):
        raise InvalidGroundingResult("Evidence must use immutable typed tuples.")


@dataclass(frozen=True)
class GlossaryEntry:
    id: str
    canonical: str
    category: str
    domain: str
    description: str
    aliases: tuple[str, ...] = ()
    asr_aliases: tuple[str, ...] = ()
    acronym_expansion: str | None = None
    source: str = "global"
    priority: float = 0.5
    ambiguous_aliases: tuple[str, ...] = ()
    requires_context: bool = False
    safe_normalization: bool = False
    sense: str | None = None
    symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or re.fullmatch(r"[a-zA-Z0-9_.-]{1,120}", self.id) is None
            or any(
                not isinstance(v, str) or not v.strip()
                for v in (self.canonical, self.category, self.description)
            )
            or not isinstance(self.domain, str)
            or self.domain not in DOMAINS
            or not isinstance(self.source, str)
            or self.source not in SCOPES
        ):
            raise InvalidGlossary(
                "Entries need valid IDs, canonical names, domains, scopes and descriptions."
            )
        for values in (self.aliases, self.asr_aliases, self.ambiguous_aliases, self.symbols):
            if not isinstance(values, tuple) or any(
                not isinstance(v, str) or not v.strip() for v in values
            ):
                raise InvalidGlossary("Aliases must be nonempty strings in immutable tuples.")
        if any(not isinstance(v, bool) for v in (self.requires_context, self.safe_normalization)):
            raise InvalidGlossary("Entry policies must be booleans.")
        if (
            not isinstance(self.priority, (float, int))
            or isinstance(self.priority, bool)
            or not math.isfinite(self.priority)
            or not 0 <= self.priority <= 1
        ):
            raise InvalidGlossary("Priority must be between zero and one.")
        if any(
            v is not None and (not isinstance(v, str) or not v.strip())
            for v in (self.acronym_expansion, self.sense)
        ):
            raise InvalidGlossary("Optional acronym and sense fields must contain text.")


@dataclass(frozen=True)
class GroundingCandidate:
    entry_id: str
    canonical: str
    scope: str
    domain: str
    category: str
    description: str
    matched_variant: str
    match_type: str
    score: float
    lexical_score: float
    phonetic_score: float
    semantic_score: float
    scope_score: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(v, str) or not v
                for v in (
                    self.entry_id,
                    self.canonical,
                    self.category,
                    self.description,
                    self.matched_variant,
                )
            )
            or self.scope not in SCOPES
            or self.domain not in DOMAINS
        ):
            raise InvalidGroundingResult("Invalid candidate identity or provenance.")
        if self.match_type not in (
            "canonical_exact",
            "alias_exact",
            "asr_alias_exact",
            "normalized_exact",
            "fuzzy",
        ):
            raise InvalidGroundingResult("Unknown retrieval match type.")
        for v in (
            self.score,
            self.lexical_score,
            self.phonetic_score,
            self.semantic_score,
            self.scope_score,
        ):
            _number(v, 0, 1)
        _tuple(self.reasons, str)


@dataclass(frozen=True)
class GroundingRecord:
    id: str
    utterance_id: str
    speaker_id: str | None
    start: float
    end: float
    char_start: int
    char_end: int
    observed_text: str
    normalized_text: str
    context: str
    source_word_references: tuple[WordReference, ...]
    source_segment_ids: tuple[str, ...]
    candidates: tuple[GroundingCandidate, ...]

    def __post_init__(self) -> None:
        _number(self.start)
        _number(self.end, self.start)
        if (
            not isinstance(self.id, str)
            or not isinstance(self.utterance_id, str)
            or re.fullmatch(r"grnd_\d{6,}", self.id) is None
            or re.fullmatch(r"utt_\d{6,}", self.utterance_id) is None
        ):
            raise InvalidGroundingResult("Record and utterance IDs must be deterministic.")
        if self.speaker_id is not None and (
            not isinstance(self.speaker_id, str)
            or re.fullmatch(r"SPEAKER_\d{2,}", self.speaker_id) is None
        ):
            raise InvalidGroundingResult("Speaker references must use anonymous labels.")
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in (self.char_start, self.char_end)
            )
            or not 0 <= self.char_start < self.char_end
        ):
            raise InvalidGroundingResult("Invalid observed-text offsets.")
        if any(
            not isinstance(v, str) or not v
            for v in (self.observed_text, self.normalized_text, self.context)
        ):
            raise InvalidGroundingResult("Observed evidence and context must be text.")
        _tuple(self.source_word_references, WordReference)
        _tuple(self.source_segment_ids, str)
        _tuple(self.candidates, GroundingCandidate)
        if (
            not self.candidates
            or not self.source_segment_ids
            or len(set(c.entry_id for c in self.candidates)) != len(self.candidates)
        ):
            raise InvalidGroundingResult(
                "Candidate and source references must be nonempty and unique."
            )
        if any(
            ref.segment_id not in self.source_segment_ids for ref in self.source_word_references
        ):
            raise InvalidGroundingResult("Word references must belong to cited ASR segments.")


@dataclass(frozen=True)
class GroundingProcessingInfo:
    embedding_model_load_seconds: float
    glossary_load_seconds: float
    index_load_seconds: float
    grounding_seconds: float
    total_seconds: float
    utterance_count: int
    considered_spans: int
    embedding_cache_hit: bool

    def __post_init__(self) -> None:
        for v in (
            self.embedding_model_load_seconds,
            self.glossary_load_seconds,
            self.index_load_seconds,
            self.grounding_seconds,
            self.total_seconds,
        ):
            _number(v)
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0
            for v in (self.utterance_count, self.considered_spans)
        ) or not isinstance(self.embedding_cache_hit, bool):
            raise InvalidGroundingResult("Invalid processing statistics.")


@dataclass(frozen=True)
class RetrievalPolicy:
    top_k: int
    min_score: float
    lexical_floor: float
    phonetic_floor: float
    context_floor: float
    max_span_words: int
    context_chars: int
    neighbor_gap_seconds: float
    lexical_weight: float
    phonetic_weight: float
    semantic_weight: float
    scope_weight: float
    policy_version: str = "alias-first-v1"

    def __post_init__(self) -> None:
        from .config import GroundingConfig

        try:
            GroundingConfig(
                **{
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "policy_version"
                }
            )
        except GroundingConfigurationError as exc:
            raise InvalidGroundingResult("Invalid recorded retrieval policy.") from exc
        if self.policy_version != "alias-first-v1":
            raise InvalidGroundingResult("Unsupported retrieval policy.")


@dataclass(frozen=True)
class GroundingResult:
    grounding_id: str
    source_speaker_transcript_id: str
    source_speaker_sha256: str
    source_raw_transcript_id: str
    source_raw_sha256: str
    glossary_version: str
    embedding_model: str
    embedding_revision: str
    records: tuple[GroundingRecord, ...]
    processing_info: GroundingProcessingInfo
    retrieval_policy: RetrievalPolicy
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        try:
            for value in (
                self.grounding_id,
                self.source_speaker_transcript_id,
                self.source_raw_transcript_id,
            ):
                UUID(value)
        except (ValueError, TypeError, AttributeError) as exc:
            raise InvalidGroundingResult("Transcript IDs must be UUIDs.") from exc
        for value in (self.source_speaker_sha256, self.source_raw_sha256, self.glossary_version):
            if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
                raise InvalidGroundingResult("Provenance requires SHA-256 digests.")
        if (
            self.schema_version != "1.0"
            or not isinstance(self.processing_info, GroundingProcessingInfo)
            or not isinstance(self.retrieval_policy, RetrievalPolicy)
            or not isinstance(self.embedding_model, str)
            or not self.embedding_model
            or not isinstance(self.embedding_revision, str)
            or not self.embedding_revision
        ):
            raise InvalidGroundingResult("Invalid grounding schema/metadata.")
        _tuple(self.records, GroundingRecord)
        last = {}
        for index, record in enumerate(self.records, 1):
            if record.id != f"grnd_{index:06d}" or record.char_start < last.get(
                record.utterance_id, 0
            ):
                raise InvalidGroundingResult(
                    "Records must be ordered and nonoverlapping within utterances."
                )
            last[record.utterance_id] = record.char_end


@dataclass(frozen=True)
class GroundingArtifacts:
    json_path: Path
    text_path: Path
