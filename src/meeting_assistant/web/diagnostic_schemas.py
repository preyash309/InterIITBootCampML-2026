"""Path-free presentation schemas; canonical and ML schemas stay unchanged."""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict


class DTO(BaseModel):
    model_config = ConfigDict(extra="forbid")


T = TypeVar("T")


class Sidecar(DTO, Generic[T]):
    state: Literal["available", "not_generated", "unavailable", "failed"]
    message: str
    data: T | None = None


class SpeakerObservation(DTO):
    utterance_id: str
    primary_speaker: str | None
    secondary_mapped_speaker: str | None
    status: Literal["RELIABLE", "MIXED", "UNCERTAIN", "UNAVAILABLE"]
    agreement_fraction: float
    disagreement_fraction: float
    primary_overlap_fraction: float
    secondary_overlap_fraction: float
    boundary_disagreement_seconds: float | None
    reasons: list[str]


class SpeakerReliabilityResult(DTO):
    primary_model: str
    secondary_model: str | None
    primary_count: int | None
    secondary_count: int | None
    count_mismatch: bool
    possible_merges: list[str]
    utterances: list[SpeakerObservation]


class ContextAudit(DTO):
    id: str
    utterance_ids: list[str]
    start: float
    end: float
    pass1_text: str
    pass2_text: str
    candidates: list[str]
    sources: list[str]
    provider: str
    model: str
    status: str
    outcomes: list[str]
    unresolved: bool


class ContextualASRResult(DTO):
    title: str | None
    terms: list[str]
    participants: list[str]
    sources: list[str]
    hypotheses: list[ContextAudit]


class MeetingEvent(DTO):
    id: str
    event_type: str
    text: str
    start: float
    end: float
    speaker_id: str | None
    evidence_utterance_ids: list[str]


class MeetingEventRelation(DTO):
    id: str
    source_event_id: str
    target_event_id: str
    relation_type: str
    evidence_utterance_ids: list[str]


class DecisionEvolution(DTO):
    issue_id: str
    title: str
    ordered_event_ids: list[str]
    current_decision_ids: list[str]
    historical_decision_ids: list[str]


class SemanticCheck(DTO):
    question: str
    choice: str
    provider: str
    model: str


class SemanticVerification(DTO):
    item_id: str
    status: Literal["SUPPORTED", "REVIEW", "UNSUPPORTED", "UNAVAILABLE"]
    evidence_utterance_ids: list[str]
    dimensions: list[SemanticCheck]


class CoverageObservation(DTO):
    event_id: str
    event_type: str
    status: str
    matched_record_ids: list[str]


class SemanticResult(DTO):
    availability: str
    models: list[str]
    events: list[MeetingEvent]
    relations: list[MeetingEventRelation]
    decision_evolution: list[DecisionEvolution]
    verification: list[SemanticVerification]
    coverage: list[CoverageObservation]
