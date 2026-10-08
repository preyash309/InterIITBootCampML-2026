"""Frozen observations, explicit provider scores, evidence IDs and source bindings."""

import math
import re
from dataclasses import dataclass, fields, is_dataclass
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

from meeting_assistant.intelligence.models import AudioReference

from .exceptions import InvalidSemantics
from .ontology import EVENT_ONTOLOGY, POLICIES, RELATION_ONTOLOGY


def require(condition, message="Invalid semantic observation."):
    if not condition:
        raise InvalidSemantics(message)


def check_type(value, kind):
    origin, args = get_origin(kind), get_args(kind)
    if origin in (UnionType, Union):
        for option in args:
            try:
                check_type(value, option)
                return
            except InvalidSemantics:
                pass
        raise InvalidSemantics("Invalid optional field.")
    if origin is Literal:
        require(value in args)
    elif origin is tuple:
        require(type(value) is tuple, "Sequences must be immutable tuples.")
        if len(args) == 2 and args[1] is Ellipsis:
            for item in value:
                check_type(item, args[0])
        else:
            require(len(value) == len(args))
            for item, item_type in zip(value, args):
                check_type(item, item_type)
    elif kind is float:
        require(type(value) in (float, int) and math.isfinite(value) and value >= 0)
    elif is_dataclass(kind):
        require(isinstance(value, kind))
    else:
        require(type(value) is kind)


class Validated:
    def __post_init__(self):
        hints = get_type_hints(type(self))
        for field in fields(self):
            check_type(getattr(self, field.name), hints[field.name])


@dataclass(frozen=True)
class Question(Validated):
    id: str
    instructions: str
    criteria: tuple[tuple[str, str], ...]

    def __post_init__(self):
        super().__post_init__()
        require(bool(self.id) and bool(self.instructions) and 2 <= len(self.criteria) <= 255)
        require(len(dict(self.criteria)) == len(self.criteria))
        require(all(k and v for k, v in self.criteria))


@dataclass(frozen=True)
class ProviderDecision(Validated):
    question_id: str
    choice: str
    probabilities: tuple[tuple[str, float], ...]
    provider_score: float
    provider: str
    model: str
    api_schema: str
    request_sha256: str
    response_sha256: str
    latency_seconds: float
    policy_version: str

    def __post_init__(self):
        super().__post_init__()
        values = dict(self.probabilities)
        require(len(values) == len(self.probabilities) and self.choice in values)
        require(all(0 <= p <= 1 for p in values.values()))
        # Native FP16 softmax may sum to 1 +/- 0.001 after conversion to Python floats.
        # Preserve exposed scores; do not round or renormalize them.
        tolerance = 1e-3 if self.provider == "gliner" else 1e-4
        require(abs(sum(values.values()) - 1) <= tolerance and self.provider_score <= 1)
        require(values[self.choice] >= max(values.values()) - 1e-6)
        for value in (self.request_sha256, self.response_sha256):
            require(re.fullmatch(r"[a-f0-9]{64}", value) is not None)
        require(self.policy_version in POLICIES)

    @property
    def selected_probability(self):
        return dict(self.probabilities)[self.choice]


@dataclass(frozen=True)
class ProviderCall(Validated):
    purpose: str
    status_code: int | None
    latency_seconds: float
    retry: bool
    request_sha256: str
    response_sha256: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    retry_after_seconds: float | None = None

    def __post_init__(self):
        super().__post_init__()
        require(self.purpose in ("events", "relations", "verification", "coverage"))
        require(self.status_code is None or 100 <= self.status_code <= 599)
        require(all(x is None or x >= 0 for x in (self.input_tokens, self.output_tokens)))


@dataclass(frozen=True)
class Event(Validated):
    id: str
    event_type: str
    outcome: Literal["accepted", "ambiguous", "abstained", "unavailable"]
    text: str
    start: float
    end: float
    speaker_id: str | None
    evidence_utterance_ids: tuple[str, ...]
    context_utterance_ids: tuple[str, ...]
    decision: ProviderDecision | None
    speaker_reliability_status: Literal["RELIABLE", "MIXED", "UNCERTAIN", "UNAVAILABLE"] | None = (
        None
    )
    speaker_agreement_fraction: float | None = None

    def __post_init__(self):
        super().__post_init__()
        require(re.fullmatch(r"evt_\d{6}", self.id) is not None)
        require(self.event_type in dict(EVENT_ONTOLOGY) and self.start <= self.end)
        require(bool(self.text.strip()) and bool(self.evidence_utterance_ids))
        require(self.speaker_id is None or re.fullmatch(r"SPEAKER_\d{2,}", self.speaker_id))
        require(self.speaker_agreement_fraction is None or self.speaker_agreement_fraction <= 1)
        require(self.outcome != "accepted" or self.event_type not in ("NONE", "AMBIGUOUS"))
        require(self.decision is not None or self.outcome == "unavailable")
        require(self.decision is None or self.decision.choice == self.event_type)


@dataclass(frozen=True)
class Relation(Validated):
    id: str
    source_event_id: str
    target_event_id: str
    relation_type: str
    evidence_utterance_ids: tuple[str, ...]
    decision: ProviderDecision

    def __post_init__(self):
        super().__post_init__()
        require(self.relation_type in dict(RELATION_ONTOLOGY))
        require(self.relation_type not in ("NONE", "AMBIGUOUS"))
        require(self.source_event_id != self.target_event_id and bool(self.evidence_utterance_ids))
        require(self.decision.choice == self.relation_type)


@dataclass(frozen=True)
class RelationObservation(Validated):
    source_event_id: str
    target_event_id: str
    evidence_utterance_ids: tuple[str, ...]
    decision: ProviderDecision | None
    outcome: Literal["accepted", "ambiguous", "abstained", "unavailable"]

    def __post_init__(self):
        super().__post_init__()
        require(self.source_event_id != self.target_event_id and bool(self.evidence_utterance_ids))
        require(self.decision is not None or self.outcome == "unavailable")
        require(self.decision is None or self.decision.choice in dict(RELATION_ONTOLOGY))
        require(self.outcome != "accepted" or self.decision.choice not in ("NONE", "AMBIGUOUS"))


@dataclass(frozen=True)
class IssueThread(Validated):
    id: str
    representative_text: str
    event_ids: tuple[str, ...]


@dataclass(frozen=True)
class EventGraph(Validated):
    event_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    issue_threads: tuple[IssueThread, ...]
    cycles: tuple[tuple[str, ...], ...] = ()
    warnings: tuple[str, ...] = ()
    version: Literal["meeting_event_graph_v1"] = "meeting_event_graph_v1"


@dataclass(frozen=True)
class DecisionEvolution(Validated):
    issue_id: str
    ordered_event_ids: tuple[str, ...]
    current_decision_ids: tuple[str, ...]
    historical_decision_ids: tuple[str, ...]
    task_event_ids: tuple[str, ...]
    unresolved_proposal_ids: tuple[str, ...]


@dataclass(frozen=True)
class Verification(Validated):
    item_id: str
    status: Literal["SUPPORTED", "REVIEW", "UNSUPPORTED", "UNAVAILABLE"]
    evidence_utterance_ids: tuple[str, ...]
    dimensions: tuple[ProviderDecision, ...]
    warnings: tuple[str, ...] = ()
    policy_version: Literal["semantic_verification_v1"] = "semantic_verification_v1"


@dataclass(frozen=True)
class CoverageObservation(Validated):
    event_id: str
    event_type: str
    matched_record_ids: tuple[str, ...]
    status: Literal["covered", "missing", "ambiguous", "unavailable"]
    decisions: tuple[ProviderDecision, ...] = ()
    warnings: tuple[str, ...] = ()
    policy_version: Literal["coverage_v1"] = "coverage_v1"


@dataclass(frozen=True)
class Provenance(Validated):
    refined_id: str
    refined_sha256: str
    speaker_id: str
    speaker_sha256: str
    meeting_record_id: str
    meeting_record_sha256: str
    audio: AudioReference | None
    reliability_sha256: str | None = None
    context_sha256: str | None = None

    def __post_init__(self):
        super().__post_init__()
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("sha256") and value is not None:
                require(re.fullmatch(r"[a-f0-9]{64}", value))


@dataclass(frozen=True)
class Processing(Validated):
    stage_seconds: tuple[tuple[str, float], ...]
    calls: tuple[ProviderCall, ...]
    cached_requests: int
    event_candidates: int
    possible_relation_pairs: int
    evaluated_relation_pairs: int
    total_seconds: float
    runtime: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        super().__post_init__()
        require(
            all(
                getattr(self, name) >= 0
                for name in (
                    "cached_requests",
                    "event_candidates",
                    "possible_relation_pairs",
                    "evaluated_relation_pairs",
                )
            )
        )


@dataclass(frozen=True)
class SemanticResult(Validated):
    availability: Literal["available", "partial", "unavailable", "disabled"]
    provenance: Provenance
    events: tuple[Event, ...]
    relations: tuple[Relation, ...]
    event_graph: EventGraph
    decision_evolution: tuple[DecisionEvolution, ...]
    verification: tuple[Verification, ...]
    coverage: tuple[CoverageObservation, ...]
    processing: Processing
    configuration: tuple[tuple[str, str], ...] = ()
    relation_observations: tuple[RelationObservation, ...] = ()
    policies: tuple[str, ...] = POLICIES
    warnings: tuple[str, ...] = ()
    schema_version: Literal["1.0"] = "1.0"

    def __post_init__(self):
        super().__post_init__()
        require(self.policies == POLICIES)
        events = {e.id: e for e in self.events}
        require(
            len({(o.source_event_id, o.target_event_id) for o in self.relation_observations})
            == len(self.relation_observations)
        )
        for observation in self.relation_observations:
            a, b = events.get(observation.source_event_id), events.get(observation.target_event_id)
            require(a is not None and b is not None and a.outcome == b.outcome == "accepted")
            require(
                set(a.evidence_utterance_ids + b.evidence_utterance_ids)
                <= set(observation.evidence_utterance_ids)
            )
        require(len({v.item_id for v in self.verification}) == len(self.verification))
        require(len({o.event_id for o in self.coverage}) == len(self.coverage))
        from .graph import build_evolution, validate_graph

        validate_graph(self.events, self.relations, self.event_graph)
        require(
            self.decision_evolution
            == build_evolution(self.events, self.relations, self.event_graph),
            "Evolution does not match graph evidence.",
        )

    def get_verification(self, item_id: str):
        return next((v for v in self.verification if v.item_id == item_id), None)

    def get_decision_evolution(self, issue_id: str):
        return next((v for v in self.decision_evolution if v.issue_id == issue_id), None)
