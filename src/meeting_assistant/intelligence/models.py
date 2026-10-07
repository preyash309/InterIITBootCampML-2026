"""Immutable canonical claims reference source IDs, never model-generated quotes/times."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from meeting_assistant.diarization.models import WordReference
from meeting_assistant.refinement.models import RefinedUtterance

from .exceptions import IntelligenceValidationError

POLICY_VERSION = "intelligence_v1"
SCHEMA_VERSION = "meeting_record_schema_v1"
EXAMPLES_VERSION = "meeting_examples_v1"
EVIDENCE_VERSION = "evidence_policy_v1"
CONSOLIDATION_VERSION = "consolidation_v1"
SECTIONS = ("summary", "minutes", "decisions", "action_items")
KINDS = ("discussion", "proposal", "concern", "information", "decision", "action")


def require(condition, message):
    if not condition:
        raise IntelligenceValidationError(message)


def finite(value, minimum=0):
    require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= minimum,
        "Invalid finite measurement.",
    )


def text(value, maximum=4000):
    require(
        isinstance(value, str) and bool(value.strip()) and len(value) <= maximum,
        "Expected bounded nonempty text.",
    )


def digest(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "Expected a SHA-256 provenance digest.",
    )


def uuid(value):
    try:
        UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise IntelligenceValidationError("Expected a UUID.") from exc


def typed_tuple(value, kind):
    require(
        isinstance(value, tuple) and all(isinstance(v, kind) for v in value),
        "Evidence must use immutable typed tuples.",
    )


@dataclass(frozen=True)
class ActionOwner:
    kind: str
    speaker_id: str | None = None
    display_text: str | None = None

    def __post_init__(self):
        if self.kind == "speaker":
            require(
                isinstance(self.speaker_id, str)
                and re.fullmatch(r"SPEAKER_\d{2,}", self.speaker_id) is not None
                and self.display_text is None,
                "Malformed anonymous speaker owner.",
            )
        else:
            require(
                self.kind == "named_entity" and self.speaker_id is None,
                "Owner must be an anonymous speaker or explicit named entity/team.",
            )
            text(self.display_text, 200)


def item_fields(item):
    require(
        isinstance(item.id, str)
        and re.fullmatch(r"(?:p\d{4}_)?(?:sum|min|dec|act)_\d{4,}", item.id) is not None,
        "Invalid deterministic item ID.",
    )
    text(item.text)
    typed_tuple(item.evidence_utterance_ids, str)
    require(
        bool(item.evidence_utterance_ids)
        and len(set(item.evidence_utterance_ids)) == len(item.evidence_utterance_ids)
        and all(re.fullmatch(r"utt_\d{6,}", u) for u in item.evidence_utterance_ids),
        "Every item requires unique source utterance IDs.",
    )


@dataclass(frozen=True)
class SummaryPoint:
    id: str
    text: str
    evidence_utterance_ids: tuple[str, ...]

    def __post_init__(self):
        item_fields(self)


@dataclass(frozen=True)
class Decision(SummaryPoint):
    pass


@dataclass(frozen=True)
class MinuteItem(SummaryPoint):
    topic: str
    kind: str

    def __post_init__(self):
        super().__post_init__()
        text(self.topic, 200)
        require(self.kind in KINDS, "Unknown minute classification.")


@dataclass(frozen=True)
class ActionItem:
    id: str
    task: str
    evidence_utterance_ids: tuple[str, ...]
    owner: ActionOwner | None = None
    deadline_text: str | None = None

    @property
    def text(self):
        return self.task

    def __post_init__(self):
        item_fields(self)
        require(self.owner is None or isinstance(self.owner, ActionOwner), "Invalid owner type.")
        if self.deadline_text is not None:
            text(self.deadline_text, 200)


@dataclass(frozen=True)
class MeetingContent:
    summary: tuple[SummaryPoint, ...] = ()
    minutes: tuple[MinuteItem, ...] = ()
    decisions: tuple[Decision, ...] = ()
    action_items: tuple[ActionItem, ...] = ()

    def __post_init__(self):
        for name, kind in zip(SECTIONS, (SummaryPoint, MinuteItem, Decision, ActionItem)):
            typed_tuple(getattr(self, name), kind)
        require(len({i.id for i in self.items}) == len(self.items), "Duplicate item IDs.")

    @property
    def items(self):
        return tuple(item for name in SECTIONS for item in getattr(self, name))


@dataclass(frozen=True)
class IntelligenceModelInfo:
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    structured_mode: str = "json_schema"
    reasoning_effort: str | None = "low"
    max_completion_tokens: int = 8192
    include_reasoning: bool = False
    prompt_version: str = POLICY_VERSION
    examples_version: str = EXAMPLES_VERSION
    schema_version: str = SCHEMA_VERSION
    consolidation_version: str = CONSOLIDATION_VERSION

    def __post_init__(self):
        text(self.provider, 100)
        text(self.model, 120)
        require(
            self.temperature == 0
            and not isinstance(self.temperature, bool)
            and self.structured_mode == "json_schema"
            and self.reasoning_effort in (None, "low")
            and self.include_reasoning is False,
            "Invalid generation baseline.",
        )
        require(
            type(self.max_completion_tokens) is int and 256 <= self.max_completion_tokens <= 16000,
            "Invalid token budget.",
        )
        require(
            (
                self.prompt_version,
                self.examples_version,
                self.schema_version,
                self.consolidation_version,
            )
            == (POLICY_VERSION, EXAMPLES_VERSION, SCHEMA_VERSION, CONSOLIDATION_VERSION),
            "Unsupported policy versions.",
        )


@dataclass(frozen=True)
class IntelligenceRequest:
    request_id: str
    utterances: tuple[RefinedUtterance, ...]
    stage: str = "extraction"
    partial: MeetingContent | None = None

    def __post_init__(self):
        text(self.request_id, 100)
        typed_tuple(self.utterances, RefinedUtterance)
        require(
            len({u.utterance_id for u in self.utterances}) == len(self.utterances),
            "Duplicate request utterances.",
        )
        require(self.stage in ("extraction", "consolidation"), "Invalid extraction role.")
        require(
            (self.stage == "extraction" and self.partial is None)
            or (
                self.stage == "consolidation"
                and isinstance(self.partial, MeetingContent)
                and not self.utterances
            ),
            "Malformed consolidation request.",
        )


@dataclass(frozen=True)
class IntelligenceCall:
    request_id: str
    stage: str
    latency_seconds: float
    request_bytes: int
    response_bytes: int
    status_code: int
    schema_repair: bool = False
    transport_retry: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self):
        text(self.request_id, 100)
        require(self.stage in ("extraction", "consolidation"), "Invalid call stage.")
        finite(self.latency_seconds)
        for name in (
            "request_bytes",
            "response_bytes",
            "status_code",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            value = getattr(self, name)
            require(
                value is None and name.endswith("tokens") or type(value) is int and value >= 0,
                "Invalid provider measurement.",
            )
        require(
            type(self.schema_repair) is bool and type(self.transport_retry) is bool,
            "Invalid retry measurements.",
        )


@dataclass(frozen=True)
class IntelligenceResponse:
    request_id: str
    content: MeetingContent
    model_info: IntelligenceModelInfo
    calls: tuple[IntelligenceCall, ...] = ()

    def __post_init__(self):
        require(
            isinstance(self.content, MeetingContent)
            and isinstance(self.model_info, IntelligenceModelInfo),
            "Invalid backend result.",
        )
        typed_tuple(self.calls, IntelligenceCall)


@dataclass(frozen=True)
class IntelligenceProcessingInfo:
    total_seconds: float
    chunk_count: int
    calls: tuple[IntelligenceCall, ...] = ()

    def __post_init__(self):
        finite(self.total_seconds)
        require(type(self.chunk_count) is int and self.chunk_count >= 0, "Invalid chunk count.")
        typed_tuple(self.calls, IntelligenceCall)

    @property
    def provider_latency_seconds(self):
        return sum(c.latency_seconds for c in self.calls)

    def usage(self, field="total_tokens"):
        require(
            field in ("prompt_tokens", "completion_tokens", "total_tokens"),
            "Unknown token measurement.",
        )
        values = tuple(getattr(c, field) for c in self.calls)
        return sum(values) if all(v is not None for v in values) else None


@dataclass(frozen=True)
class AudioReference:
    sha256: str
    duration_seconds: float
    frame_count: int
    artifact_name: str = "canonical.wav"

    def __post_init__(self):
        digest(self.sha256)
        finite(self.duration_seconds, 1 / 16000)
        require(
            type(self.frame_count) is int
            and self.frame_count > 0
            and abs(self.duration_seconds - self.frame_count / 16000) < 1e-9
            and self.artifact_name == "canonical.wav",
            "Invalid canonical audio binding.",
        )


@dataclass(frozen=True)
class MeetingRecord:
    id: str
    source_refined_id: str
    source_refined_sha256: str
    source_speaker_id: str
    source_speaker_sha256: str
    source_grounding_id: str
    source_grounding_sha256: str
    source_raw_id: str
    source_raw_sha256: str
    content: MeetingContent
    model_info: IntelligenceModelInfo
    processing_info: IntelligenceProcessingInfo
    audio: AudioReference | None = None
    policy_version: str = POLICY_VERSION
    evidence_policy_version: str = EVIDENCE_VERSION
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        for value in (
            self.id,
            self.source_refined_id,
            self.source_speaker_id,
            self.source_grounding_id,
            self.source_raw_id,
        ):
            uuid(value)
        for value in (
            self.source_refined_sha256,
            self.source_speaker_sha256,
            self.source_grounding_sha256,
            self.source_raw_sha256,
        ):
            digest(value)
        require(
            isinstance(self.content, MeetingContent)
            and isinstance(self.model_info, IntelligenceModelInfo)
            and isinstance(self.processing_info, IntelligenceProcessingInfo)
            and (self.audio is None or isinstance(self.audio, AudioReference)),
            "Meeting record requires typed metadata.",
        )
        require(
            (self.policy_version, self.evidence_policy_version, self.schema_version)
            == (POLICY_VERSION, EVIDENCE_VERSION, SCHEMA_VERSION),
            "Unsupported record version.",
        )
        for section, prefix in zip(SECTIONS, ("sum", "min", "dec", "act")):
            require(
                tuple(i.id for i in getattr(self.content, section))
                == tuple(
                    f"{prefix}_{n:04d}" for n in range(1, len(getattr(self.content, section)) + 1)
                ),
                "Final IDs must be sequential application-assigned IDs.",
            )

    @property
    def summary(self):
        return self.content.summary

    @property
    def minutes(self):
        return self.content.minutes

    @property
    def decisions(self):
        return self.content.decisions

    @property
    def action_items(self):
        return self.content.action_items


@dataclass(frozen=True)
class EvidenceSpan:
    utterance_id: str
    speaker_id: str | None
    start: float
    end: float
    raw_text: str
    refined_text: str
    source_word_refs: tuple[WordReference, ...]
    applied_edit_ids: tuple[str, ...]
    source_segment_ids: tuple[str, ...]

    def __post_init__(self):
        finite(self.start)
        finite(self.end, self.start)
        typed_tuple(self.source_word_refs, WordReference)
        typed_tuple(self.applied_edit_ids, str)
        typed_tuple(self.source_segment_ids, str)


@dataclass(frozen=True)
class EvidenceClip:
    path: Path
    source_audio_sha256: str
    evidence_start: float
    evidence_end: float
    playback_start: float
    playback_end: float
    start_frame: int
    end_frame: int

    @property
    def duration_seconds(self):
        return (self.end_frame - self.start_frame) / 16000


@dataclass(frozen=True)
class MeetingArtifacts:
    json_path: Path
    markdown_path: Path
    evidence_manifest_path: Path
