"""Frozen decisions and derived text; original words are references, never retimed."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from meeting_assistant.diarization.models import WordReference
from meeting_assistant.grounding.models import GroundingRecord

from .exceptions import RefinementSchemaError, RefinementValidationError

ACTIONS = ("KEEP", "REPLACE", "UNCERTAIN")
REASONS = (
    "exact_registered_alias",
    "strong_phonetic_context",
    "canonical_formatting",
    "original_supported",
    "candidate_ambiguous",
    "context_insufficient",
    "protected_meaning",
    "candidate_not_justified",
)
POLICY_VERSION = "conservative_v1"
PROMPT_VERSION = "refiner_v1"
FEWSHOT_VERSION = "refiner_examples_v1"
SCHEMA_VERSION = "refiner_decisions_v1"


def typed_tuple(value, kind):
    if not isinstance(value, tuple) or any(not isinstance(item, kind) for item in value):
        raise RefinementValidationError("Refinement evidence requires immutable typed tuples.")


def finite(value, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise RefinementValidationError("Invalid finite refinement measurement.")


def identifier(value, prefix):
    if not isinstance(value, str) or re.fullmatch(prefix + r"_\d{6,}", value) is None:
        raise RefinementValidationError("Invalid deterministic evidence ID.")


@dataclass(frozen=True)
class UtteranceContext:
    utterance_id: str
    speaker_id: str | None
    start: float
    end: float
    text: str

    def __post_init__(self):
        identifier(self.utterance_id, "utt")
        if self.speaker_id is not None and (
            not isinstance(self.speaker_id, str)
            or re.fullmatch(r"SPEAKER_\d{2,}", self.speaker_id) is None
        ):
            raise RefinementValidationError("Speakers must remain anonymous source IDs.")
        finite(self.start)
        finite(self.end, self.start)
        if not isinstance(self.text, str):
            raise RefinementValidationError("Utterance context must be text.")


@dataclass(frozen=True)
class ContextualRefinementEvidence:
    hypothesis_id: str
    grounding_record_id: str
    candidate_entry_id: str
    pass2_text: str
    window_start: float
    window_end: float
    context_source_ids: tuple[str, ...]

    def __post_init__(self):
        identifier(self.hypothesis_id, "casr")
        identifier(self.grounding_record_id, "grnd")
        finite(self.window_start)
        finite(self.window_end, self.window_start)
        typed_tuple(self.context_source_ids, str)
        if (
            not isinstance(self.pass2_text, str)
            or len(self.pass2_text) > 10000
            or not self.candidate_entry_id
        ):
            raise RefinementValidationError("Invalid bounded contextual evidence.")


@dataclass(frozen=True)
class RefinementRequest:
    target: UtteranceContext
    neighbors: tuple[UtteranceContext, ...]
    records: tuple[GroundingRecord, ...]
    contextual_evidence: tuple[ContextualRefinementEvidence, ...] = ()

    def __post_init__(self):
        if not isinstance(self.target, UtteranceContext):
            raise RefinementValidationError("Request requires a typed target.")
        typed_tuple(self.neighbors, UtteranceContext)
        typed_tuple(self.records, GroundingRecord)
        typed_tuple(self.contextual_evidence, ContextualRefinementEvidence)
        if any(
            not any(
                r.id == e.grounding_record_id
                and any(c.entry_id == e.candidate_entry_id for c in r.candidates)
                for r in self.records
            )
            for e in self.contextual_evidence
        ):
            raise RefinementValidationError(
                "Contextual evidence must reference an existing supplied candidate."
            )
        if any(
            r.utterance_id != self.target.utterance_id
            or r.speaker_id != self.target.speaker_id
            or self.target.text[r.char_start : r.char_end] != r.observed_text
            for r in self.records
        ):
            raise RefinementValidationError("Request evidence does not belong to target.")
        if len({r.id for r in self.records}) != len(self.records) or any(
            n.utterance_id == self.target.utterance_id for n in self.neighbors
        ):
            raise RefinementValidationError("Duplicate target or grounding records.")


@dataclass(frozen=True)
class RefinementDecision:
    grounding_record_id: str
    action: str
    candidate_entry_id: str | None
    reason_code: str

    def __post_init__(self):
        identifier(self.grounding_record_id, "grnd")
        if self.action not in ACTIONS or self.reason_code not in REASONS:
            raise RefinementSchemaError("Unknown action or reason code.")
        if (
            self.action == "REPLACE"
            and (not isinstance(self.candidate_entry_id, str) or not self.candidate_entry_id)
        ) or (self.action != "REPLACE" and self.candidate_entry_id is not None):
            raise RefinementSchemaError(
                "REPLACE requires a candidate ID; other actions require null."
            )


@dataclass(frozen=True)
class RefinerModelInfo:
    provider: str
    model: str
    temperature: float = 0.0
    structured_mode: str = "json_schema"
    prompt_version: str = PROMPT_VERSION
    fewshot_version: str = FEWSHOT_VERSION
    schema_version: str = SCHEMA_VERSION
    reasoning_effort: str | None = None
    max_completion_tokens: int = 4096
    include_reasoning: bool = False

    def __post_init__(self):
        if (
            self.reasoning_effort not in (None, "low")
            or self.include_reasoning is not False
            or isinstance(self.max_completion_tokens, bool)
            or not isinstance(self.max_completion_tokens, int)
            or not 1 <= self.max_completion_tokens <= 16000
        ):
            raise RefinementValidationError("Invalid recorded generation parameters.")
        if (
            any(not isinstance(v, str) or not v for v in (self.provider, self.model))
            or self.temperature != 0
            or isinstance(self.temperature, bool)
            or self.structured_mode not in ("json_schema", "json_object")
            or (self.prompt_version, self.fewshot_version, self.schema_version)
            != (PROMPT_VERSION, FEWSHOT_VERSION, SCHEMA_VERSION)
        ):
            raise RefinementValidationError(
                "Invalid recorded refiner model or prompt configuration."
            )


@dataclass(frozen=True)
class ProviderCall:
    utterance_id: str
    latency_seconds: float
    request_bytes: int
    response_bytes: int
    status_code: int
    schema_repair: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self):
        identifier(self.utterance_id, "utt")
        finite(self.latency_seconds)
        for value in (
            self.request_bytes,
            self.response_bytes,
            self.status_code,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise RefinementValidationError("Invalid provider count.")
        if not isinstance(self.schema_repair, bool) or not 100 <= self.status_code <= 599:
            raise RefinementValidationError("Invalid provider status.")


@dataclass(frozen=True)
class RefinementResponse:
    utterance_id: str
    decisions: tuple[RefinementDecision, ...]
    model_info: RefinerModelInfo
    calls: tuple[ProviderCall, ...] = ()

    def __post_init__(self):
        identifier(self.utterance_id, "utt")
        typed_tuple(self.decisions, RefinementDecision)
        typed_tuple(self.calls, ProviderCall)
        if not isinstance(self.model_info, RefinerModelInfo) or any(
            c.utterance_id != self.utterance_id for c in self.calls
        ):
            raise RefinementValidationError("Response provenance is inconsistent.")


@dataclass(frozen=True)
class TranscriptEdit:
    id: str
    utterance_id: str
    grounding_record_id: str
    action: str
    source_text: str
    replacement_text: str | None
    candidate_entry_id: str | None
    source_word_refs: tuple[WordReference, ...]
    start: float
    end: float
    char_start: int
    char_end: int
    model_reason_code: str
    validation_status: str
    rejection_reason: str | None = None

    def __post_init__(self):
        identifier(self.id, "edit")
        identifier(self.utterance_id, "utt")
        RefinementDecision(
            self.grounding_record_id, self.action, self.candidate_entry_id, self.model_reason_code
        )
        typed_tuple(self.source_word_refs, WordReference)
        finite(self.start)
        finite(self.end, self.start)
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in (self.char_start, self.char_end)
            )
            or not 0 <= self.char_start < self.char_end
            or not isinstance(self.source_text, str)
            or not self.source_text
        ):
            raise RefinementValidationError("Invalid edit offsets or source text.")
        expected = (
            ("applied", "rejected")
            if self.action == "REPLACE"
            else ("kept",)
            if self.action == "KEEP"
            else ("uncertain",)
        )
        if (
            self.validation_status not in expected
            or (
                self.action == "REPLACE"
                and (not isinstance(self.replacement_text, str) or not self.replacement_text)
            )
            or (self.action != "REPLACE" and self.replacement_text is not None)
        ):
            raise RefinementValidationError("Edit status and replacement are inconsistent.")
        if (
            self.validation_status == "rejected"
            and not (isinstance(self.rejection_reason, str) and self.rejection_reason)
        ) or (self.validation_status != "rejected" and self.rejection_reason is not None):
            raise RefinementValidationError(
                "Rejected edits need a reason; other edits cannot have one."
            )


@dataclass(frozen=True)
class RefinedUtterance:
    utterance_id: str
    speaker_id: str | None
    start: float
    end: float
    raw_text: str
    refined_text: str
    applied_edit_ids: tuple[str, ...]
    uncertain_record_ids: tuple[str, ...]
    source_word_refs: tuple[WordReference, ...]
    source_segment_ids: tuple[str, ...]

    def __post_init__(self):
        UtteranceContext(self.utterance_id, self.speaker_id, self.start, self.end, self.raw_text)
        if not isinstance(self.refined_text, str):
            raise RefinementValidationError("Refined text must be text.")
        for value, kind in (
            (self.applied_edit_ids, str),
            (self.uncertain_record_ids, str),
            (self.source_word_refs, WordReference),
            (self.source_segment_ids, str),
        ):
            typed_tuple(value, kind)
            if len(set(value)) != len(value):
                raise RefinementValidationError("Duplicate utterance evidence references.")


@dataclass(frozen=True)
class RefinementProcessingInfo:
    total_seconds: float
    utterance_count: int
    sent_utterance_count: int
    calls: tuple[ProviderCall, ...]

    def __post_init__(self):
        finite(self.total_seconds)
        typed_tuple(self.calls, ProviderCall)
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for v in (self.utterance_count, self.sent_utterance_count)
            )
            or self.sent_utterance_count > self.utterance_count
        ):
            raise RefinementValidationError("Invalid utterance counts.")

    def usage(self, field: str) -> int | None:
        if field not in ("prompt_tokens", "completion_tokens", "total_tokens"):
            raise ValueError("Unknown token metric.")
        values = [getattr(call, field) for call in self.calls]
        return sum(values) if all(v is not None for v in values) else None


@dataclass(frozen=True)
class RefinedTranscript:
    id: str
    source_speaker_transcript_id: str
    source_speaker_sha256: str
    grounding_result_id: str
    grounding_sha256: str
    source_raw_transcript_id: str
    source_raw_sha256: str
    model_info: RefinerModelInfo
    policy_version: str
    utterances: tuple[RefinedUtterance, ...]
    edit_log: tuple[TranscriptEdit, ...]
    processing_info: RefinementProcessingInfo
    schema_version: str = "1.0"

    def __post_init__(self):
        try:
            for value in (
                self.id,
                self.source_speaker_transcript_id,
                self.grounding_result_id,
                self.source_raw_transcript_id,
            ):
                UUID(value)
        except (ValueError, TypeError, AttributeError) as exc:
            raise RefinementValidationError("Refinement provenance requires UUIDs.") from exc
        for value in (self.source_speaker_sha256, self.grounding_sha256, self.source_raw_sha256):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise RefinementValidationError("Invalid provenance digest.")
        typed_tuple(self.utterances, RefinedUtterance)
        typed_tuple(self.edit_log, TranscriptEdit)
        if (
            self.policy_version != POLICY_VERSION
            or self.schema_version != "1.0"
            or not isinstance(self.model_info, RefinerModelInfo)
            or not isinstance(self.processing_info, RefinementProcessingInfo)
            or self.processing_info.utterance_count != len(self.utterances)
        ):
            raise RefinementValidationError("Unsupported refinement policy or schema.")
        if len({e.grounding_record_id for e in self.edit_log}) != len(self.edit_log):
            raise RefinementValidationError("Duplicate decisions in edit log.")
        if any(e.id != f"edit_{i:06d}" for i, e in enumerate(self.edit_log, 1)):
            raise RefinementValidationError("Edit IDs must be sequential.")
        utterances = {u.utterance_id: u for u in self.utterances}
        if any(u.utterance_id != f"utt_{i:06d}" for i, u in enumerate(self.utterances, 1)) or any(
            e.utterance_id not in utterances for e in self.edit_log
        ):
            raise RefinementValidationError("Utterance ordering or edit target is invalid.")
        for u in self.utterances:
            from .validation import protected_signature

            if protected_signature(u.raw_text) != protected_signature(u.refined_text):
                raise RefinementValidationError("Refined text changed protected source meaning.")
            edits = [e for e in self.edit_log if e.utterance_id == u.utterance_id]
            last = 0
            for e in edits:
                if (
                    e.char_start < last
                    or u.raw_text[e.char_start : e.char_end] != e.source_text
                    or not u.start <= e.start <= e.end <= u.end
                    or any(r not in u.source_word_refs for r in e.source_word_refs)
                ):
                    raise RefinementValidationError("Edit span or word evidence is inconsistent.")
                last = e.char_end
            applied = [e for e in edits if e.validation_status == "applied"]
            rendered = u.raw_text
            for e in reversed(applied):
                rendered = rendered[: e.char_start] + e.replacement_text + rendered[e.char_end :]
            if (
                rendered != u.refined_text
                or u.applied_edit_ids != tuple(e.id for e in applied)
                or u.uncertain_record_ids
                != tuple(e.grounding_record_id for e in edits if e.validation_status == "uncertain")
            ):
                raise RefinementValidationError(
                    "Refined text must be reproduced exactly by the edit log."
                )
        if any(call.utterance_id not in utterances for call in self.processing_info.calls):
            raise RefinementValidationError("Provider call does not refer to a source utterance.")


@dataclass(frozen=True)
class RefinementArtifacts:
    json_path: Path
    text_path: Path
    edit_log_path: Path
