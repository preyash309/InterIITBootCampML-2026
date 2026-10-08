"""Provenance checks, bounded requests and offset-based application; no provider internals."""

import hashlib
import logging
import time
from dataclasses import replace
from uuid import uuid4

from meeting_assistant.diarization.models import SpeakerAwareTranscript
from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.grounding.exceptions import InvalidGroundingResult
from meeting_assistant.grounding.models import GroundingResult
from meeting_assistant.grounding.serialization import grounding_to_json
from meeting_assistant.grounding.service import validate_grounding_source

from .base import TranscriptRefinerBackend
from .config import RefinementConfig
from .exceptions import RefinementSourceMismatch, RefinementValidationError
from .models import (
    POLICY_VERSION,
    RefinedTranscript,
    RefinedUtterance,
    RefinementProcessingInfo,
    RefinementRequest,
    RefinerModelInfo,
    TranscriptEdit,
    UtteranceContext,
)
from .prompt import request_data
from .validation import replacement_rejection, validate_response

logger = logging.getLogger(__name__)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _event(name, **values):
    logger.info(name, extra={"event": name, **values})


def _validate_source(source, grounding):
    if not isinstance(source, SpeakerAwareTranscript) or not isinstance(grounding, GroundingResult):
        raise RefinementSourceMismatch("Refinement requires typed Phase III and IV evidence.")
    try:
        validate_grounding_source(grounding, source)
    except InvalidGroundingResult as exc:
        raise RefinementSourceMismatch(
            "Grounding provenance does not match the speaker transcript."
        ) from exc


def build_requests(
    source: SpeakerAwareTranscript, grounding: GroundingResult, config: RefinementConfig
) -> tuple[RefinementRequest, ...]:
    """Prevalidate all budgets before the first billed request. No grounding means no call."""
    _validate_source(source, grounding)
    requests = []
    for i, u in enumerate(source.utterances):
        records = tuple(
            replace(r, candidates=r.candidates[: config.top_k])
            for r in grounding.records
            if r.utterance_id == u.id
        )
        if not records:
            continue
        neighbors = []
        for n in source.utterances[max(0, i - 1) : i + 2]:
            if (
                n.id != u.id
                and max(n.start - u.end, u.start - n.end, 0) <= config.neighbor_gap_seconds
            ):
                neighbors.append(
                    UtteranceContext(
                        n.id, n.speaker_id, n.start, n.end, n.text[: config.context_chars // 2]
                    )
                )
        request = RefinementRequest(
            UtteranceContext(u.id, u.speaker_id, u.start, u.end, u.text), tuple(neighbors), records
        )
        request_data(request, config)
        requests.append(request)
    return tuple(requests)


def refine_transcript(
    source: SpeakerAwareTranscript,
    grounding: GroundingResult,
    *,
    backend: TranscriptRefinerBackend | None = None,
    config: RefinementConfig | None = None,
    contextual_asr=None,
) -> RefinedTranscript:
    started = time.perf_counter()
    _event("refinement_started")
    config = (
        config
        or (getattr(backend, "config", None) if backend else None)
        or RefinementConfig.from_env()
    )
    requests = build_requests(source, grounding, config)
    if contextual_asr is not None:
        from meeting_assistant.contextual_asr.integration import (
            attach_request_evidence,
            refinement_evidence,
        )

        requests = attach_request_evidence(
            requests, refinement_evidence(contextual_asr, source, grounding)
        )
        for request in requests:
            request_data(request, config)
    _event(
        "source_validated",
        utterance_count=len(source.utterances),
        sent_utterance_count=len(requests),
    )
    if backend is None and requests:
        from .groq_backend import GroqRefinerBackend

        backend = GroqRefinerBackend(config)
    model_info = (
        backend.model_info
        if backend
        else RefinerModelInfo(
            config.provider,
            config.model,
            config.temperature,
            config.structured_mode,
            reasoning_effort="low"
            if config.model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
            else None,
            max_completion_tokens=config.max_completion_tokens,
        )
    )
    edits = []
    calls = []
    records = {r.id: r for r in grounding.records}
    for request in requests:
        response = backend.refine(request)
        validate_response(request, response)
        if response.model_info != model_info:
            raise RefinementValidationError("Backend model identity changed during refinement.")
        calls.extend(response.calls)
        decisions = {d.grounding_record_id: d for d in response.decisions}
        for sent_record in request.records:
            record = records[sent_record.id]
            d = decisions[record.id]
            candidate = next(
                (c for c in record.candidates if c.entry_id == d.candidate_entry_id), None
            )
            rejection = (
                replacement_rejection(record, candidate, request.target.text)
                if d.action == "REPLACE"
                else None
            )
            status = (
                ("rejected" if rejection else "applied")
                if d.action == "REPLACE"
                else "kept"
                if d.action == "KEEP"
                else "uncertain"
            )
            edit = TranscriptEdit(
                f"edit_{len(edits) + 1:06d}",
                record.utterance_id,
                record.id,
                d.action,
                record.observed_text,
                candidate.canonical if candidate else None,
                d.candidate_entry_id,
                record.source_word_references,
                record.start,
                record.end,
                record.char_start,
                record.char_end,
                d.reason_code,
                status,
                rejection,
            )
            edits.append(edit)
    # Sort to original grounding order; IDs follow evidence, not provider decision order.
    edits.sort(key=lambda e: int(e.grounding_record_id.removeprefix("grnd_")))
    edits = tuple(replace(e, id=f"edit_{i:06d}") for i, e in enumerate(edits, 1))
    utterances = []
    for u in source.utterances:
        selected = [e for e in edits if e.utterance_id == u.id]
        applied = [e for e in selected if e.validation_status == "applied"]
        text = u.text
        for e in reversed(applied):
            text = text[: e.char_start] + e.replacement_text + text[e.char_end :]
        utterances.append(
            RefinedUtterance(
                u.id,
                u.speaker_id,
                u.start,
                u.end,
                u.text,
                text,
                tuple(e.id for e in applied),
                tuple(
                    e.grounding_record_id for e in selected if e.validation_status == "uncertain"
                ),
                tuple(w.source_word_reference for w in u.words),
                u.source_segment_ids,
            )
        )
    result = RefinedTranscript(
        str(uuid4()),
        source.transcript_id,
        digest(speaker_transcript_to_json(source)),
        grounding.grounding_id,
        digest(grounding_to_json(grounding)),
        source.source_raw_transcript_id,
        source.source_raw_sha256,
        model_info,
        POLICY_VERSION,
        tuple(utterances),
        edits,
        RefinementProcessingInfo(
            time.perf_counter() - started, len(source.utterances), len(requests), tuple(calls)
        ),
    )
    validate_refined_source(result, source, grounding)
    for edit in result.edit_log:
        if edit.validation_status in ("rejected", "applied"):
            _event("edit_" + edit.validation_status, edit_id=edit.id, reason=edit.rejection_reason)
    return result


def validate_refined_source(
    result: RefinedTranscript, source: SpeakerAwareTranscript, grounding: GroundingResult
) -> None:
    """Use again when reading saved refinement, before a downstream phase trusts it."""
    _validate_source(source, grounding)
    if not isinstance(result, RefinedTranscript) or (
        result.source_speaker_transcript_id,
        result.source_speaker_sha256,
        result.grounding_result_id,
        result.grounding_sha256,
        result.source_raw_transcript_id,
        result.source_raw_sha256,
    ) != (
        source.transcript_id,
        digest(speaker_transcript_to_json(source)),
        grounding.grounding_id,
        digest(grounding_to_json(grounding)),
        source.source_raw_transcript_id,
        source.source_raw_sha256,
    ):
        raise RefinementSourceMismatch(
            "Refined transcript provenance does not match source evidence."
        )
    if len(result.utterances) != len(source.utterances) or {
        e.grounding_record_id for e in result.edit_log
    } != {r.id for r in grounding.records}:
        raise RefinementSourceMismatch(
            "Refinement omitted source utterances or grounding decisions."
        )
    record_map = {r.id: r for r in grounding.records}
    for u, original in zip(result.utterances, source.utterances):
        if (
            u.utterance_id,
            u.speaker_id,
            u.start,
            u.end,
            u.raw_text,
            u.source_word_refs,
            u.source_segment_ids,
        ) != (
            original.id,
            original.speaker_id,
            original.start,
            original.end,
            original.text,
            tuple(w.source_word_reference for w in original.words),
            original.source_segment_ids,
        ):
            raise RefinementSourceMismatch(
                "Refinement changed source text, speaker, times or word references."
            )
    for e in result.edit_log:
        r = record_map[e.grounding_record_id]
        if (
            e.utterance_id,
            e.source_text,
            e.char_start,
            e.char_end,
            e.start,
            e.end,
            e.source_word_refs,
        ) != (
            r.utterance_id,
            r.observed_text,
            r.char_start,
            r.char_end,
            r.start,
            r.end,
            r.source_word_references,
        ):
            raise RefinementSourceMismatch("Edit does not resolve to original grounding evidence.")
        if e.action == "REPLACE":
            c = next((c for c in r.candidates if c.entry_id == e.candidate_entry_id), None)
            if c is None or e.replacement_text != c.canonical:
                raise RefinementSourceMismatch(
                    "Edit replacement is not the cited canonical candidate."
                )
            text = next(u.text for u in source.utterances if u.id == r.utterance_id)
            rejection = replacement_rejection(r, c, text)
            if e.rejection_reason != rejection or e.validation_status != (
                "rejected" if rejection else "applied"
            ):
                raise RefinementValidationError(
                    "Saved edit contradicts deterministic safety policy."
                )
