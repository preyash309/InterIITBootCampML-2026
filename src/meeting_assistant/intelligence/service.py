"""Validate provenance before API usage, bound windows, publish only complete records."""

import hashlib
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from meeting_assistant.diarization.models import DiarizationResult, SpeakerAwareTranscript
from meeting_assistant.grounding.models import GroundingResult
from meeting_assistant.refinement.models import RefinedTranscript
from meeting_assistant.refinement.serialization import refined_to_json
from meeting_assistant.refinement.service import validate_refined_source

from .base import MeetingIntelligenceBackend
from .config import IntelligenceConfig
from .exceptions import (
    ConsolidationError,
    IntelligenceSourceMismatch,
    IntelligenceValidationError,
    MeetingTooLargeError,
)
from .models import (
    SECTIONS,
    IntelligenceProcessingInfo,
    IntelligenceRequest,
    IntelligenceResponse,
    MeetingContent,
    MeetingRecord,
)
from .prompt import request_data
from .validation import content_identity, validate_content

logger = logging.getLogger(__name__)


def refined_digest(refined):
    return hashlib.sha256(refined_to_json(refined).encode("utf-8")).hexdigest()


def _request_size(request):
    return len(json.dumps(request_data(request), ensure_ascii=False, sort_keys=True))


def build_chunks(refined, config):
    """Whole utterances only; overlap shrinks as needed to ensure forward progress."""
    utterances = refined.utterances
    result = []
    start = 0
    while start < len(utterances):
        end = start
        while end < len(utterances):
            request = IntelligenceRequest(
                f"chunk_{len(result) + 1:04d}", utterances[start : end + 1]
            )
            if _request_size(request) > config.max_chunk_chars:
                break
            end += 1
        if end == start:
            raise MeetingTooLargeError(
                "A complete utterance exceeds the request budget; no text was truncated."
            )
        result.append(IntelligenceRequest(f"chunk_{len(result) + 1:04d}", utterances[start:end]))
        if len(result) > config.max_chunks:
            raise MeetingTooLargeError("Meeting exceeds the configured chunk/call budget.")
        if end == len(utterances):
            break
        start = max(start + 1, end - config.overlap_utterances)
    return tuple(result)


def _check_response(response, request, model):
    if not isinstance(response, IntelligenceResponse) or response.request_id != request.request_id:
        raise IntelligenceValidationError("Backend returned an inconsistent request identity.")
    if response.model_info != model or any(
        c.request_id != request.request_id or c.stage != request.stage for c in response.calls
    ):
        raise IntelligenceValidationError("Backend changed model or call identity.")
    return response.content


def _combine(responses, source):
    order = {u.utterance_id: index for index, u in enumerate(source.utterances)}
    sections = {}
    for section, prefix in zip(SECTIONS, ("sum", "min", "dec", "act")):
        collected = {}
        for chunk, content in enumerate(responses, 1):
            for index, item in enumerate(getattr(content, section), 1):
                key = content_identity(item)
                if key in collected:
                    old = collected[key]
                    refs = tuple(
                        sorted(
                            set(old.evidence_utterance_ids + item.evidence_utterance_ids),
                            key=order.__getitem__,
                        )
                    )
                    collected[key] = replace(old, evidence_utterance_ids=refs)
                else:
                    collected[key] = replace(item, id=f"p{chunk:04d}_{prefix}_{index:04d}")
        sections[section] = tuple(collected.values())
    return MeetingContent(**sections)


def validate_consolidated(content, partial):
    # Revalidate pluggable backends as well as the Groq JSON parser.
    for section in SECTIONS:
        available = getattr(partial, section)
        for item in getattr(content, section):
            primary = next((p for p in available if p.id == item.id), None)
            if primary is None or content_identity(item) != content_identity(primary):
                raise ConsolidationError("Consolidation introduced a new or modified claim.")
            if not set(primary.evidence_utterance_ids).issubset(item.evidence_utterance_ids):
                raise ConsolidationError("Consolidation discarded primary support.")
            allowed = set(uid for p in available for uid in p.evidence_utterance_ids)
            if not set(item.evidence_utterance_ids).issubset(allowed):
                raise ConsolidationError("Consolidation introduced new evidence.")
    return content


def _finalize(content, refined):
    order = {u.utterance_id: n for n, u in enumerate(refined.utterances)}
    sections = {}
    for section, prefix in zip(SECTIONS, ("sum", "min", "dec", "act")):
        ordered = sorted(
            getattr(content, section),
            key=lambda item: (
                min(order[uid] for uid in item.evidence_utterance_ids),
                content_identity(item),
            ),
        )
        sections[section] = tuple(
            replace(
                item,
                id=f"{prefix}_{n:04d}",
                evidence_utterance_ids=tuple(
                    sorted(item.evidence_utterance_ids, key=order.__getitem__)
                ),
            )
            for n, item in enumerate(ordered, 1)
        )
    return MeetingContent(**sections)


def _extract_meeting_record(
    refined,
    speaker,
    grounding,
    *,
    backend=None,
    config=None,
    canonical_audio_path=None,
    diarization=None,
):
    started = time.perf_counter()
    config = config or IntelligenceConfig.from_env()
    try:
        validate_refined_source(refined, speaker, grounding)
    except Exception as exc:
        raise IntelligenceSourceMismatch(
            "Refined/speaker/grounding provenance does not match."
        ) from exc
    logger.info("intelligence_source_validated", extra={"source_refined_id": refined.id})
    chunks = build_chunks(refined, config)  # All source budgets checked before billed requests.
    audio = None
    if canonical_audio_path is not None:
        from .evidence import bind_audio
        from .exceptions import EvidenceAudioError

        if diarization is None:
            raise EvidenceAudioError(
                "Audio binding requires matching saved diarization provenance."
            )
        audio = bind_audio(canonical_audio_path, speaker, diarization=diarization)
    if backend is None and chunks:
        from .groq_backend import GroqMeetingIntelligenceBackend

        backend = GroqMeetingIntelligenceBackend(config)
    if backend is None:
        from .models import IntelligenceModelInfo

        model = IntelligenceModelInfo(
            provider=config.provider,
            model=config.model,
            max_completion_tokens=config.max_completion_tokens,
            reasoning_effort="low"
            if config.model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
            else None,
        )
    else:
        model = backend.model_info
    calls = []
    partials = []
    for request in chunks:
        response = backend.extract(request)
        content = _check_response(response, request, model)
        validate_content(content, request.utterances)
        if any(len(getattr(content, name)) > config.max_items_per_section for name in SECTIONS):
            raise IntelligenceValidationError("Backend exceeds the item budget.")
        partials.append(content)
        calls.extend(response.calls)
    partial = _combine(partials, refined)
    if len(chunks) > 1 and partial.items:
        request = IntelligenceRequest("consolidation_0001", (), "consolidation", partial)
        if _request_size(request) > config.max_consolidation_chars:
            raise MeetingTooLargeError(
                "Validated partials exceed the consolidation budget; no final record published."
            )
        response = backend.extract(request)
        content = _check_response(response, request, model)
        validate_consolidated(content, partial)
        calls.extend(response.calls)
    else:
        content = partial
    validate_content(content, refined.utterances)
    content = _finalize(content, refined)
    record = MeetingRecord(
        str(uuid4()),
        refined.id,
        refined_digest(refined),
        refined.source_speaker_transcript_id,
        refined.source_speaker_sha256,
        refined.grounding_result_id,
        refined.grounding_sha256,
        refined.source_raw_transcript_id,
        refined.source_raw_sha256,
        content,
        model,
        IntelligenceProcessingInfo(time.perf_counter() - started, len(chunks), tuple(calls)),
        audio,
    )
    logger.info(
        "meeting_record_completed",
        extra={"meeting_record_id": record.id, "item_count": len(content.items)},
    )
    return record


def extract_meeting_record(
    refined: RefinedTranscript,
    speaker: SpeakerAwareTranscript,
    grounding: GroundingResult,
    *,
    backend: MeetingIntelligenceBackend | None = None,
    config: IntelligenceConfig | None = None,
    canonical_audio_path: Path | str | None = None,
    diarization: DiarizationResult | None = None,
) -> MeetingRecord:
    logger.info("intelligence_started")
    try:
        return _extract_meeting_record(
            refined,
            speaker,
            grounding,
            backend=backend,
            config=config,
            canonical_audio_path=canonical_audio_path,
            diarization=diarization,
        )
    except Exception:
        logger.info("intelligence_failed")
        raise
