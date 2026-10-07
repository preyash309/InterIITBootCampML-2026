"""Attach candidate evidence to unchanged speaker utterances and raw word references."""

import hashlib
import logging
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from meeting_assistant.diarization.models import SpeakerAwareTranscript
from meeting_assistant.diarization.serialization import speaker_transcript_to_json

from .config import GroundingConfig
from .exceptions import InvalidGroundingResult
from .glossary import load_glossary
from .models import (
    GlossaryEntry,
    GroundingProcessingInfo,
    GroundingRecord,
    GroundingResult,
    RetrievalPolicy,
)
from .normalization import normalize
from .retrieval import GroundingRetriever

logger = logging.getLogger(__name__)
_factory_lock = threading.Lock()


@lru_cache(maxsize=2)
def _default_retriever(config: GroundingConfig) -> GroundingRetriever:
    return GroundingRetriever(load_glossary(), config)


def ground_span(
    span: str,
    context: str,
    *,
    retriever: GroundingRetriever | None = None,
    config: GroundingConfig | None = None,
):
    with _factory_lock:
        engine = retriever or _default_retriever(config or GroundingConfig.from_env())
    return engine.candidates(span, context)


def _contexts(source: SpeakerAwareTranscript, config: GroundingConfig) -> list[str]:
    contexts = []
    for index, utterance in enumerate(source.utterances):
        current = utterance.text[: config.context_chars]
        # Current text always gets the first and largest part of the context budget.
        spare = config.context_chars - len(current)
        neighbors = []
        for other in source.utterances[max(0, index - 1) : index + 2]:
            gap = max(other.start - utterance.end, utterance.start - other.end, 0)
            if other.id != utterance.id and gap <= config.neighbor_gap_seconds:
                neighbors.append(other.text)
        contexts.append((current + " " + " ".join(neighbors)[: max(0, spare - 1)]).strip())
    return contexts


def _spans(utterance, max_words):
    tokens = list(
        re.finditer(r"[\w]+(?:[.'’+-][\w]+)*(?:\+\+|#|\*)?", utterance.text, flags=re.UNICODE)
    )
    # Align actual raw-word text to the speaker display text; punctuation/leading spaces are retained upstream.
    word_intervals = []
    cursor = 0
    for word in utterance.words:
        needle = word.text.strip()
        start = utterance.text.find(needle, cursor)
        if start < 0:
            raise InvalidGroundingResult("Speaker display text cannot be linked to its raw words.")
        word_intervals.append((start, start + len(needle), word))
        cursor = start + len(needle)
    for index, token in enumerate(tokens):
        for count in range(1, min(max_words, len(tokens) - index) + 1):
            end = tokens[index + count - 1].end()
            selected = [
                w for start, finish, w in word_intervals if start < end and finish > token.start()
            ]
            yield token.start(), end, selected


def ground_transcript(
    source: SpeakerAwareTranscript,
    *,
    retriever: GroundingRetriever | None = None,
    config: GroundingConfig | None = None,
    project_glossary: Path | None = None,
    meeting_entries: tuple[GlossaryEntry, ...] = (),
) -> GroundingResult:
    if not isinstance(source, SpeakerAwareTranscript):
        raise InvalidGroundingResult("Grounding requires a typed Phase III speaker transcript.")
    if retriever and (project_glossary or meeting_entries):
        raise InvalidGroundingResult(
            "Supply glossary layers when constructing the custom retriever."
        )
    if retriever and config is not None and config != retriever.config:
        raise InvalidGroundingResult("Service and retriever configurations must agree.")
    started = time.perf_counter()
    config = config or (retriever.config if retriever else GroundingConfig.from_env())
    glossary_started = time.perf_counter()
    if retriever:
        engine = retriever
    elif project_glossary or meeting_entries:
        engine = GroundingRetriever(
            load_glossary(project_glossary=project_glossary, meeting_entries=meeting_entries),
            config,
        )
    else:
        with _factory_lock:
            engine = _default_retriever(config)
    glossary_seconds = time.perf_counter() - glossary_started
    logger.info(
        "grounding_started",
        extra={"event": "grounding_started", "utterance_count": len(source.utterances)},
    )
    index_started = time.perf_counter()
    previously_prepared = engine._matrix is not None
    engine.prepare()
    index_seconds = max(
        0.0,
        time.perf_counter()
        - index_started
        - (0.0 if previously_prepared else engine.model_load_seconds),
    )
    contexts = _contexts(source, config)
    inference_started = time.perf_counter()
    semantic = engine.context_scores(contexts)
    records, considered = [], 0
    for utterance, context, scores in zip(source.utterances, contexts, semantic):
        proposals = []
        for start, end, words in _spans(utterance, config.max_span_words):
            observed = utterance.text[start:end]
            candidates = engine.candidates(observed, context, semantic_scores=scores)
            considered += 1
            if candidates:
                proposals.append((start, end, words, observed, candidates))
        # Deterministic nonoverlap policy: exact evidence first, then confidence and span length.
        selected = []
        for proposal in sorted(
            proposals,
            key=lambda p: (
                p[4][0].match_type == "fuzzy",
                -p[4][0].score,
                -(p[1] - p[0]),
                p[0],
                p[4][0].entry_id,
            ),
        ):
            refs = {w.source_word_reference for w in proposal[2]}
            if all(
                (proposal[0] >= old[1] or proposal[1] <= old[0])
                and not refs.intersection(w.source_word_reference for w in old[2])
                for old in selected
            ):
                selected.append(proposal)
        for start, end, words, observed, candidates in sorted(selected, key=lambda p: p[0]):
            references = tuple(dict.fromkeys(w.source_word_reference for w in words))
            segments = (
                tuple(dict.fromkeys(r.segment_id for r in references))
                or utterance.source_segment_ids
            )
            records.append(
                GroundingRecord(
                    f"grnd_{len(records) + 1:06d}",
                    utterance.id,
                    utterance.speaker_id,
                    min(w.start for w in words) if words else utterance.start,
                    max(w.end for w in words) if words else utterance.end,
                    start,
                    end,
                    observed,
                    normalize(observed),
                    context,
                    references,
                    segments,
                    candidates,
                )
            )
    elapsed = time.perf_counter() - inference_started
    result = GroundingResult(
        str(uuid4()),
        source.transcript_id,
        hashlib.sha256(speaker_transcript_to_json(source).encode("utf-8")).hexdigest(),
        source.source_raw_transcript_id,
        source.source_raw_sha256,
        engine.glossary.version,
        engine.embeddings.model,
        engine.embeddings.revision,
        tuple(records),
        GroundingProcessingInfo(
            0.0 if previously_prepared else engine.model_load_seconds,
            glossary_seconds,
            index_seconds,
            elapsed,
            time.perf_counter() - started,
            len(source.utterances),
            considered,
            engine.cache_hit,
        ),
        RetrievalPolicy(
            **{
                name: getattr(config, name)
                for name in RetrievalPolicy.__dataclass_fields__
                if name != "policy_version"
            }
        ),
    )
    logger.info(
        "grounding_completed", extra={"event": "grounding_completed", "record_count": len(records)}
    )
    return result


def validate_grounding_source(result: GroundingResult, source: SpeakerAwareTranscript) -> None:
    """Reject mismatched/tampered evidence before a later phase consumes it."""
    if (
        result.source_speaker_transcript_id != source.transcript_id
        or result.source_speaker_sha256
        != hashlib.sha256(speaker_transcript_to_json(source).encode("utf-8")).hexdigest()
        or result.source_raw_transcript_id != source.source_raw_transcript_id
        or result.source_raw_sha256 != source.source_raw_sha256
    ):
        raise InvalidGroundingResult("Grounding belongs to a different source transcript.")
    utterances = {u.id: u for u in source.utterances}
    used_refs = set()
    span_maps = {}
    for record in result.records:
        utterance = utterances.get(record.utterance_id)
        if (
            utterance is None
            or record.speaker_id != utterance.speaker_id
            or utterance.text[record.char_start : record.char_end] != record.observed_text
            or normalize(record.observed_text) != record.normalized_text
        ):
            raise InvalidGroundingResult("Observed span or speaker does not match source evidence.")
        word_map = {w.source_word_reference: w for w in utterance.words}
        if any(ref not in word_map or ref in used_refs for ref in record.source_word_references):
            raise InvalidGroundingResult("Raw word references are missing or reused.")
        if utterance.id not in span_maps:
            span_maps[utterance.id] = {
                (start, end): tuple(dict.fromkeys(w.source_word_reference for w in words))
                for start, end, words in _spans(utterance, result.retrieval_policy.max_span_words)
            }
        if (
            span_maps[utterance.id].get((record.char_start, record.char_end))
            != record.source_word_references
        ):
            raise InvalidGroundingResult("Word references do not cover the observed span.")
        words = [word_map[ref] for ref in record.source_word_references]
        if record.source_segment_ids != (
            tuple(dict.fromkeys(ref.segment_id for ref in record.source_word_references))
            or utterance.source_segment_ids
        ):
            raise InvalidGroundingResult("Source segment references changed.")
        expected_start = min(w.start for w in words) if words else utterance.start
        expected_end = max(w.end for w in words) if words else utterance.end
        if record.start != expected_start or record.end != expected_end:
            raise InvalidGroundingResult("Timestamp evidence changed.")
        used_refs.update(record.source_word_references)
