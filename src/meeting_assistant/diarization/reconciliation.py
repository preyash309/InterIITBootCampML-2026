"""Deterministic temporal evidence attribution. Never rewrites raw ASR records."""

import hashlib
import logging
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from uuid import uuid4

from meeting_assistant.asr.models import TranscriptResult
from meeting_assistant.asr.serialization import transcript_to_json

from .config import ReconciliationConfig
from .exceptions import ReconciliationError
from .models import (
    AttributionMethod,
    DiarizationResult,
    OverlapRegion,
    ReconciliationInfo,
    SpeakerAwareTranscript,
    SpeakerTurn,
    SpeakerUtterance,
    SpeakerWord,
    WordReference,
)

logger = logging.getLogger(__name__)


def temporal_overlap(start: float, end: float, turn: SpeakerTurn) -> float:
    return max(0.0, min(end, turn.end) - max(start, turn.start))


@dataclass(frozen=True)
class _Attribution:
    speaker: str | None
    method: AttributionMethod
    fraction: float | None
    turn_ids: tuple[str, ...]


class _ExclusiveIndex:
    """Binary-search exclusive turns; avoid scanning every turn for every word."""

    def __init__(self, turns: tuple[SpeakerTurn, ...]):
        self.turns = turns
        self.starts = [turn.start for turn in turns]
        self.ends = [turn.end for turn in turns]

    def candidates(self, start: float, end: float, padding: float = 0) -> tuple[SpeakerTurn, ...]:
        begin = bisect_right(self.ends, start - padding)
        stop = (
            bisect_right(self.starts, end + padding)
            if start == end
            else bisect_left(self.starts, end + padding)
        )
        return self.turns[begin:stop]

    def choose(self, start: float, end: float, config: ReconciliationConfig) -> _Attribution:
        candidates = self.candidates(start, end)
        duration = end - start
        if duration == 0:
            # A point has no overlap fraction; use half-open turn containment.
            contained = [turn for turn in candidates if turn.start <= start < turn.end]
            if contained:
                turn = contained[0]
                return _Attribution(turn.speaker_id, "overlap", None, (turn.id,))
        else:
            direct = self._maximum_overlap(start, end, candidates)
            if direct:
                speaker, matched, overlap = direct
                return _Attribution(speaker, "overlap", min(1.0, overlap / duration), matched)
        tolerance = config.alignment_tolerance_seconds
        if tolerance:
            padded = self._maximum_overlap(
                start - tolerance, end + tolerance, self.candidates(start, end, tolerance)
            )
            if padded:
                speaker, matched, _ = padded
                return _Attribution(speaker, "tolerance", 0.0 if duration else None, matched)
        threshold = config.nearest_turn_max_gap_seconds
        # Only immediate neighbours can be nearest in a disjoint exclusive timeline.
        position = bisect_right(self.starts, start)
        nearest = self.turns[max(0, position - 1) : min(len(self.turns), position + 1)]
        if nearest and threshold:
            turn = min(
                nearest,
                key=lambda item: (
                    max(item.start - end, start - item.end, 0.0),
                    item.start,
                    item.speaker_id,
                ),
            )
            gap = max(turn.start - end, start - turn.end, 0.0)
            if gap <= threshold + 1e-12:
                return _Attribution(
                    turn.speaker_id, "nearest", 0.0 if duration else None, (turn.id,)
                )
        return _Attribution(None, "unknown", 0.0 if duration else None, ())

    @staticmethod
    def _maximum_overlap(start, end, candidates):
        totals: dict[str, float] = defaultdict(float)
        matches: dict[str, list[SpeakerTurn]] = defaultdict(list)
        for turn in candidates:
            overlap = temporal_overlap(start, end, turn)
            if overlap > 0:
                totals[turn.speaker_id] += overlap
                matches[turn.speaker_id].append(turn)
        if not totals:
            return None
        best = max(totals.values())
        # Numerically equal overlaps tie by earliest contributing turn, then label.
        tied = [
            speaker
            for speaker, value in totals.items()
            if math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12)
        ]
        speaker = min(tied, key=lambda name: (matches[name][0].start, name))
        return speaker, tuple(turn.id for turn in matches[speaker]), totals[speaker]


def find_overlap_regions(turns: tuple[SpeakerTurn, ...]) -> tuple[OverlapRegion, ...]:
    """Sweep regular turns, counting distinct simultaneous speakers, not adjacent turns."""
    events = defaultdict(list)
    for turn in turns:
        events[turn.start].append((turn.speaker_id, 1))
        events[turn.end].append((turn.speaker_id, -1))
    active = Counter()
    previous = 0.0
    regions: list[OverlapRegion] = []
    for timestamp in sorted(events):
        speakers = tuple(sorted(name for name, count in active.items() if count > 0))
        if timestamp > previous and len(speakers) >= 2:
            if regions and regions[-1].end == previous and regions[-1].speakers == speakers:
                last = regions.pop()
                regions.append(OverlapRegion(last.start, timestamp, speakers))
            else:
                regions.append(OverlapRegion(previous, timestamp, speakers))
        for speaker, change in events[timestamp]:
            active[speaker] += change
        previous = timestamp
    return tuple(regions)


class _OverlapIndex:
    def __init__(self, regions):
        self.regions = regions
        self.ends = [region.end for region in regions]

    def contains_overlap(self, start, end):
        index = bisect_right(self.ends, start)
        if index >= len(self.regions):
            return False
        region = self.regions[index]
        return region.start <= start < region.end if start == end else region.start < end


def render_word_sequence(words: tuple[SpeakerWord, ...]) -> str:
    """Retain every token exactly, adding only separators when the backend omitted them."""
    text = ""
    for word in words:
        token = word.text
        if text and not text[-1].isspace() and not token[0].isspace():
            text += " "
        text += token
    return text


def reconcile_transcript(
    raw_transcript: TranscriptResult,
    diarization: DiarizationResult,
    *,
    config: ReconciliationConfig | None = None,
) -> SpeakerAwareTranscript:
    if not isinstance(raw_transcript, TranscriptResult) or not isinstance(
        diarization, DiarizationResult
    ):
        raise ReconciliationError("Reconciliation requires typed raw ASR and diarization results.")
    config = config if config is not None else ReconciliationConfig()
    if not isinstance(config, ReconciliationConfig):
        raise ReconciliationError("Reconciliation configuration must be typed.")
    if (
        abs(raw_transcript.duration_seconds - diarization.duration_seconds)
        > config.duration_tolerance_seconds
    ):
        raise ReconciliationError(
            "ASR and diarization must refer to the same canonical audio duration."
        )
    if raw_transcript.segments and not diarization.exclusive_turns:
        raise ReconciliationError("No exclusive speaker turns are available for this transcript.")
    context = {
        "source_raw_transcript_id": raw_transcript.transcript_id,
        "diarization_id": diarization.diarization_id,
    }
    logger.info("reconciliation_started", extra={**context, "event": "reconciliation_started"})
    exclusive = _ExclusiveIndex(diarization.exclusive_turns)
    overlap_regions = find_overlap_regions(diarization.regular_turns)
    overlap = _OverlapIndex(overlap_regions)
    utterances: list[SpeakerUtterance] = []
    pending: list[SpeakerWord] = []
    counts = Counter()
    fallback_count = 0

    def flush():
        if not pending:
            return
        words = tuple(pending)
        utterances.append(
            SpeakerUtterance(
                id=f"utt_{len(utterances) + 1:06d}",
                speaker_id=words[0].speaker_id,
                start=words[0].start,
                end=max(word.end for word in words),
                text=render_word_sequence(words),
                words=words,
                source_segment_ids=tuple(
                    dict.fromkeys(word.source_word_reference.segment_id for word in words)
                ),
                overlap_present=any(word.overlap_present for word in words),
            )
        )
        pending.clear()

    for segment in raw_transcript.segments:
        if not segment.words:
            flush()
            attribution = exclusive.choose(segment.start, segment.end, config)
            utterances.append(
                SpeakerUtterance(
                    id=f"utt_{len(utterances) + 1:06d}",
                    speaker_id=attribution.speaker,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    words=(),
                    source_segment_ids=(segment.id,),
                    overlap_present=overlap.contains_overlap(segment.start, segment.end),
                    attribution_method="unknown"
                    if attribution.method == "unknown"
                    else f"segment_{attribution.method}",
                )
            )
            fallback_count += 1
            continue
        for index, word in enumerate(segment.words):
            attribution = exclusive.choose(word.start, word.end, config)
            derived = SpeakerWord(
                word=word,
                source_word_reference=WordReference(segment.id, index),
                speaker_id=attribution.speaker,
                attribution_method=attribution.method,
                overlap_fraction=attribution.fraction,
                source_turn_ids=attribution.turn_ids,
                overlap_present=overlap.contains_overlap(word.start, word.end),
            )
            counts[attribution.method] += 1
            if pending and (
                pending[-1].speaker_id != derived.speaker_id
                or derived.start - pending[-1].end > config.utterance_max_gap_seconds
            ):
                flush()
            pending.append(derived)
    flush()
    known = [utterance.speaker_id for utterance in utterances if utterance.speaker_id is not None]
    info = ReconciliationInfo(
        total_words=sum(counts.values()),
        direct_assignments=counts["overlap"],
        tolerance_assignments=counts["tolerance"],
        nearest_assignments=counts["nearest"],
        unassigned_words=counts["unknown"],
        segment_fallback_count=fallback_count,
        speaker_switches=sum(left != right for left, right in zip(known, known[1:])),
        config=config,
    )
    result = SpeakerAwareTranscript(
        transcript_id=str(uuid4()),
        source_raw_transcript_id=raw_transcript.transcript_id,
        source_raw_sha256=hashlib.sha256(
            transcript_to_json(raw_transcript).encode("utf-8")
        ).hexdigest(),
        diarization_id=diarization.diarization_id,
        duration_seconds=raw_transcript.duration_seconds,
        speakers=diarization.speakers,
        utterances=tuple(utterances),
        overlap_regions=overlap_regions,
        reconciliation_info=info,
    )
    logger.info(
        "reconciliation_completed",
        extra={
            **context,
            "event": "reconciliation_completed",
            "word_count": info.total_words,
            "unassigned_word_count": info.unassigned_words,
        },
    )
    if info.unassigned_words or info.tolerance_assignments or info.nearest_assignments:
        logger.info(
            "reconciliation_alignment_anomalies",
            extra={
                **context,
                "event": "reconciliation_alignment_anomalies",
                "tolerance_assignments": info.tolerance_assignments,
                "nearest_assignments": info.nearest_assignments,
                "unassigned_words": info.unassigned_words,
            },
        )
    return result
