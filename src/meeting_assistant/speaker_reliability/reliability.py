"""Time-weighted word/utterance observations without retiming or relabeling evidence."""

from bisect import bisect_right
from collections import Counter

from .models import SpeakerWordReliability, UtteranceSpeakerReliability


def measures(start, end, speaker, intervals, ends):
    totals = Counter()
    speakers = Counter()
    # Zero-duration ASR words carry no measurable temporal support; do not invent it.
    duration = end - start
    if duration <= 0:
        return dict(
            secondary_mapped_speaker=None,
            **dict.fromkeys(
                (
                    "agreement_fraction",
                    "disagreement_fraction",
                    "primary_only_fraction",
                    "secondary_only_fraction",
                    "silence_fraction",
                    "primary_overlap_fraction",
                    "secondary_overlap_fraction",
                    "overlap_agreement_fraction",
                    "unmapped_fraction",
                ),
                0.0,
            ),
        )
    for index in range(bisect_right(ends, start), len(intervals)):
        item = intervals[index]
        if item.start >= end:
            break
        time = max(0.0, min(end, item.end) - max(start, item.start))
        sole = (
            item.mapped_secondary_speakers[0]
            if len(item.secondary_speakers) == 1 and len(item.mapped_secondary_speakers) == 1
            else None
        )
        if sole:
            speakers[sole] += time
        if item.state == "SILENCE":
            totals["silence"] += time
        elif item.state == "PRIMARY_ONLY":
            totals["primary_only"] += time
        elif item.state == "SECONDARY_ONLY":
            totals["secondary_only"] += time
        elif item.state == "UNMAPPED_SECONDARY":
            totals["unmapped"] += time
        elif speaker is not None and sole == speaker and item.primary_exclusive_speaker == speaker:
            totals["agreement"] += time
        else:
            totals["disagreement"] += time
        if len(item.primary_speakers) > 1:
            totals["primary_overlap"] += time
        if len(item.secondary_speakers) > 1:
            totals["secondary_overlap"] += time
        if item.state == "OVERLAP_AGREE":
            totals["overlap_agreement"] += time
    dominant = min(speakers, key=lambda x: (-speakers[x], x)) if speakers else None
    return dict(
        secondary_mapped_speaker=dominant,
        **{
            key + "_fraction": min(1.0, totals[key] / duration)
            for key in (
                "agreement",
                "disagreement",
                "primary_only",
                "secondary_only",
                "silence",
                "primary_overlap",
                "secondary_overlap",
                "overlap_agreement",
                "unmapped",
            )
        },
    )


def classify(metrics, config, *, zero_duration=False, count_mismatch=False):
    reasons = []
    if zero_duration:
        reasons.append("zero_duration_no_temporal_support")
    if metrics["primary_overlap_fraction"] > 0 or metrics["secondary_overlap_fraction"] > 0:
        reasons.append("overlapping_speech")
    if metrics["disagreement_fraction"] > 0:
        reasons.append("speaker_disagreement")
    if metrics["primary_only_fraction"] or metrics["secondary_only_fraction"]:
        reasons.append("speech_presence_disagreement")
    if metrics["silence_fraction"]:
        reasons.append("shared_silence")
    if metrics["unmapped_fraction"]:
        reasons.append("unmapped_secondary")
    if count_mismatch:
        reasons.append("speaker_count_mismatch")
    overlap = max(metrics["primary_overlap_fraction"], metrics["secondary_overlap_fraction"])
    if zero_duration or metrics["unmapped_fraction"] > 1 - config.mixed_threshold:
        status = "UNCERTAIN"
    elif (
        metrics["agreement_fraction"] >= config.reliable_threshold
        and overlap <= config.low_overlap_threshold
    ):
        status = "RELIABLE"
        reasons.append("high_model_agreement")
    elif metrics["agreement_fraction"] >= config.mixed_threshold:
        status = "MIXED"
    else:
        status = "UNCERTAIN"
    return status, tuple(reasons)


def boundary_deltas(utterance, primary, secondary, alignment):
    mapping = {x.secondary_id: x.primary_id for x in alignment.mappings}

    def best(turns, labels):
        candidates = [
            (max(0, min(x.end, utterance.end) - max(x.start, utterance.start)), x)
            for x in turns
            if labels(x) == utterance.speaker_id
        ]
        candidates = [x for x in candidates if x[0] > 0]
        return (
            min(candidates, key=lambda x: (-x[0], x[1].start, x[1].end))[1] if candidates else None
        )

    a = best(primary.exclusive_turns, lambda x: x.speaker_id)
    b = best(secondary.segments, lambda x: mapping.get(x.speaker_id))
    if a is None or b is None:
        return None, None, None
    start, end = b.start - a.start, b.end - a.end
    return start, end, abs(start) + abs(end)


def reliability_records(speaker, primary, secondary, comparison, config):
    words, utterances = [], []
    ends = tuple(x.end for x in comparison.intervals)
    for utterance in speaker.utterances:
        for word in utterance.words:
            metrics = measures(word.start, word.end, word.speaker_id, comparison.intervals, ends)
            status, reasons = classify(
                metrics,
                config,
                zero_duration=word.end == word.start,
                count_mismatch=comparison.alignment.count_mismatch,
            )
            words.append(
                SpeakerWordReliability(
                    word.source_word_reference,
                    word.speaker_id,
                    **metrics,
                    status=status,
                    reasons=reasons,
                )
            )
        metrics = measures(
            utterance.start, utterance.end, utterance.speaker_id, comparison.intervals, ends
        )
        status, reasons = classify(
            metrics, config, count_mismatch=comparison.alignment.count_mismatch
        )
        start, end, error = boundary_deltas(utterance, primary, secondary, comparison.alignment)
        utterances.append(
            UtteranceSpeakerReliability(
                utterance.id,
                utterance.speaker_id,
                **metrics,
                start_delta_seconds=start,
                end_delta_seconds=end,
                boundary_disagreement_seconds=error,
                status=status,
                reasons=reasons,
            )
        )
    return tuple(words), tuple(utterances)
