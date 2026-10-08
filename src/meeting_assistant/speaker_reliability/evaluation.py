"""Reference-aware A/B evaluation. Scheduled synthesis clips are not acoustic DER truth."""

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .alignment import optimal_assignment
from .exceptions import InvalidReliability
from .serialization import save_speaker_reliability, to_json


def reference_mapping(turns, reference_segments):
    model_labels = sorted({x.speaker_id for x in turns})
    truth_labels = sorted({x["speaker_id"] for x in reference_segments})

    # Merge same-label intervals before summing intersections, avoiding duplicate mass.
    def union(items):
        merged = []
        for start, end in sorted(items):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return merged

    matrix = []
    for label in model_labels:
        actual = union((x.start, x.end) for x in turns if x.speaker_id == label)
        row = []
        for truth in truth_labels:
            expected = union(
                (x["start"], x["end"]) for x in reference_segments if x["speaker_id"] == truth
            )
            row.append(sum(max(0, min(b, d) - max(a, c)) for a, b in actual for c, d in expected))
        matrix.append(row)
    assignment = optimal_assignment(matrix)
    return {
        label: truth_labels[col] if col is not None else None
        for label, col in zip(model_labels, assignment, strict=True)
    }


def acoustic_metrics(primary, secondary, reference):
    if reference.get("verified_acoustic_annotations") is not True:
        return {
            "primary_der": None,
            "secondary_der": None,
            "primary_jer": None,
            "secondary_jer": None,
            "reason": "No verified acoustic speech/overlap annotation; scheduled clips are insufficient for DER/JER.",
        }
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    def annotation(turns):
        a = Annotation()
        for index, (label, start, end) in enumerate(turns):
            a[Segment(start, end), index] = label
        return a

    expected = annotation((x["speaker_id"], x["start"], x["end"]) for x in reference["segments"])
    result = {"collar_seconds": reference.get("collar_seconds", 0), "skip_overlap": False}
    for name, turns in (("primary", primary.regular_turns), ("secondary", secondary.segments)):
        actual = annotation((x.speaker_id, x.start, x.end) for x in turns)
        result[name + "_der"] = DiarizationErrorRate(
            collar=result["collar_seconds"], skip_overlap=False
        )(expected, actual)
        result[name + "_jer"] = JaccardErrorRate(
            collar=result["collar_seconds"], skip_overlap=False
        )(expected, actual)
    return result


def evaluate_reference(primary, speaker, result, reference):
    """Speaker correctness conditional on unambiguous authored source ownership.

    Real ASR-word midpoints or SAPI word-onset events are used as fixed probes. These
    count attribution correctness, not lexical correctness and not acoustic DER.
    """
    if result.availability != "available":
        return {"availability": "unavailable", "warnings": result.warnings}
    if abs(reference["duration_seconds"] - primary.duration_seconds) > 0.05:
        raise InvalidReliability("Reference recording duration does not match the canonical audio.")
    if reference.get("audio_sha256") != primary.audio_sha256:
        raise InvalidReliability("Reference must explicitly identify this canonical audio SHA256.")
    turns = reference["segments"]
    pmap = reference_mapping(primary.regular_turns, turns)
    smap = reference_mapping(result.secondary.segments, turns)
    mapping = {x.secondary_id: x.primary_id for x in result.comparison.alignment.mappings}
    refs = {
        (x.source_word_reference.segment_id, x.source_word_reference.word_index): x
        for x in result.words
    }
    probes = []
    for utterance in speaker.utterances:
        for word in utterance.words:
            sidecar = refs[
                (word.source_word_reference.segment_id, word.source_word_reference.word_index)
            ]
            probes.append(
                (
                    (word.start + word.end) / 2,
                    word.text,
                    word.speaker_id,
                    sidecar.secondary_mapped_speaker,
                    sidecar.agreement_fraction,
                )
            )
    mode = "existing_asr_word_midpoints"
    if not probes:
        mode = "authored_sapi_word_onset_probes"
        for event in reference.get("word_events", ()):
            t = event["time_seconds"]
            primary_labels = {x.speaker_id for x in primary.exclusive_turns if x.start <= t < x.end}
            secondary_labels = {
                x.speaker_id for x in result.secondary.segments if x.start <= t < x.end
            }
            a = next(iter(primary_labels)) if len(primary_labels) == 1 else None
            b = mapping.get(next(iter(secondary_labels))) if len(secondary_labels) == 1 else None
            probes.append((t, event["text"], a, b, 1.0 if a is not None and a == b else 0.0))
    counts = Counter()
    buckets = {
        key: Counter()
        for key in (
            "high",
            "medium",
            "low",
            "interjections",
            "labels_agree",
            "labels_disagree",
            "one_sided_or_ambiguous",
        )
    }
    details = []
    for time, text, primary_label, secondary_mapped, agreement in probes:
        expected = {x["speaker_id"] for x in turns if x["start"] <= time < x["end"]}
        if len(expected) != 1:
            counts["excluded_gap_or_overlap"] += 1
            continue
        truth = next(iter(expected))
        secondary_labels = {
            x.speaker_id for x in result.secondary.segments if x.start <= time < x.end
        }
        secondary_label = next(iter(secondary_labels)) if len(secondary_labels) == 1 else None
        pc, sc = pmap.get(primary_label) == truth, smap.get(secondary_label) == truth
        at_point = mapping.get(secondary_label)
        bucket = "high" if agreement >= 0.90 else "medium" if agreement >= 0.65 else "low"
        counts["scored_words"] += 1
        counts["primary_correct"] += pc
        counts["secondary_correct"] += sc
        counts["primary_unattributed"] += primary_label is None
        counts["secondary_unattributed_or_ambiguous"] += secondary_label is None
        groups = [bucket]
        groups.append(
            "one_sided_or_ambiguous"
            if primary_label is None or at_point is None
            else "labels_agree"
            if primary_label == at_point
            else "labels_disagree"
        )
        if text.lower().strip(" .,!?;") in ("yes", "no", "okay", "right", "sure"):
            groups.append("interjections")
        for key in groups:
            buckets[key]["words"] += 1
            buckets[key]["primary_correct"] += pc
            buckets[key]["secondary_correct"] += sc
            buckets[key]["primary_unattributed"] += primary_label is None
            buckets[key]["secondary_unattributed_or_ambiguous"] += secondary_label is None
        details.append(
            dict(
                time_seconds=time,
                text=text,
                truth=truth,
                primary_correct=pc,
                secondary_correct=sc,
                agreement_fraction=agreement,
                bucket=bucket,
            )
        )

    def summarize(c):
        n = c.get("words", c.get("scored_words", 0))
        return dict(
            c,
            primary_accuracy=c["primary_correct"] / n if n else None,
            secondary_accuracy=c["secondary_correct"] / n if n else None,
        )

    statuses = Counter(x.status for x in result.words)
    state_seconds = dict(result.comparison.state_seconds)
    intervals = result.comparison.intervals
    overlap_seconds = sum(
        x.end - x.start
        for x in intervals
        if len(x.primary_speakers) > 1 or len(x.secondary_speakers) > 1
    )
    return dict(
        reference_kind=reference["reference_kind"],
        probe_kind=mode,
        limitations=reference.get("limitations"),
        primary_count=len(primary.speakers),
        secondary_count=result.secondary.speaker_count,
        expected_count=len({x["speaker_id"] for x in turns}),
        agreement_fraction=result.comparison.agreement_fraction,
        state_seconds=state_seconds,
        overlap_union_seconds=overlap_seconds,
        mapping=result.comparison.alignment.mappings,
        primary_reference_mapping=tuple(pmap.items()),
        secondary_reference_mapping=tuple(smap.items()),
        word_accuracy=summarize(counts),
        agreement_strata={k: summarize(v) for k, v in buckets.items()},
        word_status_counts=tuple(sorted(statuses.items())),
        word_status_fractions=tuple((k, v / len(result.words)) for k, v in sorted(statuses.items()))
        if result.words
        else (),
        primary_runtime_seconds=primary.processing_info.diarization_seconds,
        primary_peak_gpu_memory_bytes=primary.processing_info.peak_gpu_memory_bytes,
        secondary_runtime_seconds=result.secondary.total_seconds,
        secondary_inference_seconds=result.secondary.inference_seconds,
        secondary_load_seconds=result.secondary.model_load_seconds,
        secondary_peak_gpu_memory_bytes=result.secondary.peak_gpu_memory_bytes,
        comparison_seconds=result.comparison.comparison_seconds,
        total_phase9_seconds=result.total_seconds,
        acoustic_metrics=acoustic_metrics(primary, result.secondary, reference),
        word_details=details,
    )


def _run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--speaker", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from meeting_assistant.diarization.serialization import (
        diarization_from_json,
        speaker_transcript_from_json,
    )

    from .config import SpeakerReliabilityConfig
    from .service import assess_speaker_reliability

    primary = diarization_from_json(args.primary.read_text(encoding="utf-8"))
    speaker = speaker_transcript_from_json(args.speaker.read_text(encoding="utf-8"))
    result = assess_speaker_reliability(
        args.audio,
        primary,
        speaker,
        config=replace(SpeakerReliabilityConfig.from_env(), enabled=True),
    )
    bundle = save_speaker_reliability(result, args.output_dir)
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8-sig"))
        report = evaluate_reference(primary, speaker, result, reference)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "evaluation.json").write_text(to_json(report), encoding="utf-8")
    print(f"Reliability: {result.availability}\nSidecars: {bundle}")
    return 0 if result.availability == "available" else 1


def main(argv=None):
    try:
        return _run(argv)
    except Exception as exc:
        logging.getLogger(__name__).debug("reliability_evaluation_failed", exc_info=True)
        message = (
            str(exc)
            if isinstance(exc, InvalidReliability)
            else "Check saved inputs and the local secondary-model setup."
        )
        print(f"Reliability evaluation failed: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
