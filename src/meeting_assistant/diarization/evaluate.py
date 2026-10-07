"""Saved-evidence diagnostics; DER only when the developer supplies real annotations."""

import argparse
import json
import sys
from pathlib import Path

from .exceptions import DiarizationError, DiarizationInferenceError, InvalidDiarizationResult
from .models import DiarizationResult, SpeakerAwareTranscript
from .serialization import diarization_from_json, speaker_transcript_from_json


def diagnostics(
    diarization: DiarizationResult,
    transcript: SpeakerAwareTranscript | None = None,
    *,
    expected_speakers: int | None = None,
) -> dict[str, object]:
    if expected_speakers is not None and (
        isinstance(expected_speakers, bool)
        or not isinstance(expected_speakers, int)
        or expected_speakers < 1
    ):
        raise InvalidDiarizationResult("Expected speaker count must be a positive integer.")
    if transcript is not None and transcript.diarization_id != diarization.diarization_id:
        raise InvalidDiarizationResult("Diagnostics require matching diarization provenance.")
    result: dict[str, object] = {
        "detected_speakers": diarization.detected_speaker_count,
        "expected_speakers": expected_speakers,
        "speaker_count_matches": (
            diarization.detected_speaker_count == expected_speakers
            if expected_speakers is not None
            else None
        ),
        "regular_turn_count": len(diarization.regular_turns),
        "exclusive_turn_count": len(diarization.exclusive_turns),
        "duration_seconds": diarization.duration_seconds,
        "diarization_seconds": diarization.processing_info.diarization_seconds,
        "real_time_factor": diarization.real_time_factor,
    }
    if transcript is not None:
        info = transcript.reconciliation_info
        result.update(
            total_words=info.total_words,
            direct_assignments=info.direct_assignments,
            tolerance_assignments=info.tolerance_assignments,
            nearest_assignments=info.nearest_assignments,
            unassigned_words=info.unassigned_words,
            assignment_coverage=info.assignment_coverage,
            utterance_count=len(transcript.utterances),
            speaker_switches=info.speaker_switches,
            overlap_region_count=len(transcript.overlap_regions),
        )
    return result


def score_der(
    diarization: DiarizationResult,
    reference_rttm: Path,
    *,
    recording_id: str,
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> dict[str, object]:
    """Standard optimal speaker mapping via pyannote.metrics, never synthesized ground truth."""
    if collar < 0:
        raise InvalidDiarizationResult("DER collar must be nonnegative.")
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.database.util import load_rttm
        from pyannote.metrics.diarization import DiarizationErrorRate

        references = load_rttm(reference_rttm)
        if recording_id not in references:
            raise InvalidDiarizationResult("Reference RTTM has no requested recording ID.")
        hypothesis = Annotation(uri=recording_id)
        for index, turn in enumerate(diarization.regular_turns):
            hypothesis[Segment(turn.start, turn.end), index] = turn.speaker_id
        metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        score = metric(references[recording_id], hypothesis)
        return {
            "der": float(score),
            "collar_seconds": collar,
            "skip_overlap": skip_overlap,
            "recording_id": recording_id,
            "scored_output": "regular_turns",
        }
    except (ImportError, OSError, ValueError, KeyError) as exc:
        raise DiarizationInferenceError(
            "Cannot score DER; check optional dependencies and RTTM."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect saved speaker evidence; coverage is not accuracy."
    )
    parser.add_argument("--diarization", type=Path, required=True)
    parser.add_argument("--speaker-transcript", type=Path)
    parser.add_argument("--expected-speakers", type=int)
    parser.add_argument("--reference-rttm", type=Path)
    parser.add_argument("--recording-id")
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args(argv)
    if args.reference_rttm and not args.recording_id:
        parser.error("--recording-id is required with --reference-rttm")
    try:
        diarization = diarization_from_json(args.diarization.read_text(encoding="utf-8"))
        transcript = (
            speaker_transcript_from_json(args.speaker_transcript.read_text(encoding="utf-8"))
            if args.speaker_transcript
            else None
        )
        result = diagnostics(diarization, transcript, expected_speakers=args.expected_speakers)
        if args.reference_rttm:
            result.update(
                score_der(
                    diarization,
                    args.reference_rttm,
                    recording_id=args.recording_id,
                    collar=args.collar,
                    skip_overlap=args.skip_overlap,
                )
            )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        if not args.reference_rttm:
            print("DER not scored: no reference annotations supplied.")
        return 0
    except (DiarizationError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
