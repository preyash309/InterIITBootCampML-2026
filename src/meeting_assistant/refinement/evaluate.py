"""Small labeled synthetic decision benchmark; never a fabricated speech corpus."""

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from meeting_assistant.asr.models import (
    ModelInfo,
    ProcessingInfo,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)
from meeting_assistant.asr.serialization import transcript_to_json
from meeting_assistant.diarization.config import ReconciliationConfig
from meeting_assistant.diarization.models import (
    ReconciliationInfo,
    SpeakerAwareTranscript,
    SpeakerUtterance,
    SpeakerWord,
    WordReference,
)
from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.grounding.config import GroundingConfig
from meeting_assistant.grounding.models import (
    GroundingCandidate,
    GroundingProcessingInfo,
    GroundingRecord,
    GroundingResult,
    RetrievalPolicy,
)
from meeting_assistant.grounding.normalization import normalize

from .exceptions import RefinementError
from .groq_backend import GroqRefinerBackend
from .service import digest, refine_transcript
from .validation import protected_signature, safety_violations


def controlled_evidence(text: str, spans: list[dict]):
    """Original synthetic text/word evidence and explicitly controlled candidate hypotheses.

    This bypasses retrieval intentionally to measure the decision/application stage alone.
    It is not an ASR accuracy test or measured embedding confidence.
    """
    matches = list(re.finditer(r"\S+", text))
    words = tuple(
        TranscriptWord(m.group(), i * 0.3, (i + 1) * 0.3, None) for i, m in enumerate(matches)
    )
    duration = max(1, len(words) * 0.3 + 1)
    raw = TranscriptResult(
        str(uuid4()),
        "en",
        None,
        duration,
        text,
        (TranscriptSegment("seg_000001", 0, len(words) * 0.3, text, words),) if words else (),
        ModelInfo("synthetic_fixture", "original_text"),
        ProcessingInfo(0, 0, 0, 1),
    )
    attributed = tuple(
        SpeakerWord(
            w,
            WordReference("seg_000001", i),
            "SPEAKER_00",
            "overlap",
            1.0,
            ("exclusive_turn_000001",),
            False,
        )
        for i, w in enumerate(words)
    )
    utterances = (
        (
            SpeakerUtterance(
                "utt_000001",
                "SPEAKER_00",
                0,
                len(words) * 0.3,
                text,
                attributed,
                ("seg_000001",),
                False,
            ),
        )
        if words
        else ()
    )
    source = SpeakerAwareTranscript(
        str(uuid4()),
        raw.transcript_id,
        digest(transcript_to_json(raw)),
        str(uuid4()),
        duration,
        ("SPEAKER_00",) if words else (),
        utterances,
        (),
        ReconciliationInfo(len(words), len(words), 0, 0, 0, 0, 0, ReconciliationConfig()),
    )
    records = []
    for item in sorted(spans, key=lambda s: text.index(s["span"])):
        observed = item["span"]
        start = text.index(observed)
        end = start + len(observed)
        selected = [w for m, w in zip(matches, attributed) if m.start() < end and m.end() > start]
        candidates = []
        for j, values in enumerate(item["candidates"]):
            defaults = dict(
                entry_id=f"fixture.candidate_{len(records) + 1}_{j + 1}",
                canonical=values["canonical"],
                scope="global",
                domain="software",
                category="controlled_term",
                description="Explicitly controlled candidate evidence for a synthetic regression example.",
                matched_variant=observed,
                match_type="fuzzy",
                score=0.9,
                lexical_score=0.9,
                phonetic_score=0.95,
                semantic_score=0.8,
                scope_score=0.3,
                reasons=("controlled_fixture",),
            )
            candidates.append(GroundingCandidate(**{**defaults, **values}))
        records.append(
            GroundingRecord(
                f"grnd_{len(records) + 1:06d}",
                "utt_000001",
                "SPEAKER_00",
                min(w.start for w in selected),
                max(w.end for w in selected),
                start,
                end,
                observed,
                normalize(observed),
                text,
                tuple(w.source_word_reference for w in selected),
                ("seg_000001",),
                tuple(candidates),
            )
        )
    config = GroundingConfig()
    policy = RetrievalPolicy(
        **{
            name: getattr(config, name)
            for name in RetrievalPolicy.__dataclass_fields__
            if name != "policy_version"
        }
    )
    grounding = GroundingResult(
        str(uuid4()),
        source.transcript_id,
        digest(speaker_transcript_to_json(source)),
        raw.transcript_id,
        source.source_raw_sha256,
        "0" * 64,
        "synthetic_fixture",
        "synthetic_v1",
        tuple(records),
        GroundingProcessingInfo(0, 0, 0, 0, 0, len(utterances), len(records), False),
        policy,
    )
    return source, grounding


def evaluate_cases(
    cases: list[dict], backend, *, response_dir: Path | None = None, pause_seconds: float = 0
) -> dict:
    if not 0 <= pause_seconds <= 60:
        raise ValueError("Benchmark pacing must be between 0 and 60 seconds.")
    outcomes = []
    calls = []
    for index, case in enumerate(cases):
        if index and pause_seconds:
            time.sleep(pause_seconds)
        source, grounding = controlled_evidence(case["text"], [case])
        result = refine_transcript(source, grounding, backend=backend)
        e = result.edit_log[0]
        applied = e.validation_status == "applied"
        expected = case.get("replacement", case["span"])
        correct_edit = (
            applied and e.replacement_text == expected and case["expected_action"] == "REPLACE"
        )
        protected_changed = protected_signature(
            result.utterances[0].raw_text
        ) != protected_signature(result.utterances[0].refined_text)
        speaker_time_changed = any(
            (r.speaker_id, r.start, r.end) != (s.speaker_id, s.start, s.end)
            for r, s in zip(result.utterances, source.utterances)
        )
        outcomes.append(
            {
                "id": case["id"],
                "expected_action": case["expected_action"],
                "actual_action": e.action,
                "status": e.validation_status,
                "correct_edit": correct_edit,
                "applied": applied,
                "replacement": e.replacement_text,
                "rejection_reason": e.rejection_reason,
                "protected_content_violation": protected_changed,
                "speaker_time_violation": speaker_time_changed,
                "meaning_changing_edit": applied and not correct_edit,
                "safety_violations": safety_violations(
                    result.utterances[0].raw_text, result.utterances[0].refined_text
                ),
            }
        )
        calls.extend(result.processing_info.calls)
        if response_dir:
            from .serialization import save_refined_transcript

            save_refined_transcript(result, response_dir)
    tp = sum(o["correct_edit"] for o in outcomes)
    predicted = sum(o["applied"] for o in outcomes)
    positives = sum(o["expected_action"] == "REPLACE" for o in outcomes)
    negatives = len(outcomes) - positives
    precision = tp / predicted if predicted else None
    recall = tp / positives if positives else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    return {
        "benchmark": "controlled_refinement_v1",
        "origin": "Original synthetic text and labeled candidate evidence; not recorded ASR ground truth.",
        "model_info": asdict(backend.model_info),
        "case_count": len(outcomes),
        "edit_precision": precision,
        "edit_recall": recall,
        "edit_f1": f1,
        "false_correction_rate": sum(
            o["applied"] and o["expected_action"] != "REPLACE" for o in outcomes
        )
        / negatives
        if negatives
        else None,
        "correct_decision_rates": {
            action: sum(
                o["actual_action"] == action for o in outcomes if o["expected_action"] == action
            )
            / sum(o["expected_action"] == action for o in outcomes)
            if any(o["expected_action"] == action for o in outcomes)
            else None
            for action in ("KEEP", "REPLACE", "UNCERTAIN")
        },
        "protected_content_violations": sum(o["protected_content_violation"] for o in outcomes),
        "safety_violations": {
            key: sum(o["safety_violations"][key] for o in outcomes)
            for key in (
                "number_changes",
                "negation_changes",
                "date_time_changes",
                "modality_changes",
                "protected_content_changes",
            )
        },
        "speaker_time_violations": sum(o["speaker_time_violation"] for o in outcomes),
        "meaning_changing_edits": sum(o["meaning_changing_edit"] for o in outcomes),
        "meaning_changing_edit_rate": sum(o["meaning_changing_edit"] for o in outcomes) / predicted
        if predicted
        else 0.0,
        "api_calls": len(calls),
        "calls": [asdict(c) for c in calls],
        "outcomes": outcomes,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded live Groq decision benchmark (billed API calls)."
    )
    parser.add_argument("--cases", type=Path, default=Path("data/refinement_benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/refinement.json"))
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0,
        help="Explicit pacing for account token limits; at most 60 s between cases.",
    )
    args = parser.parse_args(argv)
    try:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        report = evaluate_cases(
            cases,
            GroqRefinerBackend(),
            response_dir=args.artifacts,
            pause_seconds=args.pause_seconds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (RefinementError, OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    print(json.dumps({k: v for k, v in report.items() if k not in ("outcomes", "calls")}, indent=2))
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
