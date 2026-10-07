"""Small explicitly labeled text benchmark, separate from recording/ASR evaluation."""

import argparse
import json
import re
import time
from pathlib import Path

from .fixtures import text_evidence
from .groq_backend import GroqMeetingIntelligenceBackend
from .serialization import meeting_record_to_json
from .service import extract_meeting_record


def _matches(text, label):
    tokens = set(re.findall(r"\w+", text.casefold()))
    return all(
        any(alternative.casefold() in tokens for alternative in alternatives)
        for alternatives in label["keywords"]
    )


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def score_cases(cases, records):
    """One-to-one labeled keyword matches; inspect saved predictions to adjudicate synonyms.

    Supporting-ID precision/recall measures annotated reference overlap only, not semantic
    entailment. Unmatched outputs count as false positives, even conservative extra points.
    """
    totals = {section: [0, 0, 0] for section in ("decisions", "action_items")}
    owner_correct = deadline_correct = matched_actions = false_owner = false_deadline = 0
    evidence_expected = evidence_correct = evidence_cited = valid_refs = refs = cited_items = (
        items
    ) = 0
    outcomes = []
    for case, record in zip(cases, records):
        known = {f"utt_{n:06d}" for n in range(1, len(case["turns"]) + 1)}
        case_details = {"case_id": case["id"], "matches": []}
        for section in totals:
            gold = case.get(section, [])
            used = set()
            tp = 0
            for item in getattr(record, section):
                candidate = next(
                    (
                        n
                        for n, label in enumerate(gold)
                        if n not in used and _matches(item.text, label)
                    ),
                    None,
                )
                if candidate is None:
                    if section == "action_items":
                        false_owner += item.owner is not None
                        false_deadline += item.deadline_text is not None
                    case_details["matches"].append({"item_id": item.id, "matched": False})
                    continue
                label = gold[candidate]
                used.add(candidate)
                tp += 1
                supporting = set(label["evidence"])
                cited = set(item.evidence_utterance_ids)
                evidence_correct += len(supporting & cited)
                evidence_expected += len(supporting)
                evidence_cited += len(cited)
                if section == "action_items":
                    matched_actions += 1
                    actual_owner = (
                        None
                        if item.owner is None
                        else (
                            item.owner.speaker_id
                            if item.owner.kind == "speaker"
                            else item.owner.display_text
                        )
                    )
                    expected_owner = label.get("owner")
                    expected_deadline = label.get("deadline")
                    owner_correct += actual_owner == expected_owner
                    deadline_correct += item.deadline_text == expected_deadline
                    false_owner += actual_owner is not None and actual_owner != expected_owner
                    false_deadline += (
                        item.deadline_text is not None and item.deadline_text != expected_deadline
                    )
                case_details["matches"].append({"item_id": item.id, "matched": True})
            totals[section][0] += tp
            totals[section][1] += len(getattr(record, section)) - tp
            totals[section][2] += len(gold) - tp
        for item in record.content.items:
            items += 1
            cited_items += bool(item.evidence_utterance_ids)
            refs += len(item.evidence_utterance_ids)
            valid_refs += sum(uid in known for uid in item.evidence_utterance_ids)
        outcomes.append(case_details)
    return {
        "dataset_kind": "original_labeled_synthetic_text",
        "case_count": len(cases),
        "decisions": _prf(*totals["decisions"]),
        "action_items": _prf(*totals["action_items"]),
        "owner_accuracy_on_matched_actions": owner_correct / matched_actions
        if matched_actions
        else None,
        "deadline_accuracy_on_matched_actions": deadline_correct / matched_actions
        if matched_actions
        else None,
        "false_owner_count": false_owner,
        "false_deadline_count": false_deadline,
        "unknown_evidence_references": refs - valid_refs,
        "evidence_validity": valid_refs / refs if refs else None,
        "evidence_coverage": cited_items / items if items else None,
        "supporting_id_precision_on_matched_claims": evidence_correct / evidence_cited
        if evidence_cited
        else None,
        "supporting_id_recall_on_matched_claims": evidence_correct / evidence_expected
        if evidence_expected
        else None,
        "unmatched_decision_count": totals["decisions"][1],
        "unmatched_task_count": totals["action_items"][1],
        "summary_review": "Manually inspect saved source/predictions for unsupported claims and missed major points; no automatic entailment metric.",
        "outcomes": outcomes,
    }


def evaluate_cases(cases, backend, *, output_dir=None, pause_seconds=0):
    if not 0 <= pause_seconds <= 60:
        raise ValueError("Benchmark pacing must be bounded to 0–60 seconds.")
    records = []
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=False)
    for n, case in enumerate(cases):
        if n and pause_seconds:
            time.sleep(pause_seconds)
        refined, speaker, grounding = text_evidence(case["turns"])
        result = extract_meeting_record(refined, speaker, grounding, backend=backend)
        records.append(result)
        if output_dir:
            # Generated filenames use indices, never untrusted case IDs.
            Path(output_dir, f"case_{n + 1:04d}.json").write_text(
                meeting_record_to_json(result), encoding="utf-8"
            )
    report = score_cases(cases, records)
    report["api_calls"] = sum(len(r.processing_info.calls) for r in records)
    report["provider_latency_seconds"] = sum(
        r.processing_info.provider_latency_seconds for r in records
    )
    usage = [r.processing_info.usage() for r in records]
    report["total_tokens"] = sum(usage) if all(u is not None for u in usage) else None
    if output_dir:
        Path(output_dir, "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Opt-in paid extraction benchmark on original labeled text."
    )
    parser.add_argument("--cases", type=Path, default=Path("data/intelligence_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pause-seconds", type=float, default=10)
    args = parser.parse_args(argv)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    print(
        json.dumps(
            evaluate_cases(
                cases,
                GroqMeetingIntelligenceBackend(),
                output_dir=args.output_dir,
                pause_seconds=args.pause_seconds,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
