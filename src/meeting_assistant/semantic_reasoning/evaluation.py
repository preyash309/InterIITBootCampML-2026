"""Exact annotation metrics; unmeasured accuracy/calibration remain unavailable."""

from collections import Counter

from .models import require
from .ontology import HIGH_VALUE


def precision_recall_f1(expected, observed):
    expected, observed = set(expected), set(observed)
    correct = len(expected & observed)
    precision = correct / len(observed) if observed else float(not expected)
    recall = correct / len(expected) if expected else float(not observed)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "expected": len(expected),
        "observed": len(observed),
        "correct": correct,
    }


def calibration(samples):
    """Multiclass Brier (sum over labels), top-choice ECE and ten bins."""
    if not samples:
        return {"status": "unavailable", "sample_count": 0}
    errors, bins = [], {n: [] for n in range(10)}
    for expected, decision in samples:
        probabilities = dict(decision.probabilities)
        require(expected in probabilities)
        errors.append(sum((p - int(label == expected)) ** 2 for label, p in probabilities.items()))
        p = decision.selected_probability
        bins[min(9, int(p * 10))].append((p, decision.choice == expected))
    rows = [
        {
            "lower": n / 10,
            "upper": (n + 1) / 10,
            "count": len(values),
            "mean_probability": sum(p for p, _ in values) / len(values),
            "accuracy": sum(c for _, c in values) / len(values),
        }
        for n, values in bins.items()
        if values
    ]
    return {
        "status": "measured_small_sample_not_calibrated",
        "sample_count": len(samples),
        "brier": sum(errors) / len(errors),
        "bins": rows,
        "ece": sum(row["count"] * abs(row["mean_probability"] - row["accuracy"]) for row in rows)
        / len(samples),
    }


def evaluate_cases(cases, results):
    expected, observed, expected_edges, observed_edges = set(), set(), set(), set()
    confusion, samples = Counter(), []
    final, supersession, chronological, valid, false_proposals, rejected_retained = (0,) * 6
    coverage_counts = {label: [0, 0] for label in HIGH_VALUE}
    completed = 0
    for case in cases:
        result = results.get(case["id"])
        if result is None or result.availability not in ("available", "partial"):
            continue
        completed += 1
        cid = case["id"]
        by_uid = {e.evidence_utterance_ids[0]: e for e in result.events}
        for annotation in case["events"]:
            uid, label = annotation["utterance_id"], annotation["type"]
            e = by_uid.get(uid)
            output = (
                e.event_type
                if e and e.outcome == "accepted"
                else ("AMBIGUOUS" if e and e.outcome == "ambiguous" else "NONE")
            )
            confusion[label, output] += 1
            if label not in ("NONE", "AMBIGUOUS"):
                expected.add((cid, uid, label))
            if output not in ("NONE", "AMBIGUOUS"):
                observed.add((cid, uid, output))
            if e and e.decision and e.decision.provider in ("typesafe", "julia", "gliner"):
                samples.append((label, e.decision))
            false_proposals += label == "PROPOSAL" and output == "DECISION"
        by_id = {e.id: e for e in result.events}
        expected_edges.update(
            (cid, r["source_utterance_id"], r["target_utterance_id"], r["type"])
            for r in case["relations"]
        )
        observed_edges.update(
            (
                cid,
                by_id[r.source_event_id].evidence_utterance_ids[0],
                by_id[r.target_event_id].evidence_utterance_ids[0],
                r.relation_type,
            )
            for r in result.relations
        )
        current = {
            by_id[eid].evidence_utterance_ids[0]
            for chain in result.decision_evolution
            for eid in chain.current_decision_ids
        }
        historical = {
            by_id[eid].evidence_utterance_ids[0]
            for chain in result.decision_evolution
            for eid in chain.historical_decision_ids
        }
        final += current == set(case["current_decision_utterance_ids"])
        supersession += historical == set(case["historical_decision_utterance_ids"])
        chronological += all(
            list(c.ordered_event_ids)
            == sorted(c.ordered_event_ids, key=lambda eid: (by_id[eid].start, eid))
            for c in result.decision_evolution
        )
        allowed = {t["id"] for t in case["turns"]}
        valid += all(set(e.evidence_utterance_ids) <= allowed for e in result.events) and all(
            set(r.evidence_utterance_ids) <= allowed for r in result.relations
        )
        rejected_retained += len(
            current
            & {r["source_utterance_id"] for r in case["relations"] if r["type"] == "REJECTS"}
        )
        # Coverage gold must be manually annotated for the actual baseline MeetingRecord.
        reference = case.get("coverage_reference")
        if reference is not None:
            links = {by_id[o.event_id].evidence_utterance_ids[0]: o for o in result.coverage}
            for annotation in case["events"]:
                label, uid = annotation["type"], annotation["utterance_id"]
                if label in HIGH_VALUE:
                    counts = coverage_counts[label]
                    counts[1] += 1
                    counts[0] += bool(links.get(uid)) and set(links[uid].matched_record_ids) == set(
                        reference[uid]
                    )

    def per_class(a, b):
        return {
            label: precision_recall_f1(
                {x for x in a if x[-1] == label}, {x for x in b if x[-1] == label}
            )
            for label in sorted({x[-1] for x in a | b})
        }

    event_metrics = per_class(expected, observed)
    return {
        "cases_completed": completed,
        "cases_unavailable": len(cases) - completed,
        "events": precision_recall_f1(expected, observed) if completed else None,
        "macro_f1": sum(v["f1"] for v in event_metrics.values()) / len(event_metrics)
        if event_metrics
        else None,
        "per_class": event_metrics,
        "confusion_matrix": [
            {"reference": a, "observed": b, "count": n} for (a, b), n in sorted(confusion.items())
        ],
        "relations": precision_recall_f1(expected_edges, observed_edges) if completed else None,
        "relation_per_class": per_class(expected_edges, observed_edges),
        "decision_evolution": {
            "sample_count": completed,
            "final_correct": final,
            "supersession_correct": supersession,
            "chronological": chronological,
            "evidence_valid": valid,
            "proposal_false_decisions": false_proposals,
            "rejected_options_retained": rejected_retained,
        },
        "coverage_exact_links": {
            label: {"correct": a, "annotated_events": b, "accuracy": a / b if b else None}
            for label, (a, b) in coverage_counts.items()
        },
        "calibration": calibration(samples),
        "raw_argmax": {
            "sample_count": len(samples),
            "correct": sum(d.choice == label for label, d in samples),
            "accuracy": sum(d.choice == label for label, d in samples) / len(samples)
            if samples
            else None,
        },
    }


def evaluate_verification(challenges, observations):
    counts = {}
    for challenge in challenges:
        observation = observations.get(challenge["id"])
        if observation is None or not observation.dimensions:
            continue
        dimensions = {d.question_id: d for d in observation.dimensions}
        for name, expected in challenge["dimensions"].items():
            d = dimensions.get(name)
            correct, total = counts.get(name, (0, 0))
            counts[name] = (correct + int(d is not None and d.choice == expected), total + 1)
    return {name: {"correct": a, "count": b, "accuracy": a / b} for name, (a, b) in counts.items()}
