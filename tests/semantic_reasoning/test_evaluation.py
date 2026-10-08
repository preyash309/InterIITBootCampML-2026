import unittest

from meeting_assistant.semantic_reasoning import SemanticConfig, analyze_meeting_semantics
from meeting_assistant.semantic_reasoning.evaluation import (
    calibration,
    evaluate_cases,
    evaluate_verification,
    precision_recall_f1,
)
from tests.semantic_reasoning.helpers import evidence, scripted


class EvaluationTests(unittest.TestCase):
    def test_unmeasured_is_not_zero_accuracy(self):
        result = evaluate_cases([{"id": "missing"}], {})
        self.assertIsNone(result["macro_f1"])
        self.assertEqual(result["calibration"]["status"], "unavailable")
        self.assertEqual(evaluate_verification([], {}), {})

    def test_precision_recall_handles_misses_and_spurious_events(self):
        metrics = precision_recall_f1({"a", "b"}, {"a", "c", "d"})
        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.4)

    def test_calibration_uses_full_distribution_without_truth_claim(self):
        backend = scripted([("PROPOSAL", 0.9)])
        result = analyze_meeting_semantics(
            *evidence(["We could use Redis."]),
            backend=backend,
            config=SemanticConfig(enabled=True),
            environ={},
        )
        decision = result.events[0].decision
        metrics = calibration([("PROPOSAL", decision)])
        self.assertEqual(metrics["sample_count"], 1)
        self.assertAlmostEqual(metrics["ece"], 0.1)
        self.assertAlmostEqual(metrics["brier"], 0.01 + 0.01 / 13)
        self.assertIn("not_calibrated", metrics["status"])

    def test_fake_provider_not_counted_as_live_model_accuracy(self):
        result = analyze_meeting_semantics(
            *evidence(["We could use Redis."]),
            backend=scripted(["PROPOSAL"]),
            config=SemanticConfig(enabled=True),
            environ={},
        )
        case = {
            "id": "c",
            "events": [{"utterance_id": "utt_000001", "type": "PROPOSAL"}],
            "relations": [],
            "current_decision_utterance_ids": [],
            "historical_decision_utterance_ids": [],
            "turns": [{"id": "utt_000001"}],
            "coverage_reference": None,
        }
        metrics = evaluate_cases([case], {"c": result})
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["calibration"]["sample_count"], 0)

    def test_graph_sidecar_cannot_forge_cycle_metadata(self):
        from dataclasses import replace

        from meeting_assistant.semantic_reasoning.exceptions import InvalidSemantics

        result = analyze_meeting_semantics(
            *evidence(["We could use Redis."]),
            backend=scripted(["PROPOSAL"]),
            config=SemanticConfig(enabled=True),
            environ={},
        )
        with self.assertRaises(InvalidSemantics):
            replace(result, event_graph=replace(result.event_graph, cycles=(("evt_000001",),)))
