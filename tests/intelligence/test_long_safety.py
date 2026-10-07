import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.intelligence.config import IntelligenceConfig
from meeting_assistant.intelligence.evaluate import evaluate_cases, score_cases
from meeting_assistant.intelligence.exceptions import ConsolidationError, EvidenceAudioError
from meeting_assistant.intelligence.fixtures import text_evidence
from meeting_assistant.intelligence.models import (
    ActionItem,
    ActionOwner,
    Decision,
    IntelligenceRequest,
    MeetingContent,
    MinuteItem,
    SummaryPoint,
)
from meeting_assistant.intelligence.service import extract_meeting_record, validate_consolidated
from meeting_assistant.intelligence.validation import parse_consolidation

from .helpers import FakeBackend, blank, evidence


class LongSafetyTests(unittest.TestCase):
    def test_malicious_backend_new_claim(self):
        partial = MeetingContent(
            summary=(SummaryPoint("p0001_sum_0001", "Existing", ("utt_000001",)),)
        )
        changed = MeetingContent(summary=(replace(partial.summary[0], text="New invented claim"),))
        with self.assertRaises(ConsolidationError):
            validate_consolidated(changed, partial)

    def test_malicious_backend_new_evidence(self):
        partial = MeetingContent(
            summary=(SummaryPoint("p0001_sum_0001", "Existing", ("utt_000001",)),)
        )
        changed = MeetingContent(
            summary=(
                replace(partial.summary[0], evidence_utterance_ids=("utt_000001", "utt_999999")),
            )
        )
        with self.assertRaises(ConsolidationError):
            validate_consolidated(changed, partial)

    def test_cannot_drop_primary_evidence(self):
        partial = MeetingContent(
            summary=(SummaryPoint("p0001_sum_0001", "Existing", ("utt_000001",)),)
        )
        changed = MeetingContent(
            summary=(replace(partial.summary[0], evidence_utterance_ids=("utt_000002",)),)
        )
        with self.assertRaises(ConsolidationError):
            validate_consolidated(changed, partial)

    def test_conflicting_owners_deadlines(self):
        first = ActionItem(
            "p0001_act_0001",
            "Benchmark",
            ("utt_000001",),
            ActionOwner("speaker", "SPEAKER_00"),
            "Friday",
        )
        for changes in ({"owner": None}, {"deadline_text": "Monday"}):
            second = replace(
                first, id="p0002_act_0001", evidence_utterance_ids=("utt_000002",), **changes
            )
            request = IntelligenceRequest(
                "consolidation", (), "consolidation", MeetingContent(action_items=(first, second))
            )
            data = blank()
            data["action_items"] = [{"primary_item_id": first.id, "merge_item_ids": [second.id]}]
            with self.subTest(changes=changes), self.assertRaises(ConsolidationError):
                parse_consolidation(json.dumps(data), request)

    def test_cross_section_partial_rejected(self):
        content = MeetingContent(
            summary=(SummaryPoint("p0001_sum_0001", "Proposal", ("utt_000001",)),)
        )
        request = IntelligenceRequest("consolidation", (), "consolidation", content)
        data = blank()
        data["decisions"] = [{"primary_item_id": "p0001_sum_0001", "merge_item_ids": []}]
        with self.assertRaises(ConsolidationError):
            parse_consolidation(json.dumps(data), request)

    def test_cannot_merge_proposal_with_decision_minutes(self):
        first = MinuteItem("p0001_min_0001", "Proposal", ("utt_000001",), "Database", "proposal")
        second = replace(first, id="p0002_min_0001", kind="decision")
        request = IntelligenceRequest(
            "consolidation", (), "consolidation", MeetingContent(minutes=(first, second))
        )
        data = blank()
        data["minutes"] = [{"primary_item_id": first.id, "merge_item_ids": [second.id]}]
        with self.assertRaises(ConsolidationError):
            parse_consolidation(json.dumps(data), request)

    def test_empty_without_credentials(self):
        refined, speaker, grounding = text_evidence([])
        with patch(
            "meeting_assistant.intelligence.groq_backend.GroqMeetingIntelligenceBackend"
        ) as backend:
            result = extract_meeting_record(
                refined, speaker, grounding, config=IntelligenceConfig()
            )
        backend.assert_not_called()
        self.assertEqual(result.content.items, ())

    def test_audio_without_strong_provenance_before_call(self):
        refined, speaker, grounding = evidence()
        backend = FakeBackend()
        with self.assertRaises(EvidenceAudioError):
            extract_meeting_record(
                refined, speaker, grounding, backend=backend, canonical_audio_path="audio.wav"
            )
        self.assertEqual(backend.requests, [])

    def test_distinct_noncontiguous_evidence_preserved(self):
        refined, speaker, grounding = text_evidence(
            [
                {"text": "We could switch."},
                {"text": "Another topic."},
                {"text": "Agreed to switch."},
            ]
        )
        backend = FakeBackend(
            lambda _: MeetingContent(
                decisions=(Decision("dec_0001", "Switch", ("utt_000003", "utt_000001")),)
            )
        )
        result = extract_meeting_record(refined, speaker, grounding, backend=backend)
        self.assertEqual(result.decisions[0].evidence_utterance_ids, ("utt_000001", "utt_000003"))
        from meeting_assistant.intelligence.evidence import get_item_evidence

        spans = get_item_evidence(result, "dec_0001", refined)
        self.assertEqual(len(spans), 2)
        self.assertLess(spans[0].end, spans[1].start)

    def test_benchmark_original_labels_and_metrics(self):
        cases = json.loads(Path("data/intelligence_benchmark.json").read_text())
        self.assertEqual(len(cases), 18)
        case = next(c for c in cases if c["id"] == "self_commitment")
        refined, speaker, grounding = text_evidence(case["turns"])
        backend = FakeBackend(
            lambda _: MeetingContent(
                action_items=(
                    ActionItem(
                        "act_0001",
                        "Benchmark both models",
                        ("utt_000001",),
                        ActionOwner("speaker", "SPEAKER_00"),
                    ),
                )
            )
        )
        result = extract_meeting_record(refined, speaker, grounding, backend=backend)
        report = score_cases([case], [result])
        self.assertEqual(report["action_items"]["f1"], 1)
        self.assertEqual(report["owner_accuracy_on_matched_actions"], 1)
        self.assertEqual(report["deadline_accuracy_on_matched_actions"], 1)
        self.assertEqual(report["false_owner_count"], 0)

    def test_benchmark_output_and_invalid_pause(self):
        cases = [{"id": "original", "turns": [{"text": "Discussion only."}]}]
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "results"
            result = evaluate_cases(cases, FakeBackend(), output_dir=destination)
            self.assertTrue((destination / "report.json").exists())
            self.assertEqual(result["api_calls"], 0)
        with self.assertRaises(ValueError):
            evaluate_cases(cases, FakeBackend(), pause_seconds=61)
