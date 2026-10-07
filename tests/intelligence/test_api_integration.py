"""Explicit paid opt-in; ordinary CI never contacts Groq."""

import os
import unittest

from meeting_assistant.intelligence.fixtures import text_evidence
from meeting_assistant.intelligence.groq_backend import GroqMeetingIntelligenceBackend
from meeting_assistant.intelligence.service import extract_meeting_record


@unittest.skipUnless(os.environ.get("RUN_INTELLIGENCE_API_TESTS") == "1", "Paid Groq opt-in")
class LiveIntelligenceTests(unittest.TestCase):
    def run_case(self, text):
        refined, speaker, grounding = text_evidence([{"text": text}])
        return extract_meeting_record(
            refined, speaker, grounding, backend=GroqMeetingIntelligenceBackend()
        )

    def test_proposal_empty_decisions_tasks(self):
        result = self.run_case(
            "We could switch to PostgreSQL; this is a proposal, not a decision or assignment."
        )
        self.assertEqual(result.decisions, ())
        self.assertEqual(result.action_items, ())

    def test_explicit_decision(self):
        result = self.run_case(
            "Agreed. Keep PostgreSQL as the database. This is our final decision."
        )
        self.assertEqual(len(result.decisions), 1)
        self.assertEqual(result.decisions[0].evidence_utterance_ids, ("utt_000001",))

    def test_task_missing_owner_deadline(self):
        result = self.run_case(
            "Benchmarking both models is confirmed work. No owner or deadline has been assigned."
        )
        self.assertEqual(len(result.action_items), 1)
        self.assertIsNone(result.action_items[0].owner)
        self.assertIsNone(result.action_items[0].deadline_text)
