"""Explicit live Groq tests. No network or model load in normal test discovery."""

import json
import os
import unittest
from pathlib import Path

from meeting_assistant.refinement import GroqRefinerBackend, refine_transcript
from meeting_assistant.refinement.evaluate import controlled_evidence


@unittest.skipUnless(
    os.environ.get("RUN_REFINER_API_TESTS") == "1",
    "Set RUN_REFINER_API_TESTS=1 for billed live Groq requests.",
)
class LiveRefinerTests(unittest.TestCase):
    def check_case(self, case_id, expected):
        case = next(
            c
            for c in json.loads(Path("data/refinement_benchmark.json").read_text())
            if c["id"] == case_id
        )
        source, g = controlled_evidence(case["text"], [case])
        result = refine_transcript(source, g, backend=GroqRefinerBackend())
        self.assertEqual(result.edit_log[0].action, expected)
        self.assertGreater(len(result.processing_info.calls), 0)
        self.assertIsNotNone(result.processing_info.usage("total_tokens"))
        if expected == "REPLACE":
            self.assertEqual(result.edit_log[0].validation_status, "applied")
        else:
            self.assertEqual(result.utterances[0].refined_text, case["text"])

    def test_keep(self):
        self.check_case("gpu", "KEEP")

    def test_replace(self):
        self.check_case("kubernetes", "REPLACE")

    def test_uncertain(self):
        self.check_case("missing", "UNCERTAIN")
