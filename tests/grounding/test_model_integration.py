"""Opt-in local MiniLM tests: no downloads and no API calls."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.diarization.serialization import (
    speaker_transcript_from_json,
    speaker_transcript_to_json,
)
from meeting_assistant.grounding.config import GroundingConfig
from meeting_assistant.grounding.embeddings import MiniLMBackend
from meeting_assistant.grounding.evaluate import evaluate_cases
from meeting_assistant.grounding.glossary import load_glossary
from meeting_assistant.grounding.retrieval import GroundingRetriever
from meeting_assistant.grounding.serialization import grounding_from_json, grounding_to_json
from meeting_assistant.grounding.service import ground_transcript, validate_grounding_source


@unittest.skipUnless(
    os.environ.get("RUN_GROUNDING_MODEL_TESTS") == "1",
    "Requires explicitly prepared local MiniLM model",
)
class OfflineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocker = patch(
            "socket.socket.connect", side_effect=AssertionError("Phase IV must work offline")
        )
        cls.blocker.start()
        cls.addClassCleanup(cls.blocker.stop)
        cls.backend = MiniLMBackend(GroundingConfig.from_env())
        cls.engine = GroundingRetriever(load_glossary(), cls.backend.config, embeddings=cls.backend)

    def test_local_model_reuse_shape(self):
        self.backend.load_model()
        self.assertEqual(self.backend.load_model(), 0)
        self.assertEqual(self.backend.encode(["Local glossary retrieval."]).shape, (1, 384))

    def test_controlled_cases_and_negative_controls(self):
        data = json.loads(Path("data/grounding_benchmark.json").read_text(encoding="utf-8"))
        report = evaluate_cases(data, self.engine)
        self.assertGreaterEqual(report["recall_at_3"], 0.75)
        # Important safety gates, not a tuned perfect-accuracy assertion.
        controls = {case["span"]: case for case in report["negative_details"]}
        self.assertFalse(controls["react"]["forbidden_returned"])
        self.assertFalse(controls["15 percent"]["candidates"])

    def test_real_phase3_artifact_preserved(self):
        path = os.environ.get("GROUNDING_SPEAKER_FIXTURE")
        if not path:
            self.skipTest("Set GROUNDING_SPEAKER_FIXTURE to retained real Phase III JSON")
        payload = Path(path).read_bytes()
        source = speaker_transcript_from_json(payload.decode("utf-8"))
        before = speaker_transcript_to_json(source)
        result = ground_transcript(source, retriever=self.engine)
        validate_grounding_source(result, source)
        self.assertEqual(before, speaker_transcript_to_json(source))
        self.assertEqual(payload, Path(path).read_bytes())
        self.assertEqual(result, grounding_from_json(grounding_to_json(result)))
