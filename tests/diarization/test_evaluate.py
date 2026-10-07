import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from meeting_assistant.diarization.evaluate import diagnostics, main
from meeting_assistant.diarization.exceptions import InvalidDiarizationResult
from meeting_assistant.diarization.reconciliation import reconcile_transcript
from meeting_assistant.diarization.serialization import diarization_to_json

from .helpers import diar, raw, word


class DiagnosticTests(unittest.TestCase):
    def test_count_and_coverage(self):
        turns = diar()
        result = diagnostics(
            turns, reconcile_transcript(raw((word(),)), turns), expected_speakers=2
        )
        self.assertFalse(result["speaker_count_matches"])
        self.assertEqual(result["assignment_coverage"], 1)
        self.assertNotIn("accuracy", result)
        self.assertNotIn("der", result)

    def test_unknown_expected_count(self):
        self.assertIsNone(diagnostics(diar())["speaker_count_matches"])

    def test_wrong_provenance(self):
        with self.assertRaises(InvalidDiarizationResult):
            diagnostics(diar(), reconcile_transcript(raw((word(),)), diar()))

    def test_invalid_count(self):
        with self.assertRaises(InvalidDiarizationResult):
            diagnostics(diar(), expected_speakers=0)

    def test_cli_no_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "diarization.json"
            path.write_text(diarization_to_json(diar()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--diarization", str(path)]), 0)
            self.assertIn("DER not scored", output.getvalue())
