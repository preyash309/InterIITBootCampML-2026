import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from meeting_assistant.diarization.reconciliation import reconcile_transcript
from meeting_assistant.speaker_reliability import compare_diarization, reliability_from_json
from meeting_assistant.speaker_reliability.evaluation import acoustic_metrics, evaluate_reference
from meeting_assistant.speaker_reliability.exceptions import InvalidReliability
from meeting_assistant.speaker_reliability.serialization import to_json
from tests.diarization.helpers import diar, raw, word
from tests.speaker_reliability.test_reliability import secondary


class ReferenceTests(unittest.TestCase):
    def test_cli_failure_is_concise_nonzero(self):
        import io
        from contextlib import redirect_stderr

        from meeting_assistant.speaker_reliability.evaluation import main

        output = io.StringIO()
        with redirect_stderr(output):
            code = main(
                [
                    "--audio",
                    "missing.wav",
                    "--primary",
                    "missing-primary.json",
                    "--speaker",
                    "missing-speaker.json",
                    "--output-dir",
                    "unused-output",
                ]
            )
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", output.getvalue())
        self.assertIn("Check saved inputs", output.getvalue())

    def setUp(self):
        self.primary = diar((("a", 0, 5), ("b", 5, 10)))
        self.speaker = reconcile_transcript(
            raw((word("Yes", 1, 1.5), word("save", 6, 6.5))), self.primary
        )
        self.secondary = secondary((("x", 0, 5), ("y", 5, 10)))
        self.result = compare_diarization(self.primary, self.speaker, self.secondary)
        self.reference = dict(
            audio_sha256="a" * 64,
            duration_seconds=10,
            reference_kind="authored_source_schedule",
            verified_acoustic_annotations=False,
            segments=[
                dict(speaker_id="truth_a", start=0, end=5),
                dict(speaker_id="truth_b", start=5, end=10),
            ],
        )

    def test_known_source_word_accuracy(self):
        r = evaluate_reference(self.primary, self.speaker, self.result, self.reference)
        self.assertEqual(r["word_accuracy"]["scored_words"], 2)
        self.assertEqual(r["word_accuracy"]["primary_accuracy"], 1)
        self.assertEqual(r["word_accuracy"]["secondary_accuracy"], 1)
        self.assertEqual(r["agreement_strata"]["high"]["words"], 2)
        self.assertEqual(r["agreement_strata"]["interjections"]["words"], 1)
        self.assertIsNone(r["acoustic_metrics"]["primary_der"])

    def test_no_fabricated_der_from_schedule(self):
        self.assertIsNone(
            acoustic_metrics(self.primary, self.secondary, self.reference)["secondary_der"]
        )

    def test_high_agreement_can_still_be_wrong(self):
        # Both models miss a brief true B turn inside predominantly A speech.
        reference = {
            **self.reference,
            "segments": [
                dict(speaker_id="truth_a", start=0, end=1),
                dict(speaker_id="truth_b", start=1, end=1.5),
                dict(speaker_id="truth_a", start=1.5, end=5),
                dict(speaker_id="truth_b", start=5, end=10),
            ],
        }
        r = evaluate_reference(self.primary, self.speaker, self.result, reference)
        self.assertEqual(r["agreement_strata"]["high"]["words"], 2)
        self.assertEqual(r["agreement_strata"]["high"]["primary_accuracy"], 0.5)

    def test_unverified_reference_hash_rejected(self):
        with self.assertRaises(InvalidReliability):
            evaluate_reference(
                self.primary,
                self.speaker,
                self.result,
                {**self.reference, "audio_sha256": "b" * 64},
            )

    def test_reference_duration_rejected(self):
        with self.assertRaises(InvalidReliability):
            evaluate_reference(
                self.primary, self.speaker, self.result, {**self.reference, "duration_seconds": 11}
            )

    def test_gap_overlap_excluded(self):
        reference = {
            **self.reference,
            "segments": [
                dict(speaker_id="a", start=0, end=2),
                dict(speaker_id="b", start=0, end=2),
            ],
        }
        r = evaluate_reference(self.primary, self.speaker, self.result, reference)
        self.assertEqual(r["word_accuracy"]["excluded_gap_or_overlap"], 2)
        self.assertIsNone(r["word_accuracy"]["primary_accuracy"])

    def test_authored_word_probes_without_asr(self):
        speaker = reconcile_transcript(raw(()), self.primary)
        result = compare_diarization(self.primary, speaker, self.secondary)
        reference = {
            **self.reference,
            "word_events": [dict(time_seconds=1, text="Yes", speaker_id="truth_a")],
        }
        r = evaluate_reference(self.primary, speaker, result, reference)
        self.assertEqual(r["probe_kind"], "authored_sapi_word_onset_probes")
        self.assertEqual(r["word_accuracy"]["scored_words"], 1)

    def test_json_roundtrip_immutable(self):
        result = reliability_from_json(to_json(self.result))
        self.assertEqual(result, self.result)
        self.assertIsInstance(result.words, tuple)
        self.assertIsInstance(
            result.provenance.configuration.python_path, __import__("pathlib").Path
        )

    def test_schema_version_rejected(self):
        data = json.loads(to_json(self.result))
        data["schema_version"] = "2.0"
        with self.assertRaises(InvalidReliability):
            reliability_from_json(json.dumps(data))

    def test_nested_invalid_json_rejected(self):
        data = json.loads(to_json(self.result))
        data["words"][0]["agreement_fraction"] = "certain"
        with self.assertRaises(InvalidReliability):
            reliability_from_json(json.dumps(data))

    def test_bad_optional_configuration_does_not_abort_workflow(self):
        from meeting_assistant.speaker_reliability.service import run_optional

        with patch("meeting_assistant.speaker_reliability.service.logger"):
            self.assertIsNone(
                run_optional(
                    None,
                    None,
                    None,
                    None,
                    environ={
                        "SPEAKER_RELIABILITY_ENABLED": "true",
                        "SECONDARY_DIARIZER_DEVICE": "bad",
                    },
                )
            )

    def test_secondary_inference_error_preserves_canonical_record(self):
        from meeting_assistant.speaker_reliability.service import run_optional

        with (
            patch(
                "meeting_assistant.speaker_reliability.service.assess_speaker_reliability",
                side_effect=RuntimeError,
            ),
            patch("meeting_assistant.speaker_reliability.service.logger"),
        ):
            self.assertIsNone(run_optional(None, None, None, None, environ={}, enabled=True))

    def test_unavailable_json_roundtrip(self):
        result = replace(
            self.result,
            availability="unavailable",
            secondary=None,
            comparison=None,
            words=(),
            utterances=(),
            warnings=("unavailable",),
        )
        self.assertEqual(reliability_from_json(to_json(result)), result)


if __name__ == "__main__":
    unittest.main()
