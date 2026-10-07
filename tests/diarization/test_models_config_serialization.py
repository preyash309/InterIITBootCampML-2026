import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.diarization.config import (
    DiarizationConfig,
    DiarizationOptions,
    ReconciliationConfig,
)
from meeting_assistant.diarization.exceptions import (
    DiarizationConfigurationError,
    InvalidDiarizationResult,
    SpeakerTranscriptWriteError,
)
from meeting_assistant.diarization.models import SpeakerTurn
from meeting_assistant.diarization.reconciliation import reconcile_transcript
from meeting_assistant.diarization.serialization import (
    diarization_from_json,
    diarization_to_json,
    render_speaker_transcript_text,
    save_speaker_transcript,
    speaker_transcript_from_json,
    speaker_transcript_to_json,
)

from .helpers import diar, raw, word


class ConfigurationTests(unittest.TestCase):
    def test_defaults(self):
        config = DiarizationConfig.from_env(environ={})
        self.assertEqual(config.device, "cuda")
        self.assertTrue(config.offline_only)
        self.assertEqual(config.options.pipeline_arguments(), {})

    def test_environment_overrides(self):
        config = DiarizationConfig.from_env(
            environ={
                "DIARIZATION_DEVICE": "cpu",
                "DIARIZATION_NUM_SPEAKERS": "2",
                "DIARIZATION_OFFLINE_ONLY": "false",
                "DIARIZATION_MODEL_CACHE": "some cache",
                "HF_TOKEN": "private-token",
            }
        )
        self.assertEqual(config.options.pipeline_arguments(), {"num_speakers": 2})
        self.assertEqual(config.model_cache, Path("some cache"))
        self.assertNotIn("private-token", repr(config))

    def test_count_bounds(self):
        self.assertEqual(
            DiarizationOptions(min_speakers=2, max_speakers=4).pipeline_arguments(),
            {"min_speakers": 2, "max_speakers": 4},
        )

    def test_invalid_counts(self):
        for options in (
            {"num_speakers": 0},
            {"num_speakers": True},
            {"num_speakers": 2.5},
            {"num_speakers": 2, "max_speakers": 3},
            {"min_speakers": 3, "max_speakers": 2},
        ):
            with self.subTest(options=options), self.assertRaises(DiarizationConfigurationError):
                DiarizationOptions(**options)

    def test_invalid_configuration(self):
        for values in (
            {"DIARIZATION_DEVICE": "magic"},
            {"DIARIZATION_OFFLINE_ONLY": "yes"},
            {"DIARIZATION_NUM_SPEAKERS": "no"},
            {"DIARIZATION_MODEL_REVISION": "main"},
            {"DIARIZATION_MODEL": "cloud"},
        ):
            with self.subTest(values=values), self.assertRaises(DiarizationConfigurationError):
                DiarizationConfig.from_env(environ=values)

    def test_invalid_tolerances(self):
        for value in (-1, float("nan"), True, 1.1):
            with self.subTest(value=value), self.assertRaises(DiarizationConfigurationError):
                ReconciliationConfig(alignment_tolerance_seconds=value)

    def test_env_file_precedence_without_global_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("DIARIZATION_DEVICE=cpu\nDIARIZATION_NUM_SPEAKERS=2\n")
            with patch.dict("os.environ", {"DIARIZATION_DEVICE": "cuda"}):
                config = DiarizationConfig.from_env(env_file=path)
            self.assertEqual(config.device, "cuda")
            self.assertEqual(config.options.num_speakers, 2)


class SchemaTests(unittest.TestCase):
    def test_negative_and_nonfinite_turns(self):
        for start, end in ((-1, 1), (2, 1), (1, 1), (0, float("inf")), (True, 2)):
            with self.subTest(start=start, end=end), self.assertRaises(InvalidDiarizationResult):
                SpeakerTurn("turn", "speaker", start, end)

    def test_exclusive_overlap_rejected(self):
        with self.assertRaises(InvalidDiarizationResult):
            diar((("a", 0, 5), ("b", 4, 10)))

    def test_unsorted_turns_rejected(self):
        with self.assertRaises(InvalidDiarizationResult):
            diar((("a", 2, 4), ("b", 1, 2)))

    def test_out_of_bounds_turn_rejected(self):
        with self.assertRaises(InvalidDiarizationResult):
            diar((("a", 0, 20),))

    def test_model_results_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            diar().duration_seconds = 20

    def test_bad_provenance_rejected(self):
        with self.assertRaises(InvalidDiarizationResult):
            replace(diar(), audio_sha256="missing")

    def test_mutable_sequences_rejected(self):
        with self.assertRaises(InvalidDiarizationResult):
            replace(diar(), speakers=["SPEAKER_00"])


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.diarization = diar()
        self.result = reconcile_transcript(raw((word("café", 1.42, 1.81),)), self.diarization)

    def test_diarization_round_trip(self):
        self.assertEqual(
            diarization_from_json(diarization_to_json(self.diarization)), self.diarization
        )

    def test_speaker_round_trip_utf8_deterministic(self):
        content = speaker_transcript_to_json(self.result)
        self.assertIn("café", content)
        self.assertEqual(content, speaker_transcript_to_json(self.result))
        self.assertEqual(speaker_transcript_from_json(content), self.result)

    def test_text_render(self):
        self.assertEqual(
            render_speaker_transcript_text(self.result),
            "[00:00:01.420 --> 00:00:01.810] SPEAKER_00:\ncafé\n\n",
        )

    def test_unknown_text_render(self):
        result = reconcile_transcript(raw((word(start=5, end=6),)), diar((("a", 0, 2),)))
        self.assertIn("UNKNOWN_SPEAKER", render_speaker_transcript_text(result))

    def test_malformed_json(self):
        for loader in (speaker_transcript_from_json, diarization_from_json):
            for content in ("null", "[]", "{}", "nonsense"):
                with (
                    self.subTest(loader=loader, content=content),
                    self.assertRaises(InvalidDiarizationResult),
                ):
                    loader(content)

    def test_version_rejected(self):
        data = json.loads(speaker_transcript_to_json(self.result))
        data["schema_version"] = "99"
        with self.assertRaises(InvalidDiarizationResult):
            speaker_transcript_from_json(json.dumps(data))

    def test_duplicate_word_reference_rejected(self):
        data = json.loads(speaker_transcript_to_json(self.result))
        data["utterances"][0]["words"] *= 2
        with self.assertRaises(InvalidDiarizationResult):
            speaker_transcript_from_json(json.dumps(data))

    def test_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            raw_path = Path(folder) / "raw_transcript.json"
            raw_path.write_text("untouched")
            artifacts = save_speaker_transcript(self.result, self.diarization, folder)
            self.assertEqual(
                speaker_transcript_from_json(artifacts.json_path.read_text(encoding="utf-8")),
                self.result,
            )
            self.assertTrue(artifacts.diarization_path.exists())
            self.assertEqual(raw_path.read_text(), "untouched")
            self.assertFalse(list(Path(folder).glob(".pending-*")))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            artifacts = save_speaker_transcript(self.result, self.diarization, folder)
            before = artifacts.json_path.read_bytes()
            with self.assertRaises(SpeakerTranscriptWriteError):
                save_speaker_transcript(self.result, self.diarization, folder)
            self.assertEqual(artifacts.json_path.read_bytes(), before)

    def test_atomic_failure_cleans_partial_files(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch(
                "meeting_assistant.diarization.serialization.os.rename",
                side_effect=OSError("forced"),
            ):
                with self.assertRaises(SpeakerTranscriptWriteError):
                    save_speaker_transcript(self.result, self.diarization, folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_wrong_diarization_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(InvalidDiarizationResult):
                save_speaker_transcript(self.result, diar(), folder)
            self.assertEqual(list(Path(folder).iterdir()), [])
