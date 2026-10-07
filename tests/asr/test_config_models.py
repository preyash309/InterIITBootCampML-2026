import json
import math
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.asr import (
    ASRConfig,
    TranscriptionOptions,
    TranscriptSegment,
    TranscriptWord,
    render_transcript_text,
    save_transcript,
    transcript_from_json,
    transcript_to_json,
)
from meeting_assistant.asr.config import read_environment
from meeting_assistant.asr.evaluate import calculate_wer
from meeting_assistant.asr.exceptions import (
    ASRConfigurationError,
    InvalidTranscript,
    TranscriptWriteError,
)

from .helpers import transcript


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_alternate_provider(self):
        self.assertEqual(ASRConfig().model, "whisper-large-v3")
        self.assertEqual(ASRConfig().options.language, "en")
        self.assertEqual(ASRConfig(provider="openai").model, "whisper-1")

    def test_secret_excluded_from_repr(self):
        self.assertNotIn("secret-token", repr(ASRConfig(api_key="secret-token")))

    def test_environment_overrides(self):
        config = ASRConfig.from_env(
            environ={
                "ASR_PROVIDER": "openai",
                "OPENAI_API_KEY": "token",
                "ASR_LANGUAGE": "auto",
                "ASR_CHUNK_SECONDS": "123",
                "ASR_REQUEST_TIMEOUT": "20",
            }
        )
        self.assertEqual(config.model, "whisper-1")
        self.assertEqual(config.api_key, "token")
        self.assertIsNone(config.options.language)
        self.assertEqual(config.chunk_duration_seconds, 123)

    def test_dotenv_and_process_precedence_without_global_mutation(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            env_file = Path(temporary) / ".env"
            env_file.write_text('# comment\nGROQ_API_KEY="file-token"\nASR_LANGUAGE=en\n')
            self.assertEqual(ASRConfig.from_env(env_file=env_file).api_key, "file-token")
            self.assertNotIn("GROQ_API_KEY", os.environ)
            os.environ["GROQ_API_KEY"] = "process-token"
            self.assertEqual(ASRConfig.from_env(env_file=env_file).api_key, "process-token")

    def test_dotenv_missing_is_optional(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read_environment(Path("nonexistent-env-file")), {})

    def test_invalid_dotenv(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            for content in ("not-an-assignment", "BAD KEY=secret", "a" * 65_537):
                env_file.write_text(content)
                with self.subTest(content=content[:20]), self.assertRaises(ASRConfigurationError):
                    read_environment(env_file)

    def test_invalid_numeric_environment(self):
        with self.assertRaises(ASRConfigurationError):
            ASRConfig.from_env(environ={"ASR_REQUEST_TIMEOUT": "wrong"})

    def test_invalid_provider_model_limits_and_keys(self):
        for values in (
            {"provider": "local"},
            {"provider": []},
            {"model": {}},
            {"model": "whisper-1"},
            {"request_timeout_seconds": float("inf")},
            {"total_timeout_seconds": -1},
            {"chunk_duration_seconds": 0},
            {"max_upload_size_bytes": 25_000_000},
            {"max_upload_size_bytes": True},
            {"max_input_size_bytes": 0},
            {"api_key": "bad\r\nHeader: injection"},
            {"options": {}},
        ):
            with self.subTest(values=values), self.assertRaises(ASRConfigurationError):
                ASRConfig(**values)

    def test_invalid_options(self):
        for values in (
            {"language": "English"},
            {"language": "en\r\nInjected"},
            {"temperature": float("nan")},
            {"temperature": True},
            {"temperature": 1.1},
        ):
            with self.subTest(values=values), self.assertRaises(ASRConfigurationError):
                TranscriptionOptions(**values)


class SchemaTests(unittest.TestCase):
    def test_invalid_model_metadata(self):
        for values in (
            {"model": 123},
            {"provider": ""},
            {"temperature": float("nan")},
            {"word_timestamps": "true"},
            {"vad_enabled": 1},
        ):
            with self.subTest(values=values), self.assertRaises(InvalidTranscript):
                replace(transcript().model_info, **values)

    def test_frozen_and_counts(self):
        result = transcript()
        with self.assertRaises(FrozenInstanceError):
            result.text = "corrected"
        self.assertIsInstance(result.segments, tuple)
        self.assertIsInstance(result.segments[0].words, tuple)
        self.assertEqual(result.word_count, 2)
        self.assertEqual(result.real_time_factor, 0.2)

    def test_invalid_word_values(self):
        for values in (
            ("word", -0.1, 0.1, None),
            ("word", 0.2, 0.1, None),
            ("word", math.nan, 0.1, None),
            ("word", 0.0, 0.1, 1.1),
            ("", 0.0, 0.1, None),
        ):
            with self.subTest(values=values), self.assertRaises(InvalidTranscript):
                TranscriptWord(*values)

    def test_segment_word_order_bounds_and_mutability(self):
        for words in (
            [TranscriptWord("one", 0.0, 0.1)],
            (TranscriptWord("one", 0.0, 1.01),),
            (TranscriptWord("one", 0.2, 0.3), TranscriptWord("two", 0.1, 0.2)),
        ):
            with self.subTest(words=words), self.assertRaises(InvalidTranscript):
                TranscriptSegment("seg_000001", 0.0, 0.5, "one", words=words)

    def test_segment_small_native_timestamp_discrepancy_allowed(self):
        segment = TranscriptSegment(
            "seg_000001", 0.0, 0.5, "one", (TranscriptWord("one", 0.0, 0.98),)
        )
        self.assertEqual(segment.words[0].end, 0.98)

    def test_segment_ids_and_audio_bounds(self):
        for segment in (
            replace(transcript().segments[0], id="wrong"),
            replace(transcript().segments[0], end=2.0),
        ):
            with self.subTest(segment=segment), self.assertRaises(InvalidTranscript):
                replace(transcript(), segments=(segment,))

    def test_reject_missing_segments_and_invalid_processing(self):
        with self.assertRaises(InvalidTranscript):
            replace(transcript(), segments=())
        with self.assertRaises(InvalidTranscript):
            replace(transcript().processing_info, total_seconds=0.01)


class SerializationTests(unittest.TestCase):
    def test_json_round_trip_and_stability(self):
        result = transcript()
        content = transcript_to_json(result)
        self.assertEqual(content, transcript_to_json(result))
        self.assertEqual(transcript_from_json(content), result)
        self.assertNotIn("api_key", content)
        self.assertIsNone(json.loads(content)["segments"][0]["words"][0]["probability"])

    def test_text_preserves_raw_spacing_and_timestamps(self):
        self.assertEqual(
            render_transcript_text(transcript()),
            "[00:00:00.000 --> 00:00:00.500]  Hello, world.\n",
        )

    def test_utf8_and_atomic_pair_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = replace(transcript(), text=" Héllo, 世界.")
            artifacts = save_transcript(result, root)
            self.assertEqual(transcript_from_json(artifacts.json_path.read_text("utf-8")), result)
            self.assertTrue(artifacts.text_path.is_file())
            with self.assertRaises(TranscriptWriteError):
                save_transcript(result, root)
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_failed_pair_leaves_no_partial_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch(
                    "meeting_assistant.asr.serialization.render_transcript_text",
                    side_effect=OSError,
                ),
                self.assertRaises(TranscriptWriteError),
            ):
                save_transcript(transcript(), root)
            self.assertEqual(list(root.iterdir()), [])

    def test_publication_failure_cleans_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch("meeting_assistant.asr.serialization.os.rename", side_effect=OSError),
                self.assertRaises(TranscriptWriteError),
            ):
                save_transcript(transcript(), root)
            self.assertEqual(list(root.iterdir()), [])

    def test_path_traversal_id_rejected(self):
        with self.assertRaises(InvalidTranscript):
            save_transcript(replace(transcript(), transcript_id="../../outside"))

    def test_malformed_json_schema(self):
        for content in ("no JSON", "[]", '{"schema_version":"2.0"}', '{"schema_version":"1.0"}'):
            with self.subTest(content=content), self.assertRaises(InvalidTranscript):
                transcript_from_json(content)


class EvaluationTests(unittest.TestCase):
    def test_wer_correct_substitution_insertion_deletion(self):
        self.assertEqual(calculate_wer("Hello, world!", "hello WORLD").wer, 0.0)
        self.assertAlmostEqual(calculate_wer("a b c", "a d c").wer, 1 / 3)
        self.assertEqual(calculate_wer("a b", "a b c").wer, 0.5)
        self.assertEqual(calculate_wer("a b", "a").wer, 0.5)

    def test_wer_empty_reference_and_no_number_rewriting(self):
        self.assertEqual(calculate_wer("", "").wer, 0.0)
        self.assertIsNone(calculate_wer("", "hallucination").wer)
        self.assertGreater(calculate_wer("three", "3").wer, 0)
