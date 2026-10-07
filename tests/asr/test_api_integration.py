"""Opt-in real, billed API tests. Standard tests never contact either provider."""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from meeting_assistant.asr import (
    ASRConfig,
    WhisperAPIBackend,
    save_transcript,
    transcript_from_json,
)
from tests.audio.helpers import write_wav


@unittest.skipUnless(
    os.environ.get("RUN_ASR_API_TESTS") == "1", "Explicit API test opt-in required."
)
class APIIntegrationTests(unittest.TestCase):
    def test_real_speech_with_timestamped_artifacts(self):
        sample = os.environ.get("ASR_TEST_AUDIO")
        self.assertTrue(sample, "Set ASR_TEST_AUDIO to a canonical English speech WAV.")
        result = WhisperAPIBackend(ASRConfig.from_env()).transcribe(Path(sample))
        self.assertTrue(result.text.strip())
        self.assertGreater(result.word_count, 0)
        self.assertTrue(result.model_info.word_timestamps)
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = save_transcript(result, temporary)
            saved = transcript_from_json(artifacts.json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, result)
            self.assertTrue(artifacts.text_path.stat().st_size)

    def test_real_chunked_speech(self):
        sample = os.environ.get("ASR_TEST_AUDIO")
        self.assertTrue(sample, "Set ASR_TEST_AUDIO to a canonical English speech WAV.")
        config = replace(ASRConfig.from_env(), chunk_duration_seconds=8)
        result = WhisperAPIBackend(config).transcribe(Path(sample))
        self.assertGreater(result.processing_info.chunk_count, 1)
        self.assertTrue(result.text.strip())
        self.assertGreater(max(segment.start for segment in result.segments), 8)

    def test_real_silence_raw_behavior(self):
        with tempfile.TemporaryDirectory() as temporary:
            audio = write_wav(Path(temporary) / "silence.wav", duration=1.0, silent=True)
            result = WhisperAPIBackend(ASRConfig.from_env()).transcribe(audio)
            self.assertEqual(result.duration_seconds, 1.0)
            # If Whisper hallucinates speech, preserve that raw baseline; never invent silence.
            self.assertIsInstance(result.text, str)
