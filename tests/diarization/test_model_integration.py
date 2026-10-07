"""Opt-in offline real model tests. No model acquisition occurs in this suite."""

import hashlib
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.asr.serialization import transcript_from_json, transcript_to_json
from meeting_assistant.diarization import (
    DiarizationConfig,
    PyannoteBackend,
    reconcile_transcript,
    save_speaker_transcript,
    speaker_transcript_from_json,
)
from tests.audio.helpers import write_wav


@unittest.skipUnless(os.environ.get("RUN_DIARIZATION_MODEL_TESTS") == "1", "Opt-in GPU/model tests")
class RealCommunityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio = Path(os.environ["DIARIZATION_TEST_AUDIO"])
        cls.overlap = Path(os.environ["DIARIZATION_TEST_OVERLAP_AUDIO"])
        cls.raw_path = Path(os.environ["DIARIZATION_TEST_RAW_JSON"])
        cls.network = patch.object(
            socket.socket, "connect", side_effect=AssertionError("Network disabled")
        )
        cls.network.start()
        cls.addClassCleanup(cls.network.stop)
        cls.backend = PyannoteBackend(DiarizationConfig.from_env())
        cls.load_seconds = cls.backend.load_model()

    def test_offline_multispeaker_and_raw_provenance(self):
        before = self.raw_path.read_bytes()
        raw = transcript_from_json(before.decode("utf-8"))
        result = self.backend.diarize(self.audio)
        self.assertEqual(result.detected_speaker_count, 2)
        derived = reconcile_transcript(raw, result)
        self.assertEqual(derived.reconciliation_info.total_words, raw.word_count)
        self.assertEqual(self.raw_path.read_bytes(), before)
        self.assertEqual(
            derived.source_raw_sha256, hashlib.sha256(transcript_to_json(raw).encode()).hexdigest()
        )
        with tempfile.TemporaryDirectory() as folder:
            artifacts = save_speaker_transcript(derived, result, folder)
            self.assertEqual(
                speaker_transcript_from_json(artifacts.json_path.read_text(encoding="utf-8")),
                derived,
            )

    def test_regular_overlap_exclusive_no_overlap(self):
        result = self.backend.diarize(self.overlap)
        self.assertEqual(result.detected_speaker_count, 2)
        from meeting_assistant.diarization.reconciliation import find_overlap_regions

        self.assertGreater(len(find_overlap_regions(result.regular_turns)), 0)
        for previous, current in zip(result.exclusive_turns, result.exclusive_turns[1:]):
            self.assertLessEqual(previous.end, current.start + 1e-9)

    def test_silence_empty_speaker_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_wav(Path(folder) / "silence.wav", silent=True, duration=1)
            result = self.backend.diarize(path)
        self.assertEqual(result.speakers, ())
        self.assertEqual(result.regular_turns, ())
        self.assertEqual(result.exclusive_turns, ())

    def test_reused_model_and_gpu_metadata(self):
        pipeline = self.backend._pipeline
        self.assertEqual(self.backend.load_model(), 0)
        self.assertIs(self.backend._pipeline, pipeline)
        self.assertEqual(self.backend.model_info.device, "cuda")
        self.assertGreater(self.load_seconds, 0)


@unittest.skipUnless(
    os.environ.get("RUN_DIARIZATION_CPU_TESTS") == "1", "Opt-in explicit CPU fallback"
)
class RealCPUFallbackTests(unittest.TestCase):
    def test_explicit_cpu_multispeaker(self):
        from dataclasses import replace

        backend = PyannoteBackend(replace(DiarizationConfig.from_env(), device="cpu"))
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network disabled")):
            result = backend.diarize(Path(os.environ["DIARIZATION_TEST_AUDIO"]))
        self.assertEqual(result.detected_speaker_count, 2)
        self.assertEqual(result.model_info.device, "cpu")
        self.assertIsNone(result.processing_info.peak_gpu_memory_bytes)
