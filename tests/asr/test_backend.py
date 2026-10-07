import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.asr import ASRConfig, WhisperAPIBackend, transcribe_audio
from meeting_assistant.asr.api_backend import parse_response
from meeting_assistant.asr.audio import ensure_unchanged, inspect_audio, iter_audio_chunks
from meeting_assistant.asr.exceptions import (
    ASRAuthenticationError,
    ASRConfigurationError,
    ASRTimeout,
    ASRTranscriptionError,
    InvalidASRAudio,
    InvalidTranscript,
)
from tests.audio.helpers import write_wav

from .helpers import FakeBackend, response, transcript


class AudioBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = ASRConfig(api_key="test-token")

    def test_valid_header_and_silence(self):
        for silent in (False, True):
            path = write_wav(self.root / "recording.wav", silent=silent)
            self.assertEqual(inspect_audio(path, self.config).duration_seconds, 0.5)

    def test_invalid_paths_and_bytes(self):
        bad = self.root / "bad.wav"
        for data in (b"", b"not WAV"):
            bad.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(InvalidASRAudio):
                inspect_audio(bad, self.config)
        for path in (self.root / "missing", self.root):
            with self.subTest(path=path), self.assertRaises(InvalidASRAudio):
                inspect_audio(path, self.config)

    def test_noncanonical_audio_and_zero_frames(self):
        for kwargs in ({"rate": 44100}, {"channels": 2}, {"duration": 0}):
            path = write_wav(self.root / "bad.wav", **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(InvalidASRAudio):
                inspect_audio(path, self.config)

    def test_truncated_frames(self):
        path = write_wav(self.root / "truncated.wav")
        path.write_bytes(path.read_bytes()[:-200])
        with self.assertRaises(InvalidASRAudio):
            inspect_audio(path, self.config)

    def test_input_size_and_access_errors(self):
        path = write_wav(self.root / "valid.wav")
        with self.assertRaises(InvalidASRAudio):
            inspect_audio(path, replace(self.config, max_input_size_bytes=100))
        with (
            patch("pathlib.Path.lstat", side_effect=PermissionError),
            self.assertRaises(InvalidASRAudio),
        ):
            inspect_audio(path, self.config)

    def test_chunks_are_exact_pcm_and_source_untouched(self):
        import wave

        path = write_wav(self.root / "spaces 世界.wav", duration=1.1)
        original = path.read_bytes()
        audio = inspect_audio(path, self.config)
        config = replace(self.config, chunk_duration_seconds=0.5)
        collected = bytearray()
        offsets = []
        for chunk, offset, duration in iter_audio_chunks(audio, config, self.root):
            offsets.append(offset)
            self.assertLessEqual(chunk.stat().st_size, config.max_upload_size_bytes)
            with wave.open(str(chunk), "rb") as wav:
                self.assertEqual(wav.getframerate(), 16000)
                collected.extend(wav.readframes(wav.getnframes()))
        with wave.open(str(path), "rb") as wav:
            self.assertEqual(bytes(collected), wav.readframes(wav.getnframes()))
        self.assertEqual(offsets, [0.0, 0.5, 1.0])
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / "chunk.wav").exists())

    def test_upload_size_also_limits_chunk_frames(self):
        path = write_wav(self.root / "audio.wav")
        config = replace(self.config, max_upload_size_bytes=4096)
        chunks = 0
        for chunk, _, _ in iter_audio_chunks(inspect_audio(path, config), config, self.root):
            self.assertLessEqual(chunk.stat().st_size, 4096)
            chunks += 1
        self.assertGreater(chunks, 1)

    def test_detect_source_changed(self):
        path = write_wav(self.root / "audio.wav")
        audio = inspect_audio(path, self.config)
        path.write_bytes(b"changed")
        with self.assertRaises(InvalidASRAudio):
            ensure_unchanged(audio)


class ResponseParsingTests(unittest.TestCase):
    def parse(self, payload):
        return parse_response(payload, offset=0, chunk_duration=0.5, segment_offset=0)

    def test_raw_text_metadata_and_offsets(self):
        text, language, confidence, segments = parse_response(
            response(), offset=100, chunk_duration=0.5, segment_offset=7
        )
        self.assertEqual(text, " Hello, world.")
        self.assertEqual(language, "English")
        self.assertIsNone(confidence)
        self.assertEqual(segments[0].id, "seg_000008")
        self.assertEqual(segments[0].words[1].start, 100.25)
        self.assertIsNone(segments[0].words[0].probability)
        self.assertEqual(segments[0].avg_logprob, -0.15)

    def test_empty_transcript(self):
        self.assertEqual(self.parse({"text": "", "segments": [], "words": []})[3], ())

    def test_malformed_output_and_missing_timestamps(self):
        variants = []
        for key in ("text", "segments", "words"):
            payload = response()
            payload.pop(key)
            variants.append(payload)
        for key, value in (("text", 123), ("segments", {}), ("words", {}), ("duration", 99)):
            payload = response()
            payload[key] = value
            variants.append(payload)
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(InvalidTranscript):
                self.parse(payload)

    def test_invalid_word_times_and_confidences(self):
        for key, value in (("start", -1), ("end", 9), ("start", True), ("probability", 2)):
            payload = response()
            payload["words"][0][key] = value
            with self.subTest(key=key), self.assertRaises(InvalidTranscript):
                self.parse(payload)

    def test_words_order_and_bounds(self):
        payload = response()
        payload["words"].reverse()
        with self.assertRaises(InvalidTranscript):
            self.parse(payload)

    def test_language_probability_only_when_provided(self):
        payload = response()
        payload["language_probability"] = 0.9
        self.assertEqual(self.parse(payload)[2], 0.9)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = write_wav(self.root / "audio.wav")
        self.config = ASRConfig(api_key="test-token")

    def test_missing_key(self):
        with self.assertRaises(ASRAuthenticationError):
            WhisperAPIBackend(ASRConfig())

    @patch("meeting_assistant.asr.api_backend.request_transcription")
    def test_concurrent_calls_have_isolated_workspaces(self, request):
        from concurrent.futures import ThreadPoolExecutor

        seen = []

        def perform(chunk, *_):
            seen.append(chunk.parent)
            return response(0.1)

        request.side_effect = perform
        backend = WhisperAPIBackend(replace(self.config, chunk_duration_seconds=0.1))
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(backend.transcribe, [self.path, self.path]))
        self.assertEqual(len(set(seen)), 2)
        self.assertNotEqual(results[0].transcript_id, results[1].transcript_id)
        self.assertTrue(all(not directory.exists() for directory in seen))
        self.assertEqual(results[0].segments, results[1].segments)

    @patch("meeting_assistant.asr.api_backend.request_transcription")
    def test_backend_reuse_and_raw_schema(self, request):
        request.return_value = response()
        backend = WhisperAPIBackend(self.config)
        first = transcribe_audio(self.path, backend=backend)
        second = transcribe_audio(self.path, backend=backend)
        self.assertEqual(first.text, second.text)
        self.assertNotEqual(first.transcript_id, second.transcript_id)
        self.assertEqual(first.segments[0].id, "seg_000001")
        self.assertEqual(first.model_info.device, "remote")
        self.assertIsNone(first.model_info.compute_type)
        self.assertEqual(request.call_count, 2)

    @patch("meeting_assistant.asr.api_backend.request_transcription")
    def test_multiple_chunks_offsets_counts_and_no_leftover(self, request):
        self.path = write_wav(self.path, duration=1.1)
        config = replace(self.config, chunk_duration_seconds=0.5)
        seen = []

        def perform(chunk, config, options, timeout):
            import wave

            seen.append(chunk)
            with wave.open(str(chunk), "rb") as audio:
                return response(audio.getnframes() / audio.getframerate())

        request.side_effect = perform
        result = WhisperAPIBackend(config).transcribe(self.path)
        self.assertEqual(result.processing_info.chunk_count, 3)
        self.assertEqual([segment.start for segment in result.segments], [0, 0.5, 1.0])
        self.assertEqual(result.word_count, 6)
        self.assertEqual(result.text, "\n".join([" Hello, world."] * 3))
        self.assertTrue(all(not chunk.exists() for chunk in seen))

    @patch("meeting_assistant.asr.api_backend.request_transcription")
    def test_failure_cleanup_no_retry_source_preserved(self, request):
        config = replace(self.config, chunk_duration_seconds=0.1)
        original = hashlib.sha256(self.path.read_bytes()).hexdigest()
        seen = []

        def fail(chunk, *_):
            seen.append(chunk)
            raise ASRTranscriptionError("provider failure")

        request.side_effect = fail
        with self.assertRaises(ASRTranscriptionError):
            WhisperAPIBackend(config).transcribe(self.path)
        self.assertEqual(request.call_count, 1)
        self.assertFalse(seen[0].exists())
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), original)

    @patch("meeting_assistant.asr.api_backend.request_transcription")
    def test_invalid_input_never_calls_provider(self, request):
        with self.assertRaises(InvalidASRAudio):
            WhisperAPIBackend(self.config).transcribe(self.root / "missing")
        request.assert_not_called()

    @patch("meeting_assistant.asr.api_backend.request_transcription")
    def test_invalid_output_is_failure(self, request):
        request.return_value = {"text": "words without timestamps"}
        with self.assertRaises(InvalidTranscript):
            WhisperAPIBackend(self.config).transcribe(self.path)

    def test_backend_injection_and_invalid_contract(self):
        self.assertEqual(
            transcribe_audio("path-unused-by-fake", backend=FakeBackend()), transcript()
        )
        with self.assertRaises(ASRConfigurationError):
            transcribe_audio(self.path, backend=FakeBackend(), config=self.config)
        fake = FakeBackend()
        fake.transcribe = lambda *_: {}
        with self.assertRaises(InvalidTranscript):
            transcribe_audio(self.path, backend=fake)

    @patch("meeting_assistant.asr.api_backend.request_transcription")
    def test_total_timeout_does_not_send_request(self, request):
        with (
            patch("meeting_assistant.asr.api_backend.time.perf_counter", side_effect=[0, 8000]),
            self.assertRaises(ASRTimeout),
        ):
            WhisperAPIBackend(self.config).transcribe(self.path)
        request.assert_not_called()
