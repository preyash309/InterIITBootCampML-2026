import contextlib
import http.client
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from meeting_assistant.__main__ import main as integrated_main
from meeting_assistant.asr import ASRConfig, TranscriptionOptions
from meeting_assistant.asr.__main__ import main as asr_main
from meeting_assistant.asr.evaluate import main as evaluate_main
from meeting_assistant.asr.exceptions import (
    ASRAuthenticationError,
    ASRConnectionError,
    ASRRateLimitError,
    ASRTimeout,
    ASRTranscriptionError,
    InvalidTranscript,
)
from meeting_assistant.asr.transport import _remaining, request_transcription
from tests.audio.helpers import write_wav

from .helpers import response, transcript


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = write_wav(Path(self.temporary.name) / "untrusted ' & 世界.wav")
        self.config = ASRConfig(api_key="never-log-this-key")
        self.connection = MagicMock()
        self.response = MagicMock(status=200)
        self.response.read1.side_effect = [json.dumps(response()).encode(), b""]
        self.connection.getresponse.return_value = self.response
        self.factory = patch(
            "meeting_assistant.asr.transport.http.client.HTTPSConnection",
            return_value=self.connection,
        )
        self.mock_factory = self.factory.start()
        self.addCleanup(self.factory.stop)

    def perform(self, **kwargs):
        return request_transcription(
            self.path,
            kwargs.get("config", self.config),
            kwargs.get("options", TranscriptionOptions()),
            30,
        )

    def test_streaming_multipart_headers_body_and_timestamps(self):
        self.assertEqual(self.perform(), response())
        self.mock_factory.assert_called_once_with("api.groq.com", timeout=30)
        self.connection.putrequest.assert_called_once_with(
            "POST", "/openai/v1/audio/transcriptions"
        )
        headers = dict(call.args for call in self.connection.putheader.call_args_list)
        sent = b"".join(call.args[0] for call in self.connection.send.call_args_list)
        self.assertEqual(len(sent), int(headers["Content-Length"]))
        self.assertIn(self.path.read_bytes(), sent)
        self.assertIn(b'name="timestamp_granularities[]"\r\n\r\nsegment', sent)
        self.assertIn(b'name="timestamp_granularities[]"\r\n\r\nword', sent)
        self.assertIn(b'filename="canonical.wav"', sent)
        self.assertNotIn(self.path.name.encode(), sent)
        self.assertNotIn(b"prompt", sent)
        self.assertNotIn(b"translate", sent)
        self.connection.close.assert_called_once()

    def test_openai_endpoint_and_auto_language(self):
        config = ASRConfig(provider="openai", api_key="token")
        self.perform(config=config, options=TranscriptionOptions(language=None))
        self.mock_factory.assert_called_once_with("api.openai.com", timeout=30)
        self.connection.putrequest.assert_called_once_with("POST", "/v1/audio/transcriptions")
        sent = b"".join(call.args[0] for call in self.connection.send.call_args_list)
        self.assertNotIn(b'name="language"', sent)
        self.assertIn(b"whisper-1", sent)

    def test_status_errors_are_sanitized_and_not_retried(self):
        for status, expected in (
            (401, ASRAuthenticationError),
            (403, ASRAuthenticationError),
            (429, ASRRateLimitError),
            (413, ASRTranscriptionError),
            (400, ASRTranscriptionError),
            (503, ASRTranscriptionError),
            (302, ASRTranscriptionError),
        ):
            self.response.status = status
            with self.subTest(status=status), self.assertRaises(expected) as caught:
                self.perform()
            self.assertEqual(caught.exception.status_code, status)
            self.assertNotIn(self.config.api_key, str(caught.exception))
        self.response.read1.assert_not_called()

    def test_request_timeout_preserves_cause_and_closes(self):
        self.connection.getresponse.side_effect = TimeoutError("private-host-information")
        with self.assertRaises(ASRTimeout) as caught:
            self.perform()
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
        self.assertNotIn("private-host", str(caught.exception))
        self.connection.close.assert_called_once()

    def test_connection_failures(self):
        for exception in (OSError("private"), http.client.RemoteDisconnected("private")):
            self.connection.getresponse.side_effect = exception
            with self.subTest(exception=exception), self.assertRaises(ASRConnectionError):
                self.perform()

    def test_invalid_json_and_nonobject(self):
        for content in (b"not JSON", b"[]", b"\xff"):
            self.response.read1.side_effect = [content, b""]
            with self.subTest(content=content), self.assertRaises(InvalidTranscript):
                self.perform()

    def test_response_size_limit(self):
        with patch("meeting_assistant.asr.transport.MAX_RESPONSE_BYTES", 10):
            self.response.read1.side_effect = [b"x" * 11]
            with self.assertRaises(InvalidTranscript):
                self.perform()

    def test_oversize_chunk_not_uploaded(self):
        config = replace(self.config, max_upload_size_bytes=4096)
        with self.assertRaises(ASRTranscriptionError):
            self.perform(config=config)
        self.connection.putrequest.assert_not_called()

    def test_deadline_expiry(self):
        with patch("meeting_assistant.asr.transport.time.perf_counter", return_value=10):
            with self.assertRaises(ASRTimeout):
                _remaining(self.connection, 9)

    def test_logs_exclude_key_and_transcript(self):
        with self.assertLogs("meeting_assistant.asr.transport", level="INFO") as logs:
            self.perform()
        content = "\n".join(logs.output)
        self.assertNotIn(self.config.api_key, content)
        self.assertNotIn("Hello", content)


class CLITests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = self.root / ".env"
        self.env.write_text("ASR_PROVIDER=groq\nGROQ_API_KEY=test-token\n")

    @patch("meeting_assistant.asr.__main__.transcribe_audio", return_value=transcript())
    def test_canonical_cli_publishes_files_and_metrics(self, transcribe):
        out = self.root / "out"
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            status = asr_main(
                ["canonical.wav", "--env-file", str(self.env), "--output-dir", str(out)]
            )
        self.assertEqual(status, 0)
        self.assertEqual(len(list(out.glob("*/raw_transcript.json"))), 1)
        self.assertIn("RTF", stream.getvalue())
        transcribe.assert_called_once()

    @patch("meeting_assistant.asr.__main__.transcribe_audio", side_effect=ASRTimeout("timeout"))
    def test_failed_cli_does_not_publish(self, transcribe):
        out = self.root / "out"
        with contextlib.redirect_stderr(io.StringIO()):
            status = asr_main(
                ["canonical.wav", "--env-file", str(self.env), "--output-dir", str(out)]
            )
        self.assertEqual(status, 1)
        self.assertFalse(out.exists())

    @patch("meeting_assistant.asr.__main__.ingest_audio")
    def test_missing_key_fails_before_ingestion(self, ingest):
        with (
            patch("meeting_assistant.asr.config.read_environment", return_value={}),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            status = asr_main(["meeting.mp4", "--ingest"])
        self.assertEqual(status, 1)
        ingest.assert_not_called()

    @patch("meeting_assistant.asr.__main__.ingest_audio")
    @patch("meeting_assistant.asr.__main__.transcribe_audio", return_value=transcript())
    def test_integrated_ingestion_contract_and_job_transcript_path(self, transcribe, ingest):
        canonical = self.root / "jobs" / "job" / "audio" / "canonical.wav"
        ingest.return_value.canonical_audio_path = canonical
        with contextlib.redirect_stdout(io.StringIO()):
            status = integrated_main(["transcribe", "meeting.mp4", "--env-file", str(self.env)])
        self.assertEqual(status, 0)
        self.assertEqual(transcribe.call_args.args[0], canonical)
        self.assertTrue(
            list((canonical.parent.parent / "transcripts").glob("*/raw_transcript.json"))
        )

    def test_root_help_and_unknown_command(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(integrated_main([]), 0)
            self.assertEqual(integrated_main(["unknown"]), 2)

    def test_evaluation_saved_transcript_without_network(self):
        from meeting_assistant.asr import transcript_to_json

        saved = self.root / "raw.json"
        saved.write_text(transcript_to_json(transcript()), encoding="utf-8")
        ref = self.root / "reference.txt"
        ref.write_text("Hello world", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(
                evaluate_main(["--transcript", str(saved), "--reference", str(ref)]), 0
            )
        self.assertEqual(json.loads(stream.getvalue())["wer"], 0.0)

    def test_evaluation_missing_reference_reports_runtime_only(self):
        from meeting_assistant.asr import transcript_to_json

        saved = self.root / "raw.json"
        saved.write_text(transcript_to_json(transcript()), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(evaluate_main(["--transcript", str(saved)]), 0)
        self.assertIn("No reference supplied", stream.getvalue())
