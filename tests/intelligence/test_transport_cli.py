import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from meeting_assistant.__main__ import main as root_main
from meeting_assistant.intelligence.__main__ import evidence_main, main
from meeting_assistant.intelligence.config import IntelligenceConfig
from meeting_assistant.intelligence.evaluate import score_cases
from meeting_assistant.intelligence.exceptions import (
    IntelligenceAuthenticationError,
    IntelligenceProviderError,
    IntelligenceRateLimitError,
    IntelligenceSchemaError,
    IntelligenceTimeoutError,
)
from meeting_assistant.intelligence.groq_backend import GroqMeetingIntelligenceBackend, _exchange
from meeting_assistant.intelligence.models import IntelligenceRequest
from meeting_assistant.intelligence.serialization import save_meeting_record
from meeting_assistant.intelligence.service import extract_meeting_record
from meeting_assistant.refinement.serialization import refined_to_json

from .helpers import FakeBackend, completion, evidence, payload


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.config = IntelligenceConfig(api_key="test-secret", max_backoff_seconds=0)
        self.backend = GroqMeetingIntelligenceBackend(self.config)
        self.refined, _, _ = evidence()
        self.request = IntelligenceRequest("chunk_0001", self.refined.utterances)

    def test_requires_key(self):
        with self.assertRaises(IntelligenceAuthenticationError):
            GroqMeetingIntelligenceBackend(IntelligenceConfig())

    def test_valid_payload_instrumentation(self):
        with patch(
            "meeting_assistant.intelligence.groq_backend._exchange",
            return_value=completion(payload()),
        ) as exchange:
            result = self.backend.extract(self.request)
        sent = json.loads(exchange.call_args.args[0])
        self.assertEqual(sent["response_format"]["type"], "json_schema")
        self.assertTrue(sent["response_format"]["json_schema"]["strict"])
        self.assertEqual(sent["reasoning_effort"], "low")
        self.assertFalse(sent["include_reasoning"])
        self.assertNotIn("test-secret", repr(sent))
        self.assertEqual(result.calls[0].total_tokens, 130)
        self.assertEqual(result.calls[0].stage, "extraction")

    def test_rate_retry_bounded(self):
        with (
            patch(
                "meeting_assistant.intelligence.groq_backend._exchange",
                side_effect=[(429, b"{}"), completion(payload())],
            ) as exchange,
            patch("meeting_assistant.intelligence.groq_backend.time.sleep"),
        ):
            result = self.backend.extract(self.request)
        self.assertEqual(exchange.call_count, 2)
        self.assertTrue(result.calls[1].transport_retry)
        self.assertIsNone(result.calls[0].total_tokens)

    def test_rate_limit_exceeds_backoff_no_retry(self):
        def response(*_, retry_hint, **__):
            retry_hint["exceeds_limit"] = True
            return 429, b"{}"

        with patch(
            "meeting_assistant.intelligence.groq_backend._exchange", side_effect=response
        ) as exchange:
            with self.assertRaises(IntelligenceRateLimitError):
                self.backend.extract(self.request)
        self.assertEqual(exchange.call_count, 1)

    def test_auth_sanitized(self):
        for code in (401, 403):
            with (
                self.subTest(code=code),
                patch(
                    "meeting_assistant.intelligence.groq_backend._exchange",
                    return_value=(code, b"test-secret-provider-diagnostics"),
                ),
            ):
                with self.assertRaises(IntelligenceAuthenticationError) as ctx:
                    self.backend.extract(self.request)
                self.assertNotIn("test-secret", str(ctx.exception))

    def test_server_retry_exhausted(self):
        with (
            patch(
                "meeting_assistant.intelligence.groq_backend._exchange", return_value=(503, b"{}")
            ) as exchange,
            patch("meeting_assistant.intelligence.groq_backend.time.sleep"),
        ):
            with self.assertRaises(IntelligenceProviderError):
                self.backend.extract(self.request)
        self.assertEqual(exchange.call_count, 2)

    def test_schema_repair_one(self):
        with patch(
            "meeting_assistant.intelligence.groq_backend._exchange",
            side_effect=[completion("{}"), completion(payload())],
        ) as exchange:
            result = self.backend.extract(self.request)
        self.assertEqual(exchange.call_count, 2)
        self.assertTrue(result.calls[-1].schema_repair)
        self.assertIn(
            "schema_repair_only", json.loads(exchange.call_args.args[0])["messages"][-1]["content"]
        )

    def test_schema_failure_exhausted(self):
        with patch(
            "meeting_assistant.intelligence.groq_backend._exchange", return_value=completion("{}")
        ) as exchange:
            with self.assertRaises(IntelligenceSchemaError):
                self.backend.extract(self.request)
        self.assertEqual(exchange.call_count, 2)

    def test_provider_rejected_json_not_directly_accepted(self):
        bad = json.dumps(
            {"error": {"code": "json_validate_failed", "failed_generation": payload()}}
        ).encode()
        backend = GroqMeetingIntelligenceBackend(replace(self.config, schema_repair_retries=0))
        with patch(
            "meeting_assistant.intelligence.groq_backend._exchange", return_value=(400, bad)
        ):
            with self.assertRaises(IntelligenceSchemaError):
                backend.extract(self.request)

    def test_wrong_model_truncation_malformed_metadata(self):
        for result in (
            completion(payload(), model="different"),
            completion(payload(), choices=[]),
            completion(
                payload(), choices=[{"finish_reason": "length", "message": {"content": payload()}}]
            ),
            (200, b"not JSON"),
        ):
            with (
                self.subTest(result=result),
                patch("meeting_assistant.intelligence.groq_backend._exchange", return_value=result),
                self.assertRaises(IntelligenceProviderError),
            ):
                self.backend.extract(self.request)

    def test_transport_verified_host_no_redirect(self):
        response = Mock(status=302)
        response.getheader.return_value = "0"
        response.read1.side_effect = [b"{}", b""]
        connection = Mock()
        connection.sock = None
        connection.getresponse.return_value = response
        with patch(
            "meeting_assistant._groq_transport.http.client.HTTPSConnection", return_value=connection
        ) as constructor:
            self.assertEqual(_exchange(b"{}", self.config), (302, b"{}"))
        self.assertEqual(constructor.call_args.args, ("api.groq.com",))
        self.assertEqual(
            connection.request.call_args.args[:2], ("POST", "/openai/v1/chat/completions")
        )
        connection.close.assert_called_once()

    def test_timeout_connection_causes(self):
        for error, expected in (
            (TimeoutError("secret"), IntelligenceTimeoutError),
            (OSError("secret"), IntelligenceProviderError),
        ):
            connection = Mock(sock=None)
            connection.request.side_effect = error
            with patch(
                "meeting_assistant._groq_transport.http.client.HTTPSConnection",
                return_value=connection,
            ):
                with self.assertRaises(expected) as ctx:
                    _exchange(b"{}", self.config)
                self.assertIs(ctx.exception.__cause__, error)
                self.assertNotIn("secret", str(ctx.exception))
            connection.close.assert_called_once()

    def test_response_size_limit(self):
        response = Mock(status=200)
        response.read1.return_value = b"x" * 65536
        response.getheader.return_value = "0"
        connection = Mock(sock=None)
        connection.getresponse.return_value = response
        with patch(
            "meeting_assistant._groq_transport.http.client.HTTPSConnection", return_value=connection
        ):
            with self.assertRaises(IntelligenceProviderError):
                _exchange(b"{}", self.config)


class CLIEvaluationTests(unittest.TestCase):
    def test_root_dispatch_and_help(self):
        with patch("meeting_assistant.intelligence.__main__.main", return_value=0) as child:
            self.assertEqual(root_main(["intelligence", "meeting.wav"]), 0)
            child.assert_called_once_with(["meeting.wav"])
        with patch("sys.stdout", new=io.StringIO()) as out:
            self.assertEqual(root_main(["--help"]), 0)
            self.assertIn("intelligence", out.getvalue())

    def test_saved_mode_invalid_provenance_no_api(self):
        from meeting_assistant.diarization.serialization import speaker_transcript_to_json
        from meeting_assistant.grounding.serialization import grounding_to_json

        refined, speaker, grounding = evidence()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name, content in (
                ("r", refined_to_json(refined)),
                ("s", speaker_transcript_to_json(speaker)),
                ("g", grounding_to_json(replace(grounding, source_speaker_sha256="0" * 64))),
            ):
                (root / name).write_text(content, encoding="utf-8")
            with (
                patch("meeting_assistant.intelligence.groq_backend._exchange") as exchange,
                patch("sys.stderr", new=io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            str(root / "r"),
                            "--speaker",
                            str(root / "s"),
                            "--grounding",
                            str(root / "g"),
                            "--env-file",
                            str(root / "none"),
                        ]
                    ),
                    1,
                )
                exchange.assert_not_called()

    def test_evidence_cli_no_api(self):
        refined, speaker, grounding = evidence()
        result = extract_meeting_record(refined, speaker, grounding, backend=FakeBackend())
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            refined_path = root / "refined.json"
            refined_path.write_text(refined_to_json(refined), encoding="utf-8")
            files = save_meeting_record(result, refined, root)
            with (
                patch("sys.stdout", new=io.StringIO()) as out,
                patch("meeting_assistant.intelligence.groq_backend._exchange") as exchange,
            ):
                self.assertEqual(
                    evidence_main(
                        [str(files.json_path), "sum_0001", "--refined", str(refined_path)]
                    ),
                    0,
                )
                exchange.assert_not_called()
                self.assertIn(refined.utterances[0].refined_text, out.getvalue())

    def test_evaluation_false_positive_metrics(self):
        refined, speaker, grounding = evidence("A proposal only.")
        from meeting_assistant.intelligence.models import (
            ActionItem,
            ActionOwner,
            Decision,
            MeetingContent,
        )

        backend = FakeBackend(
            lambda _: MeetingContent(
                decisions=(Decision("dec_0001", "Unsupported migration", ("utt_000001",)),),
                action_items=(
                    ActionItem(
                        "act_0001",
                        "Invented task",
                        ("utt_000001",),
                        ActionOwner("speaker", "SPEAKER_00"),
                    ),
                ),
            )
        )
        result = extract_meeting_record(refined, speaker, grounding, backend=backend)
        report = score_cases(
            [{"id": "negative", "turns": [{"text": "A proposal only."}]}], [result]
        )
        self.assertEqual(report["decisions"]["fp"], 1)
        self.assertEqual(report["action_items"]["fp"], 1)
        self.assertEqual(report["false_owner_count"], 1)
        self.assertEqual(report["evidence_coverage"], 1)
        self.assertEqual(report["unknown_evidence_references"], 0)
