import http.client
import json
import socket
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import MagicMock, patch

from meeting_assistant.refinement import GroqRefinerBackend, RefinementConfig
from meeting_assistant.refinement.exceptions import (
    RefinementAuthenticationError,
    RefinementProviderError,
    RefinementRateLimitError,
    RefinementSchemaError,
    RefinementTimeoutError,
)
from meeting_assistant.refinement.groq_backend import ENDPOINT, HOST, _exchange
from meeting_assistant.refinement.service import build_requests

from .helpers import completion, decision_json, evidence


class BackendTests(unittest.TestCase):
    def setUp(self):
        source, g = evidence()
        self.request = build_requests(source, g, RefinementConfig())[0]
        self.backend = GroqRefinerBackend(
            RefinementConfig(api_key="private-test-key", transport_retries=0)
        )

    def test_credentials_required_and_hidden(self):
        with self.assertRaises(RefinementAuthenticationError):
            GroqRefinerBackend(RefinementConfig())
        self.assertNotIn("private-test-key", repr(self.backend.config))

    def test_valid_metadata_and_payload(self):
        with patch(
            "meeting_assistant.refinement.groq_backend._exchange",
            return_value=completion(decision_json(self.request, "REPLACE")),
        ) as exchange:
            result = self.backend.refine(self.request)
        self.assertEqual(result.decisions[0].action, "REPLACE")
        self.assertEqual(result.calls[0].total_tokens, 125)
        payload = json.loads(exchange.call_args.args[0])
        self.assertEqual(payload["temperature"], 0)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertFalse(payload["include_reasoning"])
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertNotIn("private-test-key", json.dumps(payload))

    def test_absent_usage_stays_null(self):
        with patch(
            "meeting_assistant.refinement.groq_backend._exchange",
            return_value=completion(decision_json(self.request), usage=None),
        ):
            result = self.backend.refine(self.request)
        self.assertIsNone(result.calls[0].prompt_tokens)

    def test_repair_once(self):
        with patch(
            "meeting_assistant.refinement.groq_backend._exchange",
            side_effect=[completion("bad JSON"), completion(decision_json(self.request))],
        ) as exchange:
            result = self.backend.refine(self.request)
        self.assertEqual(exchange.call_count, 2)
        self.assertEqual([c.schema_repair for c in result.calls], [False, True])
        repair = json.loads(exchange.call_args.args[0])["messages"]
        self.assertIn("untrusted_invalid_response", repair[-1]["content"])
        self.assertIn("Repair JSON structure/references only", repair[-1]["content"])

    def test_repair_failure_bounded(self):
        with (
            patch(
                "meeting_assistant.refinement.groq_backend._exchange",
                return_value=completion("bad JSON"),
            ) as exchange,
            self.assertRaises(RefinementSchemaError) as error,
        ):
            self.backend.refine(self.request)
        self.assertEqual(exchange.call_count, 2)
        self.assertIsNotNone(error.exception.__cause__)
        self.assertNotIn("bad JSON", str(error.exception))

    def test_repair_disabled(self):
        backend = GroqRefinerBackend(replace(self.backend.config, schema_repair_retries=0))
        with (
            patch(
                "meeting_assistant.refinement.groq_backend._exchange",
                return_value=completion("bad JSON"),
            ) as exchange,
            self.assertRaises(RefinementSchemaError),
        ):
            backend.refine(self.request)
        self.assertEqual(exchange.call_count, 1)

    def test_provider_schema_rejection_repaired_not_applied_directly(self):
        error = json.dumps(
            {
                "error": {
                    "code": "json_validate_failed",
                    "failed_generation": decision_json(self.request, "REPLACE"),
                }
            }
        ).encode()
        with patch(
            "meeting_assistant.refinement.groq_backend._exchange",
            side_effect=[(400, error), completion(decision_json(self.request, "UNCERTAIN"))],
        ) as exchange:
            result = self.backend.refine(self.request)
        self.assertEqual(exchange.call_count, 2)
        self.assertEqual(result.decisions[0].action, "UNCERTAIN")
        self.assertEqual([c.status_code for c in result.calls], [400, 200])

    def test_provider_schema_rejection_bounded(self):
        error = json.dumps(
            {"error": {"code": "json_validate_failed", "failed_generation": "{}"}}
        ).encode()
        with (
            patch(
                "meeting_assistant.refinement.groq_backend._exchange", return_value=(400, error)
            ) as exchange,
            self.assertRaises(RefinementSchemaError),
        ):
            self.backend.refine(self.request)
        self.assertEqual(exchange.call_count, 2)

    def test_invalid_candidate_triggers_schema_repair(self):
        with patch(
            "meeting_assistant.refinement.groq_backend._exchange",
            side_effect=[
                completion(decision_json(self.request, "REPLACE", candidate_entry_id="invented")),
                completion(decision_json(self.request, "UNCERTAIN")),
            ],
        ):
            result = self.backend.refine(self.request)
        self.assertEqual(result.decisions[0].action, "UNCERTAIN")
        self.assertEqual(len(result.calls), 2)

    def test_authentication_no_retry(self):
        with (
            patch(
                "meeting_assistant.refinement.groq_backend._exchange",
                return_value=(401, b"private provider body"),
            ) as exchange,
            self.assertRaises(RefinementAuthenticationError) as error,
        ):
            self.backend.refine(self.request)
        self.assertEqual(exchange.call_count, 1)
        self.assertEqual(error.exception.status_code, 401)
        self.assertNotIn("private provider body", str(error.exception))

    def test_rate_limit(self):
        with (
            patch("meeting_assistant.refinement.groq_backend._exchange", return_value=(429, b"")),
            self.assertRaises(RefinementRateLimitError),
        ):
            self.backend.refine(self.request)

    def test_status_retry_bounded(self):
        backend = GroqRefinerBackend(replace(self.backend.config, transport_retries=1))
        with (
            patch(
                "meeting_assistant.refinement.groq_backend._exchange",
                side_effect=[(503, b""), completion(decision_json(self.request))],
            ) as exchange,
            patch("meeting_assistant.refinement.groq_backend.time.sleep"),
        ):
            result = backend.refine(self.request)
        self.assertEqual(exchange.call_count, 2)
        self.assertEqual([c.status_code for c in result.calls], [503, 200])
        self.assertIsNone(result.calls[0].total_tokens)

    def test_status_retry_exhausted(self):
        backend = GroqRefinerBackend(replace(self.backend.config, transport_retries=1))
        with (
            patch(
                "meeting_assistant.refinement.groq_backend._exchange", return_value=(500, b"")
            ) as exchange,
            patch("meeting_assistant.refinement.groq_backend.time.sleep"),
            self.assertRaises(RefinementProviderError),
        ):
            backend.refine(self.request)
        self.assertEqual(exchange.call_count, 2)

    def test_nonretryable_model_error(self):
        with (
            patch(
                "meeting_assistant.refinement.groq_backend._exchange",
                return_value=(400, b"private details"),
            ) as exchange,
            self.assertRaises(RefinementProviderError),
        ):
            self.backend.refine(self.request)
        self.assertEqual(exchange.call_count, 1)

    def test_truncated_completion(self):
        choices = [{"finish_reason": "length", "message": {"content": decision_json(self.request)}}]
        with (
            patch(
                "meeting_assistant.refinement.groq_backend._exchange",
                return_value=completion("", choices=choices),
            ),
            self.assertRaises(RefinementProviderError),
        ):
            self.backend.refine(self.request)

    def test_provider_model_mismatch(self):
        with (
            patch(
                "meeting_assistant.refinement.groq_backend._exchange",
                return_value=completion(decision_json(self.request), model="another"),
            ),
            self.assertRaises(RefinementProviderError),
        ):
            self.backend.refine(self.request)

    def test_outer_json_malformed(self):
        with (
            patch("meeting_assistant.refinement.groq_backend._exchange", return_value=(200, b"no")),
            self.assertRaises(RefinementProviderError),
        ):
            self.backend.refine(self.request)

    def test_no_semantic_retry(self):
        with patch(
            "meeting_assistant.refinement.groq_backend._exchange",
            return_value=completion(decision_json(self.request, "UNCERTAIN")),
        ) as exchange:
            self.backend.refine(self.request)
        self.assertEqual(exchange.call_count, 1)

    def test_json_object_explicit(self):
        backend = GroqRefinerBackend(
            replace(self.backend.config, structured_mode="json_object", model="explicit-model")
        )
        with patch(
            "meeting_assistant.refinement.groq_backend._exchange",
            return_value=completion(decision_json(self.request), model="explicit-model"),
        ) as exchange:
            backend.refine(self.request)
        payload = json.loads(exchange.call_args.args[0])
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("reasoning_effort", payload)

    def test_backend_reused_concurrently(self):
        with patch(
            "meeting_assistant.refinement.groq_backend._exchange",
            return_value=completion(decision_json(self.request)),
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(self.backend.refine, [self.request] * 2))
        self.assertEqual([len(r.calls) for r in results], [1, 1])
        self.assertEqual(results[0].model_info, results[1].model_info)


class TransportTests(unittest.TestCase):
    def test_fixed_verified_https_no_redirect(self):
        connection = MagicMock()
        response = connection.getresponse.return_value
        response.status = 200
        response.read1.side_effect = [b"{}", b""]
        with patch(
            "meeting_assistant.refinement.groq_backend.http.client.HTTPSConnection",
            return_value=connection,
        ) as constructor:
            status, body = _exchange(b"{}", RefinementConfig(api_key="secret"))
        constructor.assert_called_once_with(HOST, timeout=90)
        self.assertEqual(connection.request.call_args.args[:2], ("POST", ENDPOINT))
        self.assertEqual((status, body), (200, b"{}"))
        connection.close.assert_called_once()

    def test_socket_timeout_cause(self):
        connection = MagicMock()
        connection.request.side_effect = socket.timeout("private transport details")
        with (
            patch(
                "meeting_assistant.refinement.groq_backend.http.client.HTTPSConnection",
                return_value=connection,
            ),
            self.assertRaises(RefinementTimeoutError) as error,
        ):
            _exchange(b"{}", RefinementConfig(api_key="secret"))
        self.assertIsNotNone(error.exception.__cause__)
        self.assertNotIn("private", str(error.exception))
        connection.close.assert_called_once()

    def test_connection_failure(self):
        connection = MagicMock()
        connection.request.side_effect = http.client.RemoteDisconnected("gone")
        with (
            patch(
                "meeting_assistant.refinement.groq_backend.http.client.HTTPSConnection",
                return_value=connection,
            ),
            self.assertRaises(RefinementProviderError),
        ):
            _exchange(b"{}", RefinementConfig(api_key="secret"))
        connection.close.assert_called_once()

    def test_response_size_limit(self):
        connection = MagicMock()
        response = connection.getresponse.return_value
        response.status = 200
        response.read1.return_value = b"abcdef"
        with (
            patch(
                "meeting_assistant.refinement.groq_backend.http.client.HTTPSConnection",
                return_value=connection,
            ),
            patch("meeting_assistant.refinement.groq_backend.MAX_RESPONSE_BYTES", 5),
            self.assertRaises(RefinementProviderError),
        ):
            _exchange(b"{}", RefinementConfig(api_key="secret"))

    def test_retry_after_hint(self):
        connection = MagicMock()
        response = connection.getresponse.return_value
        response.status = 429
        response.getheader.return_value = "7.5"
        response.read1.side_effect = [b"{}", b""]
        hint = {}
        with patch(
            "meeting_assistant.refinement.groq_backend.http.client.HTTPSConnection",
            return_value=connection,
        ):
            _exchange(b"{}", RefinementConfig(api_key="secret"), retry_hint=hint)
        self.assertEqual(hint, {"seconds": 7.5})

    def test_excessive_retry_after_hint(self):
        connection = MagicMock()
        response = connection.getresponse.return_value
        response.status = 429
        response.getheader.return_value = "3600"
        response.read1.side_effect = [b"{}", b""]
        hint = {}
        with patch(
            "meeting_assistant.refinement.groq_backend.http.client.HTTPSConnection",
            return_value=connection,
        ):
            _exchange(b"{}", RefinementConfig(api_key="secret"), retry_hint=hint)
        self.assertEqual(hint, {"exceeds_limit": True})

    def test_deadline(self):
        connection = MagicMock()
        with (
            patch(
                "meeting_assistant.refinement.groq_backend.http.client.HTTPSConnection",
                return_value=connection,
            ),
            patch(
                "meeting_assistant.refinement.groq_backend.time.perf_counter", side_effect=[0, 100]
            ),
            self.assertRaises(RefinementTimeoutError),
        ):
            _exchange(b"{}", RefinementConfig(api_key="secret"))
