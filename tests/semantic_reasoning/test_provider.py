import json
import os
import unittest
from unittest.mock import patch

from meeting_assistant.semantic_reasoning import SemanticConfig
from meeting_assistant.semantic_reasoning.exceptions import InvalidSemantics, ProviderUnavailable
from meeting_assistant.semantic_reasoning.models import Question
from meeting_assistant.semantic_reasoning.provider import JevBackend, request_body
from meeting_assistant.semantic_reasoning.service import DecisionSession
from meeting_assistant.semantic_reasoning.verification import YES_NO


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.questions = (Question("support", "Is the claim supported?", YES_NO),)

    def response(self, **changes):
        data = {
            "model": "jev-1.13.0",
            "answers": {
                "support": {
                    "type": "choice",
                    "choice": "YES",
                    "probabilities": {"YES": 0.9, "NO": 0.05, "AMBIGUOUS": 0.05},
                    "confidence": 0.85,
                }
            },
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
        data.update(changes)
        return json.dumps(data).encode()

    def call(self, transport, questions=None, attempts=1):
        backend = JevBackend("test-secret", transport=transport, sleep=lambda _: None)
        output = backend.decide(
            {"evidence": "Use Redis."},
            questions or self.questions,
            purpose="verification",
            policy="semantic_verification_v1",
            timeout_seconds=10,
            max_attempts=attempts,
        )
        return backend, output

    def test_official_request_schema(self):
        body = request_body({"text": "Evidence"}, self.questions, "jev-1.13.0")
        self.assertEqual(set(body), {"state", "model", "questions"})
        self.assertEqual(body["questions"]["support"]["type"], "choice")
        self.assertEqual(body["questions"]["support"]["criteria"], dict(YES_NO))
        self.assertNotIn("test-secret", json.dumps(body))

    def test_multiple_atomic_questions(self):
        q = self.questions + (Question("owner", "Is the owner explicit?", YES_NO),)
        payload = json.loads(self.response())
        payload["answers"]["owner"] = payload["answers"]["support"]
        backend, output = self.call(lambda *_: (200, {}, json.dumps(payload).encode()), q)
        self.assertEqual(tuple(d.question_id for d in output), ("support", "owner"))
        self.assertEqual(backend.calls[0].input_tokens, 20)
        self.assertNotIn("test-secret", str(backend.calls) + str(output))

    def test_missing_answer(self):
        with self.assertRaises(InvalidSemantics):
            self.call(lambda *_: (200, {}, self.response(answers={})))

    def test_unknown_question(self):
        payload = json.loads(self.response())
        payload["answers"]["other"] = payload["answers"]["support"]
        with self.assertRaises(InvalidSemantics):
            self.call(lambda *_: (200, {}, json.dumps(payload).encode()))

    def test_unknown_choice(self):
        payload = json.loads(self.response())
        payload["answers"]["support"]["choice"] = "invented"
        with self.assertRaises(InvalidSemantics):
            self.call(lambda *_: (200, {}, json.dumps(payload).encode()))

    def test_unknown_probability_option(self):
        payload = json.loads(self.response())
        payload["answers"]["support"]["probabilities"]["invented"] = 0
        with self.assertRaises(InvalidSemantics):
            self.call(lambda *_: (200, {}, json.dumps(payload).encode()))

    def test_nonfinite_and_invalid_probabilities(self):
        for value in (float("nan"), float("inf"), -1, 2, True, "0.9"):
            with self.subTest(value=value):
                payload = json.loads(self.response())
                payload["answers"]["support"]["probabilities"]["YES"] = value
                with self.assertRaises(InvalidSemantics):
                    self.call(lambda *_: (200, {}, json.dumps(payload).encode()))

    def test_invalid_confidence(self):
        payload = json.loads(self.response())
        payload["answers"]["support"]["confidence"] = float("nan")
        with self.assertRaises(InvalidSemantics):
            self.call(lambda *_: (200, {}, json.dumps(payload).encode()))

    def test_distribution_must_sum_one(self):
        payload = json.loads(self.response())
        payload["answers"]["support"]["probabilities"]["YES"] = 0.8
        with self.assertRaises(InvalidSemantics):
            self.call(lambda *_: (200, {}, json.dumps(payload).encode()))

    def test_selected_choice_must_be_highest(self):
        payload = json.loads(self.response())
        payload["answers"]["support"]["choice"] = "NO"
        with self.assertRaises(InvalidSemantics):
            self.call(lambda *_: (200, {}, json.dumps(payload).encode()))

    def test_wrong_type_and_model(self):
        for data in (
            self.response(model="jev-latest"),
            self.response(answers={"support": {"type": "noul", "noul": 1}}),
        ):
            with self.assertRaises(InvalidSemantics):
                self.call(lambda *_: (200, {}, data))

    def test_missing_malformed_usage(self):
        for usage in ({}, {"input_tokens": True, "output_tokens": 10}, None):
            with self.assertRaises(InvalidSemantics):
                self.call(lambda *_: (200, {}, self.response(usage=usage)))

    def test_malformed_json(self):
        for content in (b"not json", b'{"model":1,"model":2}', b"[]", b"null"):
            with self.assertRaises(InvalidSemantics):
                self.call(lambda *_: (200, {}, content))

    def test_401_and_403_no_retry(self):
        for status in (401, 403):
            backend = JevBackend(
                "test", transport=lambda *_: (status, {}, b"secret internal diagnostic")
            )
            with self.assertRaises(ProviderUnavailable) as caught:
                backend.decide(
                    {},
                    self.questions,
                    purpose="verification",
                    policy="semantic_verification_v1",
                    timeout_seconds=1,
                    max_attempts=2,
                )
            self.assertEqual(len(backend.calls), 1)
            self.assertNotIn("secret", str(caught.exception))

    def test_rate_limit_retry_honors_header(self):
        responses = iter(((429, {"Retry-After": "0.2"}, b"{}"), (200, {}, self.response())))
        waits = []
        backend = JevBackend("test", transport=lambda *_: next(responses), sleep=waits.append)
        backend.decide(
            {},
            self.questions,
            purpose="verification",
            policy="semantic_verification_v1",
            timeout_seconds=1,
            max_attempts=2,
        )
        self.assertEqual(waits, [0.2])
        self.assertTrue(backend.calls[1].retry)
        self.assertEqual(backend.calls[0].retry_after_seconds, 0.2)

    def test_server_error_bounded_retry(self):
        for status in (500, 502, 503, 504, 529):
            backend = JevBackend(
                "test", transport=lambda *_: (status, {}, b"{}"), sleep=lambda _: None
            )
            with self.assertRaises(ProviderUnavailable):
                backend.decide(
                    {},
                    self.questions,
                    purpose="verification",
                    policy="semantic_verification_v1",
                    timeout_seconds=2,
                    max_attempts=2,
                )
            self.assertEqual(len(backend.calls), 2)

    def test_huge_retry_after_no_early_retry(self):
        backend = JevBackend("test", transport=lambda *_: (429, {"retry-after": "3600"}, b"{}"))
        with self.assertRaises(ProviderUnavailable):
            backend.decide(
                {},
                self.questions,
                purpose="verification",
                policy="semantic_verification_v1",
                timeout_seconds=10,
                max_attempts=2,
            )
        self.assertEqual(len(backend.calls), 1)

    def test_timeout_sanitized_and_counted(self):
        def fail(*_):
            raise TimeoutError("sensitive diagnostic")

        backend = JevBackend("test", transport=fail)
        with self.assertRaises(ProviderUnavailable) as caught:
            backend.decide(
                {},
                self.questions,
                purpose="verification",
                policy="semantic_verification_v1",
                timeout_seconds=1,
                max_attempts=2,
            )
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
        self.assertEqual(len(backend.calls), 1)
        self.assertNotIn("sensitive", str(caught.exception))

    def test_no_key_no_transport(self):
        with patch.object(JevBackend, "_post") as post:
            backend = JevBackend(None)
            with self.assertRaises(ProviderUnavailable):
                backend.decide(
                    {},
                    self.questions,
                    purpose="verification",
                    policy="semantic_verification_v1",
                    timeout_seconds=1,
                    max_attempts=1,
                )
            post.assert_not_called()

    def test_physical_retry_budget(self):
        backend = JevBackend("test", transport=lambda *_: (429, {}, b"{}"), sleep=lambda _: None)
        session = DecisionSession(backend, SemanticConfig(max_total_calls=1))
        self.assertIsNone(
            session.ask({}, self.questions, "verification", "semantic_verification_v1")
        )
        self.assertEqual(len(backend.calls), 1)

    def test_http_origin_auth_and_no_redirect(self):
        with patch(
            "meeting_assistant.semantic_reasoning.provider.http.client.HTTPSConnection"
        ) as connection:
            connection.return_value.getresponse.return_value.status = 302
            connection.return_value.getresponse.return_value.getheaders.return_value = [
                ("Location", "https://untrusted.test")
            ]
            connection.return_value.getresponse.return_value.read.return_value = b""
            backend = JevBackend("test")
            with self.assertRaises(ProviderUnavailable):
                backend.decide(
                    {},
                    self.questions,
                    purpose="verification",
                    policy="semantic_verification_v1",
                    timeout_seconds=1,
                    max_attempts=1,
                )
            self.assertEqual(connection.call_args.args[0], "api.typesafe.ai")
            self.assertEqual(connection.return_value.request.call_args.args[1], "/v1/systemone")
            connection.return_value.close.assert_called_once()


@unittest.skipUnless(
    os.environ.get("RUN_JEV_API_TESTS") == "1", "Explicit live Jev opt-in required."
)
class LiveJevTests(unittest.TestCase):
    def test_real_typed_decision(self):
        from meeting_assistant.asr.config import read_environment

        values = read_environment()
        self.assertTrue(
            values.get("TYPESAFE_API_KEY"), "Live flag requires an actual TypeSafe credential."
        )
        q = (Question("support", "Does this evidence explicitly choose Redis?", YES_NO),)
        backend = JevBackend(values["TYPESAFE_API_KEY"])
        result = backend.decide(
            {"evidence": "Agreed, use Redis."},
            q,
            purpose="verification",
            policy="semantic_verification_v1",
            timeout_seconds=20,
            max_attempts=1,
        )
        self.assertEqual(result[0].choice, "YES")
