import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from meeting_assistant.semantic_reasoning import GLiNERBackend, JuliaBackend, SemanticConfig
from meeting_assistant.semantic_reasoning.exceptions import InvalidSemantics, ProviderUnavailable
from meeting_assistant.semantic_reasoning.models import Question
from meeting_assistant.semantic_reasoning.verification import YES_NO


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.q = (Question("q", "Is this supported?", YES_NO),)

    def response(self):
        return {
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "YES",
                    "probabilities": {"YES": 0.9, "NO": 0.05, "AMBIGUOUS": 0.05},
                    "max_probability": 0.9,
                }
            },
            "runtime": {"device": "cuda"},
        }

    def prepared(self, response=None):
        backend = JuliaBackend()
        worker = Mock()
        worker.stdin, worker.stdout = io.StringIO(), io.StringIO()
        worker.poll.return_value = None
        backend._worker = worker
        backend._queue.put(json.dumps(response or self.response()))
        return backend, worker

    def call(self, backend, timeout=1):
        return backend.decide(
            {"text": "Use Redis."},
            self.q,
            purpose="verification",
            policy="semantic_verification_v1",
            timeout_seconds=timeout,
            max_attempts=1,
        )

    def test_local_defaults_and_legacy_configuration(self):
        self.assertEqual(SemanticConfig().provider, "julia")
        self.assertFalse(SemanticConfig().enabled)
        self.assertEqual(
            SemanticConfig.from_env(environ={"JEV_PROVIDER": "typesafe"}).model, "jev-1.13.0"
        )
        self.assertTrue(SemanticConfig.from_env(environ={"SEMANTIC_ENABLED": "true"}).enabled)
        self.assertFalse(
            SemanticConfig.from_env(
                environ={"JEV_ENABLED": "true", "SEMANTIC_ENABLED": "false"}
            ).enabled
        )
        with self.assertRaises(InvalidSemantics):
            SemanticConfig(local_device="invented")
        gliner = SemanticConfig.from_env(environ={"SEMANTIC_PROVIDER": "gliner"})
        self.assertEqual(gliner.model, "fastino/GLiNER2.5-Decide")
        self.assertIn("gliner-decide", gliner.model_cache)

    def test_gliner_preserves_native_half_precision_rounding(self):
        backend, _ = self.prepared()
        gliner = GLiNERBackend()
        gliner._worker = backend._worker
        response = self.response()
        response["answers"]["q"]["probabilities"] = {
            "YES": 0.8999,
            "NO": 0.05,
            "AMBIGUOUS": 0.04985,
        }
        response["answers"]["q"]["max_probability"] = 0.8999
        gliner._queue.put(json.dumps(response))
        with gliner:
            d = self.call(gliner)[0]
            self.assertEqual(d.provider, "gliner")
            self.assertEqual(dict(d.probabilities)["AMBIGUOUS"], 0.04985)

    def test_lossless_encoding_rejection_does_not_become_outage(self):
        backend, _ = self.prepared({"error": "ValueError"})
        with backend, self.assertRaises(InvalidSemantics):
            self.call(backend)

    def test_missing_local_cache_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as root:
            backend = JuliaBackend(
                SemanticConfig(model_cache=root, local_python=str(Path(root) / "absent"))
            )
            with self.assertRaises(ProviderUnavailable):
                self.call(backend)
            self.assertEqual(len(backend.calls), 1)
            self.assertIsNone(backend._worker)

    def test_wrong_weights_rejected_before_worker_creation(self):
        with tempfile.TemporaryDirectory() as root, patch("subprocess.Popen") as popen:
            (Path(root) / "model.safetensors").write_bytes(b"corrupt")
            backend = JuliaBackend(SemanticConfig(model_cache=root, local_python=__file__))
            with self.assertRaises(ProviderUnavailable):
                self.call(backend)
            popen.assert_not_called()

    def test_native_probabilities_and_model_identity(self):
        backend, _ = self.prepared()
        with backend:
            decision = self.call(backend)[0]
            self.assertEqual(decision.provider, "julia")
            self.assertEqual(decision.provider_score, 0.9)
            self.assertEqual(decision.selected_probability, 0.9)
            self.assertEqual(backend.runtime, {"device": "cuda"})
            self.assertIsNone(backend.calls[0].input_tokens)

    def test_resident_model_reused_for_two_calls(self):
        backend, worker = self.prepared()
        with backend, patch.object(backend, "_start") as start:
            self.call(backend)
            backend._queue.put(json.dumps(self.response()))
            self.call(backend)
            start.assert_not_called()
            self.assertIs(backend._worker, worker)
            self.assertEqual(len(backend.calls), 2)
        worker.terminate.assert_called_once()

    def test_timeout_closes_worker_without_cpu_fallback(self):
        backend, worker = self.prepared()
        backend._queue.get()
        with self.assertRaises(ProviderUnavailable):
            self.call(backend, timeout=0.001)
        self.assertIsNone(backend._worker)
        worker.terminate.assert_called_once()

    def test_worker_failure_is_sanitized(self):
        backend, _ = self.prepared({"error": "RuntimeError"})
        with backend, self.assertRaises(ProviderUnavailable) as failure:
            self.call(backend)
        self.assertNotIn("Traceback", str(failure.exception))

    def test_malformed_output_rejected(self):
        for change in (
            {"choice": "invented"},
            {"probabilities": {"YES": 1.0}},
            {"max_probability": float("nan")},
            {"type": "noul"},
        ):
            with self.subTest(change=change):
                data = self.response()
                data["answers"]["q"].update(change)
                backend, _ = self.prepared(data)
                with backend, self.assertRaises(InvalidSemantics):
                    self.call(backend)

    def test_context_exit_cleans_failure(self):
        backend, worker = self.prepared()
        with self.assertRaises(RuntimeError), backend:
            raise RuntimeError("application error")
        worker.terminate.assert_called_once()
        self.assertIsNone(backend._worker)


@unittest.skipUnless(
    os.environ.get("RUN_SEMANTIC_MODEL_TESTS") == "1", "Explicit cached Julia GPU test"
)
class RealJuliaTests(unittest.TestCase):
    def test_offline_cuda_typed_decision_and_reuse(self):
        config = replace(SemanticConfig(), request_timeout_seconds=60)
        questions = (
            Question(
                "q",
                "Which department should handle this request?",
                (
                    ("billing", "Billing and payment disputes"),
                    ("shipping", "Shipping and delivery"),
                ),
            ),
        )
        with JuliaBackend(config) as backend:
            for _ in range(2):
                output = backend.decide(
                    {"text": "I was charged twice for the same order."},
                    questions,
                    purpose="events",
                    policy="event_classification_v1",
                    timeout_seconds=60,
                    max_attempts=1,
                )
                self.assertIn(output[0].choice, dict(questions[0].criteria))
                self.assertAlmostEqual(sum(dict(output[0].probabilities).values()), 1.0)
            self.assertEqual(backend.runtime["device"], "cuda")
            self.assertGreater(backend.runtime["peak_cuda_memory_bytes"], 0)
            self.assertEqual(len(backend.calls), 2)
            self.assertTrue(backend.runtime["python_socket_connections_blocked"])

    def test_gliner_offline_cuda_all_scores(self):
        with GLiNERBackend() as backend:
            output = backend.decide(
                {"text": "My subscription was billed twice."},
                (
                    Question(
                        "intent",
                        "Which intent describes this request?",
                        (
                            ("billing", "Payment or refund request"),
                            ("shipping", "Delivery or shipping request"),
                        ),
                    ),
                ),
                purpose="events",
                policy="event_classification_v1",
                timeout_seconds=60,
                max_attempts=1,
            )
            self.assertEqual(set(dict(output[0].probabilities)), {"billing", "shipping"})
            self.assertAlmostEqual(sum(dict(output[0].probabilities).values()), 1.0, places=3)
            self.assertEqual(backend.runtime["device"], "cuda")
            self.assertTrue(backend.runtime["python_socket_connections_blocked"])
