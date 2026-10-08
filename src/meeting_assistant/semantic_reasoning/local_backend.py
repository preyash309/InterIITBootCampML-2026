"""Resident isolated Julia worker. No CUDA/Transformers changes in the main process."""

import hashlib
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .config import SemanticConfig
from .exceptions import InvalidSemantics, ProviderUnavailable
from .models import ProviderCall, ProviderDecision, require
from .provider import request_body
from .serialization import _object, fingerprint

logger = logging.getLogger(__name__)

JULIA_REVISION = "a85b127321d580d65176c89ced8273f305745d85"
JULIA_WEIGHTS_SHA256 = "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"
JULIA_SCHEMA = "julia_native_" + JULIA_REVISION
GLINER_REVISION = "5a7adf72a23b4d311abae6ce050d7f0012bb3416"
GLINER_WEIGHTS_SHA256 = "40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997"


class LocalDecisionBackend:
    """Shared process transport; model-specific inference stays in the worker."""

    def __init__(self, config=None):
        self.config = config or SemanticConfig.from_env(
            environ={"SEMANTIC_PROVIDER": self.provider}
        )
        require(self.config.provider == self.provider)
        self.model = self.config.model
        self.calls, self.runtime = [], {}
        self._worker, self._queue = None, queue.Queue()
        self._diagnostics = None
        self._lock = threading.Lock()

    def _start(self):
        root, executable = (
            Path(self.config.model_cache).resolve(),
            Path(self.config.local_python).resolve(),
        )
        if not executable.is_file() or not (root / "model.safetensors").is_file():
            raise ProviderUnavailable(
                "Local Julia cache/runtime missing; run scripts/setup_semantic_model.py."
            )
        with (root / "model.safetensors").open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if sha != self.weights_sha256:
            raise ProviderUnavailable("Local weights do not match the pinned checkpoint.")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_TF": "0"}
        # No shell and no API credentials passed intentionally; worker receives only local paths/options.
        for key in tuple(env):
            if key.endswith("API_KEY") or key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                env.pop(key)
        command = [
            str(executable),
            "-u",
            str(Path(__file__).with_name("worker.py")),
            self.provider,
            str(root),
            self.config.local_device,
            str(self.config.local_max_length),
            str(self.config.local_head_length),
        ]
        self._diagnostics = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._worker = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._diagnostics,
            text=True,
            encoding="utf-8",
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        process, mailbox = self._worker, self._queue

        def reader():
            try:
                while True:
                    line = process.stdout.readline(2_000_001)
                    if not line:
                        break
                    mailbox.put(line)
            finally:
                mailbox.put(None)

        threading.Thread(target=reader, daemon=True).start()

    def decide(self, state, questions, *, purpose, policy, timeout_seconds, max_attempts):
        started = time.monotonic()
        body = request_body(state, questions, self.model)
        request_sha = fingerprint(
            (
                body,
                self.revision,
                self.api_schema,
                self.config.local_device,
                self.config.local_max_length,
                self.config.local_head_length,
            )
        )
        response_sha = None
        try:
            with self._lock:
                if self._worker is None:
                    self._start()
                self._worker.stdin.write(
                    json.dumps(body, ensure_ascii=False, allow_nan=False) + "\n"
                )
                self._worker.stdin.flush()
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise queue.Empty
                line = self._queue.get(timeout=remaining)
                if line is None or len(line) > 2_000_000:
                    raise ProviderUnavailable(
                        "Local Julia worker exited or exceeded response budget."
                    )
                data = json.loads(line, object_pairs_hook=_object)
                response_sha = fingerprint(data)
                if data.get("error"):
                    self._log_diagnostics()
                    if data["error"] == "ValueError":
                        raise InvalidSemantics(
                            "Local request exceeded the model's lossless contract."
                        )
                    raise ProviderUnavailable("Local inference unavailable: " + data["error"])
                answers = data["answers"]
                require(set(answers) == {q.id for q in questions})
                self.runtime = data["runtime"]
                decisions = []
                for q in questions:
                    answer = answers[q.id]
                    require(
                        answer["type"] == "choice"
                        and set(answer["probabilities"]) == set(dict(q.criteria))
                    )
                    decisions.append(
                        ProviderDecision(
                            q.id,
                            answer["choice"],
                            tuple(
                                (label, answer["probabilities"][label]) for label, _ in q.criteria
                            ),
                            answer["max_probability"],
                            self.provider,
                            self.model,
                            self.api_schema,
                            request_sha,
                            response_sha,
                            time.monotonic() - started,
                            policy,
                        )
                    )
                return tuple(decisions)
        except queue.Empty as exc:
            self._log_diagnostics()
            self.close()
            raise ProviderUnavailable(
                "Local Julia worker timed out; no CPU fallback was made."
            ) from exc
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._log_diagnostics()
            self.close()
            raise InvalidSemantics(
                "Local Julia returned an invalid response or path/runtime failed."
            ) from exc
        finally:
            self.calls.append(
                ProviderCall(
                    purpose, None, time.monotonic() - started, False, request_sha, response_sha
                )
            )

    def _log_diagnostics(self):
        if self._diagnostics:
            self._diagnostics.seek(0)
            logger.warning("semantic_local_worker_diagnostics: %s", self._diagnostics.read(16000))

    def close(self):
        worker, self._worker = self._worker, None
        if worker is not None:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            for stream in (worker.stdin, worker.stdout):
                if stream:
                    stream.close()
        self._queue = queue.Queue()
        if self._diagnostics:
            self._diagnostics.close()
            self._diagnostics = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class JuliaBackend(LocalDecisionBackend):
    provider = "julia"
    revision = JULIA_REVISION
    weights_sha256 = JULIA_WEIGHTS_SHA256
    api_schema = JULIA_SCHEMA + "_readable_state_v1"


class GLiNERBackend(LocalDecisionBackend):
    provider = "gliner"
    revision = GLINER_REVISION
    weights_sha256 = GLINER_WEIGHTS_SHA256
    api_schema = "gliner2_2.0.0_softmax_all_labels_" + GLINER_REVISION + "_readable_state_v1"
