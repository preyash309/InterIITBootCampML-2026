"""Official TypeSafe HTTP contract inspected 2026-10-08; no SDK dependency.

Reference: https://docs.typesafe.ai/api and /models. Fixed official origin prevents
accidental credential forwarding. Redirects are disabled. Scores are retained,
not advertised as meeting-domain confidence.
"""

import http.client
import json
import time

from .config import SemanticConfig
from .exceptions import InvalidSemantics, ProviderUnavailable
from .models import ProviderCall, ProviderDecision, Question, require
from .serialization import _object, fingerprint

API_SCHEMA = "typesafe_http_2026-10-08"


def request_body(state, questions, model):
    require(type(state) is dict and bool(questions))
    require(all(isinstance(q, Question) for q in questions))
    require(len({q.id for q in questions}) == len(questions))
    return {
        "model": model,
        "state": state,
        "questions": {
            q.id: {"type": "choice", "instructions": q.instructions, "criteria": dict(q.criteria)}
            for q in questions
        },
    }


def parse_response(
    data, questions, *, model, request_sha, response_sha, latency, policy, provider="typesafe"
):
    try:
        require(type(data) is dict and data["model"] == model, "Provider changed the pinned model.")
        answers = data["answers"]
        require(
            type(answers) is dict and set(answers) == {q.id for q in questions},
            "Missing/unknown question IDs.",
        )
        usage = data["usage"]
        require(type(usage) is dict)
        require(
            all(type(usage[k]) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens"))
        )
        output = []
        for q in questions:
            answer = answers[q.id]
            require(type(answer) is dict and answer["type"] == "choice")
            probabilities = answer["probabilities"]
            require(type(probabilities) is dict and set(probabilities) == set(dict(q.criteria)))
            output.append(
                ProviderDecision(
                    q.id,
                    answer["choice"],
                    tuple((k, probabilities[k]) for k, _ in q.criteria),
                    answer["confidence"],
                    provider,
                    model,
                    API_SCHEMA,
                    request_sha,
                    response_sha,
                    latency,
                    policy,
                )
            )
        return tuple(output)
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidSemantics("Malformed typed provider response.") from exc


class JevBackend:
    provider = "typesafe"

    def __init__(self, api_key: str | None, config=None, *, transport=None, sleep=time.sleep):
        self.config = config or SemanticConfig(provider="typesafe", model="jev-1.13.0")
        self.model = self.config.model
        if api_key is not None:
            api_key = api_key.strip()
            require(
                all(33 <= ord(c) <= 126 for c in api_key), "Invalid TypeSafe credential format."
            )
        self._api_key = api_key
        self._transport = transport or self._post
        self._sleep = sleep
        self.calls = []

    def _post(self, body, timeout):
        connection = http.client.HTTPSConnection("api.typesafe.ai", timeout=timeout)
        try:
            connection.request(
                "POST",
                "/v1/systemone",
                body,
                {"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"},
            )
            response = connection.getresponse()
            content = response.read(2_000_001)
            require(len(content) <= 2_000_000, "Provider response exceeded size budget.")
            return response.status, dict(response.getheaders()), content
        finally:
            connection.close()

    def decide(self, state, questions, *, purpose, policy, timeout_seconds, max_attempts):
        if not self._api_key:
            raise ProviderUnavailable("Configure TYPESAFE_API_KEY in the ignored backend .env.")
        body = request_body(state, questions, self.model)
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        require(len(encoded) <= self.config.max_state_chars * 8, "Request size budget exceeded.")
        request_sha = fingerprint(body)
        started = time.monotonic()
        deadline = started + timeout_seconds
        for attempt in range(max_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderUnavailable("Jev request deadline exceeded.")
            before = time.monotonic()
            status, response_sha, data, delay = None, None, None, None
            try:
                status, headers, content = self._transport(
                    encoded, min(remaining, self.config.request_timeout_seconds)
                )
                response_sha = fingerprint(content.decode("utf-8", errors="replace"))
                if status == 200:
                    data = json.loads(content, object_pairs_hook=_object)
                header = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
                if header is not None:
                    try:
                        delay = float(header)
                        require(0 <= delay <= 86400)
                    except (ValueError, InvalidSemantics):
                        delay = None
            except (OSError, TimeoutError, http.client.HTTPException) as exc:
                self.calls.append(
                    ProviderCall(
                        purpose, None, time.monotonic() - before, attempt > 0, request_sha, None
                    )
                )
                raise ProviderUnavailable("Jev connection failed or timed out.") from exc
            except (ValueError, InvalidSemantics) as exc:
                self.calls.append(
                    ProviderCall(
                        purpose,
                        status,
                        time.monotonic() - before,
                        attempt > 0,
                        request_sha,
                        response_sha,
                    )
                )
                raise InvalidSemantics("Malformed Jev JSON response.") from exc
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            # Usage only records validated integers; malformed success still fails below.
            tokens = [
                usage.get(k) if type(usage.get(k)) is int and usage[k] >= 0 else None
                for k in ("input_tokens", "output_tokens")
            ]
            self.calls.append(
                ProviderCall(
                    purpose,
                    status,
                    time.monotonic() - before,
                    attempt > 0,
                    request_sha,
                    response_sha,
                    *tokens,
                    delay,
                )
            )
            if status == 200:
                return parse_response(
                    data,
                    questions,
                    model=self.model,
                    request_sha=request_sha,
                    response_sha=response_sha,
                    latency=time.monotonic() - started,
                    policy=policy,
                )
            if status in (401, 403):
                raise ProviderUnavailable(
                    "Jev authentication/access denied; check TypeSafe access."
                )
            if status not in (429, 500, 502, 503, 504, 529) or attempt + 1 == max_attempts:
                raise ProviderUnavailable(f"Jev unavailable (HTTP {status}).")
            wait = delay if delay is not None else 0.5 * 2**attempt
            if wait > self.config.max_retry_delay_seconds or wait >= deadline - time.monotonic():
                raise ProviderUnavailable("Jev retry delay exceeds the bounded retry budget.")
            self._sleep(wait)
        raise ProviderUnavailable("Jev attempt budget exhausted.")
