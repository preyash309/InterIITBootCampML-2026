"""Bounded TLS JSON transport, not coupled to ASR multipart uploads."""

import http.client  # noqa: F401 - existing transport test patch point
import json
import logging
import time
from dataclasses import replace

from meeting_assistant._groq_transport import exchange_json

from .config import RefinementConfig
from .exceptions import (
    RefinementAuthenticationError,
    RefinementProviderError,
    RefinementRateLimitError,
    RefinementSchemaError,
    RefinementTimeoutError,
)
from .models import ProviderCall, RefinementRequest, RefinementResponse, RefinerModelInfo
from .prompt import build_messages, decision_schema
from .validation import parse_decisions

logger = logging.getLogger(__name__)
HOST = "api.groq.com"
ENDPOINT = "/openai/v1/chat/completions"
MAX_RESPONSE_BYTES = 1_000_000


def _event(name, **values):
    logger.info(name, extra={"event": name, **values})


def _status_error(status):
    if status in (401, 403):
        raise RefinementAuthenticationError(
            "Groq rejected credentials or model permissions.", status_code=status
        )
    if status == 429:
        raise RefinementRateLimitError("Groq rate limit reached; retry later.", status_code=status)
    raise RefinementProviderError(
        "Groq rejected refinement; check configured model, schema support and account status.",
        status_code=status,
    )


def _exchange(
    body: bytes, config: RefinementConfig, *, retry_hint: dict | None = None
) -> tuple[int, bytes]:
    return exchange_json(
        body,
        config,
        retry_hint=retry_hint,
        timeout_error=RefinementTimeoutError,
        provider_error=RefinementProviderError,
    )


class GroqRefinerBackend:
    """Reusable immutable config; per-call connections and no shared mutable inference state."""

    def __init__(self, config: RefinementConfig | None = None):
        self.config = config or RefinementConfig.from_env()
        if not self.config.api_key:
            raise RefinementAuthenticationError(
                "Set GROQ_API_KEY in the ignored .env or environment."
            )
        self._model_info = RefinerModelInfo(
            self.config.provider,
            self.config.model,
            self.config.temperature,
            self.config.structured_mode,
            reasoning_effort="low"
            if self.config.model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
            else None,
            max_completion_tokens=self.config.max_completion_tokens,
        )

    @property
    def model_info(self) -> RefinerModelInfo:
        return self._model_info

    def _completion(self, request, messages, repair):
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": messages,
            "max_completion_tokens": self.config.max_completion_tokens,
            "include_reasoning": False,
            "stream": False,
        }
        if self.config.model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            payload["reasoning_effort"] = "low"
        if self.config.structured_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "refiner_decisions_v1",
                    "strict": True,
                    "schema": decision_schema(request),
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
        calls = []
        for attempt in range(self.config.transport_retries + 1):
            _event(
                "request_started", utterance_id=request.target.utterance_id, schema_repair=repair
            )
            started = time.perf_counter()
            retry_hint = {}
            status, response = _exchange(body, self.config, retry_hint=retry_hint)
            call = ProviderCall(
                request.target.utterance_id,
                time.perf_counter() - started,
                len(body),
                len(response),
                status,
                repair,
            )
            calls.append(call)
            _event("request_completed", status_code=status, latency_seconds=call.latency_seconds)
            if status == 429 or 500 <= status <= 599:
                if attempt < self.config.transport_retries and not retry_hint.get("exceeds_limit"):
                    delay = min(
                        self.config.max_backoff_seconds,
                        max(retry_hint.get("seconds", 0), 2**attempt),
                    )
                    _event("transport_retry_scheduled", status_code=status, delay_seconds=delay)
                    time.sleep(delay)
                    continue
            if not 200 <= status < 300:
                if status == 400:
                    try:
                        error = json.loads(response).get("error", {})
                        if error.get("code") == "json_validate_failed":
                            invalid = error.get("failed_generation", "")
                            if isinstance(invalid, str):
                                return invalid[:32000], tuple(calls), True
                    except (ValueError, TypeError, AttributeError):
                        pass
                _status_error(status)
            try:
                data = json.loads(response)
                if (
                    not isinstance(data, dict)
                    or data.get("model") != self.config.model
                    or len(data["choices"]) != 1
                ):
                    raise ValueError
                choice = data["choices"][0]
                content = choice["message"]["content"]
                if (
                    choice["finish_reason"] != "stop"
                    or not isinstance(content, str)
                    or len(content) > 32000
                ):
                    raise ValueError
                usage = data.get("usage") or {}
                calls[-1] = replace(
                    call,
                    **{
                        name: usage.get(key)
                        for name, key in (
                            ("prompt_tokens", "prompt_tokens"),
                            ("completion_tokens", "completion_tokens"),
                            ("total_tokens", "total_tokens"),
                        )
                    },
                )
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                raise RefinementProviderError(
                    "Groq returned incomplete or inconsistent completion metadata."
                ) from exc
            return content, tuple(calls), False
        raise AssertionError("Unreachable bounded retry state")

    def refine(self, request: RefinementRequest) -> RefinementResponse:
        messages = build_messages(request, self.config)
        content, calls, provider_schema_failed = self._completion(request, messages, False)
        for attempt in range(self.config.schema_repair_retries + 1):
            try:
                if provider_schema_failed:
                    raise RefinementSchemaError("Provider rejected generated decision structure.")
                decisions = parse_decisions(content, request)
                return RefinementResponse(
                    request.target.utterance_id, decisions, self.model_info, calls
                )
            except RefinementSchemaError as exc:
                _event("response_validation_failed", utterance_id=request.target.utterance_id)
                if attempt >= self.config.schema_repair_retries:
                    raise RefinementSchemaError(
                        "Groq decisions remain invalid after bounded schema repair; no refined artifact produced."
                    ) from exc
                _event("schema_repair_attempted", utterance_id=request.target.utterance_id)
                # Original request and invalid response are data; this retries structure, not semantics.
                repair_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "schema_repair_only": True,
                                "instruction": "Repair JSON structure/references only. Do not reconsider semantics; use UNCERTAIN if an invalid decision cannot be represented safely.",
                                "untrusted_invalid_response": content,
                                "validation_error": str(exc),
                                "required_schema": decision_schema(request),
                            },
                            sort_keys=True,
                        ),
                    },
                ]
                content, new_calls, provider_schema_failed = self._completion(
                    request, repair_messages, True
                )
                calls += new_calls
        raise AssertionError("Unreachable bounded schema retry state")
