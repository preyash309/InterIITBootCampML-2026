"""Shared bounded HTTPS exchange for distinct refinement/intelligence roles."""

import http.client
import socket
import time

HOST = "api.groq.com"
ENDPOINT = "/openai/v1/chat/completions"
MAX_RESPONSE_BYTES = 1_000_000


def exchange_json(
    body: bytes, config, *, retry_hint: dict | None = None, timeout_error, provider_error
) -> tuple[int, bytes]:
    connection = http.client.HTTPSConnection(HOST, timeout=config.request_timeout_seconds)
    deadline = time.perf_counter() + config.request_timeout_seconds

    def remaining():
        seconds = deadline - time.perf_counter()
        if seconds <= 0:
            raise timeout_error("Groq request exceeded its deadline.")
        connection.timeout = seconds
        if connection.sock is not None:
            connection.sock.settimeout(seconds)

    try:
        remaining()
        connection.request(
            "POST",
            ENDPOINT,
            body=body,
            headers={
                "Authorization": "Bearer " + config.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "inter-iit-meeting-assistant/0.5",
            },
        )
        remaining()
        response = connection.getresponse()
        if retry_hint is not None:
            try:
                seconds = float(response.getheader("Retry-After", "0"))
                if 0 < seconds <= config.max_backoff_seconds:
                    retry_hint["seconds"] = seconds
                elif seconds > config.max_backoff_seconds:
                    retry_hint["exceeds_limit"] = True
            except (ValueError, TypeError):
                pass
        result = bytearray()
        while True:
            remaining()
            block = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(result)))
            if not block:
                break
            result.extend(block)
            if len(result) > MAX_RESPONSE_BYTES:
                raise provider_error("Groq response exceeded the safe size limit.")
        return response.status, bytes(result)
    except (TimeoutError, socket.timeout) as exc:
        raise timeout_error("Groq request timed out.") from exc
    except (OSError, http.client.HTTPException) as exc:
        # Connection failures are not automatically retried: server execution is ambiguous.
        raise provider_error("Cannot contact Groq over verified HTTPS.") from exc
    finally:
        connection.close()
