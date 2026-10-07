"""Streaming HTTPS multipart requests using the standard library.

Endpoints are fixed official hosts. No redirects, automatic retries, file URLs,
raw response logging, or disabling TLS. Each request has an overall deadline.
"""

import http.client
import json
import logging
import socket
import time
from pathlib import Path
from uuid import uuid4

from .config import PROVIDERS, ASRConfig, TranscriptionOptions
from .exceptions import (
    ASRAuthenticationError,
    ASRConnectionError,
    ASRRateLimitError,
    ASRTimeout,
    ASRTranscriptionError,
    InvalidTranscript,
)

logger = logging.getLogger(__name__)
MAX_RESPONSE_BYTES = 20_000_000


def _multipart(
    config: ASRConfig, options: TranscriptionOptions, boundary: str
) -> tuple[bytes, bytes]:
    fields = [
        ("model", config.model),
        ("response_format", "verbose_json"),
        ("temperature", str(options.temperature)),
        ("timestamp_granularities[]", "segment"),
        ("timestamp_granularities[]", "word"),
    ]
    if options.language is not None:
        fields.append(("language", options.language))
    prefix = "".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        for name, value in fields
    )
    prefix += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        'filename="canonical.wav"\r\nContent-Type: audio/wav\r\n\r\n'
    )
    return prefix.encode("utf-8"), f"\r\n--{boundary}--\r\n".encode("ascii")


def _remaining(connection: http.client.HTTPSConnection, deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise ASRTimeout("Transcription request exceeded its configured timeout.")
    connection.timeout = remaining
    if connection.sock is not None:
        connection.sock.settimeout(remaining)


def _status_error(status: int) -> None:
    if status in (401, 403):
        raise ASRAuthenticationError(
            "Provider rejected the API key or account permissions.", status_code=status
        )
    if status == 429:
        raise ASRRateLimitError(
            "Provider rate limit or quota reached; check your account and retry later.",
            status_code=status,
        )
    if status == 413:
        raise ASRTranscriptionError(
            "Provider upload limit exceeded; reduce ASR_MAX_UPLOAD_SIZE.", status_code=status
        )
    raise ASRTranscriptionError(
        "Provider rejected transcription; check model access, account status, or service availability.",
        status_code=status,
    )


def request_transcription(
    audio_path: Path,
    config: ASRConfig,
    options: TranscriptionOptions,
    timeout_seconds: float,
) -> dict:
    host, endpoint, _ = PROVIDERS[config.provider]
    boundary = "meeting-" + uuid4().hex
    prefix, suffix = _multipart(config, options, boundary)
    connection = http.client.HTTPSConnection(host, timeout=timeout_seconds)
    deadline = time.perf_counter() + timeout_seconds
    try:
        # Open and fstat the same file descriptor before writing the request length.
        with audio_path.open("rb") as audio:
            import os

            size = os.fstat(audio.fileno()).st_size
            if size > config.max_upload_size_bytes:
                raise ASRTranscriptionError("ASR chunk exceeds the configured upload-size limit.")
            connection.putrequest("POST", endpoint)
            connection.putheader("Authorization", f"Bearer {config.api_key}")
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(len(prefix) + size + len(suffix)))
            connection.putheader("Accept", "application/json")
            connection.putheader("User-Agent", "inter-iit-meeting-assistant/0.2")
            _remaining(connection, deadline)
            connection.endheaders()
            connection.send(prefix)
            remaining = size
            while remaining:
                block = audio.read(min(65_536, remaining))
                if not block:
                    raise ASRTranscriptionError("Audio chunk changed while uploading.")
                _remaining(connection, deadline)
                connection.send(block)
                remaining -= len(block)
            _remaining(connection, deadline)
            connection.send(suffix)
        _remaining(connection, deadline)
        response = connection.getresponse()
        logger.info(
            "asr_api_response",
            extra={
                "event": "asr_api_response",
                "provider": config.provider,
                "status": response.status,
            },
        )
        if not 200 <= response.status < 300:
            _status_error(response.status)
        body = bytearray()
        while True:
            _remaining(connection, deadline)
            block = response.read1(min(65_536, MAX_RESPONSE_BYTES + 1 - len(body)))
            if not block:
                break
            body.extend(block)
            if len(body) > MAX_RESPONSE_BYTES:
                raise InvalidTranscript("Provider response exceeds the safe response-size limit.")
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, ValueError) as exc:
            raise InvalidTranscript("Provider returned malformed transcription JSON.") from exc
        if not isinstance(payload, dict):
            raise InvalidTranscript("Provider transcription response must be a JSON object.")
        return payload
    except (TimeoutError, socket.timeout) as exc:
        raise ASRTimeout(
            "Transcription request timed out; retry or increase ASR_REQUEST_TIMEOUT."
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ASRConnectionError("Cannot contact the transcription provider over HTTPS.") from exc
    finally:
        connection.close()
