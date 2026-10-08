"""Separate window operation reusing Phase II's HTTPS transport and strict parser."""

from pathlib import Path
from typing import Protocol

from meeting_assistant.asr.api_backend import parse_response
from meeting_assistant.asr.config import ASRConfig
from meeting_assistant.asr.exceptions import ASRAuthenticationError
from meeting_assistant.asr.models import TranscriptSegment
from meeting_assistant.asr.transport import request_transcription


class ContextualASRBackend(Protocol):
    provider: str
    model: str

    def transcribe_window(
        self, path: Path, *, start: float, duration: float, prompt: str, timeout: float
    ) -> tuple[str, tuple[TranscriptSegment, ...]]: ...


class GroqContextualASRBackend:
    def __init__(self, config=None):
        self.config = config or ASRConfig.from_env()
        if self.config.provider != "groq":
            from .exceptions import ContextualConfigurationError

            raise ContextualConfigurationError("Phase VIII currently supports Groq Whisper only.")
        if not self.config.api_key:
            raise ASRAuthenticationError("Set GROQ_API_KEY for contextual transcription.")
        self.provider, self.model = self.config.provider, self.config.model

    def transcribe_window(self, path, *, start, duration, prompt, timeout):
        payload = request_transcription(
            path,
            self.config,
            self.config.options,
            min(timeout, self.config.request_timeout_seconds),
            prompt=prompt,
        )
        text, _, _, segments = parse_response(
            payload, offset=start, chunk_duration=duration, segment_offset=0
        )
        return text, segments
