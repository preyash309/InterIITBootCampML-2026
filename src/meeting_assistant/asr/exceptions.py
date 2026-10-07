"""Sanitized, stable ASR failures; providers' response bodies are never exposed."""


class ASRError(Exception):
    code = "asr_error"

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ASRConfigurationError(ASRError):
    code = "asr_configuration_error"


class ASRAuthenticationError(ASRError):
    code = "asr_authentication_error"


class InvalidASRAudio(ASRError):
    code = "invalid_asr_audio"


class ASRRateLimitError(ASRError):
    code = "asr_rate_limit"


class ASRTimeout(ASRError):
    code = "asr_timeout"


class ASRConnectionError(ASRError):
    code = "asr_connection_error"


class ASRTranscriptionError(ASRError):
    code = "asr_transcription_error"


class InvalidTranscript(ASRError):
    code = "invalid_transcript"


class TranscriptWriteError(ASRError):
    code = "transcript_write_error"
