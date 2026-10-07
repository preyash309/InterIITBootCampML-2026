"""Sanitized stage-specific failures; original causes stay available to debuggers."""


class IntelligenceError(Exception):
    code = "intelligence_error"


class IntelligenceConfigurationError(IntelligenceError):
    code = "configuration_error"


class IntelligenceSourceMismatch(IntelligenceError):
    code = "source_mismatch"


class IntelligenceValidationError(IntelligenceError):
    code = "validation_error"


class IntelligenceSchemaError(IntelligenceValidationError):
    code = "schema_error"


class UnknownEvidenceError(IntelligenceSchemaError):
    code = "unknown_evidence"


class IntelligenceProviderError(IntelligenceError):
    code = "provider_error"

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class IntelligenceAuthenticationError(IntelligenceProviderError):
    code = "authentication_error"


class IntelligenceRateLimitError(IntelligenceProviderError):
    code = "rate_limit"


class IntelligenceTimeoutError(IntelligenceProviderError):
    code = "timeout"


class MeetingTooLargeError(IntelligenceError):
    code = "meeting_too_large"


class ConsolidationError(IntelligenceSchemaError):
    code = "consolidation_error"


class EvidenceAudioError(IntelligenceError):
    code = "evidence_audio_error"


class IntelligenceWriteError(IntelligenceError):
    code = "publication_error"
