"""Sanitized errors for optional contextual evidence, never provider response bodies."""


class ContextualASRError(Exception):
    code = "contextual_asr_error"


class InvalidContext(ContextualASRError):
    code = "invalid_context"


class InvalidContextualEvidence(ContextualASRError):
    code = "invalid_contextual_evidence"


class ContextualConfigurationError(ContextualASRError):
    code = "contextual_configuration_error"
