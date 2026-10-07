"""Sanitized Phase V failures. Provider bodies and credentials are never messages."""


class RefinementError(Exception):
    code = "refinement_error"


class RefinementConfigurationError(RefinementError):
    code = "refinement_configuration_error"


class RefinementSourceMismatch(RefinementError):
    code = "refinement_source_mismatch"


class RefinementSchemaError(RefinementError):
    code = "refinement_schema_error"


class RefinementValidationError(RefinementError):
    code = "refinement_validation_error"


class RefinementProviderError(RefinementError):
    code = "refinement_provider_error"

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RefinementAuthenticationError(RefinementProviderError):
    code = "refinement_authentication_error"


class RefinementRateLimitError(RefinementProviderError):
    code = "refinement_rate_limit_error"


class RefinementTimeoutError(RefinementProviderError):
    code = "refinement_timeout_error"


class RefinementWriteError(RefinementError):
    code = "refinement_write_error"
