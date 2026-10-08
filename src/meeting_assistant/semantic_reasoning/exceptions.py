"""Sanitized Phase X errors; optional integration never invalidates Phase VI."""


class SemanticError(Exception):
    code = "semantic_error"


class InvalidSemantics(SemanticError):
    code = "invalid_semantics"


class SemanticSourceMismatch(SemanticError):
    code = "source_mismatch"


class ProviderUnavailable(SemanticError):
    code = "provider_unavailable"


class SemanticBudgetExceeded(SemanticError):
    code = "budget_exceeded"
