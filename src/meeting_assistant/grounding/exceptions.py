"""Application-facing failures for candidate retrieval only."""


class GroundingError(Exception):
    code = "grounding_error"


class GroundingConfigurationError(GroundingError):
    code = "configuration_error"


class InvalidGlossary(GroundingError):
    code = "invalid_glossary"


class EmbeddingModelUnavailable(GroundingError):
    code = "embedding_model_unavailable"


class EmbeddingError(GroundingError):
    code = "embedding_error"


class InvalidGroundingResult(GroundingError):
    code = "invalid_grounding_result"


class GroundingWriteError(GroundingError):
    code = "grounding_write_error"
