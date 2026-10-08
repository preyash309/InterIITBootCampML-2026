"""Phase X additive semantic observations; no canonical output mutation."""

from .backend import SemanticDecisionBackend
from .config import SemanticConfig
from .evidence import validate_semantic_source
from .local_backend import GLiNERBackend, JuliaBackend
from .models import SemanticResult
from .provider import JevBackend
from .serialization import from_json, save_semantics, to_json
from .service import analyze_meeting_semantics

__all__ = [
    "SemanticDecisionBackend",
    "SemanticConfig",
    "SemanticResult",
    "JevBackend",
    "JuliaBackend",
    "GLiNERBackend",
    "analyze_meeting_semantics",
    "validate_semantic_source",
    "from_json",
    "to_json",
    "save_semantics",
]
