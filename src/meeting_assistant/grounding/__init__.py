"""Phase IV candidate evidence; no editing, LLM calls, or identity inference."""

from .config import GroundingConfig
from .exceptions import GroundingError
from .glossary import build_glossary, load_glossary
from .models import GlossaryEntry, GroundingCandidate, GroundingRecord, GroundingResult
from .retrieval import GroundingRetriever
from .service import ground_span, ground_transcript, validate_grounding_source

__all__ = [
    "GroundingConfig",
    "GroundingError",
    "GlossaryEntry",
    "GroundingCandidate",
    "GroundingRecord",
    "GroundingResult",
    "GroundingRetriever",
    "build_glossary",
    "load_glossary",
    "ground_span",
    "ground_transcript",
    "validate_grounding_source",
]
