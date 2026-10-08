"""Optional contextual ASR: hints and alternative evidence, never direct transcript edits."""

from .config import ContextualASRConfig
from .context import build_meeting_context
from .models import ContextTerm, ContextualASRResult, MeetingContextPack
from .serialization import read_context, save_contextual_asr
from .service import contextual_retranscribe
from .transcribe import ContextualASRBackend, GroqContextualASRBackend
from .workflow import process_meeting

__all__ = [
    "ContextualASRConfig",
    "MeetingContextPack",
    "ContextTerm",
    "ContextualASRResult",
    "build_meeting_context",
    "contextual_retranscribe",
    "read_context",
    "save_contextual_asr",
    "ContextualASRBackend",
    "GroqContextualASRBackend",
    "process_meeting",
]
