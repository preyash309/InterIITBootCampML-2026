"""Phase VI: claims plus immutable evidence IDs; no Phase VII semantic verifier."""

from .base import MeetingIntelligenceBackend
from .config import IntelligenceConfig
from .evidence import extract_evidence_clip, get_item_evidence, resolve_meeting_record_evidence
from .groq_backend import GroqMeetingIntelligenceBackend
from .models import ActionItem, ActionOwner, EvidenceSpan, MeetingRecord
from .serialization import (
    meeting_record_from_json,
    meeting_record_to_json,
    render_meeting_markdown,
    save_meeting_record,
)
from .service import extract_meeting_record

__all__ = [
    "MeetingIntelligenceBackend",
    "IntelligenceConfig",
    "GroqMeetingIntelligenceBackend",
    "MeetingRecord",
    "ActionItem",
    "ActionOwner",
    "EvidenceSpan",
    "extract_meeting_record",
    "resolve_meeting_record_evidence",
    "get_item_evidence",
    "extract_evidence_clip",
    "meeting_record_from_json",
    "meeting_record_to_json",
    "render_meeting_markdown",
    "save_meeting_record",
]
