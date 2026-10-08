"""Phase IX optional independent speaker-reliability sidecars."""

from .backend import SecondaryDiarizationBackend
from .config import SpeakerReliabilityConfig
from .models import SecondaryDiarizationResult, SecondarySpeakerSegment, SpeakerReliabilityResult
from .serialization import reliability_from_json, save_speaker_reliability
from .service import assess_speaker_reliability, compare_diarization

__all__ = [
    "SecondaryDiarizationBackend",
    "SpeakerReliabilityConfig",
    "SecondaryDiarizationResult",
    "SecondarySpeakerSegment",
    "SpeakerReliabilityResult",
    "assess_speaker_reliability",
    "compare_diarization",
    "save_speaker_reliability",
    "reliability_from_json",
]
