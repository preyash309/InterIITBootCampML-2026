from uuid import uuid4

from meeting_assistant.asr.models import (
    ModelInfo,
    ProcessingInfo,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)
from meeting_assistant.diarization.config import DiarizationOptions
from meeting_assistant.diarization.models import (
    DiarizationModelInfo,
    DiarizationProcessingInfo,
    DiarizationResult,
    SpeakerTurn,
)


def raw(words=(), *, duration=10, segments=None):
    if segments is None:
        segments = (
            (TranscriptSegment("seg_000001", 0, duration, " ".join(w.text for w in words), words),)
            if words
            else ()
        )
    return TranscriptResult(
        str(uuid4()),
        "en",
        None,
        duration,
        " ".join(segment.text for segment in segments),
        tuple(segments),
        ModelInfo("fixture", "mock"),
        ProcessingInfo(0, 0, 0, 1),
    )


def word(text="word", start=1, end=1.5):
    return TranscriptWord(text, start, end, 0.8)


def diar(exclusive=(("a", 0, 10),), *, regular=None, duration=10):
    regular = exclusive if regular is None else regular
    ordered = sorted((*regular, *exclusive), key=lambda turn: (turn[1], turn[2], turn[0]))
    labels = dict.fromkeys(turn[0] for turn in ordered)
    mapping = {name: f"SPEAKER_{index:02d}" for index, name in enumerate(labels)}

    def turns(values, kind):
        return tuple(
            SpeakerTurn(f"{kind}_turn_{i:06d}", mapping[name], start, end)
            for i, (name, start, end) in enumerate(values, 1)
        )

    return DiarizationResult(
        str(uuid4()),
        duration,
        "a" * 64,
        tuple(mapping.values()),
        turns(regular, "regular"),
        turns(exclusive, "exclusive"),
        DiarizationModelInfo("pyannote/speaker-diarization-community-1", "3" * 40, "4.0.7", "cpu"),
        DiarizationProcessingInfo(0, 0, 0),
        DiarizationOptions(),
    )
