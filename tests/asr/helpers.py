"""Provider-shaped synthetic data, never represented as a real ASR recording."""

from meeting_assistant.asr.api_backend import parse_response
from meeting_assistant.asr.models import ModelInfo, ProcessingInfo, TranscriptResult


def response(duration: float = 0.5, *, text: str = " Hello, world.") -> dict:
    return {
        "language": "English",
        "duration": duration,
        "text": text,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": duration,
                "text": text,
                "avg_logprob": -0.15,
                "no_speech_prob": 0.01,
                "compression_ratio": 1.2,
            }
        ],
        "words": [
            {"word": "Hello,", "start": 0.0, "end": duration / 2},
            {"word": "world.", "start": duration / 2, "end": duration},
        ],
    }


def transcript() -> TranscriptResult:
    text, language, confidence, segments = parse_response(
        response(), offset=0, chunk_duration=0.5, segment_offset=0
    )
    return TranscriptResult(
        "9daa49e0-ea65-4e8e-8c9e-42672f16ec69",
        language,
        confidence,
        0.5,
        text,
        segments,
        ModelInfo("groq", "whisper-large-v3"),
        ProcessingInfo(0.0, 0.1, 0.2, 1),
    )


class FakeBackend:
    def transcribe(self, audio_path, options=None):
        return transcript()
