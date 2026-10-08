import wave
from dataclasses import replace

from meeting_assistant.asr.models import TranscriptSegment, TranscriptWord
from meeting_assistant.contextual_asr import build_meeting_context
from meeting_assistant.refinement.evaluate import controlled_evidence


def evidence(text="Use Drant for vector search.", span="Drant", canonical="Qdrant", **values):
    context = build_meeting_context(terms=[canonical], known_entries=())
    source, grounding = controlled_evidence(
        text, [{"span": span, "candidates": [{"canonical": canonical, **values}]}]
    )
    records = tuple(
        replace(
            r,
            candidates=tuple(
                replace(
                    c,
                    entry_id=context.terms[0].id,
                    scope="meeting",
                    reasons=c.reasons
                    + tuple(
                        "context_source:" + identity for identity in context.terms[0].source_ids
                    ),
                )
                for c in r.candidates
            ),
        )
        for r in grounding.records
    )
    return context, source, replace(grounding, records=records)


def wav(path, duration):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * round(duration * 16000))


class FakeASR:
    provider = "fixture"
    model = "controlled-context"

    def __init__(self, text="Use Qdrant for vector search.", error=None):
        self.text, self.error = text, error
        self.calls = []

    def transcribe_window(self, path, *, start, duration, prompt, timeout):
        self.calls.append((path, start, duration, prompt, timeout))
        if self.error:
            raise self.error
        words = tuple(
            TranscriptWord(word, start + i * 0.3, start + (i + 1) * 0.3, None)
            for i, word in enumerate(self.text.split())
        )
        return self.text, (TranscriptSegment("seg_000001", start, words[-1].end, self.text, words),)
