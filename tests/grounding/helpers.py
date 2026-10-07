import numpy as np

from meeting_assistant.diarization.reconciliation import reconcile_transcript
from meeting_assistant.grounding.models import GlossaryEntry
from tests.diarization.helpers import diar, raw, word


def entry(canonical="Qdrant", **kwargs):
    data = dict(
        id="global.qdrant",
        canonical=canonical,
        category="vector_search",
        domain="retrieval",
        description="A vector database for embedding similarity search.",
        asr_aliases=("cue drant",),
    )
    data.update(kwargs)
    return GlossaryEntry(**data)


class StubEmbedding:
    model = "controlled-test"
    revision = "fixture-v1"
    dimension = 4

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(tuple(texts))
        return np.asarray([[1, 0, 0, 0] for _ in texts], dtype=np.float32).reshape(len(texts), 4)


def speaker(text="Use cue drant for vector search."):
    words = tuple(word(value, i * 0.3, (i + 1) * 0.3) for i, value in enumerate(text.split()))
    duration = max(10, len(words) * 0.3 + 1)
    return reconcile_transcript(
        raw(words, duration=duration), diar(exclusive=(("a", 0, duration),), duration=duration)
    )
