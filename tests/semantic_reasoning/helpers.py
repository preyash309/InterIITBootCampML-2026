"""Controlled typed responses test software behavior, never live Jev accuracy."""

from meeting_assistant.intelligence.fixtures import text_evidence
from meeting_assistant.intelligence.models import MeetingContent
from meeting_assistant.intelligence.service import extract_meeting_record
from meeting_assistant.semantic_reasoning.models import ProviderCall
from meeting_assistant.semantic_reasoning.provider import parse_response
from meeting_assistant.semantic_reasoning.serialization import fingerprint
from tests.intelligence.helpers import FakeBackend as ExtractionBackend


def evidence(texts, content=None):
    refined, speaker, grounding = text_evidence(
        [{"text": t, "speaker_id": f"SPEAKER_{n % 2:02d}"} for n, t in enumerate(texts)]
    )
    record = extract_meeting_record(
        refined,
        speaker,
        grounding,
        backend=ExtractionBackend(lambda _: content or MeetingContent()),
    )
    return refined, speaker, record


class FakeBackend:
    provider = "fake"
    model = "fixture"

    def __init__(self, handler=None):
        self.calls, self.requests = [], []
        self.handler = handler or (
            lambda state, q, purpose: "NONE" if purpose in ("events", "relations") else "YES"
        )

    def decide(self, state, questions, *, purpose, policy, timeout_seconds, max_attempts):
        self.requests.append((state, questions, purpose))
        payload = {
            "model": self.model,
            "answers": {},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        for q in questions:
            answer = self.handler(state, q, purpose)
            label, probability = answer if isinstance(answer, tuple) else (answer, 1.0)
            options = dict(q.criteria)
            probabilities = {
                k: probability if k == label else (1 - probability) / (len(options) - 1)
                for k in options
            }
            payload["answers"][q.id] = {
                "type": "choice",
                "choice": label,
                "probabilities": probabilities,
                "confidence": (probability - 1 / len(options)) / (1 - 1 / len(options)),
            }
        request_sha, response_sha = fingerprint((state, questions)), fingerprint(payload)
        self.calls.append(ProviderCall(purpose, 200, 0.001, False, request_sha, response_sha, 1, 1))
        return parse_response(
            payload,
            questions,
            model=self.model,
            request_sha=request_sha,
            response_sha=response_sha,
            latency=0.001,
            policy=policy,
            provider=self.provider,
        )


def scripted(events, relations=None, verification=None, coverage="YES"):
    def handler(state, q, purpose):
        if purpose == "events":
            return events[int(state["target_id"].split("_")[1]) - 1]
        if purpose == "relations":
            return (relations or {}).get(
                (state["event_a"]["text"], state["event_b"]["text"]), "NONE"
            )
        if purpose == "verification":
            return (verification or {}).get(q.id, "YES")
        return coverage

    return FakeBackend(handler)
