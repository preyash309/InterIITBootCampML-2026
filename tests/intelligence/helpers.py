import json

from meeting_assistant.intelligence.fixtures import text_evidence
from meeting_assistant.intelligence.models import (
    IntelligenceModelInfo,
    IntelligenceResponse,
    MeetingContent,
    SummaryPoint,
)


def evidence(text="I'll benchmark both models by Friday."):
    return text_evidence([{"text": text}])


def blank():
    return {name: [] for name in ("summary", "minutes", "decisions", "action_items")}


def payload(section="summary", **changes):
    data = blank()
    item = {"text": "Benchmark both models.", "evidence_utterance_ids": ["utt_000001"]}
    if section == "minutes":
        item.update(topic="Benchmarking", kind="discussion")
    if section == "action_items":
        item = {
            "task": "Benchmark both models.",
            "owner": None,
            "deadline_text": None,
            "evidence_utterance_ids": ["utt_000001"],
        }
    item.update(changes)
    data[section].append(item)
    return json.dumps(data)


def completion(content, **changes):
    data = {
        "model": "openai/gpt-oss-120b",
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
    }
    data.update(changes)
    return 200, json.dumps(data).encode()


class FakeBackend:
    model_info = IntelligenceModelInfo("fixture", "controlled")

    def __init__(self, handler=None):
        self.requests = []
        self.handler = handler

    def extract(self, request):
        self.requests.append(request)
        if self.handler:
            content = self.handler(request)
        elif request.stage == "consolidation":
            content = request.partial
        else:
            content = MeetingContent(
                summary=tuple(
                    SummaryPoint(f"sum_{n:04d}", u.refined_text, (u.utterance_id,))
                    for n, u in enumerate(request.utterances, 1)
                )
            )
        return IntelligenceResponse(request.request_id, content, self.model_info)
