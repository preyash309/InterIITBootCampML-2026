import json

from meeting_assistant.refinement.config import RefinementConfig
from meeting_assistant.refinement.evaluate import controlled_evidence
from meeting_assistant.refinement.models import (
    RefinementDecision,
    RefinementResponse,
    RefinerModelInfo,
)


def evidence(
    text="Use cue drant for vector search.", span="cue drant", canonical="Qdrant", **candidate
):
    return controlled_evidence(
        text,
        [
            {
                "span": span,
                "candidates": [
                    {"canonical": canonical, "match_type": "asr_alias_exact", **candidate}
                ],
            }
        ],
    )


class FakeBackend:
    model_info = RefinerModelInfo("fixture", "controlled")
    config = RefinementConfig(api_key=None)

    def __init__(self, action="KEEP", candidate_index=0):
        self.action = action
        self.candidate_index = candidate_index
        self.requests = []

    def refine(self, request):
        self.requests.append(request)
        return RefinementResponse(
            request.target.utterance_id,
            tuple(
                RefinementDecision(
                    r.id,
                    self.action,
                    r.candidates[self.candidate_index].entry_id
                    if self.action == "REPLACE"
                    else None,
                    "strong_phonetic_context"
                    if self.action == "REPLACE"
                    else "context_insufficient"
                    if self.action == "UNCERTAIN"
                    else "original_supported",
                )
                for r in request.records
            ),
            self.model_info,
        )


def decision_json(request, action="KEEP", **changes):
    decision = {
        "grounding_record_id": request.records[0].id,
        "action": action,
        "candidate_entry_id": request.records[0].candidates[0].entry_id
        if action == "REPLACE"
        else None,
        "reason_code": "strong_phonetic_context" if action == "REPLACE" else "original_supported",
    }
    decision.update(changes)
    return json.dumps({"utterance_id": request.target.utterance_id, "decisions": [decision]})


def completion(content, **changes):
    data = {
        "model": "openai/gpt-oss-120b",
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
    }
    data.update(changes)
    return 200, json.dumps(data).encode()
