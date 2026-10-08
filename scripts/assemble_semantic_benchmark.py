"""Authored annotations reuse Phase VI text; no scores or model outputs fabricated."""

import json
from pathlib import Path


def assemble():
    previous = {
        c["id"]: c for c in json.loads(Path("data/intelligence_benchmark.json").read_text())
    }
    definitions = [
        ("proposal_only", None, ["PROPOSAL"], [], [], [], []),
        ("accepted_decision", None, ["PROPOSAL", "DECISION"], [(1, 2, "ACCEPTS")], [2], [], []),
        ("rejected_proposal", None, ["PROPOSAL", "DECISION"], [(1, 2, "REJECTS")], [2], [], []),
        ("named_assignment", None, ["TASK_ASSIGNMENT"], [], [], [], [(1, "Rahul", "Friday")]),
        ("self_commitment", None, ["COMMITMENT"], [], [], [], [(1, "SPEAKER_00", None)]),
        ("someone_should", None, ["PROPOSAL"], [], [], [], []),
        ("ambiguous_named_request", None, ["PROPOSAL"], [], [], [], []),
        ("explicit_deadline", None, ["COMMITMENT"], [], [], [], [(1, "SPEAKER_00", "Friday")]),
        (
            "shipping_reversal",
            ["We'll ship Friday.", "Actually, move shipping to Monday instead."],
            ["DECISION", "DECISION"],
            [(1, 2, "SUPERSEDES")],
            [2],
            [1],
            [],
        ),
        (
            "question_answer",
            ["What is the latency?", "The latency is 20 milliseconds."],
            ["QUESTION", "ANSWER"],
            [(1, 2, "ANSWERS")],
            [],
            [],
            [],
        ),
        (
            "blocker",
            ["The deployment is blocked until credentials arrive."],
            ["BLOCKER"],
            [],
            [],
            [],
            [],
        ),
        (
            "clarification",
            ["The benchmark measures latency.", "By latency, I mean the p95 response time."],
            ["INFORMATION", "CLARIFICATION"],
            [(1, 2, "CLARIFIES")],
            [],
            [],
            [],
        ),
        (
            "interleaved_topics",
            [
                "We could use Redis for the cache.",
                "We'll ship Friday.",
                "Agreed, use Redis for the cache.",
                "Actually, move shipping to Monday instead.",
            ],
            ["PROPOSAL", "DECISION", "DECISION", "DECISION"],
            [(1, 3, "ACCEPTS"), (2, 4, "SUPERSEDES")],
            [3, 4],
            [2],
            [],
        ),
        (
            "repeated_decision",
            ["Use Redis for the cache.", "Yes, use Redis for the cache."],
            ["DECISION", "DECISION"],
            [(1, 2, "SUPPORTS")],
            [1, 2],
            [],
            [],
        ),
        (
            "unresolved_proposal",
            ["Maybe we could deploy Monday.", "We need more evidence first."],
            ["PROPOSAL", "OBJECTION"],
            [(1, 2, "CONTRADICTS")],
            [],
            [],
            [],
        ),
        ("ambiguous", ["That could be it."], ["AMBIGUOUS"], [], [], [], []),
        ("negation", ["Do not deploy this service."], ["DECISION"], [], [1], [], []),
        ("conditional", ["If latency improves, we'll deploy it."], ["PROPOSAL"], [], [], [], []),
        (
            "decision_to_task",
            ["Agreed, use Redis for the cache.", "Rahul, benchmark Redis for the cache by Friday."],
            ["DECISION", "TASK_ASSIGNMENT"],
            [(1, 2, "ASSIGNS")],
            [1],
            [],
            [(2, "Rahul", "Friday")],
        ),
        (
            "support_not_acceptance",
            ["We could use Redis.", "I like the idea, but we haven't decided."],
            ["PROPOSAL", "SUPPORT"],
            [(1, 2, "SUPPORTS")],
            [],
            [],
            [],
        ),
        (
            "explicit_rejection",
            ["We could migrate to MongoDB.", "No, reject the MongoDB migration."],
            ["PROPOSAL", "REJECTION"],
            [(1, 2, "REJECTS")],
            [],
            [],
            [],
        ),
    ]
    cases = []

    def uid(n):
        return f"utt_{n:06d}"

    for cid, texts, labels, edges, current, historical, tasks in definitions:
        turns = (
            previous[cid]["turns"]
            if texts is None
            else [
                {"text": text, "speaker_id": f"SPEAKER_{n % 2:02d}"} for n, text in enumerate(texts)
            ]
        )
        turns = [
            {"id": uid(n), "speaker_id": t.get("speaker_id", "SPEAKER_00"), "text": t["text"]}
            for n, t in enumerate(turns, 1)
        ]
        cases.append(
            {
                "id": cid,
                "origin": "Phase VI reused" if texts is None else "Project-authored",
                "turns": turns,
                "events": [
                    {"utterance_id": t["id"], "type": label}
                    for t, label in zip(turns, labels, strict=True)
                ],
                "relations": [
                    {"source_utterance_id": uid(a), "target_utterance_id": uid(b), "type": label}
                    for a, b, label in edges
                ],
                "current_decision_utterance_ids": [uid(n) for n in current],
                "historical_decision_utterance_ids": [uid(n) for n in historical],
                "tasks": [
                    {"evidence_ids": [uid(n)], "owner": owner, "deadline": deadline}
                    for n, owner, deadline in tasks
                ],
                "coverage_reference": None,
            }
        )
    challenges = [
        (
            "proposal_as_decision",
            "We could use Redis.",
            "Use Redis.",
            "decision",
            None,
            None,
            {"claim_supported": "NO", "item_type_supported": "NO"},
        ),
        (
            "rejected_as_decision",
            "No, reject Redis.",
            "Use Redis.",
            "decision",
            None,
            None,
            {"claim_supported": "NO", "item_type_supported": "NO", "negation_preserved": "NO"},
        ),
        (
            "valid_decision",
            "Agreed, use Redis.",
            "Use Redis.",
            "decision",
            None,
            None,
            {"claim_supported": "YES", "item_type_supported": "YES"},
        ),
        (
            "suggestion_as_task",
            "Someone should benchmark. This is only a suggestion.",
            "Benchmark.",
            "action",
            None,
            None,
            {"claim_supported": "NO", "item_type_supported": "NO"},
        ),
        (
            "valid_task",
            "Rahul, benchmark both models by Friday.",
            "Benchmark both models.",
            "action",
            "Rahul",
            "Friday",
            {
                "claim_supported": "YES",
                "item_type_supported": "YES",
                "owner_explicit": "YES",
                "deadline_explicit": "YES",
            },
        ),
        (
            "wrong_owner",
            "Rahul suggested that Priya benchmark both models. Priya agreed to do it.",
            "Benchmark both models.",
            "action",
            "Rahul",
            None,
            {"owner_explicit": "NO"},
        ),
        (
            "wrong_deadline",
            "I will send the report Friday, not Monday.",
            "Send the report.",
            "action",
            "SPEAKER_00",
            "Monday",
            {"deadline_explicit": "NO"},
        ),
        (
            "negation_lost",
            "Do not deploy Redis.",
            "Deploy Redis.",
            "decision",
            None,
            None,
            {"claim_supported": "NO", "negation_preserved": "NO"},
        ),
        (
            "number_changed",
            "Keep the discount at 15%, not 50%.",
            "Keep the discount at 50%.",
            "decision",
            None,
            None,
            {"claim_supported": "NO", "numbers_preserved": "NO"},
        ),
        (
            "conditional_as_final",
            "If latency improves, we'll deploy it.",
            "Deploy it.",
            "decision",
            None,
            None,
            {"item_type_supported": "NO", "scope_preserved": "NO"},
        ),
        (
            "commitment",
            "I'll send the report tomorrow.",
            "Send the report.",
            "action",
            "SPEAKER_00",
            "tomorrow",
            {"item_type_supported": "YES", "owner_explicit": "YES", "deadline_explicit": "YES"},
        ),
    ]
    result = {
        "version": "semantic_benchmark_v1",
        "origin": "Authored intended semantics; not a human-meeting corpus",
        "cases": cases,
        "verification_challenges": [
            dict(
                zip(
                    ("id", "evidence", "claim", "kind", "owner", "deadline", "dimensions"),
                    c,
                    strict=True,
                )
            )
            for c in challenges
        ],
    }
    Path("data/semantic_benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    assemble()
