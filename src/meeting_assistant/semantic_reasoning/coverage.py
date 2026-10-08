"""Inverse coverage observations, with bounded same-event judgments for paraphrases."""

from .candidates import overlap
from .models import CoverageObservation
from .ontology import HIGH_VALUE
from .verification import question


def _expected(event, item, record):
    if event.event_type == "DECISION":
        return item in record.decisions
    if event.event_type in ("COMMITMENT", "TASK_ASSIGNMENT"):
        return item in record.action_items
    return item in record.summary or item in record.minutes and item.kind == "concern"


def check_coverage(events, record, session):
    observations = []
    for event in events:
        if event.outcome != "accepted" or event.event_type not in HIGH_VALUE:
            continue
        candidates = []
        for item in record.content.items:
            shared = set(item.evidence_utterance_ids) & set(event.evidence_utterance_ids)
            similarity = overlap(event.text, item.text)
            if shared or similarity >= 0.20:
                candidates.append(
                    (
                        not _expected(event, item, record),
                        not bool(shared),
                        -similarity,
                        item.id,
                        item,
                    )
                )
        candidates.sort(key=lambda c: c[:4])
        truncated = len(candidates) > session.config.max_coverage_candidates
        candidates = candidates[: session.config.max_coverage_candidates]
        matches, decisions, warnings = [], [], []
        uncertain, unavailable = False, False
        for wrong_type, _, _, _, item in candidates:
            output = session.ask(
                {
                    "event": {
                        "type": event.event_type,
                        "text": event.text,
                        "evidence_ids": event.evidence_utterance_ids,
                    },
                    "record_item": {
                        "type": "decision"
                        if item in record.decisions
                        else "action"
                        if item in record.action_items
                        else "other",
                        "text": item.text,
                        "evidence_ids": item.evidence_utterance_ids,
                    },
                },
                (
                    question(
                        "captures_event",
                        "Does `record_item` capture the SAME explicit event as `event`, "
                        "preserving scope, negation and numbers? Similar terminology alone is insufficient.",
                    ),
                ),
                "coverage",
                "coverage_v1",
            )
            if output is None:
                unavailable = True
                continue
            d = output[0]
            decisions.append(d)
            if (
                d.choice == "YES"
                and d.selected_probability >= session.config.acceptance_probability
            ):
                if wrong_type:
                    warnings.append("wrong_record_type:" + item.id)
                    uncertain = True
                else:
                    matches.append(item.id)
            elif (
                d.choice == "AMBIGUOUS"
                or d.selected_probability < session.config.acceptance_probability
            ):
                uncertain = True
        if len(matches) > 1:
            warnings.append("multiple_record_matches_review_duplicates")
        if truncated:
            warnings.append("coverage_candidates_truncated")
        status = (
            "covered"
            if matches
            else "unavailable"
            if unavailable
            else ("ambiguous" if uncertain or truncated else "missing")
        )
        observations.append(
            CoverageObservation(
                event.id,
                event.event_type,
                tuple(matches),
                status,
                tuple(decisions),
                tuple(warnings),
            )
        )
    return tuple(observations)
