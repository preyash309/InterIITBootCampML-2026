"""Recall-first utterance candidates; bounded lexical/adjacency relation retrieval."""

import re

STOP_WORDS = frozenset(
    "a an the we i you it this that to of for on in is are be and or will could should would do not no yes okay actually let lets by someone both our with if then".split()
)


def terms(text):
    return frozenset(re.findall(r"[\w]+", text.casefold())) - STOP_WORDS


def overlap(a, b):
    left, right = terms(a), terms(b)
    return len(left & right) / max(1, len(left | right))


def event_candidates(refined, config):
    # No cue filter excludes implicit acceptance, questions or blockers.
    # Whole utterances stay exact; over-budget spans are reported by the service.
    return tuple(u for u in refined.utterances if u.refined_text.strip())[: config.max_event_calls]


def relation_candidates(events, config):
    accepted = tuple(e for e in events if e.outcome == "accepted")
    pairs = []
    for index, b in enumerate(accepted):
        for a in accepted[max(0, index - config.relation_window) : index]:
            if max(0, b.start - a.end) > config.relation_gap_seconds:
                continue
            shared = set(a.evidence_utterance_ids) & set(b.evidence_utterance_ids)
            adjacent = index > 0 and a.id == accepted[index - 1].id
            if shared or adjacent or overlap(a.text, b.text) >= 0.15:
                pairs.append((a, b))
    return tuple(pairs[: config.max_relation_calls]), len(pairs)
