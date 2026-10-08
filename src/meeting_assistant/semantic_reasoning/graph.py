"""Deterministic graph, explicit cycle reporting and conservative issue components."""

from .candidates import terms
from .models import DecisionEvolution, EventGraph, IssueThread, require


def validate_graph(events, relations, graph):
    by_id = {e.id: e for e in events}
    require(len(by_id) == len(events), "Duplicate event IDs.")
    accepted = {e.id for e in events if e.outcome == "accepted"}
    require(set(graph.event_ids) == accepted and len(graph.event_ids) == len(accepted))
    require(tuple(sorted(graph.event_ids, key=lambda x: (by_id[x].start, x))) == graph.event_ids)
    require(len({r.id for r in relations}) == len(relations))
    require(graph.relation_ids == tuple(r.id for r in relations))
    for relation in relations:
        a, b = by_id.get(relation.source_event_id), by_id.get(relation.target_event_id)
        require(
            a is not None and b is not None and a.id in accepted and b.id in accepted,
            "Relation endpoint missing or not accepted.",
        )
        require(
            set(a.evidence_utterance_ids + b.evidence_utterance_ids)
            <= set(relation.evidence_utterance_ids),
            "Relation lost endpoint evidence.",
        )
        if relation.relation_type == "SUPERSEDES":
            require(
                a.event_type == b.event_type == "DECISION" and (a.start, a.id) < (b.start, b.id),
                "SUPERSEDES requires an earlier and a later confirmed decision.",
            )
    thread_ids = [x for t in graph.issue_threads for x in t.event_ids]
    require(len(thread_ids) == len(set(thread_ids)) and set(thread_ids) == accepted)
    require(len({t.id for t in graph.issue_threads}) == len(graph.issue_threads))
    for thread in graph.issue_threads:
        require(bool(thread.event_ids))
        require(
            thread.event_ids == tuple(sorted(thread.event_ids, key=lambda x: (by_id[x].start, x)))
        )
        require(thread.representative_text == by_id[thread.event_ids[0]].text[:120])
    require(
        graph.cycles == _cycles(graph.event_ids, relations), "Cycle metadata does not match edges."
    )


def _cycles(ids, relations):
    adjacency = {eid: set() for eid in ids}
    for r in relations:
        adjacency[r.source_event_id].add(r.target_event_id)
    # Small bounded graphs: mutual reachability yields strongly connected components.
    reachable = {}
    for eid in ids:
        seen, stack = set(), list(adjacency[eid])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency[current] - seen)
        reachable[eid] = seen
    components, consumed = [], set()
    for eid in ids:
        if eid in consumed:
            continue
        component = tuple(
            x for x in ids if x != eid and x in reachable[eid] and eid in reachable[x]
        )
        if component:
            group = tuple(x for x in ids if x in (eid,) + component)
            consumed.update(group)
            components.append(group)
    return tuple(components)


def build_graph(events, relations):
    accepted = sorted((e for e in events if e.outcome == "accepted"), key=lambda e: (e.start, e.id))
    ids = tuple(e.id for e in accepted)
    parent = {eid: eid for eid in ids}

    def root(eid):
        while parent[eid] != eid:
            eid = parent[eid]
        return eid

    def join(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for relation in relations:
        require(relation.source_event_id in parent and relation.target_event_id in parent)
        join(relation.source_event_id, relation.target_event_id)
    for index, b in enumerate(accepted):
        for a in accepted[max(0, index - 12) : index]:
            if b.start - a.end <= 120 and len(terms(a.text) & terms(b.text)) >= 2:
                join(a.id, b.id)
    groups = {}
    for event in accepted:
        groups.setdefault(root(event.id), []).append(event)
    threads = tuple(
        IssueThread(f"issue_{n:04d}", group[0].text[:120], tuple(e.id for e in group))
        for n, group in enumerate(groups.values(), 1)
    )
    cycles = _cycles(ids, relations)
    graph = EventGraph(
        ids,
        tuple(r.id for r in relations),
        threads,
        cycles,
        ("General relation cycle retained; evolution follows source chronology.",)
        if cycles
        else (),
    )
    validate_graph(events, relations, graph)
    return graph


def build_evolution(events, relations, graph):
    by_id = {e.id: e for e in events}
    historical = {r.source_event_id for r in relations if r.relation_type == "SUPERSEDES"}
    resolved = {r.source_event_id for r in relations if r.relation_type in ("ACCEPTS", "REJECTS")}
    return tuple(
        DecisionEvolution(
            thread.id,
            thread.event_ids,
            tuple(
                eid
                for eid in thread.event_ids
                if by_id[eid].event_type == "DECISION" and eid not in historical
            ),
            tuple(eid for eid in thread.event_ids if eid in historical),
            tuple(
                eid
                for eid in thread.event_ids
                if by_id[eid].event_type in ("TASK_ASSIGNMENT", "COMMITMENT")
            ),
            tuple(
                eid
                for eid in thread.event_ids
                if by_id[eid].event_type == "PROPOSAL" and eid not in resolved
            ),
        )
        for thread in graph.issue_threads
    )
