"""UTF-8 frozen schema reconstruction and atomic, non-overwriting sidecar bundles."""

import hashlib
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints
from uuid import uuid4

from .exceptions import InvalidSemantics
from .models import SemanticResult, check_type, require


def to_json(value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        return asdict(item)

    return (
        json.dumps(
            value, default=encode, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )


def fingerprint(value):
    return hashlib.sha256(to_json(value).encode("utf-8")).hexdigest()


def _object(pairs):
    require(len(dict(pairs)) == len(pairs), "Duplicate JSON fields.")
    return dict(pairs)


def from_json(content):
    def decode(value, kind):
        origin, args = get_origin(kind), get_args(kind)
        if origin in (UnionType, Union):
            return (
                None
                if value is None and type(None) in args
                else decode(value, next(x for x in args if x is not type(None)))
            )
        if origin is tuple:
            require(type(value) is list)
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(decode(x, args[0]) for x in value)
            require(len(args) == len(value))
            return tuple(decode(x, t) for x, t in zip(value, args))
        if is_dataclass(kind):
            require(type(value) is dict)
            hints = get_type_hints(kind)
            require(set(value) == set(hints), "Missing/unknown serialized fields.")
            return kind(**{k: decode(v, hints[k]) for k, v in value.items()})
        if origin is Literal:
            require(value in args)
        check_type(value, kind)
        return value

    try:
        return decode(json.loads(content, object_pairs_hook=_object), SemanticResult)
    except (TypeError, ValueError, KeyError, StopIteration) as exc:
        raise InvalidSemantics("Malformed Phase X JSON.") from exc


def render_evolution(result):
    events = {e.id: e for e in result.events}
    threads = {t.id: t for t in result.event_graph.issue_threads}
    lines = ["Decision evolution (additive observations; canonical MeetingRecord unchanged)"]
    for chain in result.decision_evolution:
        lines += ["", f"Issue {chain.issue_id}: {threads[chain.issue_id].representative_text}"]
        for eid in chain.ordered_event_ids:
            e = events[eid]
            state = (
                "historical"
                if eid in chain.historical_decision_ids
                else ("current" if eid in chain.current_decision_ids else e.event_type)
            )
            lines += [
                f"{eid} {e.event_type} ({state}) [{e.start:.3f} --> {e.end:.3f}]",
                f"{e.speaker_id or 'UNATTRIBUTED'}: {e.text}",
                "Evidence: " + ", ".join(e.evidence_utterance_ids),
            ]
    lines += [
        f"{r.source_event_id} -> {r.target_event_id}: {r.relation_type} (B relative to A)"
        for r in result.relations
    ]
    return "\n".join(lines) + "\n"


def render_text(result):
    lines = [
        f"Semantic reasoning: {result.availability}",
        "Provider scores are uncalibrated for this meeting domain.",
        f"Calls: {len(result.processing.calls)}; time: {result.processing.total_seconds:.3f} s",
        render_evolution(result),
    ]
    lines += [
        f"Verification {v.item_id}: {v.status}; evidence={','.join(v.evidence_utterance_ids)}"
        for v in result.verification
    ]
    lines += [
        f"Coverage {v.event_id}: {v.status}; matches={','.join(v.matched_record_ids)}; "
        f"warnings={','.join(v.warnings)}"
        for v in result.coverage
    ]
    lines += ["Warning: " + w for w in result.warnings]
    return "\n".join(lines) + "\n"


def save_semantics(result, output_dir):
    root = Path(output_dir) / "semantic_reasoning"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / fingerprint(result)
    payloads = {
        "semantic_result.json": to_json(result),
        "meeting_events.json": to_json(result.events),
        "event_relations.json": to_json(
            {"edges": result.relations, "observations": result.relation_observations}
        ),
        "meeting_event_graph.json": to_json(result.event_graph),
        "decision_evolution.json": to_json(result.decision_evolution),
        "decision_evolution.txt": render_evolution(result),
        "semantic_verification.json": to_json(result.verification),
        "coverage_report.json": to_json(result.coverage),
        "semantic_reasoning.txt": render_text(result),
    }
    # Every individual projection lives under a provenance-bound, content-addressed bundle.
    if destination.exists():
        require(not destination.is_symlink(), "Unsafe existing bundle.")
        require(
            all(
                (destination / name).read_text(encoding="utf-8") == content
                for name, content in payloads.items()
            ),
            "Existing bundle was modified.",
        )
        return destination
    staging = root / (".partial-" + uuid4().hex)
    staging.mkdir()
    try:
        for name, content in payloads.items():
            with (staging / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        try:
            staging.rename(destination)
        except FileExistsError:
            return save_semantics(result, output_dir)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)
