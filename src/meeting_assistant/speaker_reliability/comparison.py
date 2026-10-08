"""Exact event sweep, including silence and independent overlap observations."""

from collections import Counter, defaultdict

from .models import ComparisonInterval


def timeline(primary, secondary):
    events = defaultdict(list)
    duration = primary.duration_seconds
    events[0.0]
    events[duration]
    for kind, turns in (
        (0, primary.regular_turns),
        (1, primary.exclusive_turns),
        (2, secondary.segments),
    ):
        for turn in turns:
            start, end = min(duration, turn.start), min(duration, turn.end)
            if end > start:
                events[start].append((kind, turn.speaker_id, 1))
                events[end].append((kind, turn.speaker_id, -1))
    active = [Counter(), Counter(), Counter()]
    times = sorted(events)
    for index, start in enumerate(times[:-1]):
        for kind, label, delta in events[start]:
            active[kind][label] += delta
        groups = [tuple(sorted(k for k, v in group.items() if v > 0)) for group in active]
        yield start, times[index + 1], *groups


def compare_intervals(primary, secondary, alignment):
    mapping = {x.secondary_id: x.primary_id for x in alignment.mappings}
    result = []
    for start, end, regular, exclusive, labels in timeline(primary, secondary):
        mapped = tuple(sorted(mapping[x] for x in labels if mapping[x] is not None))
        unmapped = any(mapping[x] is None for x in labels)
        if not regular and not labels:
            state = "SILENCE"
        elif not labels:
            state = "PRIMARY_ONLY"
        elif not regular:
            state = "SECONDARY_ONLY"
        elif unmapped:
            state = "UNMAPPED_SECONDARY"
        elif len(regular) > 1 or len(labels) > 1:
            state = "OVERLAP_AGREE" if regular == mapped else "OVERLAP_DISAGREE"
        else:
            state = "AGREE" if regular == mapped else "DISAGREE"
        item = ComparisonInterval(
            start, end, regular, exclusive[0] if exclusive else None, labels, mapped, state
        )
        # Coalesce only when all observations, including exclusive attribution, are identical.
        if (
            result
            and result[-1].end == start
            and (
                result[-1].primary_speakers,
                result[-1].primary_exclusive_speaker,
                result[-1].secondary_speakers,
                result[-1].mapped_secondary_speakers,
                result[-1].state,
            )
            == (regular, item.primary_exclusive_speaker, labels, mapped, state)
        ):
            old = result.pop()
            item = ComparisonInterval(
                old.start, end, regular, item.primary_exclusive_speaker, labels, mapped, state
            )
        result.append(item)
    return tuple(result)
