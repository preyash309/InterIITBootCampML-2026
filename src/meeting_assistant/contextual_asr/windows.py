"""Frame-aligned windows, deterministic merging and hard per-meeting budgets."""

import math

from .models import AudioWindow, SkippedSpan


def select_windows(spans, grounding, duration, config):
    records = {r.id: r for r in grounding.records}
    regions, skipped = [], []
    for span in spans:
        record = records[span.grounding_id]
        start = max(0, math.floor((record.start - config.padding_seconds) * 16000) / 16000)
        end = min(duration, math.ceil((record.end + config.padding_seconds) * 16000) / 16000)
        if end - start > config.max_window_seconds or end <= start:
            skipped.append(SkippedSpan(record.id, "window_duration_limit"))
        else:
            regions.append((start, end, (record.id,), span.priority))
    merged = []
    for start, end, ids, priority in sorted(regions):
        if (
            merged
            and start <= merged[-1][1] + config.merge_gap_seconds
            and max(end, merged[-1][1]) - merged[-1][0] <= config.max_window_seconds
        ):
            previous = merged[-1]
            merged[-1] = (
                previous[0],
                max(end, previous[1]),
                previous[2] + ids,
                max(priority, previous[3]),
            )
        else:
            merged.append((start, end, ids, priority))
    selected, seconds = [], 0.0
    for start, end, ids, priority in sorted(merged, key=lambda row: (-row[3], row[0], row[2])):
        # Even when the merged duration exceeds the cap, never issue overlapping duplicate calls.
        overlaps = any(start < old.end and end > old.start for old in selected)
        reason = (
            "overlapping_window"
            if overlaps
            else "call_budget"
            if len(selected) >= config.max_windows
            else "audio_budget"
            if seconds + end - start > config.max_total_audio_seconds
            else None
        )
        if reason:
            skipped.extend(SkippedSpan(identity, reason) for identity in ids)
            continue
        selected.append(AudioWindow(f"win_{len(selected) + 1:06d}", start, end, ids))
        seconds += end - start
    return tuple(selected), tuple(skipped)
