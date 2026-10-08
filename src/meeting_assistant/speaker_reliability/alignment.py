"""Maximum speaking-time assignment with explicit lexicographic optimal tie resolution."""

import math
from functools import lru_cache

from .comparison import timeline
from .models import SpeakerAlignment, SpeakerMapping


def optimal_assignment(matrix):
    """Secondary rows / primary columns. Dummy columns leave unmatched speakers free.

    Re-solve the residual assignment for each lexicographic choice instead of perturbing
    costs, which could change the optimum for short genuine fragments.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        linear_sum_assignment = None

    n = len(matrix)
    if not n:
        return ()
    m = len(matrix[0])
    costs = [list(row) + [0.0] * n for row in matrix]

    def optimum(rows, columns):
        if not rows:
            return 0.0
        if linear_sum_assignment is None:
            # The supported secondary model has <=4 labels, making a small exact
            # fallback practical for dependency-free core tests.
            @lru_cache(None)
            def solve(index, remaining):
                if index == len(rows):
                    return 0.0
                return max(
                    costs[rows[index]][col]
                    + solve(index + 1, tuple(x for x in remaining if x != col))
                    for col in remaining
                )

            return solve(0, tuple(columns))
        a, b = linear_sum_assignment([[costs[r][c] for c in columns] for r in rows], maximize=True)
        return math.fsum(costs[rows[i]][columns[j]] for i, j in zip(a, b, strict=True))

    columns = list(range(m + n))
    target = optimum(list(range(n)), columns)
    chosen = []
    for row in range(n):
        for col in columns:
            remaining = [x for x in columns if x != col]
            residual = optimum(list(range(row + 1, n)), remaining)
            value = costs[row][col] + residual
            if math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12):
                chosen.append(col if col < m and costs[row][col] > 0 else None)
                columns = remaining
                target = residual
                break
    return tuple(chosen)


def align_speakers(primary, secondary, config):
    p, s = primary.speakers, secondary.speakers
    matrix = [[0.0 for _ in p] for _ in s]
    pd = dict.fromkeys(p, 0.0)
    sd = dict.fromkeys(s, 0.0)
    pi, si = {x: i for i, x in enumerate(p)}, {x: i for i, x in enumerate(s)}
    for start, end, regular, _, labels in timeline(primary, secondary):
        seconds = end - start
        for x in regular:
            pd[x] += seconds
        for y in labels:
            sd[y] += seconds
            for x in regular:
                matrix[si[y]][pi[x]] += seconds
    assignments = optimal_assignment(matrix)
    mappings = []
    for index, col in enumerate(assignments):
        overlap = matrix[index][col] if col is not None else 0.0
        mappings.append(
            SpeakerMapping(
                s[index],
                p[col] if col is not None else None,
                overlap,
                overlap / sd[s[index]] if sd[s[index]] else 0.0,
                overlap / pd[p[col]] if col is not None and pd[p[col]] else 0.0,
                "mapped" if col is not None else "unmapped",
            )
        )
    splits, merges = [], []
    for col, label in enumerate(p):
        candidates = tuple(
            s[row]
            for row in range(len(s))
            if pd[label] and matrix[row][col] / pd[label] >= config.split_merge_min_fraction
        )
        if len(candidates) > 1:
            splits.append((label, candidates))
    for row, label in enumerate(s):
        candidates = tuple(
            p[col]
            for col in range(len(p))
            if sd[label] and matrix[row][col] / sd[label] >= config.split_merge_min_fraction
        )
        if len(candidates) > 1:
            merges.append((label, candidates))
    mapped = {x.primary_id for x in mappings}
    return SpeakerAlignment(
        tuple(mappings),
        len(p),
        len(s),
        len(p) != len(s),
        tuple(x for x in p if x not in mapped),
        tuple(x.secondary_id for x in mappings if x.primary_id is None),
        tuple(splits),
        tuple(merges),
        tuple(tuple(row) for row in matrix),
    )
