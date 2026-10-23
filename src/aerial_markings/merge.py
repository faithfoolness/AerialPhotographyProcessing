"""Joining marking pieces into continuous lines.

Solid lines are interrupted by shadows, cars and worn paint; dashed lines
consist of separate strokes by design. Pieces are chained end-to-end when
the gap is short, the directions agree and the next piece lies on the
continuation of the current one (small lateral offset).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .config import MergeParams


@dataclass
class Chain:
    coords: np.ndarray                     # N x 2, metric CRS
    kind: str                              # solid | dashed | dash
    members: list[int] = field(default_factory=list)


def _end_direction(coords: np.ndarray, at_start: bool, probe: float = 1.5) -> np.ndarray:
    """Unit vector pointing outwards from the given end of a polyline."""
    pts = coords if not at_start else coords[::-1]
    end = pts[-1]
    acc = 0.0
    j = len(pts) - 1
    while j > 0 and acc < probe:
        acc += np.linalg.norm(pts[j] - pts[j - 1])
        j -= 1
    v = end - pts[j]
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(2)


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def _links(lines: list[np.ndarray], max_gap: float, p: MergeParams):
    ends, dirs = [], []
    for coords in lines:
        for at_start in (True, False):
            ends.append(coords[0] if at_start else coords[-1])
            dirs.append(_end_direction(coords, at_start))
    ends, dirs = np.array(ends), np.array(dirs)
    tree = cKDTree(ends)
    cand = []
    for a, b in tree.query_pairs(max_gap):
        if a // 2 == b // 2:
            continue
        gap_vec = ends[b] - ends[a]
        gap = float(np.linalg.norm(gap_vec))
        da, db = dirs[a], dirs[b]
        if _angle(da, -db) > p.max_angle_deg:           # pieces must be parallel
            continue
        if gap > 1e-6:
            u = gap_vec / gap
            if _angle(da, u) > p.max_angle_deg and gap > p.max_lateral_offset_m:
                continue                                 # next piece is not ahead
            lateral = abs(da[0] * gap_vec[1] - da[1] * gap_vec[0])
            if lateral > p.max_lateral_offset_m:
                continue
        cand.append((gap + 0.5 * _angle(da, -db), a, b))
    cand.sort()
    return cand


def chain_lines(lines: list[np.ndarray], max_gap: float, p: MergeParams) -> list[list[tuple[int, bool]]]:
    """Return chains as lists of (line index, reversed?)."""
    n = len(lines)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    used = set()
    neighbour: dict[int, int] = {}
    for _, a, b in _links(lines, max_gap, p):
        if a in used or b in used:
            continue
        ra, rb = find(a // 2), find(b // 2)
        if ra == rb:
            continue                                     # would close a loop
        parent[rb] = ra
        used.update((a, b))
        neighbour[a] = b
        neighbour[b] = a

    # Endpoint ids: 2*i is the start of line i, 2*i+1 its end. Every endpoint
    # has at most one link and there are no loops, so chains are simple paths.
    chains, seen = [], set()
    for line in range(n):
        if line in seen:
            continue
        free = [e for e in (2 * line, 2 * line + 1) if e not in neighbour]
        if not free:
            continue                                     # interior piece, reached later
        entry = free[0]
        chain = []
        while True:
            cur = entry // 2
            seen.add(cur)
            chain.append((cur, entry % 2 == 1))          # entered at its end -> reversed
            nxt = neighbour.get(entry ^ 1)
            if nxt is None:
                break
            entry = nxt
        chains.append(chain)
    return chains


def merge_segments(lines: list[np.ndarray], kinds: list[str], p: MergeParams) -> list[Chain]:
    out: list[Chain] = []
    for kind, gap in (("solid", p.solid_max_gap_m), ("dash", p.dash_max_gap_m)):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        subset = [lines[i] for i in idx]
        for chain in chain_lines(subset, gap, p):
            parts = []
            for j, rev in chain:
                c = subset[j][::-1] if rev else subset[j]
                parts.append(c)
            coords = np.vstack(parts)
            members = [idx[j] for j, _ in chain]
            if kind == "dash":
                label = "dashed" if len(chain) >= 2 else "dash"
            else:
                label = "solid"
            out.append(Chain(coords, label, members))
    return out
