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
    color: str = "white"


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
        if gap > p.max_lateral_offset_m:
            u = gap_vec / gap
            if _angle(da, u) > p.max_angle_deg or _angle(-db, u) > p.max_angle_deg:
                continue                                 # next piece is not straight ahead
            lateral = abs(da[0] * gap_vec[1] - da[1] * gap_vec[0])
            # direction of a short stroke is known to a few degrees, so the
            # tolerated offset grows with the gap
            if lateral > p.max_lateral_offset_m + p.lateral_per_gap * gap:
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
            members = [idx[j] for j, _ in chain]
            span = _length(np.vstack(parts))
            if kind == "dash" and (len(chain) < p.min_dashes or span < p.min_dashed_length_m
                                   or not _regular(parts, p)):
                # too few aligned strokes for a dashed line: keep them as single strokes
                out.extend(Chain(part, "dash", [m]) for part, m in zip(parts, members))
                continue
            label = "dashed" if kind == "dash" else "solid"
            out.append(Chain(np.vstack(parts), label, members))
    return out


def _length(c: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def _steady(values, tol: float) -> float:
    """Share of values within ±tol of the median: robust to a missing or broken stroke."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return 1.0
    med = np.median(v)
    return float(np.mean(np.abs(v - med) <= tol * med)) if med > 0 else 0.0


def _regular(parts: list[np.ndarray], p: MergeParams) -> bool:
    """Painted dashes repeat with a steady rhythm; debris on the shoulder does not."""
    lengths = [_length(c) for c in parts]
    gaps = [float(np.linalg.norm(b[0] - a[-1])) for a, b in zip(parts, parts[1:])]
    return (_steady(lengths, p.rhythm_tolerance) >= p.min_steady_share
            and _steady(gaps, p.rhythm_tolerance) >= p.min_steady_share)


def _direction_at(coords: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Unit direction of a polyline at the vertex pair closest to `point`."""
    seg = np.diff(coords, axis=0)
    mid = (coords[:-1] + coords[1:]) / 2
    k = int(np.argmin(np.linalg.norm(mid - point, axis=1)))
    v = seg[k]
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(2)


def _densify(coords: np.ndarray, step: float) -> np.ndarray:
    out = [coords[:1]]
    for a, b in zip(coords[:-1], coords[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.append(a + (b - a) * t)
    return np.vstack(out)


def drop_misaligned(chains: list[Chain], p: MergeParams) -> list[Chain]:
    """Remove pieces that do not follow the road.

    Poles, their shadows and kerb stones near the edge are short and bright
    like a dash, but they stand across the road; bright debris in a field
    can form a short straight "solid line" far from any road. Each short
    piece is compared with the nearest long solid line: it must be parallel
    to it, and a short solid piece must also be near it.
    """
    guides = [c.coords for c in chains if c.kind == "solid" and _length(c.coords) >= p.guide_min_length_m]
    if not guides:
        return chains
    dense = [_densify(g, 1.0) for g in guides]          # a straight line may have only two vertices
    pts = np.vstack(dense)
    owner = np.concatenate([np.full(len(g), i) for i, g in enumerate(dense)])
    tree = cKDTree(pts)
    out = []
    for c in chains:
        length = _length(c.coords)
        limit = p.short_solid_max_m if c.kind == "solid" else p.guide_check_max_m
        if length > limit or any(c.coords is g for g in guides):
            out.append(c)
            continue
        centre = c.coords.mean(axis=0)
        dist, i = tree.query(centre)
        if dist > p.guide_radius_m:
            if c.kind != "solid":
                out.append(c)                            # a dashed line far from any edge line
            continue                                     # a short solid piece alone in the field
        d = c.coords[-1] - c.coords[0]
        d = d / (np.linalg.norm(d) or 1.0)
        g = _direction_at(guides[owner[i]], centre)
        angle = np.degrees(np.arccos(np.clip(abs(float(d @ g)), 0, 1)))
        if angle <= p.max_angle_deg:
            out.append(c)
    return out


def smooth_line(coords: np.ndarray, window: float, step: float) -> np.ndarray:
    """Resample every `step` metres and apply a moving average over `window` metres.

    The skeleton of real paint is jagged by a pixel or two; road geometry is
    smooth at the scale of metres. Ends are kept in place.
    """
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 2 * step:
        return coords
    t = np.linspace(0, total, max(int(total / step), 2) + 1)
    pts = np.column_stack([np.interp(t, cum, coords[:, 0]), np.interp(t, cum, coords[:, 1])])
    k = int(round(window / step)) // 2
    if k < 1 or len(pts) < 3:
        return pts
    pad = np.vstack([2 * pts[0] - pts[k:0:-1], pts, 2 * pts[-1] - pts[-2:-k - 2:-1]])   # odd reflection
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.column_stack([np.convolve(pad[:, i], kernel, mode="valid") for i in range(2)])
    sm[0], sm[-1] = pts[0], pts[-1]
    return sm
