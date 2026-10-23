"""From a binary mask to ordered centre lines of individual markings."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely.geometry import LineString
from skimage.morphology import skeletonize

from .config import ShapeParams

SQRT2 = float(np.sqrt(2))
_OFFSETS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2))


@dataclass
class Segment:
    pixels: np.ndarray       # N x 2 (col, row) ordered along the marking, image coords
    length_m: float
    width_m: float
    kind: str                # "dash" or "solid"


def _skeleton_graph(skel: np.ndarray):
    rows, cols = np.nonzero(skel)
    n = len(rows)
    index = -np.ones(skel.shape, dtype=np.int64)
    index[rows, cols] = np.arange(n)
    src, dst, wts = [], [], []
    h, w = skel.shape
    for dr, dc, wt in _OFFSETS:
        r2, c2 = rows + dr, cols + dc
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        j = np.full(n, -1)
        j[ok] = index[r2[ok], c2[ok]]
        has = j >= 0
        src.append(np.nonzero(has)[0])
        dst.append(j[has])
        wts.append(np.full(has.sum(), wt))
    src, dst, wts = map(np.concatenate, (src, dst, wts))
    g = coo_matrix((wts, (src, dst)), shape=(n, n)).tocsr()
    return g + g.T, np.column_stack([cols, rows])


def _longest_path(g, nodes: np.ndarray) -> tuple[list[int], float]:
    """Approximate graph diameter inside `nodes` with a double sweep."""
    d0 = dijkstra(g, indices=nodes[0])
    a = nodes[np.argmax(np.where(np.isfinite(d0[nodes]), d0[nodes], -1))]
    da, pred = dijkstra(g, indices=a, return_predecessors=True)
    b = nodes[np.argmax(np.where(np.isfinite(da[nodes]), da[nodes], -1))]
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    return path[::-1], float(da[b])


def skeleton_paths(skel: np.ndarray, min_len_px: float, max_paths: int = 6) -> list[tuple[np.ndarray, float]]:
    """Split a skeleton into ordered polylines: the main axis first, then long branches."""
    if skel.sum() < 2:
        return []
    g, xy = _skeleton_graph(skel)
    alive = np.ones(len(xy), dtype=bool)
    out = []
    for _ in range(max_paths):
        idx = np.nonzero(alive)[0]
        if len(idx) < 2:
            break
        sub = g[idx][:, idx]
        ncomp, labels = connected_components(sub, directed=False)
        best = None
        for c in range(ncomp):
            members = np.nonzero(labels == c)[0]
            if len(members) < 2:
                continue
            path, length = _longest_path(sub, members)
            if best is None or length > best[1]:
                best = (path, length)
        if best is None or best[1] < min_len_px:
            break
        path_global = idx[best[0]]
        out.append((xy[path_global], best[1]))
        alive[path_global] = False
        # drop pixels touching the extracted axis so branches start cleanly
        touch = g[path_global].nonzero()[1]
        alive[touch] = False
    return out


def extract_segments(mask: np.ndarray, gsd_m: float, p: ShapeParams, simplify_px: float = 1.5) -> tuple[list[Segment], dict]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
    min_area_px = (p.min_length_m * 0.05) / gsd_m ** 2
    min_len_px = p.min_length_m / gsd_m
    counters = {"components": n - 1, "too_small": 0, "bad_shape": 0, "kept": 0}

    segments: list[Segment] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area_px or max(w, h) < min_len_px * 0.7:
            counters["too_small"] += 1
            continue
        crop = np.pad(labels[y:y + h, x:x + w] == i, 1)
        skel = skeletonize(crop)
        paths = skeleton_paths(skel, min_len_px)
        if not paths:
            counters["too_small"] += 1
            continue
        total_len_px = sum(length for _, length in paths)
        width_px = area / max(total_len_px, 1.0)
        main_len_px = paths[0][1]
        if (width_px * gsd_m > p.max_width_m
                or main_len_px / max(width_px, 1.0) < p.min_elongation):
            counters["bad_shape"] += 1
            continue
        kept_any = False
        for xy, length_px in paths:
            coords = xy + [x - 1, y - 1] + 0.5      # crop offset, pixel centres
            line = LineString(coords).simplify(simplify_px)   # also removes pixel staircase
            length_m = line.length * gsd_m
            chord = float(np.linalg.norm(xy[-1] - xy[0]))
            if length_m < p.straight_check_max_m and chord / max(line.length, 1.0) < p.min_straightness:
                continue                             # curved like a grass blade
            kept_any = True
            kind = "solid" if length_m > p.dash_max_length_m else "dash"
            segments.append(Segment(np.asarray(line.coords), length_m, width_px * gsd_m, kind))
        counters["kept" if kept_any else "bad_shape"] += 1
    return segments, counters
