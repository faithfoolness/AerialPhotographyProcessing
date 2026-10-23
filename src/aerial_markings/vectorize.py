"""From a binary mask to ordered centre lines of individual markings."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage import distance_transform_edt
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


def window_wiggle(coords: np.ndarray, window: float) -> float:
    """Median deviation from a straight chord over sliding windows of a polyline.

    Paint follows the road smoothly: even on a bend, a few metres of it are
    practically straight. Chains of grains or grass blades meander at every
    step. Returns 0 for polylines shorter than one window.
    """
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < window:
        return 0.0
    devs, start = [], 0.0
    while start + window <= cum[-1]:
        t = np.linspace(start, start + window, 12)
        pts = np.column_stack([np.interp(t, cum, coords[:, 0]), np.interp(t, cum, coords[:, 1])])
        d = pts[-1] - pts[0]
        d = d / (np.linalg.norm(d) or 1.0)
        devs.append(float(np.abs((pts - pts[0]) @ np.array([-d[1], d[0]])).max()))
        start += window / 2
    return float(np.median(devs))


def _holes_per_10m(crop: np.ndarray, gsd_m: float) -> float:
    """Enclosed gaps per 10 m of length.

    Paint is a solid stripe. A guardrail seen from above is two rails joined
    by posts every couple of metres, so its mask looks like a ladder full of
    small windows.
    """
    cnts, hier = cv2.findContours(crop.view(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0.0
    min_area = 0.01 / gsd_m ** 2
    holes = sum(1 for k, c in enumerate(cnts) if hier[0][k][3] >= 0 and cv2.contourArea(c) >= min_area)
    if holes == 0:
        return 0.0
    length_m = max(crop.shape) * gsd_m
    return holes / max(length_m, 1.0) * 10


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
        if _holes_per_10m(crop, gsd_m) > p.max_holes_per_10m:
            counters["ladder_dropped"] = counters.get("ladder_dropped", 0) + 1
            continue                                 # guardrail: two rails joined by posts
        skel = skeletonize(crop)
        paths = skeleton_paths(skel, min_len_px)
        if not paths:
            counters["too_small"] += 1
            continue
        # width = twice the median distance from the axis to the edge of the blob;
        # bright specks stuck to a line hardly change the median
        dist = distance_transform_edt(crop)
        axis = paths[0][0]
        width_px = max(2.0 * float(np.median(dist[axis[:, 1], axis[:, 0]])) - 1.0, 1.0)
        main_len_px = paths[0][1]
        # ...but a chain of grains along a gravel shoulder is narrow at the axis
        # and has a lot of area around it
        spread_m = area / max(sum(length for _, length in paths), 1.0) * gsd_m
        if (width_px * gsd_m > p.max_width_m or spread_m > p.max_spread_m
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
            if window_wiggle(np.asarray(line.coords), p.wiggle_window_m / gsd_m) * gsd_m > p.max_wiggle_m:
                continue                             # meanders like a chain of gravel grains
            kept_any = True
            kind = "solid" if length_m > p.dash_max_length_m else "dash"
            segments.append(Segment(np.asarray(line.coords), length_m, width_px * gsd_m, kind))
        counters["kept" if kept_any else "bad_shape"] += 1
    return segments, counters
