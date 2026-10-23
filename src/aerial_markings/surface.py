"""Does a candidate line lie on pavement?

Paint is applied to asphalt or concrete. Guardrails, kerbs on soil, bright
debris in the median and grass blades are long and thin too, but the ground
beside them is soil, grass or gravel. For each candidate the surface is
sampled a little to the left and to the right of its axis; at least one
side must look like pavement (grey, unsaturated, not green) along most of
the line. An edge line passes with pavement on one side only.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import SurfaceParams


def pavement_map(rgb: np.ndarray, p: SurfaceParams) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    exg = 2 * g - r - b
    return (sat <= p.max_saturation) & (exg <= p.max_excess_green) & (val >= p.min_value)


def _resample(coords: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0, cum[-1] + 1e-9, step) if cum[-1] > step else np.array([0.0, cum[-1]])
    pts = np.column_stack([np.interp(t, cum, coords[:, 0]), np.interp(t, cum, coords[:, 1])])
    tang = np.gradient(pts, axis=0)
    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = np.divide(tang, norm, out=np.zeros_like(tang), where=norm > 0)
    return pts, np.column_stack([-tang[:, 1], tang[:, 0]])


def pavement_share(pave: np.ndarray, coords_px: np.ndarray, gsd_m: float, p: SurfaceParams) -> float:
    """Share of positions along the line with pavement on at least one side."""
    pts, normals = _resample(coords_px, p.sample_step_m / gsd_m)
    h, w = pave.shape
    sides = []
    for sign in (1, -1):
        hits = np.zeros(len(pts))
        for off in p.side_offsets_m:
            q = np.round(pts + sign * normals * off / gsd_m).astype(int)
            ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
            vals = np.zeros(len(q))
            vals[ok] = pave[q[ok, 1], q[ok, 0]]
            hits += vals
        sides.append(hits / len(p.side_offsets_m) > 0.5)        # most offsets on pavement
    return float(np.mean(sides[0] | sides[1]))
