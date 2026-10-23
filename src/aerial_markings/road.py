"""Coarse mask of the road surface.

Paint only exists on pavement. Asphalt seen from above is grey, colourless
and smooth at the scale of a metre, while grass, bushes, gravel and forest
are textured or green. The mask is computed on a downsampled copy (about
25 cm per pixel), cleaned with morphology so that markings, cracks and small
stains inside the carriageway are filled, and only large regions are kept.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import RoadParams


def road_mask(rgb: np.ndarray, valid: np.ndarray, gsd_m: float, p: RoadParams) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(gsd_m / p.work_gsd_m, 1.0)
    sw, sh = max(int(w * scale), 1), max(int(h * scale), 1)
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    vsmall = cv2.resize(valid.view(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    px = lambda m: max(int(round(m / p.work_gsd_m)), 1)  # noqa: E731

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1]
    r, g, b = (small[..., i].astype(np.int16) for i in range(3))
    exg = 2 * g - r - b

    # local texture: standard deviation of brightness in a ~1 m window.
    # Paint inflates it on the road, so bright pixels are excluded from the
    # statistics by clamping them to the local median first.
    med = cv2.medianBlur(gray.astype(np.uint8), px(1.0) | 1).astype(np.float32)
    clamped = np.minimum(gray, med + 12)
    k = (px(p.texture_window_m) | 1,) * 2
    mean = cv2.blur(clamped, k)
    std = np.sqrt(np.maximum(cv2.blur(clamped * clamped, k) - mean * mean, 0))

    cand = ((sat <= p.max_saturation) & (exg <= p.max_excess_green)
            & (std <= p.max_texture) & (med >= p.min_brightness) & (med <= p.max_brightness)
            & vsmall).astype(np.uint8)

    # fill markings, cracks and tyre marks; remove thin and small false regions
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px(p.fill_holes_m) * 2 + 1,) * 2))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px(p.min_road_width_m),) * 2))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    min_area_px = p.min_area_m2 / p.work_gsd_m ** 2
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    road = keep[labels].astype(np.uint8)

    # markings sit on the very edge of the pavement: grow the mask a little
    if p.margin_m > 0:
        road = cv2.dilate(road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px(p.margin_m) * 2 + 1,) * 2))
    return cv2.resize(road, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
