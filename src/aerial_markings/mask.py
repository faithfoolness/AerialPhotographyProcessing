"""Pixel-level candidate mask of road markings."""

from __future__ import annotations

import cv2
import numpy as np

from .config import MaskParams

_DIRECTIONS = np.radians(np.arange(0, 180, 22.5))


def ridge_contrast(gray: np.ndarray, distances_px: list[int]) -> np.ndarray:
    """How much brighter each pixel is than BOTH its neighbours across a stripe.

    For every direction and distance d, a pixel is compared with the two
    points d pixels away on either side; the contrast is the brightness above
    the brighter of the two. A painted stripe is brighter than the surface on
    both sides in the direction across it. The edge of a bright slab next to
    a dark joint or shoulder is not: on one side there is more of the slab.
    """
    pad = max(distances_px) + 1
    g = cv2.GaussianBlur(gray, (3, 3), 0.7).astype(np.int16)
    big = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    h, w = g.shape
    best = np.full((h, w), -255, dtype=np.int16)
    for d in distances_px:
        for a in _DIRECTIONS:
            dx, dy = int(round(d * np.cos(a))), int(round(d * np.sin(a)))
            plus = big[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            minus = big[pad - dy:pad - dy + h, pad - dx:pad - dx + w]
            np.maximum(best, g - np.maximum(plus, minus), out=best)
    return np.clip(best, 0, 255).astype(np.uint8)


def marking_mask(rgb: np.ndarray, valid: np.ndarray, gsd_m: float, p: MaskParams) -> np.ndarray:
    """Return a boolean mask of pixels that look like paint on the road.

    A pixel qualifies when it is
      * a bright ridge: brighter than the surface on both sides of it;
      * bright in absolute terms;
      * white/grey or yellow (paint), not some other colour;
      * not vegetation (green clearly above red and blue).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dists = sorted({max(int(round(m / gsd_m)), 2) for m in p.ridge_offsets_m})
    ridge = ridge_contrast(gray, dists)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    yellow = (hue >= 10) & (hue <= 40)                  # OpenCV hue is 0-180
    colour_ok = (sat <= p.max_saturation) | (yellow & (sat <= p.max_yellow_saturation))

    mask = (ridge >= p.ridge_threshold) & (gray >= p.min_brightness) & colour_ok

    if p.exclude_vegetation:
        r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
        mask &= ~((g - r > 12) & (g - b > 12))

    mask &= valid

    # close tiny holes inside paint
    small = max(int(round(0.05 / gsd_m)), 1) * 2 + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small, small))
    return cv2.morphologyEx(mask.view(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)


def is_yellow(rgb: np.ndarray, pixels: np.ndarray, threshold: float) -> bool:
    """Paint colour of a marking from the mean b* (blue-yellow) of its axis pixels."""
    pts = np.round(pixels).astype(int)
    h, w = rgb.shape[:2]
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
    if len(pts) == 0:
        return False
    sample = rgb[pts[:, 1], pts[:, 0]].reshape(-1, 1, 3)
    lab_b = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)[..., 2].astype(float)
    return float(np.mean(lab_b)) - 128.0 >= threshold
