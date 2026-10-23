"""Pixel-level candidate mask of road markings."""

from __future__ import annotations

import cv2
import numpy as np

from .config import MaskParams


def marking_mask(rgb: np.ndarray, valid: np.ndarray, gsd_m: float, p: MaskParams) -> np.ndarray:
    """Return a boolean mask of pixels that look like paint on the road.

    A pixel qualifies when it is
      * noticeably brighter than its surroundings (white top-hat): a narrow
        bright stripe on darker asphalt, unlike large bright areas;
      * bright in absolute terms;
      * close to neutral colour (paint is white, foliage glints are not);
      * not vegetation (green clearly above red and blue).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    k = int(round(2.5 * p.max_marking_width_m / gsd_m)) | 1
    k = max(k, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]

    mask = (tophat >= p.tophat_threshold) & (gray >= p.min_brightness) & (sat <= p.max_saturation)

    if p.exclude_vegetation:
        r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
        mask &= ~((g - r > 12) & (g - b > 12))

    mask &= valid

    # close tiny holes inside paint, drop isolated specks
    small = max(int(round(0.05 / gsd_m)), 1) * 2 + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small, small))
    mask = cv2.morphologyEx(mask.view(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)
    return mask
