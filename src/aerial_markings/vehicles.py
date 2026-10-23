"""Vehicle masking with a YOLO oriented-bounding-box model trained on aerial imagery.

White cars look exactly like paint to a brightness filter, so they are
detected first and excluded from the marking mask. The default model
(YOLO11n-OBB, DOTA v1) knows "small vehicle" and "large vehicle" seen from
above; it is downloaded automatically by Ultralytics on first use.

The image is resampled to roughly the resolution of the training data and
processed in overlapping tiles, so cars keep their size in pixels no matter
how large the orthophoto is.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np

from .config import VehicleParams

log = logging.getLogger(__name__)

VEHICLE_CLASSES = {"small vehicle", "large vehicle"}


@lru_cache(maxsize=2)
def _load_model(name: str):
    from ultralytics import YOLO  # optional dependency, imported lazily

    return YOLO(name)


def _tiles(length: int, tile: int, overlap: int):
    if length <= tile:
        yield 0
        return
    step = tile - overlap
    pos = 0
    while True:
        yield pos
        if pos + tile >= length:
            return
        pos = min(pos + step, length - tile)


def detect_vehicles(rgb: np.ndarray, gsd_m: float, p: VehicleParams) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (mask of vehicle pixels, list of 4x2 polygons in pixel coords)."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not p.enabled:
        return mask.astype(bool), []
    try:
        model = _load_model(p.model)
    except Exception as exc:  # no ultralytics / no network
        log.warning("Vehicle detection skipped: %s", exc)
        return mask.astype(bool), []

    names = model.names
    keep = [i for i, n in names.items() if n in VEHICLE_CLASSES]

    scale = min(gsd_m / p.target_gsd_m, 1.0)
    small = rgb if scale == 1.0 else cv2.resize(
        rgb, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA
    )
    sh, sw = small.shape[:2]
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

    polys: list[np.ndarray] = []
    for y0 in _tiles(sh, p.tile_px, p.overlap_px):
        for x0 in _tiles(sw, p.tile_px, p.overlap_px):
            tile = bgr[y0:y0 + p.tile_px, x0:x0 + p.tile_px]
            if tile.size == 0 or tile.max() == 0:
                continue
            res = model.predict(tile, imgsz=p.tile_px, conf=p.confidence,
                                classes=keep, verbose=False)[0]
            if res.obb is None or len(res.obb) == 0:
                continue
            for poly in res.obb.xyxyxyxy.cpu().numpy():
                polys.append((poly + [x0, y0]) / scale)

    # suppress duplicates from overlapping tiles: drawing them twice is harmless
    buf = int(round(p.buffer_m / gsd_m))
    for poly in polys:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    if buf > 0 and polys:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buf + 1, 2 * buf + 1))
        mask = cv2.dilate(mask, se)
    log.info("Vehicles detected: %d", len(polys))
    return mask.astype(bool), polys
