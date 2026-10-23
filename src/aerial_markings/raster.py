"""Reading orthophotos (GeoTIFF or TIFF + world file) with a known CRS."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS

log = logging.getLogger(__name__)


class CRSUnknownError(ValueError):
    """The raster has no CRS and none was given."""


@dataclass
class Ortho:
    rgb: np.ndarray            # H x W x 3, uint8
    valid: np.ndarray          # H x W, bool (False = nodata / outside the photo)
    transform: Affine          # pixel -> raster CRS
    crs: CRS
    gsd_m: float               # ground size of one pixel in metres
    path: Path

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]


def _read_world_file(path: Path) -> Affine | None:
    for ext in (".tfw", ".tifw", ".wld"):
        wf = path.with_suffix(ext)
        if wf.exists():
            a, d, b, e, c, f = (float(v) for v in wf.read_text().split()[:6])
            # world files reference pixel centres, Affine references corners
            return Affine(a, b, c - a / 2 - b / 2, d, e, f - d / 2 - e / 2)
    return None


def _resolve_crs(src, transform: Affine, path: Path, crs_override: str | None) -> CRS:
    if crs_override:
        return CRS.from_user_input(crs_override)
    if src.crs:
        return CRS.from_user_input(src.crs.to_wkt())
    prj = path.with_suffix(".prj")
    if prj.exists():
        return CRS.from_wkt(prj.read_text())
    x, y = transform.c, transform.f
    if abs(x) <= 180 and abs(y) <= 90:
        log.warning("No CRS in %s; coordinates look like degrees, assuming EPSG:4326", path.name)
        return CRS.from_epsg(4326)
    raise CRSUnknownError(
        f"{path.name} has no coordinate reference system. "
        "Pass it explicitly, e.g. --crs EPSG:32637, or put a .prj file next to the image."
    )


def _gsd_metres(transform: Affine, crs: CRS, height: int, width: int) -> float:
    px = math.hypot(transform.a, transform.d)
    py = math.hypot(transform.b, transform.e)
    size = (px + py) / 2
    if crs.is_geographic:
        lat = transform.f + transform.d * width / 2 + transform.e * height / 2
        metres_per_deg = 111_320 * math.sqrt(
            (math.cos(math.radians(lat)) ** 2 + 1) / 2
        )
        return size * metres_per_deg
    unit = crs.axis_info[0].unit_conversion_factor if crs.axis_info else 1.0
    return size * unit


def _to_uint8(bands: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if bands.dtype == np.uint8:
        return bands
    data = bands[:, valid] if valid.any() else bands.reshape(bands.shape[0], -1)
    lo, hi = np.percentile(data, [0.5, 99.5])
    scaled = (bands.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def read_ortho(path: str | Path, crs: str | None = None) -> Ortho:
    path = Path(path)
    with rasterio.open(path) as src:
        transform = src.transform
        if transform.is_identity:
            transform = _read_world_file(path) or transform
        if transform.is_identity:
            raise CRSUnknownError(f"{path.name} is not georeferenced (no GeoTIFF tags and no .tfw)")
        raster_crs = _resolve_crs(src, transform, path, crs)

        count = src.count
        if count >= 3:
            bands = src.read([1, 2, 3])
        else:
            bands = np.repeat(src.read(1)[None], 3, axis=0)

        valid = src.dataset_mask() > 0
        if count == 4 and src.colorinterp[3].name == "alpha":
            valid &= src.read(4) > 0

    rgb = np.ascontiguousarray(np.moveaxis(_to_uint8(bands, valid), 0, -1))
    gsd = _gsd_metres(transform, raster_crs, *rgb.shape[:2])
    log.info("Loaded %s: %dx%d px, GSD %.3f m, CRS %s",
             path.name, rgb.shape[1], rgb.shape[0], gsd, raster_crs.to_string())
    return Ortho(rgb=rgb, valid=valid, transform=transform, crs=raster_crs, gsd_m=gsd, path=path)
