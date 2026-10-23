"""Writing results: GeoPackage, zipped Shapefile, GeoJSON and a web preview."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

from .pipeline import Result


def write_outputs(result: Result, out_dir: str | Path, formats=("gpkg", "shp", "geojson"),
                  preview: bool = True) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = result.ortho.path.stem
    files: dict[str, str] = {}

    if "gpkg" in formats:
        path = out / f"{stem}_markings.gpkg"
        path.unlink(missing_ok=True)
        result.markings.to_file(path, layer="markings", driver="GPKG")
        if len(result.segments):
            result.segments.to_file(path, layer="segments", driver="GPKG")
        if len(result.vehicles):
            result.vehicles.to_file(path, layer="vehicles", driver="GPKG")
        files["gpkg"] = path.name

    if "shp" in formats:
        shp_dir = out / "shp"
        shp_dir.mkdir(exist_ok=True)
        shp = shp_dir / f"{stem}_markings.shp"
        result.markings.to_file(shp, encoding="utf-8")
        zpath = out / f"{stem}_markings_shp.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(shp_dir.glob(f"{stem}_markings.*")):
                z.write(f, arcname=f.name)
        files["shp"] = zpath.name

    if "geojson" in formats or preview:
        path = out / f"{stem}_markings.geojson"
        result.markings.to_crs(4326).to_file(path, driver="GeoJSON")
        files["geojson"] = path.name

    if preview:
        files.update(write_preview(result, out))

    (out / "stats.json").write_text(json.dumps(result.stats, indent=2, ensure_ascii=False))
    files["stats"] = "stats.json"
    return files


def write_preview(result: Result, out: Path, max_side: int = 2048) -> dict[str, str]:
    """Downsampled image warped to Web Mercator, so Leaflet can overlay it exactly."""
    o = result.ortho
    h, w = o.shape
    scale = min(max_side / max(h, w), 1.0)
    sw, sh = max(int(w * scale), 1), max(int(h * scale), 1)
    small = cv2.resize(o.rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(o.valid.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
    src = np.concatenate([np.moveaxis(small, -1, 0), alpha[None]], axis=0)
    t = o.transform
    fx, fy = w / sw, h / sh
    src_transform = Affine(t.a * fx, t.b * fy, t.c, t.d * fx, t.e * fy, t.f)

    dst_crs = "EPSG:3857"
    left, bottom, right, top = rasterio.transform.array_bounds(sh, sw, src_transform)
    dst_transform, dw, dh = calculate_default_transform(o.crs, dst_crs, sw, sh, left, bottom, right, top)
    dst = np.zeros((4, dh, dw), dtype=np.uint8)
    for b in range(4):
        reproject(src[b], dst[b], src_transform=src_transform, src_crs=o.crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.nearest if b == 3 else Resampling.bilinear)

    png = out / "preview.png"
    rgba = np.moveaxis(dst, 0, -1)
    cv2.imwrite(str(png), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    l, b, r, t = rasterio.transform.array_bounds(dh, dw, dst_transform)
    w_, s_, e_, n_ = transform_bounds(dst_crs, "EPSG:4326", l, b, r, t)
    (out / "preview.json").write_text(json.dumps({"bounds": [[s_, w_], [n_, e_]]}))
    return {"preview": png.name, "preview_bounds": "preview.json"}
