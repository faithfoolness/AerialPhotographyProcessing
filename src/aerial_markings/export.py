"""Writing results: GeoPackage, zipped Shapefile, GeoJSON and a web preview."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

from .pipeline import Result


def _write_lines(gdf, out: Path, stem: str, formats, extra_layers: dict | None = None) -> dict[str, str]:
    files: dict[str, str] = {}
    if "gpkg" in formats:
        path = out / f"{stem}_markings.gpkg"
        path.unlink(missing_ok=True)
        gdf.to_file(path, layer="markings", driver="GPKG")
        for layer, data in (extra_layers or {}).items():
            if len(data):
                data.to_file(path, layer=layer, driver="GPKG")
        files["gpkg"] = path.name

    if "shp" in formats:
        shp_dir = out / "shp"
        shp_dir.mkdir(exist_ok=True)
        gdf.to_file(shp_dir / f"{stem}_markings.shp", encoding="utf-8")
        zpath = out / f"{stem}_markings_shp.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(shp_dir.glob(f"{stem}_markings.*")):
                z.write(f, arcname=f.name)
        files["shp"] = zpath.name

    if "geojson" in formats:
        path = out / f"{stem}_markings.geojson"
        gdf.to_crs(4326).to_file(path, driver="GeoJSON")
        files["geojson"] = path.name
    return files


def write_outputs(result: Result, out_dir: str | Path, formats=("gpkg", "shp", "geojson"),
                  preview: bool = True) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmts = set(formats) | ({"geojson"} if preview else set())
    files = _write_lines(result.markings, out, result.ortho.path.stem, fmts,
                         {"segments": result.segments, "vehicles": result.vehicles})
    if preview:
        files.update(write_preview(result, out))
    (out / "stats.json").write_text(json.dumps(result.stats, indent=2, ensure_ascii=False))
    files["stats"] = "stats.json"
    return files


def write_combined(results: list[tuple[str, "gpd.GeoDataFrame", dict]], out_dir: str | Path,
                   formats=("gpkg", "shp", "geojson")) -> tuple[dict[str, str], dict]:
    """One layer with the lines of all images, in the CRS of the first one.

    Each line keeps the name of its image in the `image` column, and ids are
    renumbered across the whole set. Takes (name, markings, stats) triples
    rather than full results, so the images themselves need not stay in memory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crs = results[0][1].crs
    parts = []
    for name, markings, _ in results:
        m = markings.drop(columns="id").to_crs(crs)
        m.insert(0, "image", name)
        parts.append(m)
    allm = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs)
    allm.insert(0, "id", np.arange(1, len(allm) + 1))
    files = _write_lines(allm, out, "all", formats)

    by_type = allm.groupby("type")["length_m"].agg(["count", "sum"]).round(1)
    stats = {
        "images": len(results),
        "lines": int(len(allm)),
        "by_type": {t: {"count": int(v["count"]), "length_m": float(v["sum"])} for t, v in by_type.iterrows()},
        "total_s": round(sum(st["total_s"] for _, _, st in results), 2),
        "crs": results[0][2]["crs"],
        "per_image": {name: {k: st[k] for k in ("lines", "by_type", "total_s", "gsd_m", "size_px")}
                      for name, _, st in results},
    }
    (out / "stats_all.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    files["stats"] = "stats_all.json"
    return files, stats


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
