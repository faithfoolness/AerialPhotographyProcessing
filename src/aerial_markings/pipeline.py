"""End-to-end extraction: orthophoto in, vector road markings out."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.affinity import affine_transform

from .config import Params
from .mask import is_yellow, marking_mask
from .merge import drop_misaligned, merge_segments, smooth_line
from .raster import Ortho, read_ortho
from .road import road_mask
from .surface import pavement_map, pavement_share
from .vectorize import extract_segments
from .vehicles import detect_vehicles

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


@dataclass
class Result:
    markings: gpd.GeoDataFrame      # merged lines, CRS of the input image
    segments: gpd.GeoDataFrame      # individual pieces before merging
    vehicles: gpd.GeoDataFrame      # detected vehicle footprints
    ortho: Ortho
    stats: dict = field(default_factory=dict)


def _pixel_to_crs(geom, transform):
    t = transform
    return affine_transform(geom, [t.a, t.b, t.d, t.e, t.c, t.f])


@contextmanager
def _stage(timings: dict, name: str, progress: Progress | None, frac: float, label: str):
    if progress:
        progress(frac, label)
    t0 = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - t0, 2)
    log.info("%-10s %.2f s", name, timings[name])


def run(image: str | Path, params: Params | None = None, crs: str | None = None,
        progress: Progress | None = None) -> Result:
    p = params or Params()
    timings: dict[str, float] = {}

    with _stage(timings, "read", progress, 0.02, "Reading image"):
        ortho = read_ortho(image, crs)
    gsd = ortho.gsd_m

    with _stage(timings, "vehicles", progress, 0.10, "Detecting vehicles"):
        vehicle_mask, vehicle_polys = detect_vehicles(ortho.rgb, gsd, p.vehicles)

    with _stage(timings, "road", progress, 0.40, "Finding the road surface"):
        road = road_mask(ortho.rgb, ortho.valid, gsd, p.road) if p.road.enabled else ortho.valid

    with _stage(timings, "mask", progress, 0.50, "Finding paint pixels"):
        mask = marking_mask(ortho.rgb, road, gsd, p.mask)
        mask &= ~vehicle_mask

    with _stage(timings, "vectorize", progress, 0.60, "Tracing centre lines"):
        segments, counters = extract_segments(mask, gsd, p.shape, p.merge.simplify_tolerance_px)

    with _stage(timings, "surface", progress, 0.75, "Checking what lies beside the lines"):
        if p.surface.enabled and segments:
            pave = pavement_map(ortho.rgb, p.surface)
            before = len(segments)
            segments = [s for s in segments
                        if pavement_share(pave, s.pixels, gsd, p.surface) >= p.surface.min_share]
            counters["off_pavement_dropped"] = before - len(segments)
            del pave
        colors = ["yellow" if is_yellow(ortho.rgb, s.pixels, p.mask.yellow_b_threshold) else "white"
                  for s in segments]

    with _stage(timings, "merge", progress, 0.85, "Joining pieces into lines"):
        crs_in = ortho.crs
        seg_geoms = [_pixel_to_crs(LineString(s.pixels), ortho.transform) for s in segments]
        seg_gdf = gpd.GeoDataFrame(
            {
                "type": [s.kind for s in segments],
                "color": colors,
                "length_m": [round(s.length_m, 2) for s in segments],
                "width_m": [round(s.width_m, 3) for s in segments],
            },
            geometry=seg_geoms, crs=crs_in,
        )
        work_crs = crs_in if crs_in.is_projected else seg_gdf.estimate_utm_crs() if len(seg_gdf) else crs_in
        seg_m = seg_gdf.to_crs(work_crs)
        all_lines = [np.asarray(g.coords) for g in seg_m.geometry]
        chains = []
        for color in ("white", "yellow"):                 # never join pieces of different colour
            idx = [i for i, c in enumerate(colors) if c == color]
            for ch in merge_segments([all_lines[i] for i in idx], [segments[i].kind for i in idx], p.merge):
                ch.members = [idx[m] for m in ch.members]
                ch.color = color
                chains.append(ch)
        n_before = len(chains)
        chains = drop_misaligned(chains, p.merge)
        counters["misaligned_dropped"] = n_before - len(chains)
        widths = seg_gdf["width_m"].to_numpy()
        markings = gpd.GeoDataFrame(
            {
                "type": [c.kind for c in chains],
                "color": [c.color for c in chains],
                "pieces": [len(c.members) for c in chains],
                "width_m": [round(float(np.mean(widths[c.members])), 3) for c in chains],
            },
            geometry=[LineString(smooth_line(c.coords, p.merge.smooth_window_m, p.merge.smooth_step_m))
                      for c in chains], crs=work_crs,
        )
        markings["length_m"] = markings.length.round(2)
        lone = (markings["type"] == "dash") & (
            (markings["length_m"] < p.shape.isolated_min_length_m) | (not p.shape.keep_single_strokes))
        counters["isolated_dropped"] = int(lone.sum())
        markings = markings[~lone].reset_index(drop=True).to_crs(crs_in)
        markings.insert(0, "id", np.arange(1, len(markings) + 1))

        veh_gdf = gpd.GeoDataFrame(
            geometry=[_pixel_to_crs(Polygon(pg), ortho.transform) for pg in vehicle_polys],
            crs=crs_in,
        )
        if len(veh_gdf):
            veh_gdf = veh_gdf.dissolve().explode(index_parts=False).reset_index(drop=True)

    if progress:
        progress(1.0, "Done")

    h, w = ortho.shape
    by_type = markings.groupby("type")["length_m"].agg(["count", "sum"]).round(1)
    stats = {
        "image": ortho.path.name,
        "size_px": [w, h],
        "gsd_m": round(gsd, 4),
        "crs": f"EPSG:{crs_in.to_epsg()}" if crs_in.to_epsg() else crs_in.name,
        "road_area_m2": round(float(road.sum()) * gsd * gsd, 1),
        "vehicles": int(len(veh_gdf)),
        "paint_pixels": int(mask.sum()),
        "components": counters,
        "segments": int(len(seg_gdf)),
        "lines": int(len(markings)),
        "by_type": {t: {"count": int(r["count"]), "length_m": float(r["sum"])} for t, r in by_type.iterrows()},
        "timings_s": timings,
        "total_s": round(sum(timings.values()), 2),
    }
    log.info("Extracted %d lines from %d pieces in %.1f s", stats["lines"], stats["segments"], stats["total_s"])
    return Result(markings, seg_gdf, veh_gdf, ortho, stats)
