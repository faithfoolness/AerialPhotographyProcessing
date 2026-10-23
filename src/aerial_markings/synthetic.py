"""A synthetic orthophoto of a dual carriageway, used for tests and the demo.

It contains the situations that broke the first version of the project:
solid edge lines interrupted by a shadow, dashed centre lines, a white car
standing on the markings and a forest full of bright specks.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

GSD = 0.05
SIZE = 2400                    # px -> 120 x 120 m
ANGLE = np.radians(-35)        # road direction (image coords)
CRS = "EPSG:32641"
ORIGIN = (300_000.0, 6_290_000.0)


def _px(m: float) -> int:
    return int(round(m / GSD))


def _road_points(offset_m: float):
    """Endpoints of a line parallel to the road axis, offset across it."""
    c = np.array([SIZE / 2, SIZE / 2])
    d = np.array([np.cos(ANGLE), np.sin(ANGLE)])
    n = np.array([-d[1], d[0]])
    base = c + n * _px(offset_m)
    return base - d * SIZE, base + d * SIZE, d, n


def make_scene(seed: int = 0) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:] = (88, 128, 70)                                  # grass
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

    # forest with sunlit specks in the upper-right corner
    for _ in range(260):
        x, y = rng.integers(1300, SIZE), rng.integers(0, 1000)
        r = int(rng.integers(25, 70))
        cv2.circle(img, (int(x), int(y)), r, (int(rng.integers(30, 55)), int(rng.integers(60, 90)), 40), -1)
    for _ in range(900):
        x, y = rng.integers(1300, SIZE), rng.integers(0, 1000)
        v = int(rng.integers(200, 250))
        cv2.ellipse(img, (int(x), int(y)), (int(rng.integers(2, 9)), int(rng.integers(1, 4))),
                    float(rng.uniform(0, 180)), 0, 360, (v, v, v - 10), -1)

    # dry, greyish grass in the lower-left half: mottled and full of pale straws,
    # some of them as straight as a painted dash
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dry = (yy - xx) > 300
    blotch = cv2.GaussianBlur(rng.normal(0, 1, (SIZE // 8, SIZE // 8)).astype(np.float32), (0, 0), 2)
    blotch = cv2.resize(blotch, (SIZE, SIZE))
    base = np.stack([118 + 18 * blotch, 116 + 14 * blotch, 96 + 10 * blotch], -1)
    base = np.clip(base + rng.normal(0, 14, base.shape), 0, 255).astype(np.uint8)
    img[dry] = base[dry]
    for _ in range(5000):
        x, y = int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE))
        if not dry[y, x]:
            continue
        length = _px(float(rng.uniform(0.3, 3.5)))
        ang = rng.uniform(0, np.pi)
        bend = rng.uniform(-0.4, 0.4) if rng.random() < 0.6 else 0.0
        pts = []
        for t in np.linspace(0, 1, 8):
            a_ = ang + bend * t
            pts.append((x + np.cos(a_) * length * t, y + np.sin(a_) * length * t))
        v = int(rng.integers(185, 235))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (v, v, v - 8), int(rng.integers(1, 4)))

    truth = {"solid": 0, "dashed": 0}
    for centre, asphalt in ((-6.0, (95, 95, 98)), (6.0, (62, 60, 70))):
        a, b, d, n = _road_points(centre)
        # gravel shoulders: light, grainy, slightly brown
        for side in (-4.8, 4.8):
            s0, s1, *_ = _road_points(centre + side)
            cv2.line(img, tuple(s0.astype(int)), tuple(s1.astype(int)), (150, 142, 128), _px(1.8))
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), asphalt, _px(8.0))
        for edge in (-3.5, 3.5):
            a2, b2, *_ = _road_points(centre + edge)
            cv2.line(img, tuple(a2.astype(int)), tuple(b2.astype(int)), (235, 235, 232), _px(0.2))
            truth["solid"] += 1
        a3, _, *_ = _road_points(centre)
        for k in range(0, 2 * SIZE, _px(12.0)):
            p0 = a3 + d * k
            p1 = p0 + d * _px(3.0)
            cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), (230, 230, 228), _px(0.15))
        truth["dashed"] += 1

    # shadow of a sign gantry across the left carriageway
    c, _, d, n = _road_points(-6.0)
    mid = np.array([SIZE / 2, SIZE / 2]) + n * _px(-6.0) - d * _px(25)
    shadow = cv2.boxPoints(((float(mid[0]), float(mid[1])), (_px(1.2), _px(14)), float(np.degrees(ANGLE))))
    cv2.fillPoly(img, [shadow.astype(np.int32)], (25, 25, 30))

    # a white car standing across the dashed line of the right carriageway
    c2 = np.array([SIZE / 2, SIZE / 2]) + n * _px(6.4) + d * _px(18)
    car = cv2.boxPoints(((float(c2[0]), float(c2[1])), (_px(4.5), _px(1.9)), float(np.degrees(ANGLE))))
    cv2.fillPoly(img, [car.astype(np.int32)], (240, 240, 240))
    cv2.fillPoly(img, [cv2.boxPoints(((float(c2[0]), float(c2[1])), (_px(1.6), _px(1.5)),
                                       float(np.degrees(ANGLE)))).astype(np.int32)], (60, 70, 80))

    gravel = (np.abs(img.astype(int) - (150, 142, 128)).sum(-1) < 6)
    img[gravel] = np.clip(img[gravel] + rng.normal(0, 22, (gravel.sum(), 1)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img, truth


def write_scene(path: str | Path, seed: int = 0) -> dict:
    img, truth = make_scene(seed)
    transform = from_origin(*ORIGIN, GSD, GSD)
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=3,
                       dtype="uint8", crs=CRS, transform=transform, compress="deflate",
                       photometric="RGB") as dst:
        dst.write(np.moveaxis(img, -1, 0))
    return truth


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_road.tif"
    print(write_scene(out))
