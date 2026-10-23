from pathlib import Path

import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from aerial_markings import Params, run
from aerial_markings.export import write_outputs
from aerial_markings.merge import chain_lines
from aerial_markings.config import MergeParams
from aerial_markings.raster import CRSUnknownError, read_ortho
from aerial_markings.synthetic import ANGLE, GSD, ORIGIN, SIZE, make_scene, write_scene


def _params():
    p = Params()
    p.vehicles.enabled = False          # keep tests fast and offline
    return p


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    path = tmp_path_factory.mktemp("scene") / "road.tif"
    truth = write_scene(path)
    return path, truth


@pytest.fixture(scope="module")
def result(scene):
    return run(scene[0], _params())


def test_finds_all_lines(result, scene):
    _, truth = scene
    by_type = result.stats["by_type"]
    assert by_type["solid"]["count"] == truth["solid"]
    assert by_type["dashed"]["count"] == truth["dashed"]


def test_solid_lines_cross_the_shadow(result):
    solid = result.markings[result.markings["type"] == "solid"]
    # the diagonal of a 120 m tile is ~170 m; lines clipped by the tile are ~145 m
    assert (solid.length_m > 130).all()


def test_no_lines_in_forest(result):
    x0, y0 = ORIGIN
    forest = box(x0 + 1300 * GSD, y0 - 1000 * GSD, x0 + SIZE * GSD, y0)
    m = result.markings
    stray = m[m.within(forest)]
    assert stray.length.sum() < 5


def test_nothing_off_the_road(result):
    """Dry grass full of pale straws and gravel shoulders must not produce lines."""
    x0, y0 = ORIGIN
    cx, cy = x0 + SIZE / 2 * GSD, y0 - SIZE / 2 * GSD
    d = np.array([np.cos(ANGLE), -np.sin(ANGLE)])          # image y axis points down
    n = np.array([-d[1], d[0]])
    for geom in result.markings.geometry:
        pts = np.asarray(geom.coords) - [cx, cy]
        assert np.abs(pts @ n).max() < 10.5, "line found outside the carriageways"


def test_world_file_in_degrees(tmp_path):
    """TIFF without GeoTIFF tags + .tfw in degrees, like the original project's data."""
    img, _ = make_scene()
    tif = tmp_path / "plain.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=SIZE, width=SIZE, count=3, dtype="uint8") as dst:
        dst.write(np.moveaxis(img, -1, 0))
    px = 0.05 / 111_320                               # ~5 cm in degrees of latitude
    (tmp_path / "plain.tfw").write_text(f"{px}\n0\n0\n{-px * 1.8}\n60.8\n56.7\n")
    ortho = read_ortho(tif)
    assert ortho.crs.to_epsg() == 4326
    assert 0.03 < ortho.gsd_m < 0.08
    res = run(tif, _params())
    assert res.stats["by_type"]["solid"]["count"] >= 4
    assert res.markings.crs.to_epsg() == 4326


def test_projected_without_crs_requires_crs(tmp_path):
    img, _ = make_scene()
    tif = tmp_path / "nocrs.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=SIZE, width=SIZE, count=3, dtype="uint8") as dst:
        dst.write(np.moveaxis(img, -1, 0))
    (tmp_path / "nocrs.tfw").write_text(f"{GSD}\n0\n0\n{-GSD}\n{ORIGIN[0]}\n{ORIGIN[1]}\n")
    with pytest.raises(CRSUnknownError):
        read_ortho(tif)
    assert read_ortho(tif, crs="EPSG:32641").crs.to_epsg() == 32641


def test_outputs(result, tmp_path):
    files = write_outputs(result, tmp_path)
    for name in files.values():
        assert (tmp_path / name).exists()


def test_chain_reverses_pieces():
    # three collinear pieces given in shuffled order and directions
    lines = [np.array([[10.0, 0], [14, 0]]), np.array([[3.0, 0], [0, 0]]), np.array([[5.0, 0], [8, 0]])]
    chains = chain_lines(lines, 3.0, MergeParams())
    assert len(chains) == 1 and len(chains[0]) == 3
    coords = np.vstack([lines[i][::-1] if rev else lines[i] for i, rev in chains[0]])
    xs = coords[:, 0]
    assert np.all(np.diff(xs) >= 0) or np.all(np.diff(xs) <= 0)


def _road_tile(paint=(235, 235, 232), rails=False, size=600, gsd=0.05):
    """A straight horizontal road with one painted line and optionally a guardrail."""
    import cv2
    img = np.full((size, size, 3), (90, 125, 70), np.uint8)            # grass
    cv2.rectangle(img, (0, 200), (size, 400), (120, 120, 124), -1)        # asphalt
    cv2.line(img, (0, 300), (size, 300), paint, 4)                        # 0.2 m line
    if rails:
        cv2.rectangle(img, (0, 120), (size, 190), (150, 120, 90), -1)     # soil strip
        for y in (150, 156):
            cv2.line(img, (0, y), (size, y), (225, 225, 225), 2)          # two rails
        for x in range(0, size, 40):
            cv2.line(img, (x, 150), (x, 156), (225, 225, 225), 2)         # posts
    rng = np.random.default_rng(1)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def _run_array(img, tmp_path, name):
    from rasterio.transform import from_origin
    path = tmp_path / f"{name}.tif"
    with rasterio.open(path, "w", driver="GTiff", height=img.shape[0], width=img.shape[1], count=3,
                       dtype="uint8", crs="EPSG:32641", transform=from_origin(*ORIGIN, 0.05, 0.05)) as dst:
        dst.write(np.moveaxis(img, -1, 0))
    return run(path, _params())


def test_yellow_paint(tmp_path):
    res = _run_array(_road_tile(paint=(240, 215, 90)), tmp_path, "yellow")
    solid = res.markings[res.markings["type"] == "solid"]
    assert len(solid) == 1 and solid.iloc[0]["color"] == "yellow"


def test_guardrail_is_not_a_marking(tmp_path):
    res = _run_array(_road_tile(rails=True), tmp_path, "rail")
    solid = res.markings[res.markings["type"] == "solid"]
    assert len(solid) == 1 and solid.iloc[0]["color"] == "white"
