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
from aerial_markings.synthetic import GSD, ORIGIN, SIZE, make_scene, write_scene


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
    solid = result.markings[result.markings.type == "solid"]
    # the diagonal of a 120 m tile is ~170 m; lines clipped by the tile are ~145 m
    assert (solid.length_m > 130).all()


def test_no_lines_in_forest(result):
    x0, y0 = ORIGIN
    forest = box(x0 + 1300 * GSD, y0 - 1000 * GSD, x0 + SIZE * GSD, y0)
    m = result.markings
    stray = m[(m.type == "dash") & m.within(forest)]
    assert stray.length.sum() < 5


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
