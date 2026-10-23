import time

from fastapi.testclient import TestClient

from aerial_markings.synthetic import write_scene
from aerial_markings.web.app import create_app


def test_upload_and_results(tmp_path):
    tif = tmp_path / "road.tif"
    write_scene(tif)
    client = TestClient(create_app(tmp_path / "runs"))

    assert client.get("/").status_code == 200

    with tif.open("rb") as f:
        r = client.post("/api/jobs", files={"files": ("my road (1).tif", f, "image/tiff")},
                        data={"vehicles": "false"})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    for _ in range(120):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    assert job["stats"]["lines"] >= 6

    geo = client.get(f"/api/jobs/{job_id}/files/{job['files']['geojson']}")
    assert geo.status_code == 200 and geo.json()["type"] == "FeatureCollection"
    assert client.get(f"/api/jobs/{job_id}/files/../../etc/passwd").status_code == 404


def test_rejects_without_tif(tmp_path):
    client = TestClient(create_app(tmp_path / "runs"))
    r = client.post("/api/jobs", files={"files": ("a.tfw", b"1\n0\n0\n-1\n0\n0\n")})
    assert r.status_code == 400


def _wait(client, job_id):
    for _ in range(240):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_several_images_with_world_files(tmp_path):
    import numpy as np
    import rasterio

    from aerial_markings.synthetic import GSD, ORIGIN, SIZE, make_scene

    img, _ = make_scene()
    uploads = []
    for k, name in enumerate(("north", "south")):
        tif = tmp_path / f"{name}.tif"
        with rasterio.open(tif, "w", driver="GTiff", height=SIZE, width=SIZE, count=3, dtype="uint8") as dst:
            dst.write(np.moveaxis(img, -1, 0))
        x0, y0 = ORIGIN[0], ORIGIN[1] - k * SIZE * GSD          # second tile directly below the first
        (tmp_path / f"{name}.tfw").write_text(f"{GSD}\n0\n0\n{-GSD}\n{x0 + GSD / 2}\n{y0 - GSD / 2}\n")
        (tmp_path / f"{name}.prj").write_text(rasterio.crs.CRS.from_epsg(32641).to_wkt())
        uploads += [tif, tmp_path / f"{name}.tfw", tmp_path / f"{name}.prj"]

    client = TestClient(create_app(tmp_path / "runs"))
    handles = [p.open("rb") for p in uploads]
    try:
        r = client.post("/api/jobs", files=[("files", (p.name, h)) for p, h in zip(uploads, handles)],
                        data={"vehicles": "false"})
    finally:
        for h in handles:
            h.close()
    assert r.status_code == 200, r.text
    assert r.json()["images"] == 2

    job = _wait(client, r.json()["id"])
    assert job["status"] == "done", job
    assert [im["status"] for im in job["images"]] == ["done", "done"]
    assert job["stats"]["images"] == 2
    assert job["stats"]["lines"] == sum(im["stats"]["lines"] for im in job["images"])

    geo = client.get(f"/api/jobs/{job['id']}/files/{job['files']['geojson']}").json()
    assert {f["properties"]["image"] for f in geo["features"]} == {"north", "south"}
    per_image = job["images"][1]["files"]["gpkg"]
    assert client.get(f"/api/jobs/{job['id']}/files/{per_image}").status_code == 200
