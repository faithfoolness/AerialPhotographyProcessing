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
