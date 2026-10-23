"""Web interface: upload an orthophoto, watch progress, inspect lines on a map."""

from __future__ import annotations

import logging
import re
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Params
from ..export import write_outputs
from ..pipeline import run

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
ALLOWED = {".tif", ".tiff", ".tfw", ".tifw", ".prj"}


@dataclass
class Job:
    id: str
    image: str
    status: str = "queued"           # queued | running | done | error
    progress: float = 0.0
    message: str = "Waiting in queue"
    stats: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    error: str | None = None


def _safe_name(name: str) -> str:
    name = Path(name).name
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^\w.+-]+", "_", stem, flags=re.UNICODE).strip("._") or "image"
    return stem + ext


def create_app(workdir: Path = Path("runs")) -> FastAPI:
    workdir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Aerial road markings")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    jobs: dict[str, Job] = {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=1)    # one heavy job at a time

    def work(job: Job, image: Path, crs: str | None, vehicles: bool):
        def progress(frac: float, msg: str):
            with lock:
                job.progress, job.message = round(frac, 3), msg

        with lock:
            job.status = "running"
        try:
            params = Params()
            params.vehicles.enabled = vehicles
            result = run(image, params, crs=crs or None, progress=progress)
            files = write_outputs(result, workdir / job.id / "output")
            with lock:
                job.stats, job.files = result.stats, files
                job.status, job.progress, job.message = "done", 1.0, "Done"
        except Exception as exc:
            log.error("Job %s failed\n%s", job.id, traceback.format_exc())
            with lock:
                job.status, job.error, job.message = "error", str(exc), "Failed"

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/api/jobs")
    async def create_job(
        files: list[UploadFile] = File(..., description="GeoTIFF, optionally with .tfw/.prj"),
        crs: str = Form(""),
        vehicles: bool = Form(True),
    ):
        saved: list[Path] = []
        job_id = uuid.uuid4().hex[:12]
        in_dir = workdir / job_id / "input"
        in_dir.mkdir(parents=True)
        images = [f for f in files if Path(f.filename or "").suffix.lower() in (".tif", ".tiff")]
        if len(images) != 1:
            shutil.rmtree(workdir / job_id)
            raise HTTPException(400, "Upload exactly one .tif image (plus optional .tfw / .prj).")
        stem = Path(_safe_name(images[0].filename)).stem
        for f in files:
            ext = Path(f.filename or "").suffix.lower()
            if ext not in ALLOWED:
                continue
            target = in_dir / f"{stem}{ext}"           # companions must share the image's name
            with target.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(target)
        image = next(p for p in saved if p.suffix in (".tif", ".tiff"))
        job = Job(id=job_id, image=images[0].filename)
        jobs[job_id] = job
        pool.submit(work, job, image, crs.strip(), vehicles)
        return {"id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job")
        with lock:
            return job.__dict__.copy()

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        job = jobs.get(job_id)
        if not job or name not in job.files.values():
            raise HTTPException(404, "File not found")
        return FileResponse(workdir / job_id / "output" / name, filename=name)

    return app
