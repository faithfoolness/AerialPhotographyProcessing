"""Web interface: upload orthophotos, watch progress, inspect lines on a map."""

from __future__ import annotations

import logging
import re
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Params
from ..export import write_combined, write_outputs
from ..pipeline import run

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
IMAGE_EXT = {".tif", ".tiff"}
COMPANION_EXT = {".tfw", ".tifw", ".prj"}


@dataclass
class ImageTask:
    name: str                        # original file name
    stem: str                        # safe name used on disk
    status: str = "queued"           # queued | running | done | error
    stats: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class Job:
    id: str
    images: list[ImageTask]
    status: str = "queued"           # queued | running | done | error
    progress: float = 0.0
    message: str = "Waiting in queue"
    current: int = 0                 # index of the image being processed
    stats: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    error: str | None = None

    def all_files(self) -> set[str]:
        names = set(self.files.values())
        for im in self.images:
            names.update(im.files.values())
        return names


def _safe_stem(name: str) -> str:
    stem = Path(Path(name).name).stem
    return re.sub(r"[^\w.+-]+", "_", stem, flags=re.UNICODE).strip("._") or "image"


def _group_uploads(files: list[UploadFile]) -> list[tuple[UploadFile, list[UploadFile]]]:
    """Pair every image with its .tfw/.prj by file name.

    With a single image every companion belongs to it, whatever its name, so a
    mismatched world-file name is forgiven in the common one-image case.
    """
    images = [f for f in files if Path(f.filename or "").suffix.lower() in IMAGE_EXT]
    companions = [f for f in files if Path(f.filename or "").suffix.lower() in COMPANION_EXT]
    if len(images) == 1:
        return [(images[0], companions)]
    key = lambda f: Path(f.filename or "").stem.lower()  # noqa: E731
    return [(im, [c for c in companions if key(c) == key(im)]) for im in images]


def create_app(workdir: Path = Path("runs")) -> FastAPI:
    workdir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Aerial road markings")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    jobs: dict[str, Job] = {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=1)    # one heavy job at a time

    def work(job: Job, crs: str | None, vehicles: bool):
        out_dir = workdir / job.id / "output"
        n = len(job.images)
        results = []
        with lock:
            job.status = "running"
        for i, task in enumerate(job.images):
            def progress(frac: float, msg: str, i=i, task=task):
                with lock:
                    job.current = i
                    job.progress = round((i + frac) / n, 3)
                    job.message = f"{task.name}: {msg}" if n > 1 else msg

            with lock:
                task.status = "running"
            try:
                params = Params()
                params.vehicles.enabled = vehicles
                image = next((workdir / job.id / "input").glob(f"{task.stem}.tif*"))
                result = run(image, params, crs=crs or None, progress=progress)
                files = write_outputs(result, out_dir / task.stem)
                results.append((task.stem, result.markings, result.stats))
                with lock:
                    task.stats = result.stats
                    task.files = {k: f"{task.stem}/{v}" for k, v in files.items()}
                    task.status = "done"
                del result                               # free the image before the next one
            except Exception as exc:
                log.error("Job %s, image %s failed\n%s", job.id, task.name, traceback.format_exc())
                with lock:
                    task.status, task.error = "error", str(exc)

        try:
            if results:
                files, stats = write_combined(results, out_dir)
            else:
                files, stats = {}, {}
            with lock:
                job.files, job.stats = files, stats
                failed = sum(t.status == "error" for t in job.images)
                if failed == n:
                    job.status, job.message = "error", "Failed"
                    job.error = job.images[0].error if n == 1 else "All images failed"
                else:
                    job.status, job.progress = "done", 1.0
                    job.message = "Done" if not failed else f"Done, {failed} of {n} failed"
        except Exception as exc:
            log.error("Job %s failed\n%s", job.id, traceback.format_exc())
            with lock:
                job.status, job.error, job.message = "error", str(exc), "Failed"

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/api/jobs")
    async def create_job(
        files: list[UploadFile] = File(..., description="GeoTIFFs, optionally with .tfw/.prj"),
        crs: str = Form(""),
        vehicles: bool = Form(True),
    ):
        groups = _group_uploads(files)
        if not groups:
            raise HTTPException(400, "Upload at least one .tif image (with .tfw / .prj if it has them).")
        job_id = uuid.uuid4().hex[:12]
        in_dir = workdir / job_id / "input"
        in_dir.mkdir(parents=True)
        tasks, used = [], set()
        for image, companions in groups:
            stem = _safe_stem(image.filename)
            base, k = stem, 2
            while stem.lower() in used:                  # same name from different folders
                stem, k = f"{base}_{k}", k + 1
            used.add(stem.lower())
            for f in (image, *companions):
                ext = Path(f.filename or "").suffix.lower()
                with (in_dir / f"{stem}{ext}").open("wb") as out:
                    shutil.copyfileobj(f.file, out)
            tasks.append(ImageTask(name=Path(image.filename).name, stem=stem))
        job = Job(id=job_id, images=tasks)
        jobs[job_id] = job
        pool.submit(work, job, crs.strip(), vehicles)
        return {"id": job_id, "images": len(tasks)}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job")
        with lock:
            return asdict(job)

    @app.get("/api/jobs/{job_id}/files/{name:path}")
    def job_file(job_id: str, name: str):
        job = jobs.get(job_id)
        if not job or name not in job.all_files():
            raise HTTPException(404, "File not found")
        return FileResponse(workdir / job_id / "output" / name, filename=Path(name).name)

    return app
