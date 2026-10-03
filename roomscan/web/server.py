"""Local test website: `roomscan serve`, then open http://localhost:8765.

Upload a capture (a folder of room photo folders, a video, a Stray Scanner folder or its zip), run the pipeline and
inspect the plan, every measurement with its 90% interval, and an optional comparison with tape measurements.

Safety: one job at a time, each in its own process under scripts/run_guarded.sh, with the profile's GPU-memory cap.
(Two concurrent photo runs on an 8 GB Mac would freeze it; see docs/CHANGELOG.md 0.3.1.)
Runs and uploads live in runs/web/ (git-ignored); the job list survives restarts.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from roomscan import __version__
from roomscan.config import auto_profile, available_profiles, detect_hardware

PKG = Path(__file__).resolve().parents[1]
CHECKOUT = PKG.parent if (PKG.parent / "pyproject.toml").exists() else None
STATIC = Path(__file__).resolve().parent / "static"
IMG_EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png"}
VID_EXT = {".mov", ".mp4", ".m4v"}


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.index = root / "jobs.json"
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = json.loads(self.index.read_text()) if self.index.exists() else {}
        for j in self.jobs.values():  # a restart interrupts anything that was running
            if j["status"] in ("queued", "running"):
                j["status"] = "interrupted"
        self.save()

    def save(self):
        with self.lock:
            self.index.write_text(json.dumps(self.jobs, indent=1))

    def update(self, jid: str, **kw):
        with self.lock:
            self.jobs[jid].update(kw)
        self.save()


def find_capture(inp: Path) -> tuple[Path, str]:
    """Locate the capture inside an upload and name its tier."""
    for z in list(inp.rglob("*.zip")):  # Stray Scanner shares a zip
        with zipfile.ZipFile(z) as f:
            f.extractall(z.parent / z.stem)
        z.unlink()
    odo = sorted(inp.rglob("odometry.csv"))
    if odo:
        return odo[0].parent, "lidar"
    vids = [p for p in inp.rglob("*") if p.suffix.lower() in VID_EXT]
    if vids:
        return vids[0], "video"
    imgs = [p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXT]
    if not imgs:
        raise ValueError("no photos, video or Stray Scanner capture found in the upload")
    room_dirs = {p.parent for p in imgs}
    if len(room_dirs) == 1 and next(iter(room_dirs)) == inp:  # loose photos: treat as one room
        room = inp / "room1"
        room.mkdir()
        for p in imgs:
            p.rename(room / p.name)
        return inp, "photos"
    parents = {d.parent for d in room_dirs}
    return (parents.pop() if len(parents) == 1 else inp), "photos"


class Runner(threading.Thread):
    def __init__(self, store: JobStore):
        super().__init__(daemon=True)
        self.store, self.q = store, queue.Queue()

    def run(self):
        while True:
            jid = self.q.get()
            self.run_job(jid)

    def run_job(self, jid: str):
        job = self.store.jobs[jid]
        jdir = self.store.root / jid
        out = jdir / "out"
        try:
            if job.get("local_path"):  # on-disk capture: the CLI resolves zips / wrapping folders and the tier
                from roomscan.cli import detect_tier, resolve_capture

                cap = resolve_capture(Path(job["local_path"]), out)
                tier = detect_tier(cap)
            else:
                cap, tier = find_capture(jdir / "input")
        except Exception as e:  # noqa: BLE001
            self.store.update(jid, status="failed", error=str(e))
            return
        shown = str(cap.relative_to(jdir)) if jdir in cap.parents else str(cap)
        self.store.update(jid, status="running", tier=tier, started=time.time(), capture=shown)
        cmd = [sys.executable, "-m", "roomscan.cli", "run", str(cap), "-o", str(out), "-p", job["profile"]]
        guard = CHECKOUT / "scripts/run_guarded.sh" if CHECKOUT else None
        if guard and guard.exists():
            cmd = [str(guard)] + cmd
        env = dict(os.environ, LIMIT_MB=str(job.get("limit_mb", 5500)), PYTHONUNBUFFERED="1")
        if CHECKOUT:  # run the code in this checkout, not a possibly stale installed copy
            env["PYTHONPATH"] = str(CHECKOUT) + os.pathsep + env.get("PYTHONPATH", "")
        log = jdir / "log.txt"
        with log.open("w") as fh:
            code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(CHECKOUT or jdir)).returncode
        plans = sorted(out.glob("*/plan.json"))
        text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
        if code == 0 and plans:
            self.store.update(jid, status="done", finished=time.time(), plan=str(plans[0].relative_to(jdir)))
        else:
            err = "killed by the memory watchdog" if code == 137 else _last_error(text)
            self.store.update(jid, status="failed", finished=time.time(), error=err)


def _last_error(text: str) -> str:
    lines = [ln.strip(" │") for ln in text.splitlines() if ln.strip(" │")]
    for ln in reversed(lines):
        if re.search(r"(Error|Exception|not implemented)", ln):
            return ln[:300]
    return lines[-1][:300] if lines else "failed"


def create_app(root: Path | None = None) -> FastAPI:
    root = root or ((CHECKOUT or Path.cwd()) / "runs/web")
    store = JobStore(root)
    runner = Runner(store)
    runner.start()
    app = FastAPI(title="Room-Scanner test bench")
    hw = detect_hardware()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/info")
    def info():
        gt = None
        if CHECKOUT and (CHECKOUT / "bench/ground_truth/home_tape.yaml").exists():
            import yaml

            gt = yaml.safe_load((CHECKOUT / "bench/ground_truth/home_tape.yaml").read_text())
        return {"version": __version__, "hardware": hw.describe(), "auto_profile": auto_profile(hw),
                "profiles": available_profiles(), "ground_truth": gt}

    @app.post("/api/jobs")
    async def create_job(request: Request):
        # a Stray Scanner folder is 3-20k files; Starlette's default form limit (1000) rejected every LiDAR upload
        form = await request.form(max_files=200_000, max_fields=200_000)
        files = form.getlist("files")
        if not files:
            raise HTTPException(400, "no files uploaded")
        rel = json.loads(form.get("paths") or "[]")
        profile, label = form.get("profile") or "auto", form.get("label") or ""
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        inp = root / jid / "input"
        inp.mkdir(parents=True)
        for i, f in enumerate(files):
            name = rel[i] if i < len(rel) and rel[i] else f.filename
            dest = (inp / name).resolve()
            if inp.resolve() not in dest.parents:
                raise HTTPException(400, f"bad path {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                shutil.copyfileobj(f.file, fh)
        return _queue(jid, label or (rel[0].split("/")[0] if rel else files[0].filename), profile, len(files))

    @app.post("/api/jobs/local")
    async def create_local_job(request: Request):
        """Run a capture already on this machine (no upload): the local showcase path. Restricted to the user's home."""
        body = await request.json()
        path = Path(str(body.get("path", ""))).expanduser().resolve()
        if not path.exists():
            raise HTTPException(400, f"not found: {path}")
        if Path.home().resolve() not in path.parents:
            raise HTTPException(400, "only captures inside your home folder can be run")
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        (root / jid).mkdir(parents=True)
        return _queue(jid, body.get("label") or path.name, body.get("profile") or "auto", None, str(path))

    def _queue(jid, label, profile, n_files, local_path=None):
        store.jobs[jid] = {"id": jid, "label": label, "profile": profile, "status": "queued", "created": time.time(),
                           "n_files": n_files, "pipeline_version": __version__,
                           **({"local_path": local_path} if local_path else {})}
        store.save()
        runner.q.put(jid)
        return {"id": jid}

    @app.get("/api/jobs")
    def list_jobs():
        return sorted(store.jobs.values(), key=lambda j: j["created"], reverse=True)

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        if jid not in store.jobs:
            raise HTTPException(404)
        j = dict(store.jobs[jid])
        log = root / jid / "log.txt"
        if log.exists():
            j["log_tail"] = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))[-3000:]
        if j.get("plan"):
            j["plan_json"] = json.loads((root / jid / j["plan"]).read_text())
        return JSONResponse(j)

    @app.get("/api/jobs/{jid}/plan.svg")
    def plan_svg(jid: str):
        j = store.jobs.get(jid)
        if not j or not j.get("plan"):
            raise HTTPException(404)
        return FileResponse(root / jid / Path(j["plan"]).with_suffix(".svg"), media_type="image/svg+xml")

    @app.get("/api/jobs/{jid}/plan.json")
    def plan_json(jid: str):
        j = store.jobs.get(jid)
        if not j or not j.get("plan"):
            raise HTTPException(404)
        return FileResponse(root / jid / j["plan"], filename=f"plan_{jid}.json")

    return app


def serve(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
