"""Kaggle remote runner: run `roomscan run` on a GPU for every capture in the private `roomscan-captures` dataset.

Driven by scripts/kaggle_run.sh, which stages the code and the captures (private datasets) and a jobs.json:
    [{"capture": "house_b", "args": ["-p", "cuda-16gb"]}, ...]
Outputs (plan.json, plan.svg, config.resolved.yaml, log) land in /kaggle/working/out/<capture>/ and are downloaded back.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DA3_COMMIT = "3d835ec"
WORK = Path("/kaggle/working")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def sh(cmd: str, check: bool = True) -> int:
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


def extract_all() -> None:
    for arc in glob.glob("/kaggle/input/**/*.tar", recursive=True):
        dst = WORK / "extracted"
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(arc) as t:
            t.extractall(dst)


def find(pattern: str) -> Path:
    """Kaggle unpacks uploaded archives itself (files under /kaggle/input); fall back to our own extraction."""
    for root in ("/kaggle/input", str(WORK / "extracted")):
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(pattern)


def main() -> None:
    t0 = time.time()
    extract_all()
    code = find("roomscan/__init__.py").parents[1]
    caps = find("jobs.json").parent
    # the pipeline's runtime deps (Kaggle has torch/numpy/opencv/transformers already); --no-deps for research repos
    sh("pip install -q pillow-heif typer rich jsonschema pycolmap kornia imageio-ffmpeg 'shapely>=2'")
    sh(f"pip install -q --no-deps git+https://github.com/ByteDance-Seed/depth-anything-3@{DA3_COMMIT}")
    sh("pip install -q addict einops omegaconf evo plyfile 'moviepy==1.0.3' safetensors huggingface_hub")
    print(f"setup {time.time() - t0:.0f} s", flush=True)
    env = os.environ | {"PYTHONPATH": str(code)}
    out = WORK / "out"
    summary = []
    for job in json.loads((caps / "jobs.json").read_text()):
        cap = caps / job["capture"]
        log = WORK / f"{job['capture']}.log"
        t = time.time()
        cmd = [sys.executable, "-m", "roomscan.cli", "run", str(cap), "-o", str(out), *job.get("args", [])]
        with open(log, "w") as f:
            rc = subprocess.run(cmd, env=env, cwd=code, stdout=f, stderr=subprocess.STDOUT).returncode
        summary.append({"capture": job["capture"], "args": job.get("args", []), "exit": rc,
                        "seconds": round(time.time() - t, 1)})
        print(summary[-1], flush=True)
        print(open(log).read()[-1500:], flush=True)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    for p in WORK.glob("extracted"):
        shutil.rmtree(p, ignore_errors=True)  # don't ship the inputs back


if __name__ == "__main__":
    main()
