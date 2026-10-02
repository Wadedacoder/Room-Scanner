#!/usr/bin/env bash
# Learned-geometry stack for the photo / video tiers (run after scripts/setup.sh).
#   torch + Depth Anything 3 (pinned commit, installed WITHOUT its own dependency pins: it pins numpy<2 and
#   moviepy==1.0.3 for exporters we never call) + the modules its import chain actually needs + pycolmap.
# Then fetches every weight the configured pipeline uses (DA3 geometry + metric from Hugging Face, DISK + LightGlue
# via kornia), so runs work offline afterwards (roomscan/models.py lists the repos and licences).
set -euo pipefail
cd "$(dirname "$0")/.."
DA3_COMMIT=3d835ec
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
"$UV" pip install -q torch torchvision kornia transformers pycolmap huggingface_hub safetensors addict einops omegaconf evo plyfile \
  "moviepy==1.0.3"
"$UV" pip install -q --no-deps "git+https://github.com/ByteDance-Seed/depth-anything-3@${DA3_COMMIT}"
# pycolmap and torch each bundle libomp and abort if loaded in one process (OMP Error #15): the pipeline runs
# COLMAP in a subprocess, and this check imports them separately too.
.venv/bin/python -c "import pycolmap; print('pycolmap', pycolmap.__version__)"
.venv/bin/python - <<'PY'
from roomscan.recon.da3_loader import load_da3_class
import torch
load_da3_class()
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"DA3 + torch {torch.__version__} OK on {dev}")
PY
# Prefetch weights (no-op when already cached). PROFILE=<name> fetches that profile's models instead of the default.
PROFILE="${PROFILE:-auto}" .venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
from roomscan import models
from roomscan.config import load_config

cfg = load_config(profile=os.environ["PROFILE"])
for key in ("model", "metric_model"):
    spec = models.get(cfg["recon"][key])
    path = snapshot_download(spec.hf_repo)
    print(f"weights: {cfg['recon'][key]} ({spec.hf_repo}, {spec.license}) -> {path}")
import kornia.feature as KF  # cross-room photo links; weights land in the torch hub cache
KF.DISK.from_pretrained("depth")
KF.LightGlueMatcher("disk")
print("weights: DISK + LightGlue (kornia) cached")
from roomscan.damage.detect_local import MODEL  # local damage detector (Apache-2.0)
print(f"weights: {MODEL} -> {snapshot_download(MODEL, allow_patterns=['*.json', '*.safetensors', '*.txt'])}")
PY
