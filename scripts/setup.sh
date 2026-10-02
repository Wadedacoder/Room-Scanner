#!/usr/bin/env bash
# Clean-machine setup: uv + Python 3.11 venv + package.
#   ./scripts/setup.sh            regular install (what graders / the walk-in machine should use)
#   ./scripts/setup.sh --dev      editable install for development
# Extras default to "geo,vlm,dev"; override with EXTRAS=geo,vlm,web,dev ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRAS="${EXTRAS:-geo,vlm,dev}"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv venv -p 3.11 --allow-existing -q
if [ "${1:-}" = "--dev" ]; then
  uv pip install -q -e ".[${EXTRAS}]"
  # macOS: uv marks .venv hidden and recent CPython skips *hidden* .pth files, which silently breaks an editable
  # install ("No module named roomscan"). Clear the flag. Any later `uv pip install` can re-hide it; rerun this
  # script, or run with PYTHONPATH=. as a fallback.
  if [ "$(uname)" = "Darwin" ]; then chflags nohidden .venv/lib/python*/site-packages/*.pth; fi
else
  uv pip install -q --reinstall-package roomscan ".[${EXTRAS}]"   # regular install: no .pth involved
fi
.venv/bin/roomscan hw
