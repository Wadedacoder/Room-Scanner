#!/usr/bin/env bash
# Clean-machine setup: uv + Python 3.11 venv + package. Usage: ./scripts/setup.sh [extras, default "geo,dev"]
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRAS="${1:-geo,dev}"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv venv -p 3.11 --allow-existing -q
uv pip install -q -e ".[${EXTRAS}]"
# macOS: uv flags .venv as hidden; recent CPython skips *hidden* .pth files, which silently breaks the
# editable install ("No module named roomscan"). Clear the flag on the .pth files.
if [ "$(uname)" = "Darwin" ]; then chflags nohidden .venv/lib/python*/site-packages/*.pth; fi
.venv/bin/roomscan hw
