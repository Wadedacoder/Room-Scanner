"""Import Depth Anything 3 without its export helpers.

`depth_anything_3.api` imports `depth_anything_3.utils.export` at module load, which pulls in open3d and pycolmap. On
macOS each of those ships its own OpenMP runtime next to torch's, and loading two aborts the process
("OMP: Error #15"). The pipeline only needs predictions, never DA3's file exporters, so that one module is replaced
by a stub that fails loudly if anything calls it.
"""

from __future__ import annotations

import sys
import types


def _export_disabled(*_args, **_kwargs):
    raise RuntimeError("DA3 file export is disabled in roomscan (see roomscan/recon/da3_loader.py)")


def load_da3_class():
    name = "depth_anything_3.utils.export"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.export = _export_disabled
        sys.modules[name] = stub
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3


def load_pretrained(cls, repo: str):
    """Local copy first: a run must not depend on the network once weights are downloaded (scripts/setup_models.sh).
    E15: a Hugging Face "Server disconnected" mid-run killed a photo-tier job although every weight was cached."""
    try:
        return cls.from_pretrained(repo, local_files_only=True)
    except Exception:  # noqa: BLE001  (not cached yet: download once)
        return cls.from_pretrained(repo)
