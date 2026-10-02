"""Registry of pretrained models the config may name. Licences and sizes checked on Hugging Face 2026-10-02.

Disclosure (case-study constraint): every model used for a reported number is listed in the benchmark report
with its licence; the resolved config saved next to each output names the exact one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    hf_repo: str
    params_m: int
    license: str
    metric: bool  # outputs metric-scale depth
    poses: bool  # estimates camera poses from images


MODELS: dict[str, ModelSpec] = {
    "da3-small": ModelSpec("depth-anything/DA3-SMALL", 34, "apache-2.0", False, True),
    "da3-base": ModelSpec("depth-anything/DA3-BASE", 135, "apache-2.0", False, True),
    "da3-large-1.1": ModelSpec("depth-anything/DA3-LARGE-1.1", 411, "apache-2.0", False, True),
    "da3-giant-1.1": ModelSpec("depth-anything/DA3-GIANT-1.1", 1356, "cc-by-nc-4.0", False, True),
    "da3-nested-giant-large": ModelSpec("depth-anything/DA3NESTED-GIANT-LARGE", 1690, "cc-by-nc-4.0", True, True),
    "da3metric-large": ModelSpec("depth-anything/DA3METRIC-LARGE", 334, "apache-2.0", True, False),
    "mapanything-apache": ModelSpec("facebook/map-anything-apache", 0, "apache-2.0", True, True),
}


def get(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model '{name}'; known: {sorted(MODELS)}") from None
