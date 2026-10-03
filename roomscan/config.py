"""Layered pipeline config with hardware profiles.

    default.yaml  <  profiles/<name>.yaml  <  --config file.yaml  <  --set a.b=value

`profile="auto"` picks the strongest profile the detected hardware can hold. Detection never imports torch,
so `roomscan hw` works before the ML extras are installed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from roomscan import models
from roomscan.paths import data_dir

CONFIG_DIR = data_dir("configs")
PROFILE_META_KEYS = {"description"}


@dataclass
class Hardware:
    os: str
    arch: str
    ram_gb: float
    apple_silicon: bool
    cuda_vram_gb: float  # largest GPU, 0 if none

    def describe(self) -> str:
        gpu = f"CUDA {self.cuda_vram_gb:.0f} GB" if self.cuda_vram_gb else ("Apple GPU" if self.apple_silicon else "no GPU")
        return f"{self.os}/{self.arch}, {self.ram_gb:.0f} GB RAM, {gpu}"


def detect_hardware() -> Hardware:
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        ram = 0.0
    vram = 0.0
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
            vram = max(float(x) for x in out.split()) / 1024
        except (subprocess.SubprocessError, ValueError):
            vram = 0.0
    sysname, arch = platform.system(), platform.machine()
    return Hardware(sysname, arch, round(ram, 1), sysname == "Darwin" and arch == "arm64", round(vram, 1))


def auto_profile(hw: Hardware) -> str:
    # Thresholds sit slightly under the nominal sizes: a "16 GB" machine reports ~15.x GB.
    if hw.cuda_vram_gb >= 22:
        return "cuda-24gb"
    if hw.cuda_vram_gb >= 11:
        return "cuda-16gb"
    if hw.apple_silicon and hw.ram_gb >= 30:
        return "mac-32gb"
    if hw.apple_silicon and hw.ram_gb >= 15:
        return "mac-16gb"
    return "lite"


def available_profiles() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "profiles").glob("*.yaml"))


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict, path: str = "") -> dict:
    """Deep-merge `over` into a copy of `base`, rejecting keys that base does not define."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        key = f"{path}{k}"
        if k not in base:
            raise ConfigError(f"unknown config key '{key}'")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise ConfigError(f"'{key}' must be a mapping")
            out[k] = _merge(base[k], v, key + ".")
        else:
            out[k] = v
    return out


def _parse_set(item: str) -> dict:
    """'recon.max_views=16' -> {'recon': {'max_views': 16}} (value parsed as YAML)."""
    if "=" not in item:
        raise ConfigError(f"--set expects key=value, got '{item}'")
    dotted, raw = item.split("=", 1)
    node: Any = yaml.safe_load(raw)
    for part in reversed(dotted.strip().split(".")):
        node = {part: node}
    return node


def _validate(cfg: dict) -> None:
    for key in ("model", "metric_model"):
        models.get(cfg["recon"][key])
    if cfg["recon"]["image_long_side"] % 14:
        raise ConfigError("recon.image_long_side must be a multiple of 14 (ViT patch size)")
    if cfg["runtime"]["device"] not in {"auto", "cpu", "mps", "cuda"}:
        raise ConfigError("runtime.device must be auto|cpu|mps|cuda")
    if cfg["recon"]["photo_matcher"] not in {"disk_lightglue", "mast3r"}:
        raise ConfigError("recon.photo_matcher must be disk_lightglue|mast3r")
    if cfg["recon"]["photo_outline"] not in {"walls", "box"}:
        raise ConfigError("recon.photo_outline must be walls|box")
    if cfg["recon"]["video_matcher"] not in {"sequential", "exhaustive"}:
        raise ConfigError("recon.video_matcher must be sequential|exhaustive")
    if cfg["recon"]["video_features"] not in {"sift", "disk_lightglue"}:
        raise ConfigError("recon.video_features must be sift|disk_lightglue")
    if cfg["drift"]["method"] is False:  # YAML 1.1 reads a bare `off` as boolean false
        cfg["drift"]["method"] = "off"
    if cfg["drift"]["method"] not in {"off", "posegraph"}:
        raise ConfigError("drift.method must be off|posegraph")
    if cfg["damage"]["backend"] is False:  # `-s damage.backend=off` (YAML 1.1 boolean)
        cfg["damage"]["backend"] = "off"
    if cfg["damage"]["backend"] not in {"owlv2", "vlm", "off"}:
        raise ConfigError("damage.backend must be owlv2|vlm|off")
    if cfg["damage"]["cache"] not in {"replay", "live", "replay_or_live"}:
        raise ConfigError("damage.cache must be replay|live|replay_or_live")


def resolve_device(requested: str, hw: Hardware) -> str:
    if requested != "auto":
        return requested
    if hw.cuda_vram_gb:
        return "cuda"
    return "mps" if hw.apple_silicon else "cpu"


@dataclass
class Resolved:
    profile: str
    hardware: Hardware
    cfg: dict

    def __getitem__(self, k: str) -> Any:
        return self.cfg[k]

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()[:12]

    def dump(self) -> str:
        meta = {"profile": self.profile, "hardware": asdict(self.hardware), "digest": self.digest}
        return yaml.safe_dump({"_meta": meta, **self.cfg}, sort_keys=False)


def load_config(
    profile: str = "auto",
    config_file: Path | None = None,
    overrides: list[str] | None = None,
    hw: Hardware | None = None,
) -> Resolved:
    hw = hw or detect_hardware()
    name = auto_profile(hw) if profile == "auto" else profile
    if name not in available_profiles():
        raise ConfigError(f"unknown profile '{name}'; available: {available_profiles()} or 'auto'")

    cfg = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text())
    prof = yaml.safe_load((CONFIG_DIR / "profiles" / f"{name}.yaml").read_text()) or {}
    cfg = _merge(cfg, {k: v for k, v in prof.items() if k not in PROFILE_META_KEYS})
    if config_file:
        cfg = _merge(cfg, yaml.safe_load(Path(config_file).read_text()) or {})
    for item in overrides or []:
        cfg = _merge(cfg, _parse_set(item))
    _validate(cfg)
    cfg["runtime"]["device"] = resolve_device(cfg["runtime"]["device"], hw)
    return Resolved(name, hw, cfg)
