"""Locate data that lives at the repo root in a checkout but inside the package in an installed wheel."""

from pathlib import Path

_PKG = Path(__file__).resolve().parent


def data_dir(name: str) -> Path:
    """`configs` or `schema`: wheel copy (roomscan/_data/<name>) if present, else the checkout's <repo>/<name>."""
    packaged = _PKG / "_data" / name
    return packaged if packaged.is_dir() else _PKG.parent / name
