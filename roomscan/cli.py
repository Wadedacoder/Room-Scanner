"""One command per capture: `roomscan run <capture> -o out/`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from roomscan.config import ConfigError, available_profiles, detect_hardware, load_config, auto_profile

app = typer.Typer(no_args_is_help=True, add_completion=False)

ProfileOpt = typer.Option("auto", "--profile", "-p", help="hardware profile or 'auto' (see `roomscan hw`)")
ConfigOpt = typer.Option(None, "--config", "-c", help="YAML file layered over the profile")
SetOpt = typer.Option(None, "--set", "-s", help="override a single key, e.g. -s recon.max_views=16")


def _load(profile: str, config: Optional[Path], overrides: Optional[list[str]]):
    try:
        return load_config(profile, config, overrides)
    except ConfigError as e:
        raise typer.BadParameter(str(e)) from None


def resolve_capture(path: Path, out: Path) -> Path:
    """A zip (Stray Scanner shares one) is extracted under the output folder; a folder that only wraps one Stray scan
    resolves to that scan. The original capture is never modified."""
    if path.is_file() and path.suffix.lower() == ".zip":
        import zipfile

        dest = out / "_extracted" / path.stem
        if not dest.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(dest)
        path = dest
        while True:  # a zipped folder usually wraps everything in one top-level folder
            inner = [p for p in path.iterdir() if not p.name.startswith((".", "__MACOSX"))]
            if len(inner) == 1 and inner[0].is_dir():
                path = inner[0]
            else:
                break
    if path.is_dir() and not (path / "odometry.csv").exists():
        # fixed depth (glob follows symlinked folders; rglob does not)
        scans = [p.parent for pat in ("*/odometry.csv", "*/*/odometry.csv") for p in path.glob(pat)
                 if "__MACOSX" not in p.parts]
        if len(scans) == 1:
            return scans[0]
        inner = [p for p in path.iterdir() if not p.name.startswith((".", "__MACOSX"))]
        if len(inner) == 1 and inner[0].is_file() and inner[0].suffix.lower() in {".mov", ".mp4"}:
            return inner[0]
    return path


def detect_tier(path: Path) -> str:
    if (path / "odometry.csv").exists() and (path / "depth").is_dir():
        return "lidar"
    if path.is_file() and path.suffix.lower() in {".mov", ".mp4"}:
        return "video"
    if path.is_dir() and any(p.is_dir() for p in path.iterdir()):
        return "photos"  # one sub-folder of stills per room
    if path.is_dir() and any(p.suffix.lower() in {".jpg", ".jpeg", ".heic", ".heif", ".png"} for p in path.iterdir()):
        return "photos"  # photos without room folders: one room (warned)
    raise typer.BadParameter(f"cannot infer tier for {path}")


@app.command()
def hw():
    """Show detected hardware and the profile `auto` would pick."""
    h = detect_hardware()
    typer.echo(f"hardware : {h.describe()}")
    typer.echo(f"auto     : {auto_profile(h)}")
    typer.echo(f"profiles : {', '.join(available_profiles())}")


@app.command()
def config(profile: str = ProfileOpt, config: Optional[Path] = ConfigOpt, set_: Optional[list[str]] = SetOpt):
    """Print the fully resolved config."""
    typer.echo(_load(profile, config, set_).dump())


@app.command()
def inspect(capture: Path):
    """Print basic stats for a capture."""
    tier = detect_tier(capture)
    info: dict = {"tier": tier}
    if tier == "lidar":
        from roomscan.io.stray import StrayCapture

        info |= StrayCapture.open(capture).summary()
    typer.echo(json.dumps(info, indent=2))


@app.command()
def run(
    capture: Path,
    out: Path = typer.Option(Path("out"), "-o"),
    tier: str = "auto",
    profile: str = ProfileOpt,
    config: Optional[Path] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
):
    """Capture -> plan.json (schema/plan.schema.json) + plan.svg."""
    capture = resolve_capture(capture, out)
    tier = detect_tier(capture) if tier == "auto" else tier
    cfg = _load(profile, config, set_)
    run_dir = out / capture.resolve().name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(cfg.dump())
    typer.echo(f"tier={tier} profile={cfg.profile} device={cfg['runtime']['device']} config={cfg.digest}")
    import jsonschema

    from roomscan.render.svg import render_svg

    if tier == "lidar":
        from roomscan.pipeline.lidar import run_lidar as run_tier
    elif tier == "photos":
        from roomscan.pipeline.photos import run_photos as run_tier
    elif tier == "video":
        from roomscan.pipeline.video import run_video as run_tier
    else:
        raise typer.Exit(f"pipeline for tier '{tier}' not implemented yet")
    plan, _ = run_tier(capture, cfg)
    plan["capture"]["config_digest"] = cfg.digest
    from roomscan.paths import data_dir

    schema = json.loads((data_dir("schema") / "plan.schema.json").read_text())
    jsonschema.validate(plan, schema)
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=1))
    (run_dir / "plan.svg").write_text(render_svg(plan))
    for r in plan["rooms"]:
        c = r["ceiling_height"]
        ceil = "not observed" if c is None else f"{c['value']:.3f} m"
        typer.echo(f"  {r['id']}: {r['floor_area']['value']:.2f} m2, {len(r['walls'])} walls, ceiling {ceil}")
    for w in plan["warnings"]:
        typer.echo(f"  ! {w}")
    typer.echo(f"wrote {run_dir}/plan.json, plan.svg in {plan['capture']['runtime_s']} s")


@app.command()
def serve(port: int = typer.Option(8765, help="local port"), host: str = typer.Option("127.0.0.1")):
    """Local test website: upload a capture, run it, compare with tape (http://localhost:8765)."""
    from roomscan.web.server import serve as _serve

    typer.echo(f"Room-Scanner test bench on http://{host}:{port}  (Ctrl+C to stop)")
    _serve(host, port)


if __name__ == "__main__":
    app()
