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


def detect_tier(path: Path) -> str:
    if (path / "odometry.csv").exists() and (path / "depth").is_dir():
        return "lidar"
    if path.is_file() and path.suffix.lower() in {".mov", ".mp4"}:
        return "video"
    if path.is_dir() and any(p.is_dir() for p in path.iterdir()):
        return "photos"  # one sub-folder of stills per room
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


if __name__ == "__main__":
    app()
