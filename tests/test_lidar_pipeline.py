import json
from pathlib import Path

import jsonschema
import pytest

from roomscan.config import Hardware, load_config
from roomscan.pipeline.lidar import run_lidar

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data/raw/lidar/c00a170fe1"


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/fetch_data.sh first")
def test_single_capture_end_to_end():
    cfg = load_config("lite", hw=Hardware("Darwin", "arm64", 8.0, True, 0.0))
    plan, _ = run_lidar(DATA, cfg)
    jsonschema.validate(plan, json.loads((ROOT / "schema/plan.schema.json").read_text()))
    assert len(plan["rooms"]) == 3  # living, corridor, bathroom (bench/annotations/c00a170fe1_rooms.yaml)
    # this walk never looked up: the pipeline must say so instead of inventing a ceiling
    assert all(r["ceiling_height"] is None for r in plan["rooms"])
    for r in plan["rooms"]:
        for w in r["walls"]:
            m = w["length"]
            assert m["lo"] < m["value"] < m["hi"]
