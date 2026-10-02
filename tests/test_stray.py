from pathlib import Path

import numpy as np
import pytest

from roomscan.io.stray import StrayCapture, quat_to_rot

DATA = Path(__file__).parents[1] / "data/raw/lidar/c00a170fe1"


def test_quat_identity():
    assert np.allclose(quat_to_rot(np.array([[0, 0, 0, 1.0]]))[0], np.eye(3))


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/fetch_data.sh first")
def test_points_are_gravity_aligned():
    cap = StrayCapture.open(DATA)
    pts = np.concatenate([cap.points_world(i) for i in range(0, len(cap), 50)])
    # the floor (lowest dense level) must sit 0.8-2.0 m *below* the handheld camera; a wrong camera
    # convention mirrors it above the camera
    floor = np.percentile(pts[:, 1], 1)
    cam_h = np.median(cap.poses[:, 1, 3]) - floor
    assert 0.8 < cam_h < 2.0
