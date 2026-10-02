import numpy as np
import pytest

from roomscan.recon.windows import WindowPrediction, apply_sim3, make_windows, run_windowed


def test_make_windows_cover_and_overlap():
    w = make_windows(10, 4, 2)
    assert w[0] == [0, 1, 2, 3] and w[-1][-1] == 9
    assert all(len(set(a) & set(b)) >= 2 for a, b in zip(w, w[1:]))
    assert make_windows(3, 8, 2) == [[0, 1, 2]]


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@pytest.mark.parametrize("n,size,overlap", [(7, 4, 2), (12, 5, 2), (60, 8, 2)])
def test_windowed_recovers_global_frame(n, size, overlap):
    rng = np.random.default_rng(0)
    gt = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        gt[i, :3, :3] = _rot(0.15 * i)
        gt[i, :3, 3] = [0.3 * i, 0.0, 0.1 * i]
    depth = rng.uniform(1, 4, (n, 12, 16)).astype(np.float32)

    def predict(idx):
        # each window lives in its own random frame and scale, like a scale-ambiguous model
        s = rng.uniform(0.3, 3.0)
        R, t = _rot(rng.uniform(-3, 3)), rng.normal(size=3)
        T = apply_sim3(gt[idx], 1 / s, R, t)
        return WindowPrediction(T, depth[idx] / s, np.ones_like(depth[idx]), np.tile(np.eye(3), (len(idx), 1, 1)))

    out = run_windowed(n, size, overlap, predict)
    # bring result into GT frame with ONE similarity fitted on view 0 and its depth
    s = np.median(depth[0] / out.depth[0])
    R = gt[0, :3, :3] @ out.c2w[0, :3, :3].T
    t = gt[0, :3, 3] - s * R @ out.c2w[0, :3, 3]
    aligned = apply_sim3(out.c2w, s, R, t)
    assert np.allclose(aligned, gt, atol=1e-6)
    assert np.allclose(out.depth * s, depth, rtol=1e-5)
