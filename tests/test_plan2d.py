import numpy as np

from roomscan.geometry import plan2d as p2


def _square_room(side=4.0, gap=0.0, n=4000, seed=0):
    """Wall-band points of an axis-aligned square room, optionally with a doorway-sized gap in one wall."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, side, n)
    walls = [np.c_[t, np.zeros(n)], np.c_[t, np.full(n, side)], np.c_[np.zeros(n), t], np.c_[np.full(n, side), t]]
    if gap:
        w = walls[0]
        walls[0] = w[(w[:, 0] < side / 2 - gap / 2) | (w[:, 0] > side / 2 + gap / 2)]
    uv = np.concatenate(walls)
    h = rng.uniform(0.25, 1.75, len(uv))
    return uv, h


def test_enclosed_by_walls_fills_sparse_room():
    uv, h = _square_room()
    grid = p2.Grid(np.array([-0.5, -0.5]), 0.02, (250, 250))
    walls = p2.vertical_coverage(uv, h, grid, 0.2, 1.8) >= 0.35
    cams = grid.to_px(np.array([[2.0, 2.0]]))  # one viewpoint in the middle, no rays needed
    inside = p2.enclosed_by_walls(uv, walls, grid, cams)
    area = inside.sum() * grid.res ** 2
    assert 15.0 < area < 16.2  # 4 x 4 m room, minus wall thickness


def test_enclosed_by_walls_does_not_leak_through_small_gaps():
    uv, h = _square_room(gap=0.4)  # smaller than 2 x the 0.3 m gap closing
    grid = p2.Grid(np.array([-0.5, -0.5]), 0.02, (250, 250))
    walls = p2.vertical_coverage(uv, h, grid, 0.2, 1.8) >= 0.35
    inside = p2.enclosed_by_walls(uv, walls, grid, grid.to_px(np.array([[2.0, 2.0]])))
    assert inside.sum() * grid.res ** 2 < 16.5
