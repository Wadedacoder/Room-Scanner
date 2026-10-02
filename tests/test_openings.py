import numpy as np

from roomscan.openings.detect import detect_openings


def test_door_found_where_camera_sees_through_wall():
    # 4 x 3 m room; the wall y = 3 has a 0.9 m door at x = 1.5..2.4; a camera inside sees the next room through it
    rng = np.random.default_rng(0)
    poly = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float)
    wall_x = rng.uniform(0, 4, 6000)
    wall_x = wall_x[(wall_x < 1.5) | (wall_x > 2.4)]
    wall = np.c_[wall_x, np.full_like(wall_x, 3.0)]
    beyond = np.c_[rng.uniform(1.0, 2.9, 3000), rng.uniform(3.6, 5.0, 3000)]  # next room, seen through the door
    cam = np.array([1.95, 1.0])
    # keep only beyond-points whose ray from the camera actually passes the door gap
    t = (3.0 - cam[1]) / (beyond[:, 1] - cam[1])
    xc = cam[0] + t * (beyond[:, 0] - cam[0])
    beyond = beyond[(xc > 1.5) & (xc < 2.4)]
    pts = np.r_[wall, beyond]
    h = np.r_[rng.uniform(0.3, 1.8, len(wall)), rng.uniform(0.0, 2.0, len(beyond))]
    ops = detect_openings(poly, pts, h, np.repeat(cam[None], len(pts), 0))
    doors = [o for o in ops if o.kind in ("door", "open_passage")]
    assert len(doors) == 1
    assert abs(doors[0].width - 0.9) <= 0.1 and abs(doors[0].centre_uv[0] - 1.95) < 0.1
