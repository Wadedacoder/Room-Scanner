import zipfile

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from roomscan.web.server import create_app, find_capture


def test_find_capture_photo_folder(tmp_path):
    (tmp_path / "home/kitchen").mkdir(parents=True)
    (tmp_path / "home/kitchen/a.jpg").write_bytes(b"x")
    (tmp_path / "home/hall").mkdir()
    (tmp_path / "home/hall/b.jpg").write_bytes(b"x")
    cap, tier = find_capture(tmp_path)
    assert tier == "photos" and cap == tmp_path / "home"


def test_find_capture_loose_photos_become_one_room(tmp_path):
    (tmp_path / "a.HEIC").write_bytes(b"x")
    (tmp_path / "b.HEIC").write_bytes(b"x")
    cap, tier = find_capture(tmp_path)
    assert tier == "photos" and (cap / "room1" / "a.HEIC").exists()


def test_find_capture_stray_zip(tmp_path):
    z = tmp_path / "scan.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("abc123/odometry.csv", "t\n")
        f.writestr("abc123/rgb.mp4", "")
    cap, tier = find_capture(tmp_path)
    assert tier == "lidar" and cap.name == "abc123"


def test_find_capture_video(tmp_path):
    (tmp_path / "walk.MOV").write_bytes(b"x")
    assert find_capture(tmp_path)[1] == "video"


def test_info_endpoint(tmp_path):
    c = TestClient(create_app(tmp_path))
    j = c.get("/api/info").json()
    assert j["auto_profile"] and "lite" in j["profiles"]
    assert c.get("/").status_code == 200
