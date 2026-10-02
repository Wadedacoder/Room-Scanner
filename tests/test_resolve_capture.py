import zipfile

from roomscan.cli import detect_tier, resolve_capture


def test_zip_with_wrapping_folder_and_nested_stray_scan(tmp_path):
    scan = tmp_path / "src" / "export" / "abc123"
    (scan / "depth").mkdir(parents=True)
    (scan / "odometry.csv").write_text("t\n")
    z = tmp_path / "scan.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(scan / "odometry.csv", "export/abc123/odometry.csv")
        f.writestr("export/abc123/depth/000000.png", b"")
        f.writestr("__MACOSX/._x", b"")
    got = resolve_capture(z, tmp_path / "out")
    assert got.name == "abc123" and detect_tier(got) == "lidar"
    assert resolve_capture(tmp_path / "src", tmp_path / "out") == scan  # folder wrapping one scan


def test_zipped_photo_capture_descends_to_room_folders(tmp_path):
    z = tmp_path / "house.zip"
    with zipfile.ZipFile(z, "w") as f:
        for room in ("kitchen", "study"):
            f.writestr(f"house/{room}/a.jpg", b"x")
    got = resolve_capture(z, tmp_path / "out")
    assert got.name == "house" and detect_tier(got) == "photos"
