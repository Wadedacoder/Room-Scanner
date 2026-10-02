"""Walk-in robustness: an off-protocol photo capture degrades with warnings instead of failing the run."""

import numpy as np
import pytest
from PIL import Image

from roomscan.io.photos import ASSUMED_F35, load_capture


def _jpg(path, f35=13):
    im = Image.fromarray(np.full((60, 80, 3), 128, np.uint8))
    exif = Image.Exif()
    if f35:
        exif.get_ifd(0x8769)[0xA405] = f35
    im.save(path, exif=exif)


def test_loose_photos_become_one_room(tmp_path):
    for i in range(3):
        _jpg(tmp_path / f"{i}.jpg")
    w = []
    cap = load_capture(tmp_path, w)
    assert list(cap) == [tmp_path.name] and len(cap[tmp_path.name]) == 3 and "one room" in w[0]


def test_too_many_too_few_and_unreadable(tmp_path):
    (tmp_path / "big").mkdir()
    (tmp_path / "tiny").mkdir()
    for i in range(11):
        _jpg(tmp_path / "big" / f"{i:02d}.jpg")
    _jpg(tmp_path / "tiny" / "a.jpg")
    (tmp_path / "big" / "zz.jpg").write_text("not an image")
    w = []
    cap = load_capture(tmp_path, w)
    assert list(cap) == ["big"] and len(cap["big"]) == 8
    assert any("unreadable" in x for x in w) and any("kept 8" in x for x in w) and any("tiny" in x for x in w)


def test_missing_exif_focal_assumes_protocol_lens(tmp_path):
    (tmp_path / "r").mkdir()
    for i in range(2):
        _jpg(tmp_path / "r" / f"{i}.jpg", f35=None)
    w = []
    ph = load_capture(tmp_path, w)["r"]
    assert ph[0].focal_source.startswith("assumed") and any("EXIF" in x for x in w)
    assert ph[0].K[0, 0] == pytest.approx(ASSUMED_F35 * np.hypot(80, 60) / np.hypot(36, 24))


def test_nothing_usable_is_an_error(tmp_path):
    (tmp_path / "r").mkdir()
    _jpg(tmp_path / "r" / "a.jpg")
    with pytest.raises(ValueError):
        load_capture(tmp_path, [])
