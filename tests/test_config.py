import pytest
import yaml

from roomscan.config import ConfigError, Hardware, auto_profile, available_profiles, load_config

M1_8 = Hardware("Darwin", "arm64", 8.0, True, 0.0)
M3_36 = Hardware("Darwin", "arm64", 36.0, True, 0.0)
KAGGLE_T4 = Hardware("Linux", "x86_64", 31.0, False, 15.0)
RTX4090 = Hardware("Linux", "x86_64", 64.0, False, 24.0)
INTEL_16 = Hardware("Linux", "x86_64", 15.5, False, 0.0)


@pytest.mark.parametrize(
    "hw,expected",
    [(M1_8, "lite"), (M3_36, "mac-32gb"), (KAGGLE_T4, "cuda-16gb"), (RTX4090, "cuda-24gb"), (INTEL_16, "lite")],
)
def test_auto_profile(hw, expected):
    assert auto_profile(hw) == expected


@pytest.mark.parametrize("name", available_profiles())
def test_every_profile_resolves(name):
    r = load_config(name, hw=RTX4090)
    assert r.profile == name and r["recon"]["max_views"] >= 2


def test_layering_order(tmp_path):
    f = tmp_path / "mine.yaml"
    f.write_text(yaml.safe_dump({"recon": {"max_views": 12, "video_keyframes": 33}}))
    r = load_config("lite", f, ["recon.max_views=10"], hw=M1_8)
    assert r["recon"]["max_views"] == 10  # --set beats file
    assert r["recon"]["video_keyframes"] == 33  # file beats profile
    assert r["lidar"]["frame_stride"] == 4  # profile beats default
    assert r["runtime"]["device"] == "mps"  # auto device on Apple silicon


def test_typo_is_rejected():
    with pytest.raises(ConfigError, match="recon.max_veiws"):
        load_config("lite", overrides=["recon.max_veiws=4"], hw=M1_8)


def test_unknown_model_rejected():
    with pytest.raises(KeyError):
        load_config("lite", overrides=["recon.model=vggt-xxl"], hw=M1_8)


def test_digest_changes_with_config():
    a = load_config("lite", hw=M1_8).digest
    b = load_config("lite", overrides=["lidar.voxel_m=0.01"], hw=M1_8).digest
    assert a != b
