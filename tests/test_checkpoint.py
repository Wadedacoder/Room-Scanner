import numpy as np

from roomscan.recon import checkpoint as ck


def test_key_changes_with_inputs_and_settings():
    a = np.zeros((4, 4))
    k1 = ck.key_for("da3_recon", [a], {"res": 504})
    assert k1 == ck.key_for("da3_recon", [a.copy()], {"res": 504})  # same content, same key
    assert k1 != ck.key_for("da3_recon", [a + 1], {"res": 504})  # different pixels
    assert k1 != ck.key_for("da3_recon", [a], {"res": 378})  # different settings


def test_roundtrip_and_disable(tmp_path, monkeypatch):
    p = ck.path_for(tmp_path, "x", "k")
    ck.save_npz(p, a=np.arange(3))
    assert np.array_equal(ck.load_npz(p)["a"], np.arange(3))
    monkeypatch.setenv("ROOMSCAN_NO_CACHE", "1")
    assert ck.load_npz(p) is None


def test_truncated_checkpoint_is_ignored(tmp_path):
    p = ck.path_for(tmp_path, "x", "bad")
    p.write_bytes(b"not a zip")
    assert ck.load_npz(p) is None
