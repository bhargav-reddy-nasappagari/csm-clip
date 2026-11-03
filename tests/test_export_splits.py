import json

import numpy as np
import pytest

from src.export_splits import make_manifest, write_manifest

NAMES = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "VQDM", "Wukong"]  # SD14 empty


def fake_gens(n_real, per_gen, gens):
    g = [0] * n_real
    for k in gens:
        g += [k] * per_gen
    return np.array(g)


@pytest.fixture
def data():
    train_gen = fake_gens(3000, 400, [1, 2, 3, 4, 6, 7])
    val_gen = fake_gens(1500, 500, [1, 2, 3, 4, 6, 7])
    return train_gen, val_gen


def test_contract_properties(data):
    tg, vg = data
    m = make_manifest(tg, vg, NAMES, "Midjourney", "rev")
    held = NAMES.index("Midjourney")
    assert not (tg[m["train"]] == held).any() and not (tg[m["val"]] == held).any()
    assert not set(m["train"]) & set(m["val"])
    assert (vg[m["test"]] == held).sum() == 500 and (vg[m["test"]] == 0).sum() == 500
    assert "Midjourney" not in m["seen"] and "SD14" not in m["seen"]
    assert m["per_gen"] == 300
    # 90/10, real balanced 1:1 against fake
    tr, va = tg[m["train"]], tg[m["val"]]
    assert (tr == 0).sum() == (tr != 0).sum() and (va == 0).sum() == (va != 0).sum()
    assert len(m["val"]) == pytest.approx(0.1 * (len(m["train"]) + len(m["val"])), abs=2)


def test_deterministic_and_byte_stable(data, tmp_path):
    tg, vg = data
    a = make_manifest(tg, vg, NAMES, "ADM", "rev")
    b = make_manifest(tg, vg, NAMES, "ADM", "rev")
    pa, ha = write_manifest(a, tmp_path / "a")
    pb, hb = write_manifest(b, tmp_path / "b")
    assert ha == hb and open(pa).read() == open(pb).read()
    assert json.load(open(pa))["revision"] == "rev"


def test_fails_loudly_when_contract_cannot_be_met():
    tg = fake_gens(500, 400, [1, 2, 3, 4, 6, 7])   # too few reals -> per_gen shrinks
    vg = fake_gens(1500, 500, [1, 2, 3, 4, 6, 7])
    with pytest.raises(AssertionError, match="per_gen shrank"):
        make_manifest(tg, vg, NAMES, "BigGAN", "rev")
    tg = fake_gens(3000, 400, [1, 2, 3, 4, 6, 7])
    vg = fake_gens(1500, 300, [1, 2, 3, 4, 6, 7])  # only 300 held-out fakes
    with pytest.raises(AssertionError, match="test is"):
        make_manifest(tg, vg, NAMES, "BigGAN", "rev")
