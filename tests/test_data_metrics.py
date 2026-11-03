import io
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from src import data
from src.data import audit_manifest, controlled_pre, load_manifest, preprocess
from src.metrics import bootstrap_auroc, evaluate

FOLDS = ["midjourney", "biggan", "adm"]


def rand_img(w=300, h=200, seed=0):
    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(a)


# ---- preprocessing protocols
@pytest.mark.parametrize("proto", data.PROTOCOLS)
def test_output_shape_and_dtype(proto):
    x = preprocess(rand_img(), proto)
    assert x.shape == (3, 336, 336) and x.dtype == torch.float32


def test_flip_only_in_training_and_deterministic_eval():
    img = rand_img()
    a, b = preprocess(img, "raw"), preprocess(img, "raw")
    assert torch.equal(a, b)
    flips = {preprocess(img, "raw", train=True, rng=np.random.default_rng(s)).sum().item() for s in range(8)}
    assert len(flips) <= 2  # flip preserves the sum; only crop/flip determinism matters
    seen = [torch.equal(preprocess(img, "raw", True, np.random.default_rng(s)), a) for s in range(20)]
    assert any(seen) and not all(seen)


def test_controlled_exactly_one_jpeg_encode():
    with mock.patch.object(Image.Image, "save", autospec=True, side_effect=Image.Image.save) as save:
        preprocess(rand_img(), "controlled")
    jpeg_saves = [c for c in save.call_args_list if c.kwargs.get("format") == "JPEG"]
    assert len(jpeg_saves) == 1 and jpeg_saves[0].kwargs["quality"] == 90
    with mock.patch.object(Image.Image, "save", autospec=True, side_effect=Image.Image.save) as save:
        preprocess(rand_img(), "raw")
    assert save.call_count == 0


def test_controlled_is_q90_jpeg_at_256_resize_before_jpeg():
    src = rand_img(512, 400)
    out = controlled_pre(src)
    assert out.size == (256, 256)
    # Reference: crop -> resize -> JPEG q90. Order matters; this must match exactly.
    sq = src.crop((56, 0, 456, 400)).resize((256, 256), Image.BICUBIC)
    buf = io.BytesIO(); sq.save(buf, format="JPEG", quality=90); buf.seek(0)
    assert np.array_equal(np.asarray(out), np.asarray(Image.open(buf).convert("RGB")))
    # and the encoded stream carries the q90 quantisation tables
    buf.seek(0)
    q = list(Image.open(buf).quantization[0])
    b75 = io.BytesIO(); sq.save(b75, format="JPEG", quality=75); b75.seek(0)
    assert q != list(Image.open(b75).quantization[0]) and max(q) == 24  # q90 luminance table


def test_controlled_identical_for_png_and_jpeg_origin_up_to_decoding():
    img = rand_img(256, 256)
    j = io.BytesIO(); img.save(j, format="JPEG", quality=95); j.seek(0)
    a, b = preprocess(Image.open(j), "controlled"), preprocess(Image.open(j), "controlled")
    assert torch.equal(a, b)


# ---- manifests (real files in splits/)
@pytest.mark.parametrize("fold", FOLDS)
def test_real_manifest_gate_facts(fold):
    m = load_manifest(fold)
    tr, va, te = set(m["train"]), set(m["val"]), m["test"]
    assert not tr & va
    assert len(tr) == 3240 and len(va) == 360 and len(te) == 1000
    assert m["counts"]["test_fake"] == 500 and m["counts"]["test_real"] == 500
    assert m["per_gen"] == 300 and m["seed"] == 0
    assert m["held_out"] not in m["seen"] and len(m["seen"]) == 6
    assert m["train"] == sorted(m["train"]) and len(set(te)) == len(te)


def test_manifests_share_revision():
    revs = {load_manifest(f)["revision"] for f in FOLDS}
    assert len(revs) == 1


def test_audit_manifest_catches_leak():
    names = ["Real", "ADM", "BigGAN"]
    tg = np.array([0] * 4 + [1] * 2 + [2] * 2)
    vg = np.array([0] * 2 + [2] * 2)
    ok = {"held_out": "BigGAN", "train": [0, 1, 4, 5], "val": [2, 3], "test": [0, 1, 2, 3]}
    with pytest.raises(AssertionError):  # test is 2/2, contract wants 500/500
        audit_manifest(ok, tg, vg, names)
    leak = dict(ok, train=[0, 1, 6, 7])
    with pytest.raises(AssertionError, match="held-out leaked"):
        audit_manifest(leak, tg, vg, names)


# ---- metrics
def test_auroc_perfect_and_chance():
    y = np.array([0] * 100 + [1] * 100)
    auc, lo, hi = bootstrap_auroc(y, np.r_[np.zeros(100), np.ones(100)], 200)
    assert auc == 1.0 and hi == 1.0
    rng = np.random.default_rng(0)
    auc, lo, hi = bootstrap_auroc(y, rng.normal(size=200), 500)
    assert lo < 0.5 < hi


def test_bootstrap_deterministic_and_ci_contains_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = y * 1.0 + rng.normal(size=400)
    a = bootstrap_auroc(y, s, 300, seed=3)
    assert a == bootstrap_auroc(y, s, 300, seed=3)
    assert a[1] <= a[0] <= a[2]


def test_evaluate_fields_and_threshold_zero():
    y = np.array([0, 0, 1, 1])
    out = evaluate(y, np.array([-2.0, 1.0, 3.0, -1.0]), n_boot=50)
    assert out["acc_at_0"] == 0.5 and set(out) == {"auroc", "auroc_ci95", "ap", "acc_at_0", "n"}
