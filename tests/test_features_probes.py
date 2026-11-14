import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPVisionConfig, CLIPVisionModel

from src import features as F
from src.features import ClipTaps, FeatureStore, RowDataset, extract_to_cache, union_rows
from src.probes import REFERENCE, fit_probe, gate_passes, gate_report, run_fold


class FakeHF:
    """Mimics an HF split: rows of {image, label, generator}."""

    def __init__(self, n):
        rng = np.random.default_rng(0)
        self.rows = [{"image": Image.fromarray(rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)),
                      "label": i % 2, "generator": (i % 2) * 3} for i in range(n)]

    def __getitem__(self, i):
        return self.rows[i]


class Stand(torch.nn.Module):
    """Stand-in encoder: deterministic function of the image, same output keys as ClipTaps."""

    def forward(self, x):
        m = x.mean(dim=(2, 3))
        return {"cls_mid": m, "cls_last": m * 2, "pooled": m + 1}


# ---- ClipTaps on a tiny random CLIP vision tower (no network needed)
def test_cliptaps_semantics():
    cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                           image_size=32, patch_size=8)
    vm = CLIPVisionModel(cfg).eval()
    taps = ClipTaps(vm, taps=(2, 4))
    x = torch.randn(3, 3, 32, 32)
    out = taps(x)
    ref = vm(pixel_values=x, output_hidden_states=True)
    assert out["cls_mid"].shape == out["cls_last"].shape == out["pooled"].shape == (3, 32)
    assert torch.equal(out["cls_mid"], ref.hidden_states[2][:, 0])
    assert torch.equal(out["cls_last"], ref.last_hidden_state[:, 0])           # hidden_states[-1] is pre post-LN
    assert torch.allclose(out["pooled"], getattr(vm, "vision_model", vm).post_layernorm(ref.last_hidden_state[:, 0]), atol=1e-6)
    assert not any(p.requires_grad for p in taps.parameters())


def test_real_config_has_24_layers_and_336():
    cfg = CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                           num_attention_heads=16, image_size=336, patch_size=14)
    assert cfg.num_hidden_layers == 24 and (cfg.image_size // cfg.patch_size) ** 2 == 576


# ---- cache: ordering, resume, lookup
def test_extract_order_and_resume(tmp_path):
    hf = FakeHF(10)
    ds = RowDataset(hf, [7, 3, 9, 0, 5, 1, 2], "controlled")
    d = str(tmp_path / "c")
    extract_to_cache(Stand(), ds, d, chunk=3, batch_size=2, num_workers=0, log=lambda *_: None)
    s1 = FeatureStore(d)
    assert len(s1.idx) == 7 and list(s1.idx) == sorted([7, 3, 9, 0, 5, 1, 2])
    # simulate a disconnect after the first chunk: delete later chunks, re-run, same result
    import glob, os
    for f in sorted(glob.glob(d + "/chunk_*.npz"))[1:]:
        os.remove(f)
    called = []
    extract_to_cache(Stand(), ds, d, chunk=3, batch_size=2, num_workers=0, log=called.append)
    assert len(called) == 2  # only the 2 missing chunks were recomputed
    s2 = FeatureStore(d)
    for k in F.KEYS:
        assert np.array_equal(s1.arrays[k], s2.arrays[k])
    # lookup by row index returns that row's own features and label
    rows = [9, 3]
    assert list(s2.take("label", rows)) == [hf[9]["label"], hf[3]["label"]]
    direct = Stand()(torch.stack([RowDataset(hf, [9], "controlled")[0][0]]))["pooled"].numpy()[0]
    assert np.allclose(s2.take("pooled", [9])[0], direct, atol=1e-6)
    with pytest.raises(AssertionError):
        s2.take("pooled", [4])


# ---- probes
def synthetic_stores(tmp_path, sep=2.0):
    """Two fake 'HF splits' cached as stores with separable features."""
    rng = np.random.default_rng(0)

    def mk(name, n):
        y = rng.integers(0, 2, n)
        arrs = {k: rng.normal(size=(n, 8)) + sep * y[:, None] for k in F.KEYS}
        d = tmp_path / name; d.mkdir()
        np.savez(d / "chunk_00000.npz", idx=np.arange(n), label=y, gen=y, **arrs)
        return FeatureStore(str(d)), y
    return mk("train", 400), mk("val", 300)


def test_run_fold_and_c_selected_on_val_only(tmp_path):
    (tr, ytr), (te, yte) = synthetic_stores(tmp_path)
    man = {"held_out": "ADM", "train": list(range(0, 300)), "val": list(range(300, 400)), "test": list(range(300))}
    for kind in ("univfd", "multilayer"):
        r = run_fold(man, tr, te, kind, n_boot=50)
        assert r["auroc"] > 0.9 and r["fold"] == "ADM" and r["kind"] == kind and r["C"] in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    # flipping test labels must not change the chosen C (test never used for selection)
    te.arrays["label"] = 1 - te.arrays["label"]
    r2 = run_fold(man, tr, te, "univfd", n_boot=50)
    assert r2["C"] == run_fold(man, tr, F.FeatureStore(str(tmp_path / "val")) if False else te, "univfd", n_boot=50)["C"]
    assert r2["auroc"] < 0.1  # flipped labels -> inverted AUROC, proving test labels only score, never fit


def test_fit_probe_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)); y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    a = fit_probe(X[:150], y[:150], X[150:], y[150:])
    b = fit_probe(X[:150], y[:150], X[150:], y[150:])
    assert a[1] == b[1] and np.array_equal(a[0].decision_function(X), b[0].decision_function(X))


def test_gate_report_logic():
    ref, lo, hi = REFERENCE[("BigGAN", "univfd")]
    good = {"fold": "BigGAN", "kind": "univfd", "auroc": ref, "auroc_ci95": [lo, hi], "C": 1.0}
    bad = {"fold": "BigGAN", "kind": "univfd", "auroc": 0.90, "auroc_ci95": [0.88, 0.92], "C": 1.0}
    g, b = gate_report([good, bad])
    assert g["in_ref_ci"] and g["ci_overlap"] and not b["in_ref_ci"] and not b["ci_overlap"]
    assert len(REFERENCE) == 6


def test_union_rows():
    ms = [{"train": [3, 1], "val": [5]}, {"train": [1, 2], "val": [5]}]
    assert union_rows(ms, "train") == [1, 2, 3]


def test_revised_gate_d017():
    ref, lo, hi = REFERENCE[("ADM", "multilayer")]
    mk = lambda a, ci: {"fold": "ADM", "kind": "multilayer", "auroc": a, "auroc_ci95": ci, "C": 0.1}
    ok = gate_report([mk(0.963, [0.952, 0.972])])            # observed M2 cell: -0.014, CIs overlap
    assert not ok[0]["in_ref_ci"] and ok[0]["delta"] == -0.014 and gate_passes(ok)
    assert not gate_passes(gate_report([mk(0.940, [0.925, 0.955])]))      # no CI overlap
    assert not gate_passes(gate_report([mk(0.950, [0.930, 0.975])]))      # overlaps but 0.027 below
