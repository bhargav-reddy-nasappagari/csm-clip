import json
import os

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src import experiment as E
from tests.test_train import D, D_IN, Fake, text_embeds, tiny_clip
from src.train import make_model

SMALL = dict(epochs=2, batch_size=4, lr_lora=1e-3, lr_new=3e-3, num_workers=0, ckpt_every=2, log_every=1,
             diag_every=3, amp=False, mamba_chunk=4)


def build(cfg):
    return make_model(tiny_clip(), text_embeds(), cfg, taps=(2, 4), d_in=D_IN, d=D, d_state=4)


def data():
    return Fake(16), Fake(8, seed=1, flip=False), Fake(12, seed=2, flip=False)


def run(cfg, root, **kw):
    return E.run_one(cfg, build, *data(), root, device="cpu", n_boot=30, log=lambda *_: None, **kw)


@pytest.fixture
def small_m9(monkeypatch):
    """compare / gate / export rebuild the frozen config from M9_BASE, so shrink the base itself."""
    monkeypatch.setattr(E, "M9_BASE", {**E.M9_BASE, **SMALL})


@pytest.fixture
def finished(tmp_path, small_m9):
    for arm in E.ARM_ORDER:
        run(E.m9_config(arm, 0), tmp_path)
    return tmp_path


def test_frozen_config_and_matched_arms():
    s, p = E.m9_config("steered"), E.m9_config("plain")
    assert (s.fold, s.protocol, s.epochs, s.batch_size, s.seed) == ("Midjourney", "controlled", 5, 8, 0)
    E.check_matched(s, p)
    with pytest.raises(AssertionError):
        E.check_matched(s, E.m9_config("plain", lr_new=1e-2))
    with pytest.raises(AssertionError):
        E.check_matched(s, E.m9_config("plain", seed=1))
    assert E.run_name(s) != E.run_name(p)


def test_run_one_writes_everything_and_is_not_re_evaluated(finished):
    rdir = finished / E.run_name(E.m9_config("steered", 0))
    for fn in E.RUN_FILES:
        assert os.path.exists(rdir / fn), fn
    y, lg, idx = E.load_predictions(rdir)
    res = json.load(open(rdir / "results.json"))
    assert len(y) == len(lg) == 12 and np.array_equal(idx, data()[2].indices)
    assert roc_auc_score(y, lg) == pytest.approx(res["final_epoch"]["auroc"])
    yb, lb, _ = E.load_predictions(rdir, "best_val")
    assert roc_auc_score(yb, lb) == pytest.approx(res["best_val_epoch"]["auroc"])

    def boom(cfg):
        raise AssertionError("a finished run must not build a model or touch the test set again")

    again = E.run_one(E.m9_config("steered", 0), boom, None, None, None, finished, device="cpu", log=lambda *_: None)
    assert again == res
    with pytest.raises(AssertionError, match="different config"):
        run(E.m9_config("steered", 0, lr_new=1e-2), finished)


def test_interrupted_run_resumes_to_the_same_result(tmp_path, small_m9):
    cfg = E.m9_config("steered", 0)
    a = run(cfg, tmp_path / "full")
    assert run(cfg, tmp_path / "part", stop_after_steps=5) is None
    assert not os.path.exists(tmp_path / "part" / E.run_name(cfg) / "predictions.npz")  # test set untouched
    b = run(cfg, tmp_path / "part")
    assert a["final_epoch"] == b["final_epoch"]
    assert len(open(tmp_path / "part" / E.run_name(cfg) / "sessions.jsonl").readlines()) == 2


def test_val_diagnostics_and_gates_restored(tmp_path):
    cfg = E.m9_config("steered", 0, **SMALL)
    m = build(cfg)
    val = Fake(8, seed=1, flip=False)
    kw = dict(batch_size=4, device="cpu", amp=False, num_workers=0)
    d0 = E.val_diagnostics(m, val, **kw)
    assert d0["gates_off"]["mean_abs_logit_change"] == 0 and d0["gates_off"]["sign_flips"] == 0
    assert all(g["dt_max"] == 0 and g["B_max"] == 0 and g["C_max"] == 0 for g in d0["gate_deviation"])
    with torch.no_grad():
        for layer in m.mamba.layers:
            layer.mixer.gates.b.fill_(0.5)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    d1 = E.val_diagnostics(m, val, **kw)
    assert d1["gates_off"]["mean_abs_logit_change"] > 0
    assert d1["gate_deviation"][0]["dt_mean"] == pytest.approx(float(np.tanh(0.5)), abs=1e-5)
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
    plain = E.val_diagnostics(build(E.m9_config("plain", 0, **SMALL)), val, **kw)
    assert "gate_deviation" not in plain and "gates_off" not in plain and "alpha" in plain


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    y = np.arange(400) % 2
    base = y + rng.normal(0, 1.0, 400)
    same = E.paired_bootstrap(y, base, base, n_boot=200)
    assert same["delta_auroc"] == 0 and same["ci95"] == [0.0, 0.0]
    better = E.paired_bootstrap(y, base + y, base, n_boot=200)
    assert better["delta_auroc"] == pytest.approx(roc_auc_score(y, base + y) - roc_auc_score(y, base))
    assert better["ci95"][0] > 0
    swapped = E.paired_bootstrap(y, base, base + y, n_boot=200)
    assert swapped["delta_auroc"] == pytest.approx(-better["delta_auroc"])
    two = E.paired_bootstrap(y, np.stack([base + y, base]), np.stack([base, base]), n_boot=200)
    assert two["delta_auroc"] == pytest.approx(better["delta_auroc"] / 2)


def test_verdict_rule():
    assert E.verdict([0.05], [0.01, 0.09]).startswith("preliminary")
    assert E.verdict([0.02, 0.03, 0.01], [0.005, 0.04]) == "steering helps"
    assert E.verdict([-0.02, -0.03, -0.01], [-0.04, -0.005]) == "steering hurts"
    assert E.verdict([0.02, 0.03, -0.001], [0.005, 0.04]) == "no demonstrated difference"
    assert E.verdict([0.02, 0.03, 0.01], [-0.001, 0.04]) == "no demonstrated difference"


def test_compare_gate_rows_export(finished, tmp_path_factory):
    c = E.compare(finished, seeds=(0,), n_boot=30)
    assert c["n_test"] == 12 and c["verdict"].startswith("preliminary")
    r = c["per_seed"][0]
    assert r["delta_auroc"] == pytest.approx(r["steered"]["auroc"] - r["plain"]["auroc"])
    assert c["mean_delta_auroc"] == pytest.approx(r["delta_auroc"])
    assert "mean delta" in E.format_comparison(c)
    assert E.compare(finished, which="best_val", n_boot=30)["which"] == "best_val_epoch"
    g = E.m9_gate(finished)
    assert g["gate"] == "PASS" and len(g["runs"]) == 2
    with pytest.raises(AssertionError, match="missing"):
        E.m9_gate(finished, seeds=(0, 1))
    rows = E.log_rows(finished).split("\n")
    assert rows[0].startswith("| R007 |") and "| steered | 0 |" in rows[0] and "| plain | 0 |" in rows[1]
    assert all(row.count("|") == 13 for row in rows)
    dest = tmp_path_factory.mktemp("export")
    files = E.export(finished, dest)
    assert len(files) == 2 * len(E.RUN_FILES) and not any(f.endswith(".pt") for f in files)


def test_compare_refuses_unpaired_test_sets(finished):
    p = finished / E.run_name(E.m9_config("plain", 0)) / "predictions.npz"
    z = dict(np.load(p))
    z["indices"] = z["indices"][::-1]
    np.savez(p, **z)
    with pytest.raises(AssertionError, match="not be paired"):
        E.compare(finished, n_boot=10)