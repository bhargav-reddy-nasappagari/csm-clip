import json
import os

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from transformers import CLIPVisionConfig, CLIPVisionModel

from src.model import CSMDetector
from src.train import (EpochSeeded, TrainConfig, Trainer, epoch_batches, final_test, lr_mult, make_model, param_groups,
                       predict, science_config)

D_IN, D, K = 32, 24, 16


def tiny_clip():
    cfg = CLIPVisionConfig(hidden_size=D_IN, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                           image_size=32, patch_size=8)
    torch.manual_seed(0)
    return CLIPVisionModel(cfg).eval()


def text_embeds():
    g = torch.Generator().manual_seed(0)
    return torch.nn.functional.normalize(torch.randn(K, D_IN, generator=g), dim=-1)


class Fake(Dataset):
    """Stand-in for ManifestDataset: .indices/.rng, flip augmentation drawn from .rng, label = sign of the mean."""

    def __init__(self, n=16, seed=0, flip=True):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.x += torch.where(torch.arange(n) % 2 == 0, 0.8, -0.8).view(-1, 1, 1, 1)
        self.y = (torch.arange(n) % 2 == 0).float()
        self.indices = np.arange(n) * 3 + 5  # deliberately not 0..n-1
        self.rng, self.flip = np.random.default_rng(0), flip

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.flip and self.rng.random() < 0.5:
            x = x.flip(-1)
        return x, self.y[i], 0


def cfg(**kw):
    base = dict(epochs=2, batch_size=4, lr_lora=1e-3, lr_new=3e-3, num_workers=0, ckpt_every=2, log_every=1,
                diag_every=3, amp=False, mamba_chunk=4)
    base.update(kw)
    return TrainConfig(**base)


def model_for(c):
    return make_model(tiny_clip(), text_embeds(), c, taps=(2, 4), d_in=D_IN, d=D, d_state=4)


def trainer(c, run_dir, train=None, val=None):
    return Trainer(model_for(c), train or Fake(), val if val is not None else Fake(8, seed=1, flip=False), c, run_dir,
                   device="cpu", log=lambda *_: None)


# ---- schedule / batching / groups
def test_lr_mult_shape():
    tot, wu = 100, 10
    m = [lr_mult(s, tot, wu, "cosine", 0.1) for s in range(tot)]
    assert m[0] == pytest.approx(0.1) and m[wu - 1] == pytest.approx(1.0) and m[wu] == pytest.approx(1.0)
    assert all(a >= b - 1e-9 for a, b in zip(m[wu:], m[wu + 1:]))
    assert m[-1] == pytest.approx(0.1, abs=0.01)
    assert lr_mult(50, tot, wu, "const") == 1.0


def test_epoch_batches_deterministic_and_disjoint():
    a, b = epoch_batches(23, 4, seed=1, epoch=0), epoch_batches(23, 4, seed=1, epoch=0)
    assert a == b and len(a) == 5
    flat = [i for bt in a for i in bt]
    assert len(set(flat)) == len(flat) == 20
    assert epoch_batches(23, 4, 1, 1) != a and epoch_batches(23, 4, 2, 0) != a


def test_param_groups_partition_all_trainables():
    c = cfg()
    m = model_for(c)
    gs = param_groups(m, c)
    ids = [id(p) for g in gs for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in m.parameters() if p.requires_grad}
    by = {g["name"]: g for g in gs}
    assert by["lora"]["weight_decay"] == 0 and by["gates"]["weight_decay"] == 0
    assert by["lora"]["lr"] == c.lr_lora and by["gates"]["lr"] == c.lr_new
    assert all(p.ndim >= 2 for p in by["new_decay"]["params"])
    gate_ids = {id(p) for p in m.mamba.gate_parameters()}
    assert {id(p) for p in by["gates"]["params"]} == gate_ids
    assert "gates" not in {g["name"] for g in param_groups(model_for(cfg(arm="plain")), cfg(arm="plain"))}


def test_arms_share_initial_weights_and_outputs():
    s, p = model_for(cfg(arm="steered")), model_for(cfg(arm="plain"))
    x = torch.randn(2, 3, 32, 32)
    for k, v in p.state_dict().items():
        assert torch.equal(v, s.state_dict()[k]), k
    s.eval(); p.eval()
    assert torch.allclose(s(x)["logit"], p(x)["logit"], atol=1e-5)  # gates zero at init


def test_epoch_seeded_augmentation_is_pure_function():
    ds = EpochSeeded(Fake(), seed=3)
    ds.set_epoch(1); a = [ds[i][0] for i in range(8)]
    ds.set_epoch(1); _ = ds[7]; b = [ds[i][0] for i in range(8)]  # order of access must not matter
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    b2 = [ds[i][0] for i in (3, 1, 0, 2, 4, 5, 6, 7)]
    assert torch.equal(b2[0], a[3]) and torch.equal(b2[1], a[1])
    ds.set_epoch(2); c = [ds[i][0] for i in range(8)]
    assert any(not torch.equal(u, v) for u, v in zip(a, c))


# ---- training behaviour
def test_loss_decreases_and_only_trainables_change(tmp_path):
    c = cfg(epochs=6)
    t = trainer(c, tmp_path / "r")
    frozen0 = {k: v.clone() for k, v in t.model.state_dict().items() if k.startswith("backbone.m.") and "lora_" not in k}
    gate0 = {k: v.clone() for k, v in t.model.state_dict().items() if ".gates." in k}
    t.fit()
    sl = t.step_losses
    assert np.mean(sl[-4:]) < 0.5 * np.mean(sl[:4]) or np.mean(sl[-4:]) < 0.1
    for k, v in t.model.state_dict().items():
        if k in frozen0:
            assert torch.equal(v, frozen0[k]), k
    assert any(not torch.equal(v, gate0[k]) for k, v in t.model.state_dict().items() if k in gate0)
    assert t.state["done"] and t.state["skipped"] == 0 and t.state["nonfinite"] == 0


def test_overfit_four_images(tmp_path):
    """The M8 gate, CPU version: 4 images, no flip, loss -> ~0 and AUROC 1 on the same four."""
    ds = Fake(4, flip=False)
    c = cfg(epochs=60, batch_size=4, warmup_frac=0.05, ckpt_every=1000, log_every=1000, diag_every=1000)
    t = trainer(c, tmp_path / "r", train=ds, val=Fake(4, flip=False))
    t.fit()
    assert np.mean(t.step_losses[-3:]) < 0.05, t.step_losses[-5:]
    assert t.state["history"][-1]["val_auroc"] == 1.0


def test_resume_is_bit_exact(tmp_path):
    c = cfg(epochs=3)
    full = trainer(c, tmp_path / "full")
    full.fit()
    part = trainer(c, tmp_path / "part")
    part.fit(stop_after_steps=5)           # "disconnect": last checkpoint was at step 4, so step 5 is lost
    assert not part.state["done"]
    again = trainer(c, tmp_path / "part")  # fresh process, same run_dir
    again.fit()
    assert again.state["done"] and again.state["step"] == full.state["step"]
    a, b = full.model.trainable_state_dict(), again.model.trainable_state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    assert [h["train_loss"] for h in full.state["history"]] == [h["train_loss"] for h in again.state["history"]]


def test_resume_refuses_changed_config(tmp_path):
    c = cfg()
    t = trainer(c, tmp_path / "r")
    t.fit(stop_after_steps=2); t.save()
    with pytest.raises(ValueError, match="differs"):
        trainer(cfg(lr_new=1e-2), tmp_path / "r").fit()
    trainer(cfg(num_workers=0, ckpt_every=7), tmp_path / "r")  # operational keys may differ
    assert science_config(cfg(ckpt_every=1)) == science_config(cfg(ckpt_every=99))


def test_checkpoint_excludes_frozen_clip_and_is_atomic(tmp_path):
    t = trainer(cfg(), tmp_path / "r")
    t.fit(stop_after_steps=2); t.save()
    ck = torch.load(t.last_path, weights_only=True)
    keys = ck["model"].keys()
    assert all(("lora_" in k) or not k.startswith("backbone.m.") for k in keys)
    assert not any(f.endswith(".tmp") for f in os.listdir(t.run_dir))


def test_nonfinite_loss_is_skipped_then_aborts(tmp_path):
    c = cfg(max_consecutive_nonfinite=3)
    t = trainer(c, tmp_path / "r")
    x, y, _ = next(iter(torch.utils.data.DataLoader(Fake(), batch_size=4)))
    before = t.model.trainable_state_dict()
    bad = x.clone(); bad[0, 0, 0, 0] = float("nan")
    info = t.train_step(bad, y)
    assert info["nonfinite_loss"] and t.state["nonfinite"] == 1
    for k, v in t.model.trainable_state_dict().items():
        assert torch.equal(v, before[k])
    t.train_step(x, y)
    assert t.state["consec_nonfinite"] == 0
    with pytest.raises(FloatingPointError):
        for _ in range(3):
            t.train_step(bad, y)


def test_grad_clip_applied(tmp_path):
    t = trainer(cfg(grad_clip=1e-3), tmp_path / "r")
    x, y, _ = next(iter(torch.utils.data.DataLoader(Fake(), batch_size=4)))
    t.train_step(x, y)
    after = torch.nn.utils.clip_grad_norm_(t.params, 1e9)
    assert float(after) <= 1e-3 * 1.001


def test_diagnostics_and_logging(tmp_path):
    t = trainer(cfg(), tmp_path / "r")
    t.fit()
    recs = [json.loads(l) for l in open(t.log_path)]
    diag = [r for r in recs if r.get("diag")]
    assert diag and {"alpha", "readout", "mix_weights", "affinity_scale", "gates"} <= set(diag[0]["diag"])
    assert any(r["event"] == "epoch_end" for r in recs)
    p = trainer(cfg(arm="plain"), tmp_path / "p")
    p.fit()
    assert not any("gates" in r["diag"] for r in map(json.loads, open(p.log_path)) if r.get("diag"))


# ---- held-out test guard
def test_final_test_once_only_and_after_training(tmp_path):
    t = trainer(cfg(), tmp_path / "r")
    te = Fake(8, seed=2, flip=False)
    with pytest.raises(AssertionError):
        final_test(t, te)
    t.fit()
    res = final_test(t, te, n_boot=20)
    assert res["final_epoch"]["auroc"] >= 0 and "best_val_epoch" in res and res["config"]["arm"] == "steered"
    assert os.path.exists(os.path.join(t.run_dir, "results.json"))
    with pytest.raises(RuntimeError, match="already evaluated"):
        final_test(t, te, n_boot=20)


def test_predict_shapes_and_eval_mode():
    m = model_for(cfg())
    lg, y, g = predict(m, Fake(8, flip=False), batch_size=3, device="cpu", amp=False, num_workers=0)
    assert lg.shape == y.shape == g.shape == (8,) and not m.training


def test_train_mode_keeps_clip_in_eval():
    m = model_for(cfg())
    m.train()
    assert m.training and not m.backbone.m.training