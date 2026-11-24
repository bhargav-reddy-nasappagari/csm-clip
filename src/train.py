"""M8: training loop. Resumable, deterministic in data order, fp16-safe, held-out test touched once.

Design (docs/decision_log.md D011, D018-D021):
  * Loss: plain BCE-with-logits on the additive logit. No auxiliary concept loss in v1 (polarity loss is a v2 candidate).
  * Optimiser: AdamW, four param groups (LoRA / new weights with decay / new weights without decay / gates), stateless
    schedule (linear warm-up -> cosine to min_lr_mult), grad clip, fp16 autocast + GradScaler.
  * Every random choice is a pure function of (seed, epoch, sample): the batch order of an epoch is permutation(seed,
    epoch) and the flip of a sample is rng(seed, epoch, row). So a run that dies and resumes replays exactly the batches
    it would have seen, whatever the number of DataLoader workers. There is no other randomness in training (no dropout).
  * Checkpoints hold only trainable state (+ optimiser, scaler, counters, config); written atomically (tmp + replace),
    so a disconnect mid-write cannot corrupt last.pt. fit() resumes from run_dir/last.pt automatically and refuses to
    resume under a different config.
  * Model selection uses the validation split (seen generators only). The held-out test split is evaluated by
    final_test() only, once per run, and only after training finished: final epoch is the primary result, the
    best-validation epoch is reported as secondary (contract Section 2).
"""
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.concepts import CONCEPT_POLARITY
from src.fusion import affinity_stats
from src.metrics import evaluate
from src.model import CSMDetector
from src.readout import readout_diagnostics
from src.steering import gate_stats

OPERATIONAL_KEYS = ("ckpt_every", "log_every", "diag_every", "num_workers", "max_consecutive_nonfinite")


@dataclass
class TrainConfig:
    fold: str = "Midjourney"
    arm: str = "steered"                # "steered" | "plain"
    protocol: str = "controlled"        # "raw" | "controlled"
    seed: int = 0
    epochs: int = 5                     # contract: fixed at 5 for every arm
    batch_size: int = 8                 # peak 8.54 GiB on a T4 (M7); 16 does not fit
    lr_lora: float = 1e-4
    lr_new: float = 3e-4                # front end, Mamba, readout
    lr_gate: float = None               # None -> lr_new (a higher gate lr is a v2 lever)
    weight_decay: float = 0.01          # only on >=2-D new weights; never on LoRA, gates, A_log, D, norms, biases, scalars
    warmup_frac: float = 0.05
    min_lr_mult: float = 0.1
    schedule: str = "cosine"            # "cosine" | "const"
    grad_clip: float = 1.0
    amp: bool = True                    # fp16 autocast + GradScaler (only active on CUDA)
    lora_r: int = 8
    lora_alpha: float = 16.0
    mamba_chunk: int = 8
    mamba_checkpoint: bool = False
    # operational: allowed to differ between a run and its resume
    ckpt_every: int = 100
    log_every: int = 25
    diag_every: int = 100
    num_workers: int = 2
    max_consecutive_nonfinite: int = 10


def science_config(cfg):
    return {k: v for k, v in asdict(cfg).items() if k not in OPERATIONAL_KEYS}


def make_model(vision_model, text_embeds, cfg, taps=(16, 24), d_in=1024, d=384, **mamba_kw):
    """Real sizes by default; tests pass small ones. The same cfg.seed gives both arms identical shared weights."""
    return CSMDetector(vision_model, text_embeds, arm=cfg.arm, taps=taps, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                       d_in=d_in, d=d, seed=cfg.seed, chunk=cfg.mamba_chunk, checkpoint=cfg.mamba_checkpoint,
                       **mamba_kw)


# ---- schedule, batching, parameter groups
def lr_mult(step, total, warmup, schedule="cosine", min_mult=0.1):
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if schedule == "const":
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return min_mult + (1.0 - min_mult) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def epoch_batches(n, batch_size, seed, epoch):
    """Deterministic batches of an epoch (drop_last): a function of (seed, epoch) only."""
    perm = np.random.default_rng((seed, epoch)).permutation(n)
    return [perm[i * batch_size:(i + 1) * batch_size].tolist() for i in range(n // batch_size)]


def _no_decay(name, p):
    return p.ndim < 2 or name.endswith("A_log")


def param_groups(model, cfg):
    lr_gate = cfg.lr_new if cfg.lr_gate is None else cfg.lr_gate
    buckets = {"lora": [], "new_decay": [], "new_nodecay": [], "gates": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            buckets["lora"].append(p)
        elif ".gates." in n:
            buckets["gates"].append(p)
        elif _no_decay(n, p):
            buckets["new_nodecay"].append(p)
        else:
            buckets["new_decay"].append(p)
    spec = {"lora": (cfg.lr_lora, 0.0), "new_decay": (cfg.lr_new, cfg.weight_decay),
            "new_nodecay": (cfg.lr_new, 0.0), "gates": (lr_gate, 0.0)}
    return [{"name": k, "params": v, "lr": spec[k][0], "lr_base": spec[k][0], "weight_decay": spec[k][1]}
            for k, v in buckets.items() if v]


class EpochSeeded(Dataset):
    """Makes augmentation a pure function of (seed, epoch, row). Wraps a ManifestDataset-like object (has .indices and
    .rng). `positions` optionally restricts to a subset (used by the 4-image overfit gate)."""

    def __init__(self, base, seed=0, positions=None):
        self.base, self.seed, self.epoch = base, seed, 0
        self.positions = np.arange(len(base)) if positions is None else np.asarray(positions, dtype=np.int64)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        j = int(self.positions[i])
        if hasattr(self.base, "rng"):
            self.base.rng = np.random.default_rng((self.seed, self.epoch, int(self.base.indices[j])))
        return self.base[j]


def build_datasets(manifest, hf_train, hf_val, protocol, seed):
    """(train: EpochSeeded with flip, val, test). Val/test have no augmentation."""
    from src.data import ManifestDataset
    tr = EpochSeeded(ManifestDataset(manifest, "train", hf_train, hf_val, protocol=protocol, train=True), seed)
    va = ManifestDataset(manifest, "val", hf_train, hf_val, protocol=protocol, train=False)
    te = ManifestDataset(manifest, "test", hf_train, hf_val, protocol=protocol, train=False)
    return tr, va, te


# ---- evaluation
@torch.no_grad()
def predict(model, ds, batch_size=16, device="cuda", amp=True, num_workers=2):
    """-> logits (float64 np), labels (int np), generator ids (np). Eval mode, no grad."""
    model.eval()
    dev = torch.device(device)
    use_amp = bool(amp and dev.type == "cuda")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    L, Y, G = [], [], []
    for x, y, g in dl:
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            out = model(x.to(dev))
        L.append(out["logit"].float().cpu())
        Y.append(torch.as_tensor(y))
        G.append(torch.as_tensor(g))
    return (torch.cat(L).double().numpy(), torch.cat(Y).numpy().astype(int), torch.cat(G).numpy())


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class Trainer:
    def __init__(self, model, train_ds, val_ds, cfg, run_dir, device="cuda", log=print):
        self.cfg, self.run_dir, self.log = cfg, str(run_dir), log
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_ds = train_ds if isinstance(train_ds, EpochSeeded) else EpochSeeded(train_ds, cfg.seed)
        self.val_ds = val_ds
        self.use_amp = bool(cfg.amp and self.device.type == "cuda")
        self.groups = param_groups(model, cfg)
        self.params = [p for g in self.groups for p in g["params"]]
        self.opt = torch.optim.AdamW(self.groups, betas=(0.9, 0.999), eps=1e-8)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.n_batches = len(self.train_ds) // cfg.batch_size
        assert self.n_batches > 0, "dataset smaller than one batch"
        self.total_steps = cfg.epochs * self.n_batches
        self.warmup = int(round(cfg.warmup_frac * self.total_steps))
        self.state = {"step": 0, "epoch": 0, "batch": 0, "skipped": 0, "nonfinite": 0, "consec_nonfinite": 0,
                      "ep_loss_sum": 0.0, "ep_loss_n": 0, "history": [], "best_val": -1.0, "best_epoch": -1,
                      "done": False}
        self.step_losses = []  # in-memory only (tests and notebook gates)
        os.makedirs(self.run_dir, exist_ok=True)
        self.last_path = os.path.join(self.run_dir, "last.pt")
        self.best_path = os.path.join(self.run_dir, "best.pt")
        self.log_path = os.path.join(self.run_dir, "log.jsonl")

    # ---- logging / checkpoints
    def _log(self, rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _atomic_save(self, obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save(self, path=None):
        self._atomic_save({"model": self.model.trainable_state_dict(), "opt": self.opt.state_dict(),
                           "scaler": self.scaler.state_dict(), "state": self.state,
                           "config": science_config(self.cfg)}, path or self.last_path)

    def load(self, path=None):
        ck = torch.load(path or self.last_path, map_location="cpu", weights_only=True)
        if ck["config"] != science_config(self.cfg):
            diff = {k: (ck["config"].get(k), v) for k, v in science_config(self.cfg).items() if ck["config"].get(k) != v}
            raise ValueError(f"checkpoint config differs from this run's config (saved, now): {diff}")
        self.model.load_trainable_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.scaler.load_state_dict(ck["scaler"])
        self.state = ck["state"]

    # ---- one optimisation step
    def _set_lr(self, step):
        m = lr_mult(step, self.total_steps, self.warmup, self.cfg.schedule, self.cfg.min_lr_mult)
        for g in self.opt.param_groups:
            g["lr"] = g["lr_base"] * m
        return m

    @torch.no_grad()
    def _diagnostics(self, out):
        fe = self.model.front
        d = {"alpha": affinity_stats(out["alpha"]),
             "readout": readout_diagnostics({k: out[k] for k in ("logit", "scores", "pi")}, polarity=CONCEPT_POLARITY),
             "mix_weights": fe.fusion.mix_weights().tolist(),
             "affinity_scale": float(fe.affinity.log_scale.clamp(max=fe.affinity.max_log_scale).exp())}
        if self.model.arm == "steered":
            d["gates"] = gate_stats(self.model.mamba)
        return d

    def train_step(self, x, y, want_diag=False):
        cfg = self.cfg
        self.model.train()
        x, y = x.to(self.device, non_blocking=True), y.to(self.device).float()
        mult = self._set_lr(self.state["step"])
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            out = self.model(x)
        loss = F.binary_cross_entropy_with_logits(out["logit"].float(), y)
        lv = float(loss.detach())
        info = {"loss": lv, "grad_norm": float("nan"), "skipped": False, "nonfinite_loss": False, "lr_mult": mult}
        if not math.isfinite(lv):
            self.state["nonfinite"] += 1
            self.state["consec_nonfinite"] += 1
            info["nonfinite_loss"] = True
            if self.state["consec_nonfinite"] >= cfg.max_consecutive_nonfinite:
                raise FloatingPointError(f"{self.state['consec_nonfinite']} consecutive non-finite losses")
            return info
        self.state["consec_nonfinite"] = 0
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)  # no-op when the scaler is disabled
        gn = float(torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip))
        info["grad_norm"] = gn
        if self.use_amp:
            before = self.scaler.get_scale()
            self.scaler.step(self.opt)  # skips the update itself if the unscaled grads contain inf/nan
            self.scaler.update()
            info["skipped"] = self.scaler.get_scale() < before
        elif math.isfinite(gn):
            self.opt.step()
        else:
            info["skipped"] = True
        self.state["skipped"] += int(info["skipped"])
        if want_diag:
            info["diag"] = self._diagnostics(out)
        return info

    # ---- epochs
    def _loader(self, epoch, start_batch):
        self.train_ds.set_epoch(epoch)
        batches = epoch_batches(len(self.train_ds), self.cfg.batch_size, self.cfg.seed, epoch)[start_batch:]
        return DataLoader(self.train_ds, batch_sampler=batches, num_workers=self.cfg.num_workers,
                          pin_memory=self.device.type == "cuda")

    def validate(self):
        logits, y, _ = predict(self.model, self.val_ds, self.cfg.batch_size * 2, self.device, self.use_amp,
                               self.cfg.num_workers)
        loss = F.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(y).double())
        return {"auroc": float(roc_auc_score(y, logits)), "loss": float(loss), "n": int(len(y))}

    def _end_epoch(self):
        st = self.state
        rec = {"epoch": st["epoch"] + 1, "step": st["step"],
               "train_loss": st["ep_loss_sum"] / max(1, st["ep_loss_n"])}
        if self.val_ds is not None:
            v = self.validate()
            rec.update(val_auroc=v["auroc"], val_loss=v["loss"])
            if v["auroc"] > st["best_val"] + 1e-12:  # ties keep the earlier epoch
                st["best_val"], st["best_epoch"] = v["auroc"], rec["epoch"]
                self._atomic_save({"model": self.model.trainable_state_dict(), "epoch": rec["epoch"],
                                   "val_auroc": v["auroc"]}, self.best_path)
        st["history"].append(rec)
        self._log({"event": "epoch_end", **rec})
        self.log(f"epoch {rec['epoch']}/{self.cfg.epochs}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in rec.items() if isinstance(v, float)))
        st["epoch"] += 1
        st["batch"] = 0
        st["ep_loss_sum"], st["ep_loss_n"] = 0.0, 0
        self.save()

    def fit(self, stop_after_steps=None):
        """Train to cfg.epochs, resuming from run_dir/last.pt if present. stop_after_steps simulates a disconnect:
        it returns WITHOUT saving, so everything since the last periodic checkpoint is lost, as in a real crash."""
        cfg = self.cfg
        if os.path.exists(self.last_path):
            self.load()
            self._log({"event": "resumed", "step": self.state["step"], "epoch": self.state["epoch"],
                       "batch": self.state["batch"]})
            self.log(f"resumed from step {self.state['step']} (epoch {self.state['epoch']}, batch {self.state['batch']})")
        if self.state["done"]:
            return self.state
        win = {"t": time.time(), "n": 0, "loss": [], "gn": []}
        while self.state["epoch"] < cfg.epochs:
            if self.state["batch"] < self.n_batches:
                for x, y, _ in self._loader(self.state["epoch"], self.state["batch"]):
                    want_diag = (self.state["step"] + 1) % cfg.diag_every == 0
                    info = self.train_step(x, y, want_diag)
                    st = self.state
                    st["step"] += 1
                    st["batch"] += 1
                    self.step_losses.append(info["loss"])
                    if math.isfinite(info["loss"]):
                        st["ep_loss_sum"] += info["loss"]
                        st["ep_loss_n"] += 1
                        win["loss"].append(info["loss"])
                    if math.isfinite(info["grad_norm"]):
                        win["gn"].append(info["grad_norm"])
                    win["n"] += len(y)
                    if st["step"] % cfg.log_every == 0 or "diag" in info:
                        now = time.time()
                        rec = {"event": "train", "step": st["step"], "epoch": st["epoch"], "batch": st["batch"],
                               "loss": float(np.mean(win["loss"])) if win["loss"] else float("nan"),
                               "grad_norm": float(np.mean(win["gn"])) if win["gn"] else float("nan"),
                               "lr": {g["name"]: g["lr"] for g in self.opt.param_groups},
                               "scale": self.scaler.get_scale() if self.use_amp else None,
                               "skipped_total": st["skipped"], "nonfinite_total": st["nonfinite"],
                               "img_per_s": win["n"] / max(1e-9, now - win["t"])}
                        if self.device.type == "cuda":
                            rec["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
                        if "diag" in info:
                            rec["diag"] = info["diag"]
                        self._log(rec)
                        self.log(f"step {st['step']}/{self.total_steps} loss {rec['loss']:.4f} "
                                 f"gn {rec['grad_norm']:.2f} {rec['img_per_s']:.1f} img/s skipped {st['skipped']}")
                        win = {"t": now, "n": 0, "loss": [], "gn": []}
                    if st["step"] % cfg.ckpt_every == 0 or st["batch"] == self.n_batches:
                        self.save()
                    if stop_after_steps is not None and st["step"] >= stop_after_steps:
                        return self.state
            self._end_epoch()
        self.state["done"] = True
        self.save()
        return self.state


def final_test(trainer, test_ds, n_boot=1000):
    """The held-out generator, touched ONCE per run (contract rule 5). Requires finished training. Primary result =
    final-epoch model; secondary = best-validation-epoch model. Writes run_dir/results.json."""
    path = os.path.join(trainer.run_dir, "results.json")
    if os.path.exists(path):
        raise RuntimeError(f"{path} exists: the held-out test was already evaluated for this run (contract rule 5)")
    assert trainer.state["done"], "training has not finished; the held-out test is evaluated at the end only"
    cfg, st = trainer.cfg, trainer.state
    pred = lambda: predict(trainer.model, test_ds, cfg.batch_size * 2, trainer.device, trainer.use_amp, cfg.num_workers)
    logits, y, _ = pred()
    res = {"run_id": os.path.basename(trainer.run_dir.rstrip("/")), "git_commit": git_commit(),
           "config": asdict(cfg), "epochs_done": st["epoch"], "skipped_steps": st["skipped"],
           "nonfinite_steps": st["nonfinite"], "history": st["history"],
           "final_epoch": {"epoch": st["epoch"], **evaluate(y, logits, n_boot=n_boot)}}
    if os.path.exists(trainer.best_path):
        ck = torch.load(trainer.best_path, map_location="cpu", weights_only=True)
        trainer.model.load_trainable_state_dict(ck["model"])
        logits, y, _ = pred()
        res["best_val_epoch"] = {"epoch": ck["epoch"], "val_auroc": ck["val_auroc"],
                                 **evaluate(y, logits, n_boot=n_boot)}
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    return res