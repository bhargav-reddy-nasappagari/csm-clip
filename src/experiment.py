"""M9: main experiment, steered vs plain on the same fold with matched settings.

Design (docs/decision_log.md D022, written before any M9 result existed):
  * Fold Midjourney, controlled protocol, TrainConfig defaults for everything else. The two arms of a seed differ in
    `arm` only (checked by check_matched).
  * run_one() is idempotent: a finished run (results.json) is returned, an unfinished one resumes from last.pt. So the
    whole experiment is "run the same cell again" after a Colab disconnect.
  * The held-out test is touched once per run, inside final_test(). Everything diagnostic here uses VALIDATION only.
  * The two arms are compared with a PAIRED bootstrap: the same resampled test images for both, CI on the AUROC
    difference. Overlap of the two separate CIs is not used to decide anything.
"""
import datetime
import gc
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.concepts import CONCEPT_POLARITY
from src.fusion import affinity_stats
from src.readout import readout_diagnostics
from src.train import OPERATIONAL_KEYS, TrainConfig, Trainer, final_test, git_commit, science_config

ARM_ORDER = ("steered", "plain")
M9_BASE = dict(fold="Midjourney", protocol="controlled", epochs=5, batch_size=8,
               ckpt_every=100, log_every=25, diag_every=100)
FIRST_RUN_ID = 7          # R001-R006 are the M2 probe runs in docs/experiment_log.md
SEEDS_FOR_CLAIM = 3       # contract Section 2: three seeds for the main experiment
RUN_FILES = ("results.json", "predictions.npz", "val_diagnostics.json", "sessions.jsonl", "log.jsonl")


def m9_config(arm, seed=0, **overrides):
    return TrainConfig(arm=arm, seed=seed, **{**M9_BASE, **overrides})


def run_name(cfg):
    return f"{cfg.fold.lower()}_{cfg.protocol}_{cfg.arm}_seed{cfg.seed}"


def check_matched(a, b):
    """The two arms must differ in `arm` and nothing else."""
    sa, sb = science_config(a), science_config(b)
    diff = sorted(k for k in sa if sa[k] != sb[k])
    assert diff == ["arm"], f"arms are not matched, they differ in {diff}"


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _science(d):
    return {k: v for k, v in d.items() if k not in OPERATIONAL_KEYS}


# ---- validation-only diagnostics
@contextmanager
def gates_disabled(mamba):
    """Temporarily set every gate to exactly identity (W = b = 0), restoring the trained values afterwards."""
    gates = [layer.mixer.gates for layer in mamba.layers]
    saved = [(g.W.detach().clone(), g.b.detach().clone()) for g in gates]
    try:
        with torch.no_grad():
            for g in gates:
                g.W.zero_()
                g.b.zero_()
        yield
    finally:
        with torch.no_grad():
            for g, (W, b) in zip(gates, saved):
                g.W.copy_(W)
                g.b.copy_(b)


@torch.no_grad()
def val_diagnostics(model, ds, batch_size=16, device="cuda", amp=True, num_workers=2):
    """How the trained model uses concepts and (steered arm) its gates, on the validation split.
    gate deviation: |g - 1| on real tokens, per block and per part (dt, B, C). 0 = identity, max possible 1.
    gates off: the same trained weights with identity gates. This is NOT the plain arm (which trained without gates);
    it measures how much the steered model's own output relies on steering."""
    model.eval()
    dev = torch.device(device)
    use_amp = bool(amp and dev.type == "cuda")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    steered = model.arm == "steered"

    acc = {}

    def hook(i):
        def fn(_module, _inputs, out):
            for part, g in zip(("dt", "B", "C"), out):
                d = (g.float() - 1.0).abs()
                s = acc.setdefault((i, part), [0.0, 0, 0.0])
                s[0] += d.sum().item()
                s[1] += d.numel()
                s[2] = max(s[2], d.max().item())
        return fn

    def run(collect):
        L, Y, A, R = [], [], [], {"logit": [], "scores": [], "pi": []}
        for x, y, _ in dl:
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                out = model(x.to(dev))
            L.append(out["logit"].float().cpu())
            Y.append(torch.as_tensor(y))
            if collect:
                A.append(out["alpha"].float().cpu())
                for k in R:
                    R[k].append(out[k].float().cpu())
        return torch.cat(L).double().numpy(), torch.cat(Y).numpy().astype(int), A, R

    handles = [layer.mixer.gates.register_forward_hook(hook(i))
               for i, layer in enumerate(model.mamba.layers)] if steered else []
    try:
        logits, y, A, R = run(collect=True)
    finally:
        for h in handles:
            h.remove()

    d = {"split": "val", "arm": model.arm, "n": int(len(y)), "auroc": float(roc_auc_score(y, logits)),
         "alpha": affinity_stats(torch.cat(A)),
         "readout": readout_diagnostics({k: torch.cat(v) for k, v in R.items()}, polarity=CONCEPT_POLARITY)}
    if steered:
        d["gate_deviation"] = [
            {"layer": i, **{f"{p}_{s}": v for p in ("dt", "B", "C")
                            for s, v in (("mean", acc[(i, p)][0] / acc[(i, p)][1]), ("max", acc[(i, p)][2]))}}
            for i in range(len(model.mamba.layers))]
        with gates_disabled(model.mamba):
            off = run(collect=False)[0]
        d["gates_off"] = {"auroc": float(roc_auc_score(y, off)),
                          "mean_abs_logit_change": float(np.abs(logits - off).mean()),
                          "mean_abs_logit": float(np.abs(logits).mean()),
                          "sign_flips": int(((logits > 0) != (off > 0)).sum())}
    return d


# ---- one run
def run_one(cfg, build_model, train_ds, val_ds, test_ds, root, device="cuda", n_boot=1000, log=print,
            stop_after_steps=None):
    """Train (or resume) one arm/seed, record validation diagnostics, evaluate the held-out test once.
    build_model(cfg) -> CSMDetector. train_ds is the un-wrapped training dataset (the Trainer seeds it with cfg.seed).
    Returns the results dict, or None when stop_after_steps interrupted the run (tests)."""
    rdir = os.path.join(str(root), run_name(cfg))
    res_path = os.path.join(rdir, "results.json")
    if os.path.exists(res_path):
        res = _read_json(res_path)
        assert _science(res["config"]) == science_config(cfg), f"{res_path} was produced by a different config"
        log(f"[{run_name(cfg)}] already finished, test AUROC {res['final_epoch']['auroc']:.4f} (not re-evaluated)")
        return res
    assert not hasattr(train_ds, "set_epoch"), "pass the raw training dataset; the Trainer seeds it with cfg.seed"
    os.makedirs(rdir, exist_ok=True)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = Trainer(build_model(cfg), train_ds, val_ds, cfg, rdir, device=device, log=log)

    commit = git_commit()
    sessions = os.path.join(rdir, "sessions.jsonl")
    if os.path.exists(sessions):
        with open(sessions) as f:
            first = json.loads(f.readline())["git_commit"]
        if first != commit:
            log(f"WARNING [{run_name(cfg)}] started at commit {first}, resuming at {commit}: note it in the log")
    with open(sessions, "a") as f:
        f.write(json.dumps({"started": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
                            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"}) + "\n")

    t0 = time.time()
    log(f"[{run_name(cfg)}] training")
    t.fit(stop_after_steps=stop_after_steps)
    if not t.state["done"]:
        return None
    # t.model holds the final-epoch weights here; final_test() later swaps in the best-val ones
    _write_json(val_diagnostics(t.model, t.val_ds, cfg.batch_size * 2, device, t.use_amp, cfg.num_workers),
                os.path.join(rdir, "val_diagnostics.json"))
    res = final_test(t, test_ds, n_boot=n_boot)
    mins = (time.time() - t0) / 60
    peak = f", peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB" if dev.type == "cuda" else ""
    log(f"[{run_name(cfg)}] done in {mins:.1f} min (this session){peak}: "
        f"test AUROC {res['final_epoch']['auroc']:.4f} {res['final_epoch']['auroc_ci95']}")
    del t.model, t
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


# ---- comparison
def load_predictions(rdir, which="final"):
    z = np.load(os.path.join(str(rdir), "predictions.npz"))
    key = {"final": "final_logits", "best_val": "best_val_logits"}[which]
    return z["labels"], z[key], z["indices"] if "indices" in z.files else None


def paired_bootstrap(y, a, b, n_boot=1000, seed=0, alpha=0.05):
    """AUROC(a) - AUROC(b) on the same test images. a, b: arrays (n,) or (n_seeds, n). With several seeds the
    statistic is the mean difference over seeds, and every seed sees the same resample. The CI reflects which test
    images were drawn; it does not fully reflect training-seed variance."""
    y = np.asarray(y).astype(int)
    a, b = np.atleast_2d(np.asarray(a, dtype=np.float64)), np.atleast_2d(np.asarray(b, dtype=np.float64))
    assert a.shape == b.shape and a.shape[1] == len(y), (a.shape, b.shape, len(y))

    def delta(i):
        return float(np.mean([roc_auc_score(y[i], u[i]) - roc_auc_score(y[i], v[i]) for u, v in zip(a, b)]))

    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    while len(vals) < n_boot:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(delta(i))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta_auroc": delta(np.arange(n)), "ci95": [float(lo), float(hi)]}


def verdict(per_seed_delta, mean_ci):
    """The rule fixed in D022 before the runs."""
    n = len(per_seed_delta)
    if n < SEEDS_FOR_CLAIM:
        return f"preliminary ({n} seed{'s' if n > 1 else ''}): no claim until {SEEDS_FOR_CLAIM} seeds"
    if mean_ci[0] > 0 and all(d > 0 for d in per_seed_delta):
        return "steering helps"
    if mean_ci[1] < 0 and all(d < 0 for d in per_seed_delta):
        return "steering hurts"
    return "no demonstrated difference"


def compare(root, seeds=(0,), fold=M9_BASE["fold"], protocol=M9_BASE["protocol"], which="final", n_boot=1000):
    """Steered minus plain over the given seeds. which: 'final' (primary) or 'best_val' (secondary)."""
    block = {"final": "final_epoch", "best_val": "best_val_epoch"}[which]
    y0, idx0 = None, None
    logits = {a: [] for a in ARM_ORDER}
    per_seed = []
    for s in seeds:
        row = {"seed": s}
        cfgs = []
        for arm in ARM_ORDER:
            rdir = os.path.join(str(root), run_name(m9_config(arm, s, fold=fold, protocol=protocol)))
            res = _read_json(os.path.join(rdir, "results.json"))
            cfgs.append(TrainConfig(**res["config"]))
            y, lg, idx = load_predictions(rdir, which)
            if y0 is None:
                y0, idx0 = y, idx
            assert np.array_equal(y, y0) and (idx is None) == (idx0 is None) and (idx is None or np.array_equal(idx, idx0)), \
                f"{rdir}: test images differ between runs, the comparison would not be paired"
            logits[arm].append(lg)
            row[arm] = {k: res[block][k] for k in ("epoch", "auroc", "auroc_ci95", "ap", "acc_at_0")}
        check_matched(*cfgs)
        row.update(paired_bootstrap(y0, logits["steered"][-1], logits["plain"][-1], n_boot=n_boot))
        per_seed.append(row)
    pooled = paired_bootstrap(y0, np.stack(logits["steered"]), np.stack(logits["plain"]), n_boot=n_boot)
    summary = {}
    for arm in ARM_ORDER:
        au = [r[arm]["auroc"] for r in per_seed]
        ap = [float(average_precision_score(y0, lg)) for lg in logits[arm]]
        summary[arm] = {"auroc_mean": float(np.mean(au)), "auroc_std": float(np.std(au, ddof=1)) if len(au) > 1 else None,
                        "ap_mean": float(np.mean(ap))}
    return {"fold": fold, "protocol": protocol, "which": block, "seeds": list(seeds), "n_test": int(len(y0)),
            "per_seed": per_seed, "arms": summary, "mean_delta_auroc": pooled["delta_auroc"],
            "mean_delta_ci95": pooled["ci95"],
            "verdict": verdict([r["delta_auroc"] for r in per_seed], pooled["ci95"])}


def format_comparison(c):
    lines = [f"{c['fold']} held out, {c['protocol']}, {c['which']}, n_test={c['n_test']}",
             f"{'seed':>4}  {'steered AUROC [95% CI]':<28}{'plain AUROC [95% CI]':<28}delta (paired 95% CI)"]
    for r in c["per_seed"]:
        cell = {a: "{:.4f} [{:.4f}, {:.4f}]".format(r[a]["auroc"], *r[a]["auroc_ci95"]) for a in ARM_ORDER}
        lines.append(f"{r['seed']:>4}  {cell['steered']:<28}{cell['plain']:<28}"
                     "{:+.4f} [{:+.4f}, {:+.4f}]".format(r["delta_auroc"], *r["ci95"]))
    lines.append("mean delta {:+.4f} [{:+.4f}, {:+.4f}] -> {}".format(
        c["mean_delta_auroc"], *c["mean_delta_ci95"], c["verdict"]))
    return "\n".join(lines)


# ---- gate, log rows, export
def m9_gate(root, seeds=(0,), fold=M9_BASE["fold"], protocol=M9_BASE["protocol"]):
    """Contract M9: both arms complete, results JSON written. Raises if not; returns the per-run facts."""
    runs = {}
    for s in seeds:
        cfgs = []
        for arm in ARM_ORDER:
            cfg = m9_config(arm, s, fold=fold, protocol=protocol)
            path = os.path.join(str(root), run_name(cfg), "results.json")
            assert os.path.exists(path), f"missing {path}"
            res = _read_json(path)
            assert _science(res["config"]) == science_config(cfg), f"{path}: config is not the frozen M9 config"
            assert res["epochs_done"] == cfg.epochs, f"{path}: {res['epochs_done']} epochs, expected {cfg.epochs}"
            assert math.isfinite(res["final_epoch"]["auroc"])
            runs[run_name(cfg)] = {"commit": res["git_commit"], "epochs_done": res["epochs_done"],
                                   "skipped_steps": res["skipped_steps"], "nonfinite_steps": res["nonfinite_steps"],
                                   "test_auroc": res["final_epoch"]["auroc"]}
            cfgs.append(cfg)
        check_matched(*cfgs)
    return {"milestone": "M9", "runs": runs, "gate": "PASS"}


def log_rows(root, seeds=(0,), fold=M9_BASE["fold"], protocol=M9_BASE["protocol"], first_id=FIRST_RUN_ID):
    """Markdown rows for docs/experiment_log.md (final epoch; best-val epoch in the notes)."""
    rows, rid = [], first_id
    for s in seeds:
        for arm in ARM_ORDER:
            cfg = m9_config(arm, s, fold=fold, protocol=protocol)
            path = os.path.join(str(root), run_name(cfg), "results.json")
            r = _read_json(path)
            f, b = r["final_epoch"], r.get("best_val_epoch")
            date = datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
            notes = f"skipped {r['skipped_steps']}, nonfinite {r['nonfinite_steps']}"
            if b:
                notes += f"; best-val epoch {b['epoch']} (val {b['val_auroc']:.4f}): test {b['auroc']:.4f}"
            rows.append(f"| R{rid:03d} | {date} | {r['git_commit']} | {fold} | {protocol} | {arm} | {s} "
                        f"| {r['epochs_done']} | none (M9 frozen config) "
                        f"| {f['auroc']:.4f} [{f['auroc_ci95'][0]:.4f}, {f['auroc_ci95'][1]:.4f}] | {f['ap']:.4f} "
                        f"| {notes} |")
            rid += 1
    return "\n".join(rows)


def export(root, dest, seeds=(0,), fold=M9_BASE["fold"], protocol=M9_BASE["protocol"]):
    """Copy the small run artefacts (not checkpoints) from Drive into the repo for committing."""
    out = []
    for s in seeds:
        for arm in ARM_ORDER:
            name = run_name(m9_config(arm, s, fold=fold, protocol=protocol))
            os.makedirs(os.path.join(str(dest), name), exist_ok=True)
            for fn in RUN_FILES:
                src = os.path.join(str(root), name, fn)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(str(dest), name, fn))
                    out.append(f"{name}/{fn}")
    return out