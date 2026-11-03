"""Metrics: AUROC with 95% bootstrap CI (1000 resamples), average precision, accuracy at threshold 0.
Scores are logits (fake = positive class); threshold 0 on the logit."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_auroc(y, s, n_boot=1000, seed=0, alpha=0.05):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    while len(vals) < n_boot:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():  # degenerate resample, redraw
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def evaluate(y, logits, n_boot=1000, seed=0):
    y, logits = np.asarray(y).astype(int), np.asarray(logits, dtype=np.float64)
    auc, lo, hi = bootstrap_auroc(y, logits, n_boot, seed)
    return {"auroc": auc, "auroc_ci95": [lo, hi],
            "ap": float(average_precision_score(y, logits)),
            "acc_at_0": float(((logits > 0).astype(int) == y).mean()), "n": int(len(y))}


def within_ci(value, lo, hi):
    return lo <= value <= hi
