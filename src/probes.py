"""Frozen-CLIP probes (M2). StandardScaler + L2 logistic regression; C chosen on the seen-generator val set.

  univfd     : post-layernorm CLS of layer 24                    (1024-d)
  multilayer : CLS of layer 16 || raw CLS of layer 24            (2048-d)
Test rows are only touched after C is fixed. LogisticRegression (lbfgs) is deterministic: no seed.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import evaluate, within_ci

FEATURE_SETS = {"univfd": ("pooled",), "multilayer": ("cls_mid", "cls_last")}
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

# Chethan's reported numbers (contract Section 2): (auroc, ci_lo, ci_hi)
REFERENCE = {
    ("Midjourney", "univfd"): (0.826, 0.799, 0.851), ("Midjourney", "multilayer"): (0.875, 0.855, 0.896),
    ("BigGAN", "univfd"): (0.960, 0.951, 0.969), ("BigGAN", "multilayer"): (0.976, 0.968, 0.983),
    ("ADM", "univfd"): (0.889, 0.866, 0.908), ("ADM", "multilayer"): (0.977, 0.970, 0.984),
}


def design(store, rows, kind):
    return np.concatenate([store.take(k, rows) for k in FEATURE_SETS[kind]], axis=1)


def fit_probe(Xtr, ytr, Xva, yva, Cs=C_GRID):
    """Pick C by val AUROC (ties -> smaller C), refit is the pipeline fit at that C."""
    best = None
    for C in Cs:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        pipe.fit(Xtr, ytr)
        a = roc_auc_score(yva, pipe.decision_function(Xva))
        if best is None or a > best[0] + 1e-12:
            best = (a, C, pipe)
    return best[2], best[1], best[0]


def run_fold(manifest, train_store, test_store, kind, n_boot=1000):
    tr, va, te = manifest["train"], manifest["val"], manifest["test"]
    Xtr, ytr = design(train_store, tr, kind), train_store.take("label", tr)
    Xva, yva = design(train_store, va, kind), train_store.take("label", va)
    pipe, C, val_auc = fit_probe(Xtr, ytr, Xva, yva)
    # ---- held-out test touched once, after C is fixed
    Xte, yte = design(test_store, te, kind), test_store.take("label", te)
    res = evaluate(yte, pipe.decision_function(Xte), n_boot=n_boot)
    res.update(fold=manifest["held_out"], kind=kind, C=C, val_auroc=float(val_auc))
    return res


def gate_report(results):
    """Per-cell comparison with the reference. in_ref_ci (point estimate inside reference CI) is informational;
    the gate itself is gate_passes() (revised criterion, decision D017)."""
    rows = []
    for r in results:
        ref, lo, hi = REFERENCE[(r["fold"], r["kind"])]
        olo, ohi = r["auroc_ci95"]
        rows.append({"fold": r["fold"], "kind": r["kind"], "ours": round(r["auroc"], 3),
                     "ours_ci": [round(olo, 3), round(ohi, 3)], "ref": ref, "ref_ci": [lo, hi],
                     "in_ref_ci": bool(within_ci(r["auroc"], lo, hi)),
                     "ci_overlap": bool(olo <= hi and lo <= ohi), "delta": round(r["auroc"] - ref, 3),
                     "C": r["C"]})
    return rows


MAX_SHORTFALL = 0.02


def gate_passes(rows, max_shortfall=MAX_SHORTFALL):
    """Revised M2 gate (D017): every cell's CI overlaps the reference CI AND no cell is more than
    max_shortfall below the reference AUROC."""
    return all(r["ci_overlap"] and r["delta"] > -max_shortfall for r in rows)
