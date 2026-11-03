"""Generate the canonical LOGO split manifests ONCE and write splits/<fold>.json.

Uses Chethan's shared data code (src/vendor/chethan_data.py) so both builds get identical splits.
After this runs, nobody calls build_logo_split again: both builds load the JSON manifests.

Usage (Colab):
    python -m src.export_splits --out splits
Prints a SHA-256 per file; Chethan runs the same command and the hashes must match.
"""
import argparse
import hashlib
import json
import os

import numpy as np

from src.vendor.chethan_data import build_logo_split

REPO = "TheKernel01/Tiny-GenImage"
FOLDS = ["Midjourney", "BigGAN", "ADM"]
PER_GEN = 300
SEED = 0
EXPECTED_TEST = 500  # contract: 500 fake (held-out) + 500 real


def make_manifest(train_gen, val_gen, names, held_out, revision, per_gen=PER_GEN, seed=SEED):
    s = build_logo_split(train_gen, val_gen, names, held_out, per_gen=per_gen, seed=seed)
    c = s["counts"]
    # Contract checks: fail loudly instead of silently shrinking.
    assert s["per_gen"] == per_gen, f"{held_out}: per_gen shrank to {s['per_gen']} (contract: {per_gen})"
    assert c["test_fake"] == EXPECTED_TEST and c["test_real"] == EXPECTED_TEST, \
        f"{held_out}: test is {c['test_fake']} fake / {c['test_real']} real (contract: {EXPECTED_TEST}/{EXPECTED_TEST})"
    s["dataset"] = REPO
    s["revision"] = revision
    s["index_space"] = "row positions: train/val -> HF 'train' split, test -> HF 'validation' split"
    # sorted so the file is byte-stable
    for k in ("train", "val", "test"):
        s[k] = sorted(int(i) for i in s[k])
    return s


def write_manifest(manifest, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{manifest['held_out'].lower()}.json")
    text = json.dumps(manifest, indent=1, sort_keys=True)
    with open(path, "w") as f:
        f.write(text + "\n")
    return path, hashlib.sha256((text + "\n").encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="splits")
    a = ap.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(REPO).sha
    ds = load_dataset(REPO, revision=revision)
    names = ds["train"].features["generator"].names
    train_gen = np.asarray(ds["train"]["generator"])
    val_gen = np.asarray(ds["validation"]["generator"])
    print("revision:", revision, "| names:", names)
    for fold in FOLDS:
        m = make_manifest(train_gen, val_gen, names, fold, revision)
        path, h = write_manifest(m, a.out)
        print(f"{fold:11s} {m['counts']}  seen={m['seen']}\n  {path}  sha256={h}")


if __name__ == "__main__":
    main()
