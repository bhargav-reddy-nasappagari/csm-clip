"""Data harness: manifest loading, preprocessing protocols (raw / controlled), Dataset.

Our own implementation of contract Section 2. Splits come ONLY from splits/*.json.
Manifest index space: train/val -> HF 'train' split rows; test -> HF 'validation' split rows.
HF labels: label 0=real, 1=fake.
"""
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIZE = 336
PROTOCOLS = ("raw", "controlled")
CONTROLLED_PRE_SIZE = 256
CONTROLLED_JPEG_Q = 90


def load_manifest(fold, split_dir="splits"):
    with open(os.path.join(split_dir, f"{fold.lower()}.json")) as f:
        m = json.load(f)
    assert m["held_out"].lower() == fold.lower()
    return m


def _square_crop(img):
    w, h = img.size
    m = min(w, h)
    l, t = (w - m) // 2, (h - m) // 2
    return img.crop((l, t, l + m, t + m))


def controlled_pre(img):
    """Centre-crop square -> resize 256x256 (bicubic) -> ONE JPEG q90 encode/decode.
    Resize happens before JPEG so every image has exactly one compression."""
    img = _square_crop(img.convert("RGB")).resize((CONTROLLED_PRE_SIZE,) * 2, Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=CONTROLLED_JPEG_Q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _resize_short_and_crop(img):
    w, h = img.size
    s = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
    return img.crop((left, top, left + SIZE, top + SIZE))


def preprocess(img, protocol="raw", train=False, rng=None):
    """Returns a normalised (3,336,336) float tensor. Flip is training-only, both protocols."""
    assert protocol in PROTOCOLS, protocol
    img = img.convert("RGB")
    if protocol == "controlled":
        img = controlled_pre(img)
    img = _resize_short_and_crop(img)
    if train and (rng or np.random).random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class ManifestDataset(Dataset):
    """One manifest split ('train' | 'val' | 'test') -> (image tensor, label float, generator id)."""

    def __init__(self, manifest, which, hf_train, hf_val, protocol="raw", train=None, seed=0):
        assert which in ("train", "val", "test")
        self.hf = hf_val if which == "test" else hf_train
        self.indices = np.asarray(manifest[which], dtype=np.int64)
        self.protocol = protocol
        self.train = (which == "train") if train is None else train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ex = self.hf[int(self.indices[i])]
        x = preprocess(ex["image"], self.protocol, self.train, self.rng)
        return x, torch.tensor(float(ex["label"])), int(ex["generator"])


def audit_manifest(manifest, train_gen, val_gen, names):
    """Needs the HF generator-id columns (Colab). Verifies the M1 gate facts against real data."""
    train_gen, val_gen = np.asarray(train_gen), np.asarray(val_gen)
    held = names.index(manifest["held_out"])
    tr, va, te = (np.asarray(manifest[k]) for k in ("train", "val", "test"))
    assert not set(tr.tolist()) & set(va.tolist()), "train/val overlap"
    assert not (train_gen[tr] == held).any() and not (train_gen[va] == held).any(), "held-out leaked"
    assert (val_gen[te] == held).sum() == 500 and (val_gen[te] == 0).sum() == 500, "test not 500/500"
    for name, idx in (("train", tr), ("val", va)):
        g = train_gen[idx]
        assert (g == 0).sum() == (g != 0).sum(), f"{name} not 1:1 real:fake"
    return {"train": len(tr), "val": len(va), "test": len(te)}
