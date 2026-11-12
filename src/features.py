"""Frozen-CLIP feature extraction with a resumable on-disk cache (Drive on Colab).

Features depend only on (image, protocol), not on the fold, so we extract once for the union of
rows needed by all folds and reuse them. Cache = chunked .npz files; re-running skips finished chunks.
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import preprocess

KEYS = ("cls_mid", "cls_last", "pooled")  # CLS of layer 16, CLS of layer 24 (raw), post-layernorm CLS of layer 24


class RowDataset(Dataset):
    """Arbitrary HF rows -> (image tensor, label int, generator int). No augmentation."""

    def __init__(self, hf, indices, protocol):
        self.hf, self.indices, self.protocol = hf, np.asarray(indices, dtype=np.int64), protocol

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ex = self.hf[int(self.indices[i])]
        return preprocess(ex["image"], self.protocol, train=False), int(ex["label"]), int(ex["generator"])


class ClipTaps(torch.nn.Module):
    """Frozen CLIP vision tower -> CLS features. HF hidden_states[0] is the embedding output, so
    hidden_states[16] / [24] are the outputs of encoder layers 16 / 24 (contract: taps 16 and 24)."""

    def __init__(self, vision_model, taps=(16, 24)):
        super().__init__()
        self.m, self.taps = vision_model.eval(), taps
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        out = self.m(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states
        return {"cls_mid": hs[self.taps[0]][:, 0], "cls_last": hs[self.taps[1]][:, 0],
                "pooled": out.pooler_output}  # pooler_output = post_layernorm(CLS of last layer)


def extract_to_cache(encoder, ds, out_dir, chunk=512, batch_size=32, num_workers=2, device="cpu",
                     autocast=False, log=print):
    os.makedirs(out_dir, exist_ok=True)
    n = len(ds)
    dev = torch.device(device)
    for k, s in enumerate(range(0, n, chunk)):
        path = os.path.join(out_dir, f"chunk_{k:05d}.npz")
        if os.path.exists(path):
            continue
        e = min(n, s + chunk)
        dl = DataLoader(Subset(ds, range(s, e)), batch_size=batch_size, num_workers=num_workers)
        feats, labels, gens = {key: [] for key in KEYS}, [], []
        for x, y, g in dl:
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=autocast):
                out = encoder(x.to(dev))
            for key in KEYS:
                feats[key].append(out[key].float().cpu().numpy())
            labels.append(y.numpy()); gens.append(g.numpy())
        tmp = path + ".tmp.npz"
        np.savez(tmp, idx=ds.indices[s:e], label=np.concatenate(labels), gen=np.concatenate(gens),
                 **{key: np.concatenate(v) for key, v in feats.items()})
        os.replace(tmp, path)
        log(f"{out_dir}: {e}/{n}")


class FeatureStore:
    """Cached features addressable by HF row index."""

    def __init__(self, out_dir):
        files = sorted(glob.glob(os.path.join(out_dir, "chunk_*.npz")))
        assert files, f"no cache in {out_dir}"
        parts = [np.load(f) for f in files]
        self.idx = np.concatenate([p["idx"] for p in parts])
        self.arrays = {k: np.concatenate([p[k] for p in parts]) for k in (*KEYS, "label", "gen")}
        order = np.argsort(self.idx)
        self.idx = self.idx[order]
        self.arrays = {k: v[order] for k, v in self.arrays.items()}
        assert len(np.unique(self.idx)) == len(self.idx), "duplicate rows in cache"

    def _pos(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        pos = np.searchsorted(self.idx, rows)
        assert (pos < len(self.idx)).all() and (self.idx[pos] == rows).all(), "rows missing from cache"
        return pos

    def take(self, key, rows):
        return self.arrays[key][self._pos(rows)]


def union_rows(manifests, which):
    return sorted({int(i) for m in manifests for i in m[which]})
