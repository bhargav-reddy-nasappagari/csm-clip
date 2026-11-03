"""Data utilities for Tiny-GenImage (HF: TheKernel01/Tiny-GenImage).

Labels in the HF dataset:  label: 0=real, 1=fake
                           generator: 0=Real, 1..8 = ADM, BigGAN, GLIDE, Midjourney, SD14, SD15, VQDM, Wukong

Protocol = leave-one-generator-out (LOGO):
  * train / val  : drawn from the HF *train* split, seen generators only (held-out generator never appears)
  * test         : drawn from the HF *validation* split: held-out generator's fakes + the same number of reals
  * real images are balanced 1:1 against fakes in every set
  * model selection (val) uses seen generators only - the held-out generator is touched once, at the end
Everything is deterministic given (held_out, per_gen, seed).
"""
import io
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIZE = 336


def preprocess(img, train=False, rng=None, jpeg_q=None, pre_size=None):
    """CLIP-style: RGB -> bicubic resize (short side 336) -> center crop 336 -> normalise.
    Train adds a random horizontal flip. Identical pipeline for every class.

    Optional shortcut controls (applied identically to real and fake, train and test), in this order:
      pre_size : center-crop to a square, resize to pre_size x pre_size -> removes native-size / aspect cues
                 (and, for real images, destroys the original JPEG 8x8 block grid)
      jpeg_q   : then re-encode as JPEG at this quality -> every image ends up with the same single
                 compression, removing the PNG-vs-JPEG file-format cue
    """
    img = img.convert("RGB")
    if pre_size:
        w, h = img.size
        m = min(w, h)
        l, t = (w - m) // 2, (h - m) // 2
        img = img.crop((l, t, l + m, t + m)).resize((pre_size, pre_size), Image.BICUBIC)
    if jpeg_q:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(jpeg_q))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    w, h = img.size
    s = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    if train and (rng or np.random).random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class DetectionDataset(Dataset):
    """Wraps one HF split + a list of indices -> (image tensor, label float, generator id)."""
    def __init__(self, hf_split, indices, train=False, jpeg_q=None, pre_size=None):
        self.ds, self.indices, self.train = hf_split, np.asarray(indices), train
        self.jpeg_q, self.pre_size = jpeg_q, pre_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ex = self.ds[int(self.indices[i])]
        return preprocess(ex["image"], self.train, jpeg_q=self.jpeg_q, pre_size=self.pre_size), torch.tensor(float(ex["label"])), int(ex["generator"])


def _pick(rng, pool, n):
    pool = np.asarray(pool)
    n = min(n, len(pool))
    return rng.choice(pool, size=n, replace=False)


def build_logo_split(train_gen, val_gen, names, held_out, per_gen=300, n_test=None,
                     val_frac=0.1, seed=0):
    """train_gen / val_gen: integer generator id per example of the HF train / validation split.
    names: class names ('Real','ADM',...). Returns dict of index lists + metadata."""
    train_gen, val_gen = np.asarray(train_gen), np.asarray(val_gen)
    held_id = names.index(held_out)
    assert held_id != 0, "held_out must be a generator, not 'Real'"
    # only generators that actually have images (e.g. 'SD14' is empty in Tiny-GenImage)
    seen = [g for g in range(1, len(names)) if g != held_id and (train_gen == g).any()]
    assert (val_gen == held_id).any(), f"no test images for held-out generator {held_out}"
    rng = np.random.default_rng(seed)

    # ---- train pool: seen fakes + equal number of reals (HF train split)
    # If there are fewer reals than requested fakes, shrink the fakes so classes stay balanced.
    real_pool = np.where(train_gen == 0)[0]
    per_gen_eff = min(per_gen, len(real_pool) // len(seen))
    fake_idx = np.concatenate([_pick(rng, np.where(train_gen == g)[0], per_gen_eff) for g in seen])
    real_idx = _pick(rng, real_pool, len(fake_idx))

    # ---- carve a validation set (seen generators only), stratified fake/real
    def carve(idx):
        idx = rng.permutation(idx)
        k = max(1, int(round(len(idx) * val_frac)))
        return idx[k:], idx[:k]
    f_tr, f_va = carve(fake_idx)
    r_tr, r_va = carve(real_idx)
    train = rng.permutation(np.concatenate([f_tr, r_tr]))
    val = rng.permutation(np.concatenate([f_va, r_va]))

    # ---- test: held-out fakes + equal reals (HF validation split)
    t_fake = np.where(val_gen == held_id)[0]
    t_real_pool = np.where(val_gen == 0)[0]
    n = min(len(t_fake), len(t_real_pool), n_test if n_test is not None else len(t_fake))
    t_fake = _pick(rng, t_fake, n)
    t_real = _pick(rng, t_real_pool, n)
    test = rng.permutation(np.concatenate([t_fake, t_real]))

    # ---- hard guarantees
    assert not set(train) & set(val), "train/val overlap"
    assert not np.isin(train_gen[train], [held_id]).any(), "held-out generator leaked into train"
    assert not np.isin(train_gen[val], [held_id]).any(), "held-out generator leaked into val"
    assert (val_gen[test] == held_id).sum() == len(t_fake) and (val_gen[test] == 0).sum() == len(t_real)
    return {
        "held_out": held_out, "seen": [names[g] for g in seen], "per_gen": per_gen_eff, "seed": seed,
        "train": train.tolist(), "val": val.tolist(), "test": test.tolist(),
        "counts": {"train": len(train), "val": len(val), "test": len(test),
                   "test_fake": int(len(t_fake)), "test_real": int(len(t_real))},
    }


def audit_formats(hf_split, names, per_gen=40, seed=0):
    """Look for dataset shortcuts: file format and image size per generator (decodes a small sample)."""
    gen = np.asarray(list(hf_split["generator"]))
    rng = np.random.default_rng(seed)
    rows = []
    for g, name in enumerate(names):
        idx = _pick(rng, np.where(gen == g)[0], per_gen)
        if len(idx) == 0:
            rows.append({"class": name, "n": 0, "top_format": "-", "pct_top_format": 0.0,
                         "median_w": 0, "median_h": 0, "min_side": 0, "max_side": 0})
            continue
        fmts, ws, hs = [], [], []
        for i in idx:
            im = hf_split[int(i)]["image"]
            fmts.append(im.format or "?"); ws.append(im.size[0]); hs.append(im.size[1])
        top = max(set(fmts), key=fmts.count)
        rows.append({"class": name, "n": len(idx), "top_format": top,
                     "pct_top_format": 100.0 * fmts.count(top) / len(fmts),
                     "median_w": int(np.median(ws)), "median_h": int(np.median(hs)),
                     "min_side": int(min(min(ws), min(hs))), "max_side": int(max(max(ws), max(hs)))})
    return rows


def metadata_features(hf_split, indices):
    """Header-only features (no pixels): [is_jpeg, width, height, is_square, min_side, aspect].
    Used for a 'shortcut baseline': a classifier that never looks at image content."""
    X = []
    for i in indices:
        im = hf_split[int(i)]["image"]
        w, h = im.size
        X.append([float((im.format or "") == "JPEG"), w, h, float(w == h), min(w, h), max(w, h) / max(1, min(w, h))])
    return np.asarray(X, dtype=np.float32)
