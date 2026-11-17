"""Concept bank (offline): 16 prompts (8 fake-oriented, 8 real-oriented) encoded once by the frozen CLIP text tower.

Prompts describe visual properties, never generator names (cross-generator goal; spec Section 3.6).
Order is fixed: indices 0-7 fake-oriented, 8-15 real-oriented. CONCEPT_POLARITY marks this (1 = fake) for
analysis and the v2 polarity-loss candidate; M4 itself does not use it.

The (16, 768) embedding table is cached as .npy (NOT .pt: *.pt is gitignored) so it can be committed to
assets/ and training never needs the text tower.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

TEMPLATE = "an image with {}"
TEXT_CHECKPOINT = "openai/clip-vit-large-patch14-336"  # same checkpoint as the vision tower (contract Section 2)

FAKE_CONCEPTS = (
    "unnaturally smooth, plastic-looking skin or surfaces",
    "inconsistent or physically implausible lighting and shadows",
    "distorted or malformed fine details such as hands, text and teeth",
    "repetitive, over-regular synthetic texture patterns",
    "an overly perfect, airbrushed, hyper-polished rendering",
    "blurry or melted boundaries between objects",
    "incoherent background structures that do not make geometric sense",
    "oversaturated, dreamlike colors typical of digital art",
)
REAL_CONCEPTS = (
    "natural sensor noise and film grain",
    "consistent lighting with physically plausible shadows",
    "fine natural detail such as skin pores, fabric weave and foliage",
    "natural lens blur and depth of field from a real camera",
    "an ordinary candid composition with imperfect framing",
    "realistic camera artifacts such as motion blur and lens flare",
    "coherent scene geometry with correct perspective",
    "natural, slightly muted colors of an unedited photograph",
)
CONCEPTS = FAKE_CONCEPTS + REAL_CONCEPTS
CONCEPT_POLARITY = (1,) * len(FAKE_CONCEPTS) + (0,) * len(REAL_CONCEPTS)
N_CONCEPTS = len(CONCEPTS)  # 16


def prompts(template=TEMPLATE):
    return [template.format(c) for c in CONCEPTS]


@torch.no_grad()
def encode_concepts(text_model, tokenizer, device="cpu", template=TEMPLATE, max_length=77):
    """text_model: HF CLIPTextModelWithProjection (frozen). Returns L2-normalised float32 (16, d_text) tensor."""
    text_model = text_model.eval().to(device)
    tok = tokenizer(prompts(template), padding="max_length", max_length=max_length, truncation=True,
                    return_tensors="pt")
    out = text_model(**{k: v.to(device) for k, v in tok.items()})
    return F.normalize(out.text_embeds.float(), dim=-1).cpu()


def load_or_encode(path, text_model=None, tokenizer=None, **kw):
    """Read the cached table if present, else encode once and save it. Returns a float32 torch tensor."""
    if os.path.exists(path):
        return torch.from_numpy(np.load(path)).float()
    assert text_model is not None and tokenizer is not None, "no cache and no text tower given"
    e = encode_concepts(text_model, tokenizer, **kw)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, e.numpy())
    return e