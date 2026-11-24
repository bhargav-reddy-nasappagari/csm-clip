"""M8: the full detector, assembled from the M3-M7 modules.

    image -> CLIP ViT-L/14 + LoRA (taps 16, 24, CLS dropped)            M3
          -> fusion -> X (B,576,384); concept affinity alpha (B,576,16)  M4
          -> 2 x Mamba, steered by alpha  ("steered")  or not ("plain")  M5 / M6
          -> concept-wise readout -> logit, 24x24 map                    M7

Two arms (decision D019). They share EVERYTHING (backbone, front end, readout, readout pooling by alpha); the only
difference is that the steered Mamba gates dt / B / C with alpha and the plain Mamba does not. This is baseline 6 of
the architecture doc (Section 9.1: "concept affinity + Mamba with no state conditioning"). With the same seed the two
arms start from identical weights and, because the gates are zero at init, from identical outputs.

Frozen CLIP weights are never saved to checkpoints (they are re-loaded from the hub); only the trainable parts and the
small buffers outside the CLIP tower are (see trainable_state_dict).
"""
import torch
import torch.nn as nn

from src.backbone import ClipLoRABackbone
from src.fusion import ConceptFrontEnd
from src.mamba import PlainMamba
from src.readout import ConceptReadout
from src.steering import SteeredMamba

ARMS = ("steered", "plain")
_CLIP_PREFIX = "backbone.m."


def _is_checkpointed(key):
    """LoRA tensors, and everything that is not part of the frozen CLIP tower."""
    return ("lora_" in key) or not key.startswith(_CLIP_PREFIX)


class CSMDetector(nn.Module):
    def __init__(self, vision_model, text_embeds, arm="steered", taps=(16, 24), lora_r=8, lora_alpha=16.0,
                 d_in=1024, d=384, n_mamba=2, seed=0, **mamba_kw):
        super().__init__()
        assert arm in ARMS, arm
        self.arm = arm
        n_concepts = int(text_embeds.shape[0])
        torch.manual_seed(seed)  # one seed, fixed construction order -> both arms get identical shared weights
        self.backbone = ClipLoRABackbone(vision_model, taps=taps, r=lora_r, alpha=lora_alpha)
        self.front = ConceptFrontEnd(text_embeds, d_in=d_in, d=d, n_taps=len(taps))
        if arm == "steered":
            self.mamba = SteeredMamba(n_mamba, n_concepts=n_concepts, d_model=d, **mamba_kw)
        else:
            self.mamba = PlainMamba(n_mamba, d_model=d, **mamba_kw)
        self.head = ConceptReadout(d, n_concepts=n_concepts)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.m.eval()  # CLIP stays in eval mode whatever the wrapper does (only LoRA trains)
        return self

    def forward(self, x):
        o = self.front(self.backbone(x))
        H = self.mamba(o["X"], o["alpha"]) if self.arm == "steered" else self.mamba(o["X"])
        out = self.head(H, o["alpha"])
        out["alpha"] = o["alpha"]
        return out

    # ---- checkpoint helpers
    def trainable_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if _is_checkpointed(k)}

    def load_trainable_state_dict(self, sd):
        res = self.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys[:3]}"
        missing = [k for k in res.missing_keys if _is_checkpointed(k)]
        assert not missing, f"checkpoint is missing keys: {missing[:3]}"

    def param_report(self):
        def n(prefix=None, pred=None):
            return sum(p.numel() for k, p in self.named_parameters()
                       if p.requires_grad and (prefix is None or k.startswith(prefix)) and (pred is None or pred(k)))
        return {"trainable": n(), "lora": n("backbone."), "front_end": n("front."),
                "mamba": n("mamba."), "gates": n("mamba.", lambda k: ".gates." in k), "readout": n("head.")}