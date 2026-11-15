"""M3: CLIP ViT-L/14@336 backbone with LoRA, exposing patch-token taps.

Decisions (see docs/decision_log.md D001/D002):
  * HF CLIPVisionModel 'openai/clip-vit-large-patch14-336'; taps are hidden_states[16] and [24]
    (outputs of encoder layers 16 and 24, same indexing as M2), CLS token dropped -> 2 x (B, 576, 1024).
  * LoRA (hand-written, no peft dependency) on q_proj and v_proj of every encoder layer, r=8, alpha=16.
    B is zero-initialised, so at init the wrapped model is exactly the frozen CLIP.
  * Everything else (embeddings, pre/post layernorm, MLPs, k_proj, out_proj) stays frozen.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B (A x). Base layer frozen; only A, B train."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        assert r > 0
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


class ClipLoRABackbone(nn.Module):
    """Wraps a HF CLIPVisionModel. forward(pixel_values) -> tuple of patch-token tensors, one per tap."""

    def __init__(self, vision_model, taps=(16, 24), r=8, alpha=16.0, targets=("q_proj", "v_proj")):
        super().__init__()
        self.m = vision_model
        self.taps = tuple(taps)
        n_layers = self.m.config.num_hidden_layers
        assert all(1 <= t <= n_layers for t in self.taps), f"taps {self.taps} outside 1..{n_layers}"
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.n_wrapped = 0
        for layer in self._layers():
            for name in targets:
                base = getattr(layer.self_attn, name)
                assert isinstance(base, nn.Linear), f"{name} is {type(base)}"
                setattr(layer.self_attn, name, LoRALinear(base, r, alpha))
                self.n_wrapped += 1

    def _layers(self):
        vm = getattr(self.m, "vision_model", self.m)
        return vm.encoder.layers

    def forward(self, x):
        hs = self.m(pixel_values=x, output_hidden_states=True).hidden_states
        return tuple(hs[t][:, 1:] for t in self.taps)  # drop CLS (index 0)

    # ---- parameter bookkeeping
    def lora_parameters(self):
        return [p for n, p in self.named_parameters() if "lora_" in n]

    def lora_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if "lora_" in k}

    def load_lora_state_dict(self, sd):
        missing = [k for k in self.lora_state_dict() if k not in sd]
        assert not missing, f"missing LoRA keys: {missing[:3]}"
        res = self.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys

    def param_report(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "lora_modules": self.n_wrapped}