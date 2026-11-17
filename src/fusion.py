"""M4: block-16/24 fusion -> X (B,576,384); concept bank -> C (16,384); token-to-concept affinity alpha (B,576,16).

Input is the M3 backbone output: a tuple of patch-token taps, each (B,576,1024), CLS already dropped.

Decisions (docs/decision_log.md D003-D006):
  fusion   : per-tap LayerNorm -> softmax-weighted scalar mix (learned, init equal) -> Linear(1024, 384)
             (per-tap LayerNorm because raw CLIP intermediate tokens have large outlier dimensions and the two
              taps live on very different scales)
  concepts : frozen L2-normalised CLIP text embeddings (16,768) -> own learned Linear(768, 384)
             (v1: two independent learned projections, as in the spec; a shared/frozen projection is a v2 candidate)
  affinity : alpha = softmax_k( s * cos(x_t, c_k) ), s = learnable logit scale, init 1/0.07, clamped <= 100
             computed in fp32 even under autocast
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerFusion(nn.Module):
    def __init__(self, n_taps=2, d_in=1024, d_out=384):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d_in) for _ in range(n_taps))
        self.mix_logits = nn.Parameter(torch.zeros(n_taps))  # softmax -> equal weights at init
        self.proj = nn.Linear(d_in, d_out)

    def mix_weights(self):
        return F.softmax(self.mix_logits, dim=0)

    def forward(self, taps):
        assert len(taps) == len(self.norms), f"expected {len(self.norms)} taps, got {len(taps)}"
        w = self.mix_weights()
        fused = sum(w[i] * self.norms[i](t) for i, t in enumerate(taps))
        return self.proj(fused)


class ConceptAffinity(nn.Module):
    def __init__(self, text_embeds, d=384, init_temp=0.07, max_scale=100.0):
        super().__init__()
        te = F.normalize(text_embeds.float(), dim=-1)
        self.register_buffer("text_embeds", te)  # frozen CLIP text output, never trained
        self.text_proj = nn.Linear(te.shape[1], d)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))
        self.max_log_scale = math.log(max_scale)

    def concepts(self):
        return self.text_proj(self.text_embeds)  # (K, d)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):  # keep affinity in fp32
            C = self.concepts()
            xn = F.normalize(x.float(), dim=-1)
            cn = F.normalize(C, dim=-1)
            scale = self.log_scale.clamp(max=self.max_log_scale).exp()
            logits = scale * (xn @ cn.t())  # (B, N, K)
            alpha = F.softmax(logits, dim=-1)
        return alpha, logits, C


class ConceptFrontEnd(nn.Module):
    """taps: sequence of (B,N,d_in) patch tokens. Returns dict(X, alpha, logits, C)."""

    def __init__(self, text_embeds, d_in=1024, d=384, n_taps=2):
        super().__init__()
        self.fusion = LayerFusion(n_taps, d_in, d)
        self.affinity = ConceptAffinity(text_embeds, d)

    def forward(self, taps):
        X = self.fusion(taps)
        alpha, logits, C = self.affinity(X)
        return {"X": X, "alpha": alpha, "logits": logits, "C": C}


@torch.no_grad()
def affinity_stats(alpha):
    """Diagnostics for the notebook gate: is alpha informative (not uniform, not one-hot, no dead concepts)?"""
    a = alpha.float().reshape(-1, alpha.shape[-1])
    ent = -(a * a.clamp_min(1e-9).log()).sum(-1)
    usage = a.mean(0)  # average mass per concept
    return {
        "row_sum_min": a.sum(-1).min().item(), "row_sum_max": a.sum(-1).max().item(),
        "entropy_mean": ent.mean().item(), "entropy_max_possible": math.log(a.shape[-1]),
        "max_alpha_mean": a.max(-1).values.mean().item(),
        "argmax_concepts_used": int(a.argmax(-1).unique().numel()),
        "usage_min": usage.min().item(), "usage_max": usage.max().item(),
    }