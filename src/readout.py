"""M7: concept-wise readout, additive logit and the 24x24 explanation map.

Input: H (B,L,d) final Mamba states (L = 576 = 24x24, row-major, token 0 = top-left, the scan order of D008) and
alpha (B,L,K) token-to-concept affinity (K = 16). Per concept k:

    pi_tk = alpha_tk / sum_t alpha_tk                  pooling weights, a distribution over tokens (per image, per concept)
    h_k   = sum_t pi_tk H_t                  (d,)      concept-wise pooled vector     ("16 x 384")
    s_k   = w_k . h_k + b_k                  scalar    own linear scoring head per concept
    z     = sum_k s_k                                  fake logit (positive = fake), P(fake) = sigmoid(z)

Because pooling and the heads are linear, the logit splits exactly over tokens:

    u_tk  = w_k . H_t                                  token-concept response
    c_tk  = pi_tk * u_tk                               contribution of token t through concept k
    m_t   = sum_k c_tk ,  M = reshape(m, 24, 24)       the explanation map
    z     = sum_t m_t + sum_k b_k                      (the M7 gate: map sum + biases == logit)

M is therefore the logit itself, laid out in space, not a heuristic: every cell is the signed evidence (towards fake
when > 0) that a patch adds to the decision, and c[:, :, k] is the same thing for concept k alone. The logit and the
map are computed by two different routes (pool-then-score vs score-then-pool); `decomposition_error` compares them.

Everything here runs in fp32 even under autocast (inputs are cast up), so the identity holds to ~1e-6, far inside the
fp16 tolerance the gate asks for. Decisions: docs/decision_log.md D009 (pooling, heads) and D010 (map).
"""
import math

import torch
import torch.nn as nn

N_CONCEPTS = 16


class ConceptReadout(nn.Module):
    def __init__(self, d=384, n_concepts=N_CONCEPTS, init_std=0.02, eps=1e-12):
        super().__init__()
        self.d, self.K, self.eps = d, n_concepts, eps
        # one scalar head per concept: s_k = w_k . h_k + b_k. Small random w (not zeros): zero w would give H and alpha
        # no gradient on step 0. b is zero, so z starts near 0 (P(fake) ~ 0.5).
        self.w = nn.Parameter(torch.randn(n_concepts, d) * init_std)
        self.b = nn.Parameter(torch.zeros(n_concepts))

    def pooling_weights(self, alpha):
        """alpha (B,L,K) -> pi (B,L,K); every (b,k) column sums to 1 over tokens."""
        return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def forward(self, H, alpha):
        assert H.dim() == 3 and alpha.dim() == 3, (tuple(H.shape), tuple(alpha.shape))
        B, L, d = H.shape
        assert alpha.shape == (B, L, self.K), f"alpha {tuple(alpha.shape)} vs H {tuple(H.shape)}, K={self.K}"
        assert d == self.d, f"H has d={d}, readout expects {self.d}"
        side = math.isqrt(L)
        assert side * side == L, f"{L} tokens do not form a square grid"
        with torch.autocast(device_type=H.device.type, enabled=False):  # fp32 even under fp16 autocast
            Hf, af = H.float(), alpha.float()
            pi = self.pooling_weights(af)                                    # (B,L,K)
            h = torch.einsum("blk,bld->bkd", pi, Hf)                         # (B,K,d)   concept-wise pooled vectors
            scores = (h * self.w).sum(-1) + self.b                           # (B,K)     s_k
            logit = scores.sum(-1)                                           # (B,)      z
            contrib = pi * torch.einsum("bld,kd->blk", Hf, self.w)           # (B,L,K)   c_tk = pi_tk (w_k . H_t)
            m = contrib.sum(-1)                                              # (B,L)
        return {
            "logit": logit, "scores": scores, "h": h, "pi": pi, "contrib": contrib,
            "map": m.view(B, side, side),                                    # (B,24,24), row-major like the scan
            "concept_maps": contrib.transpose(1, 2).reshape(B, self.K, side, side),   # (B,K,24,24)
        }

    def readout_stats(self):
        return {"w_absmean": self.w.abs().mean().item(), "b_absmean": self.b.abs().mean().item(),
                "w_norm_per_concept": self.w.norm(dim=-1).tolist()}


@torch.no_grad()
def decomposition_error(readout, out):
    """The M7 gate quantities (all max-abs over the batch). The logit comes from pool-then-score, the map from
    score-then-pool, so agreement is a real check, not a restatement.
      map_plus_bias_vs_logit : | sum(map) + sum_k b_k - logit |            (the gate)
      per_concept            : max_k | sum_t c_tk + b_k - s_k |            (same identity, concept by concept)"""
    z = out["map"].sum(dim=(1, 2)) + readout.b.sum()
    return {
        "map_plus_bias_vs_logit": (z - out["logit"]).abs().max().item(),
        "per_concept": (out["contrib"].sum(dim=1) + readout.b - out["scores"]).abs().max().item(),
        "map_absmax": out["map"].abs().max().item(),
        "logit_absmax": out["logit"].abs().max().item(),
    }


@torch.no_grad()
def readout_diagnostics(out, polarity=None):
    """Notebook / M9 logging: how are the 16 concepts used? polarity: optional 0/1 tuple (1 = fake-oriented prompt)."""
    s, pi = out["scores"].float(), out["pi"].float()
    ent = -(pi * pi.clamp_min(1e-12).log()).sum(1)                           # (B,K) entropy of each pooling distribution
    d = {"logit_mean": out["logit"].mean().item(), "logit_std": out["logit"].std().item() if s.shape[0] > 1 else 0.0,
         "score_absmean_per_concept": s.abs().mean(0).tolist(),
         "pooling_entropy_mean": ent.mean().item(), "pooling_entropy_max_possible": math.log(pi.shape[1])}
    if polarity is not None:
        p = torch.tensor(polarity, device=s.device, dtype=torch.bool)
        d["score_mean_fake_oriented"] = s[:, p].mean().item()
        d["score_mean_real_oriented"] = s[:, ~p].mean().item()
    return d