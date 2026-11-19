"""M6: concept-steered Mamba. Concept affinity alpha (B,L,16) modulates the step, write and read of the scan.

Per token t (alpha_t in R^K, K=16), per block:
    a_t   = K * alpha_t - 1                    centred + rescaled (alpha is a softmax over K concepts; at init it is
                                               near-uniform, so raw alpha is ~constant; K*alpha-1 is ~0 for uniform
                                               and O(1) when alpha is sharp)
    s_t   = W a_t + b                          W: (d_inner + 2N, K), b: (d_inner + 2N,), both ZERO at init
    g_t   = 1 + tanh(s_t)                      in (0, 2); exactly 1.0 when s_t = 0
    g^dt, g^B, g^C = split(g_t) into d_inner / N / N
    dt_t  = softplus(dt_raw_t) * g^dt_t        step   ("alpha sets step")
    B_t   = B_t * g^B_t                        write  ("alpha sets write")
    C_t   = C_t * g^C_t                        read   ("alpha sets read")

Everything else (conv, A, D, z-gate, projections, residual, norms) is identical to the plain block of src/mamba.py,
and src/mamba.py is not modified. With W = b = 0 every gate is exactly 1.0, so the steered stack is bit-for-bit the
plain stack (the M6 gate). The gates are positive and bounded, so dt stays > 0 and B, C never flip sign.

Gate parameters are created with torch.zeros (no RNG draw), so under the same seed SteeredMamba and PlainMamba get
identical mixer weights: the steered and plain arms of M9 differ only by the gates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mamba import MambaBlock, RMSNorm, selective_scan

N_CONCEPTS = 16


class ConceptGates(nn.Module):
    """alpha (B,L,K) -> (g_dt (B,L,d_inner), g_B (B,L,N), g_C (B,L,N)), all in (0, 2), fp32."""

    def __init__(self, d_inner, d_state, n_concepts=N_CONCEPTS, center=True):
        super().__init__()
        self.d_inner, self.d_state, self.K, self.center = d_inner, d_state, n_concepts, center
        n_out = d_inner + 2 * d_state
        self.W = nn.Parameter(torch.zeros(n_out, n_concepts))
        self.b = nn.Parameter(torch.zeros(n_out))

    def forward(self, alpha):
        assert alpha.shape[-1] == self.K, f"alpha has {alpha.shape[-1]} concepts, gates expect {self.K}"
        with torch.autocast(device_type=alpha.device.type, enabled=False):  # gates in fp32 even under autocast
            a = alpha.float()
            if self.center:
                a = self.K * a - 1.0
            g = 1.0 + torch.tanh(F.linear(a, self.W, self.b))
            return g.split([self.d_inner, self.d_state, self.d_state], dim=-1)


class SteeredMambaBlock(MambaBlock):
    """MambaBlock whose ssm parameters (dt, B, C) are gated by the concept affinity. forward(u, alpha)."""

    def __init__(self, n_concepts=N_CONCEPTS, center=True, **kw):
        super().__init__(**kw)
        self.gates = ConceptGates(self.d_inner, self.d_state, n_concepts, center)  # built after the mixer: no RNG shift

    def forward(self, u, alpha):
        L = u.shape[1]
        assert alpha.shape[:2] == u.shape[:2], f"alpha {tuple(alpha.shape)} does not align with tokens {tuple(u.shape)}"
        x, z = self.in_proj(u).chunk(2, dim=-1)
        xc = F.silu(self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))
        dt_raw, B, C = self.ssm_params(xc)
        g_dt, g_B, g_C = self.gates(alpha)
        dt = F.softplus(dt_raw.float()) * g_dt
        A = -torch.exp(self.A_log.float())
        y, _ = selective_scan(xc, dt, A, B.float() * g_B, C.float() * g_C, self.D,
                              chunk=self.chunk, checkpoint=self.checkpoint)
        return self.out_proj((y.to(z.dtype)) * F.silu(z))


class SteeredResidual(nn.Module):
    """x + SteeredMambaBlock(RMSNorm(x), alpha). Submodule names match MambaResidual (norm, mixer)."""

    def __init__(self, n_concepts=N_CONCEPTS, center=True, **kw):
        super().__init__()
        self.norm = RMSNorm(kw.get("d_model", 384))
        self.mixer = SteeredMambaBlock(n_concepts, center, **kw)

    def forward(self, x, alpha):
        return x + self.mixer(self.norm(x), alpha)


class SteeredMamba(nn.Module):
    """The spec's 'Mamba x2', concept-steered. (B,576,384) + alpha (B,576,16) -> (B,576,384).
    Every block receives the same alpha (it is token-aligned and computed once, from the fused tokens X)."""

    def __init__(self, n_layers=2, n_concepts=N_CONCEPTS, center=True, **kw):
        super().__init__()
        self.layers = nn.ModuleList(SteeredResidual(n_concepts, center, **kw) for _ in range(n_layers))
        self.norm_f = RMSNorm(kw.get("d_model", 384))

    def forward(self, x, alpha):
        for layer in self.layers:
            x = layer(x, alpha)
        return self.norm_f(x)

    # ---- parameter bookkeeping (M8 uses gate_parameters for its own learning-rate group)
    def gate_parameters(self):
        return [p for n, p in self.named_parameters() if ".gates." in n]

    def mixer_parameters(self):
        return [p for n, p in self.named_parameters() if ".gates." not in n]

    @classmethod
    def from_plain(cls, plain, **kw):
        """Steered stack with the weights of a PlainMamba copied in (gates stay at identity)."""
        first = plain.layers[0].mixer
        m = cls(n_layers=len(plain.layers), d_model=first.d_model, d_state=first.d_state, d_conv=first.d_conv,
                expand=first.d_inner // first.d_model, dt_rank=first.dt_rank, chunk=first.chunk,
                checkpoint=first.checkpoint, **kw)
        res = m.load_state_dict(plain.state_dict(), strict=False)
        assert not res.unexpected_keys and all(".gates." in k for k in res.missing_keys), res
        return m


@torch.no_grad()
def gate_stats(model):
    """Diagnostics: how far the gates have moved from identity, per block. Used in the notebook and by M8 logging."""
    out = []
    for i, layer in enumerate(model.layers):
        g = layer.mixer.gates
        di, N = g.d_inner, g.d_state
        w = g.W.float()
        parts = {"dt": slice(0, di), "B": slice(di, di + N), "C": slice(di + N, di + 2 * N)}
        out.append({"layer": i, **{f"W_{k}_absmean": w[sl].abs().mean().item() for k, sl in parts.items()},
                    **{f"b_{k}_absmean": g.b[sl].abs().mean().item() for k, sl in parts.items()}})
    return out