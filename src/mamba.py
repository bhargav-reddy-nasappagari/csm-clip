"""M5: plain (unsteered) Mamba block in pure PyTorch, no mamba_ssm / CUDA kernels.

Selective state space model (Mamba-1), per channel d and state n:
    dt_t   = softplus(dt_proj(x_t))                      (B, L, D)      step size, > 0
    a_t    = dt_t * A                      A = -exp(A_log) < 0          log of the decay, <= 0
    h_t    = exp(a_t) * h_{t-1} + dt_t * B_t * x_t                      (B, D, N) state
    y_t    = sum_n C_t[n] * h_t[:, n] + Dskip * x_t

Decisions (docs/decision_log.md D008):
  * Scan order: row-major raster over the 24x24 grid, forward only (token 0 = top-left). The scan and the
    depthwise conv are both causal, which is what the M5 gate tests. 2-D-aware / bidirectional order is v2.
  * Block: pre-norm RMSNorm -> in_proj (x, z) -> causal depthwise conv(k=4) + SiLU -> selective scan -> * SiLU(z)
    -> out_proj, residual around the whole thing. d_model 384, expand 2 (d_inner 768), d_state 16, dt_rank 24.
  * Scan implementation: chunked parallel scan. Inside a chunk of Lc tokens, with S = cumsum(a):
        h_t = exp(S_t) * (h_carry + cumsum_s(exp(-S_s) * u_s)),   u_s = dt_s * B_s * x_s
    and the carry crosses chunks sequentially (576/16 = 36 iterations instead of 576). Always fp32, even under
    autocast. exp(-S) is bounded because a_t is clamped to >= -A_CLAMP per step and chunks are short
    (Lc * A_CLAMP <= 64 -> exp(64) ~ 6e27, inside fp32). The same clamp is applied in the independent reference.
  * M6 hook: `ssm_params(xc)` returns (dt_raw, B, C); steering gates will modify these three, nothing else.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

A_CLAMP = 4.0  # per-step decay floor: exp(-4) = 0.018, i.e. already ~forgotten; keeps exp(-cumsum) inside fp32


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def selective_scan(x, dt, A, B, C, D, chunk=16, checkpoint=False, h0=None):
    """x, dt: (b,L,D)  A: (D,N) negative  B, C: (b,L,N)  D: (D,)  ->  y (b,L,D) fp32, h_last (b,D,N).

    Equivalent to the sequential recurrence in src/mamba_ref.py (tested)."""
    assert chunk * A_CLAMP <= 64.0, "chunk too long for fp32-safe exp(-cumsum)"
    x, dt, A, B, C = x.float(), dt.float(), A.float(), B.float(), C.float()
    b, L, Dm = x.shape
    N = A.shape[1]
    h = x.new_zeros(b, Dm, N) if h0 is None else h0.float()

    def one_chunk(h, xc, dtc, Bc, Cc):
        a = (dtc.unsqueeze(-1) * A).clamp(min=-A_CLAMP)          # (b,Lc,D,N)
        S = a.cumsum(1)
        u = (dtc * xc).unsqueeze(-1) * Bc.unsqueeze(2)            # (b,Lc,D,N)
        hs = S.exp() * (h.unsqueeze(1) + (u * (-S).exp()).cumsum(1))
        y = (hs * Cc.unsqueeze(2)).sum(-1)                        # (b,Lc,D)
        return y, hs[:, -1]

    ys = []
    for s in range(0, L, chunk):
        sl = slice(s, min(s + chunk, L))
        args = (h, x[:, sl], dt[:, sl], B[:, sl], C[:, sl])
        if checkpoint and torch.is_grad_enabled():
            y, h = torch.utils.checkpoint.checkpoint(one_chunk, *args, use_reentrant=False)
        else:
            y, h = one_chunk(*args)
        ys.append(y)
    return torch.cat(ys, 1) + x * D, h


class MambaBlock(nn.Module):
    """Mixer only (no norm / residual): (b,L,d_model) -> (b,L,d_model)."""

    def __init__(self, d_model=384, d_state=16, d_conv=4, expand=2, dt_rank=None,
                 dt_min=1e-3, dt_max=0.1, chunk=16, checkpoint=False):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.chunk, self.checkpoint = chunk, checkpoint
        di = self.d_inner
        self.in_proj = nn.Linear(d_model, 2 * di, bias=False)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di, bias=True)
        self.out_proj = nn.Linear(di, d_model, bias=False)
        # dt init (Mamba): dt ~ log-uniform in [dt_min, dt_max]; bias = softplus^-1(dt); weight uniform +-rank^-0.5
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(di) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        # S4D-real init: A = -(1..N) per channel; D skip = 1
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))

    def ssm_params(self, xc):
        """xc: (b,L,d_inner) post-conv activation -> dt_raw (b,L,d_inner), B, C (b,L,N). M6 steers these three."""
        dt_low, B, C = self.x_proj(xc).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        return self.dt_proj(dt_low), B, C

    def forward(self, u):
        L = u.shape[1]
        x, z = self.in_proj(u).chunk(2, dim=-1)
        xc = F.silu(self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))  # causal: trim the right padding
        dt_raw, B, C = self.ssm_params(xc)
        dt = F.softplus(dt_raw.float())
        A = -torch.exp(self.A_log.float())
        y, _ = selective_scan(xc, dt, A, B, C, self.D, chunk=self.chunk, checkpoint=self.checkpoint)
        return self.out_proj((y.to(z.dtype)) * F.silu(z))


class MambaResidual(nn.Module):
    """x + MambaBlock(RMSNorm(x))."""

    def __init__(self, **kw):
        super().__init__()
        d = kw.get("d_model", 384)
        self.norm = RMSNorm(d)
        self.mixer = MambaBlock(**kw)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class PlainMamba(nn.Module):
    """Two stacked residual Mamba blocks (the spec's 'Mamba x2') with a final RMSNorm. (B,576,384) -> (B,576,384)."""

    def __init__(self, n_layers=2, **kw):
        super().__init__()
        self.layers = nn.ModuleList(MambaResidual(**kw) for _ in range(n_layers))
        self.norm_f = RMSNorm(kw.get("d_model", 384))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)