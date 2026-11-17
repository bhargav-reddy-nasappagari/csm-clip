"""Independent reference for the M5 gate: the plain sequential recurrence, float64, one token at a time.

Deliberately shares no code with src/mamba.py (no cumsum, no chunking, no exp(-S) trick). It is the oracle the
chunked scan is compared against, in the unit tests (CPU) and in the Colab notebook (GPU data, CPU reference).
"""
import torch


@torch.no_grad()
def scan_reference(x, dt, A, B, C, D, a_clamp=4.0):
    """x, dt: (b,L,Dm)  A: (Dm,N)  B, C: (b,L,N)  D: (Dm,)  ->  y (b,L,Dm) float64."""
    x, dt, A, B, C, D = (t.detach().double().cpu() for t in (x, dt, A, B, C, D))
    b, L, Dm = x.shape
    h = torch.zeros(b, Dm, A.shape[1], dtype=torch.float64)
    y = torch.zeros(b, L, Dm, dtype=torch.float64)
    for t in range(L):
        a = torch.clamp(dt[:, t, :, None] * A[None], min=-a_clamp)           # (b,Dm,N)
        h = torch.exp(a) * h + dt[:, t, :, None] * B[:, t, None, :] * x[:, t, :, None]
        y[:, t] = (h * C[:, t, None, :]).sum(-1) + D[None] * x[:, t]
    return y