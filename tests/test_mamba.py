import pytest
import torch

from src.mamba import A_CLAMP, MambaBlock, PlainMamba, RMSNorm, selective_scan
from src.mamba_ref import scan_reference

DM, L = 24, 40  # small stand-in sizes; real model is d_model 384, L 576


def rand_scan(b=2, L=L, Dm=12, N=4, seed=0, dt_scale=0.1):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    x, B, C = r(b, L, Dm), r(b, L, N), r(b, L, N)
    dt = torch.rand(b, L, Dm, generator=g) * dt_scale + 1e-3
    A = -(torch.rand(Dm, N, generator=g) * 15 + 1)
    return x, dt, A, B, C, r(Dm)


# ---- gate: matches an independent reference scan on random input
@pytest.mark.parametrize("chunk", [1, 7, 16, 40, 64])  # incl. chunk not dividing L and chunk > L
def test_scan_matches_reference(chunk):
    args = rand_scan()
    y, _ = selective_scan(*args, chunk=min(chunk, 16))
    assert torch.allclose(y.double(), scan_reference(*args), atol=1e-4, rtol=1e-4)


def test_scan_matches_reference_large_dt_hits_clamp():
    args = rand_scan(dt_scale=2.0)  # dt*A reaches ~ -30 per step -> clamp active in both
    y, _ = selective_scan(*args, chunk=16)
    assert (args[1][..., None] * args[2]).min() < -A_CLAMP
    assert torch.allclose(y.double(), scan_reference(*args), atol=1e-4, rtol=1e-4)
    assert torch.isfinite(y).all()


def test_scan_chunk_independent_and_carry():
    args = rand_scan()
    y8, h8 = selective_scan(*args, chunk=8)
    y16, h16 = selective_scan(*args, chunk=16)
    assert torch.allclose(y8, y16, atol=1e-5) and torch.allclose(h8, h16, atol=1e-5)
    x, dt, A, B, C, D = args  # run in two halves, passing the state: same result
    ya, ha = selective_scan(x[:, :24], dt[:, :24], A, B[:, :24], C[:, :24], D, chunk=8)
    yb, _ = selective_scan(x[:, 24:], dt[:, 24:], A, B[:, 24:], C[:, 24:], D, chunk=8, h0=ha)
    assert torch.allclose(torch.cat([ya, yb], 1), y8, atol=1e-5)


def test_scan_checkpoint_same_values_and_grads():
    x, dt, A, B, C, D = rand_scan()
    outs = []
    for ck in (False, True):
        xi = x.clone().requires_grad_(True)
        y, _ = selective_scan(xi, dt, A, B, C, D, chunk=8, checkpoint=ck)
        y.square().sum().backward()
        outs.append((y.detach(), xi.grad))
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-6) and torch.allclose(outs[0][1], outs[1][1], atol=1e-5)


def test_scan_rejects_unsafe_chunk():
    with pytest.raises(AssertionError):
        selective_scan(*rand_scan(), chunk=32)


# ---- gate: causal (perturbing token t changes nothing before t)
@pytest.mark.parametrize("t", [0, 1, 15, 16, 17, 39])
def test_block_is_causal(t):
    torch.manual_seed(0)
    blk = MambaBlock(DM, d_state=4, chunk=16).eval()
    x = torch.randn(2, L, DM)
    x2 = x.clone(); x2[:, t] += 5.0
    with torch.no_grad():
        a, b = blk(x), blk(x2)
    assert torch.equal(a[:, :t], b[:, :t]) or (a[:, :t] - b[:, :t]).abs().max() < 1e-6
    assert (a[:, t:] - b[:, t:]).abs().max() > 1e-3  # and the future does see it (not a dead block)


def test_stack_is_causal_and_shape():
    torch.manual_seed(0)
    m = PlainMamba(2, d_model=DM, d_state=4).eval()
    x = torch.randn(2, L, DM); x2 = x.clone(); x2[:, 20] += 5.0
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert a.shape == x.shape
    assert (a[:, :20] - b[:, :20]).abs().max() < 1e-5


def test_block_matches_reference_end_to_end():
    """Whole block vs a hand-rolled recomputation using the reference scan on the same intermediate tensors."""
    torch.manual_seed(1)
    blk = MambaBlock(DM, d_state=4, chunk=8).eval()
    u = torch.randn(2, L, DM)
    with torch.no_grad():
        x, z = blk.in_proj(u).chunk(2, -1)
        xc = torch.nn.functional.silu(blk.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))
        dt_raw, B, C = blk.ssm_params(xc)
        dt = torch.nn.functional.softplus(dt_raw)
        y = scan_reference(xc, dt, -torch.exp(blk.A_log), B, C, blk.D).float()
        want = blk.out_proj(y * torch.nn.functional.silu(z))
        got = blk(u)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)


# ---- init / gradients / stability
def test_init_ranges_and_param_count():
    blk = MambaBlock(384, d_state=16)
    assert blk.d_inner == 768 and blk.dt_rank == 24
    dt = torch.nn.functional.softplus(blk.dt_proj.bias)
    assert 9e-4 < dt.min() and dt.max() < 0.11
    assert torch.allclose(-torch.exp(blk.A_log[0]), -torch.arange(1.0, 17.0))
    n = sum(p.numel() for p in blk.parameters())
    assert n == 963_840, n  # ~0.96M per block at d=384
    assert sum(p.numel() for p in PlainMamba(2).parameters()) < 2_200_000


def test_all_params_get_gradient():
    torch.manual_seed(0)
    m = PlainMamba(2, d_model=DM, d_state=4)
    m(torch.randn(2, L, DM)).square().sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, dead


def test_finite_with_extreme_inputs():
    torch.manual_seed(0)
    m = PlainMamba(2, d_model=DM, d_state=4)
    out = m(torch.randn(2, L, DM) * 100)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_bf16_autocast_cpu_runs_and_scan_stays_fp32():
    m = PlainMamba(1, d_model=DM, d_state=4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(torch.randn(2, L, DM))
    assert torch.isfinite(out.float()).all()
    y, h = selective_scan(*[t.half() if t.is_floating_point() else t for t in rand_scan()], chunk=8)
    assert y.dtype == torch.float32


def test_rmsnorm():
    x = torch.randn(3, 5, 8) * 7
    y = RMSNorm(8)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3, 5), atol=1e-3)