import pytest
import torch
import torch.nn.functional as F

from src.fusion import ConceptFrontEnd
from src.mamba import MambaBlock, PlainMamba
from src.mamba_ref import scan_reference
from src.steering import ConceptGates, SteeredMamba, SteeredMambaBlock, gate_stats

DM, DS, L, K = 24, 4, 40, 16  # small stand-in sizes; real model is d_model 384, d_state 16, L 576, K 16
KW = dict(d_model=DM, d_state=DS)


def rand_alpha(b=2, n=L, seed=0, sharp=1.0):
    g = torch.Generator().manual_seed(seed)
    return F.softmax(sharp * torch.randn(b, n, K, generator=g), dim=-1)


def pair(seed=0, n_layers=2, **kw):
    """Same seed -> plain and steered stacks must start with identical mixer weights."""
    torch.manual_seed(seed); plain = PlainMamba(n_layers, **KW, **kw)
    torch.manual_seed(seed); steered = SteeredMamba(n_layers, **KW, **kw)
    return plain.eval(), steered.eval()


def nudge_gates(model, scale=0.5, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.gate_parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))


# ---- gate: with gates at init, steered output equals plain output exactly
def test_init_is_exactly_plain_fp32():
    plain, steered = pair()
    x, alpha = torch.randn(2, L, DM), rand_alpha()
    with torch.no_grad():
        assert torch.equal(plain(x), steered(x, alpha))


def test_init_exact_for_any_alpha_including_sharp():
    plain, steered = pair()
    x = torch.randn(2, L, DM)
    with torch.no_grad():
        want = plain(x)
        for sharp in (0.0, 1.0, 10.0):
            assert torch.equal(want, steered(x, rand_alpha(sharp=sharp)))


def test_init_exact_under_bf16_autocast_cpu():
    plain, steered = pair()
    x, alpha = torch.randn(2, L, DM), rand_alpha()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        a, b = plain(x), steered(x, alpha)
    assert torch.equal(a, b)


def test_init_exact_with_checkpoint_and_small_chunk():
    plain, steered = pair(chunk=8, checkpoint=True)
    x, alpha = torch.randn(2, L, DM), rand_alpha()
    assert torch.equal(plain(x), steered(x, alpha))


def test_same_seed_same_mixer_weights_and_from_plain():
    plain, steered = pair(seed=3)
    for (n1, p1), (n2, p2) in zip(plain.named_parameters(), [(n, p) for n, p in steered.named_parameters() if ".gates." not in n]):
        assert n1 == n2 and torch.equal(p1, p2), n1
    cp = SteeredMamba.from_plain(plain).eval()
    x, alpha = torch.randn(2, L, DM), rand_alpha()
    with torch.no_grad():
        assert torch.equal(plain(x), cp(x, alpha))
    assert all(torch.count_nonzero(p) == 0 for p in cp.gate_parameters())


def test_gate_param_counts():
    _, steered = pair()
    di = 2 * DM
    per_block = (di + 2 * DS) * K + (di + 2 * DS)
    assert sum(p.numel() for p in steered.gate_parameters()) == 2 * per_block
    assert len(steered.gate_parameters()) == 4  # (W, b) x 2 blocks
    real = (768 + 2 * 16) * 16 + (768 + 2 * 16)
    assert real == 13_600 and 2 * real == 27_200  # real model: +27,200 trainable over the plain stack
    assert not set(map(id, steered.gate_parameters())) & set(map(id, steered.mixer_parameters()))


# ---- gate: gates receive gradient
def test_gates_get_gradient_at_init_alpha_does_not_yet():
    _, steered = pair()
    steered.train()
    alpha = rand_alpha().requires_grad_(True)
    steered(torch.randn(2, L, DM), alpha).square().mean().backward()
    for n, p in steered.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, n  # gates AND every mixer param
    assert alpha.grad.abs().sum() == 0  # W = 0 at init, so dL/dalpha is exactly 0 on step 0


def test_gradient_reaches_front_end_once_gates_are_nonzero():
    torch.manual_seed(0)
    text = F.normalize(torch.randn(K, 24), dim=-1)
    fe = ConceptFrontEnd(text, d_in=32, d=DM)
    steered = SteeredMamba(2, **KW).train()
    nudge_gates(steered)
    taps = [torch.randn(2, L, 32), torch.randn(2, L, 32) * 5]
    out = fe(taps)
    # X is detached: the ONLY path from the front end to the loss is alpha -> gates
    steered(out["X"].detach(), out["alpha"]).square().mean().backward()
    assert fe.affinity.text_proj.weight.grad.abs().sum() > 0   # text projection
    assert fe.fusion.proj.weight.grad.abs().sum() > 0          # visual projection
    assert fe.affinity.log_scale.grad.abs() > 0                # temperature
    assert fe.affinity.text_embeds.grad is None


# ---- steering is real
def test_nonzero_gates_change_output_and_depend_on_alpha():
    plain, steered = pair()
    x, alpha = torch.randn(2, L, DM), rand_alpha()
    nudge_gates(steered)
    with torch.no_grad():
        a = steered(x, alpha)
        assert not torch.allclose(a, plain(x), atol=1e-3)
        assert not torch.allclose(a, steered(x, rand_alpha(seed=7)), atol=1e-3)  # same x, different alpha
        assert torch.isfinite(a).all()


@pytest.mark.parametrize("group", ["dt", "B", "C"])
def test_each_gate_group_is_live(group):
    plain, steered = pair(n_layers=1)
    di = 2 * DM
    sl = {"dt": slice(0, di), "B": slice(di, di + DS), "C": slice(di + DS, di + 2 * DS)}[group]
    with torch.no_grad():
        steered.layers[0].mixer.gates.b[sl] = 1.0   # open one group only: g = 1 + tanh(1)
        x, alpha = torch.randn(2, L, DM), rand_alpha()
        assert not torch.allclose(plain(x), steered(x, alpha), atol=1e-4)


# ---- the steered block matches the reference scan fed the gated parameters (independent oracle)
def test_block_matches_reference_with_gated_params():
    torch.manual_seed(1)
    blk = SteeredMambaBlock(d_model=DM, d_state=DS, chunk=8).eval()
    with torch.no_grad():
        blk.gates.W.normal_(0, 0.5); blk.gates.b.normal_(0, 0.5)
    u, alpha = torch.randn(2, L, DM), rand_alpha()
    with torch.no_grad():
        x, z = blk.in_proj(u).chunk(2, -1)
        xc = F.silu(blk.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))
        dt_raw, B, C = blk.ssm_params(xc)
        a = K * alpha - 1.0                                       # re-derive the gates by hand
        g = 1.0 + torch.tanh(a @ blk.gates.W.t() + blk.gates.b)
        g_dt, g_B, g_C = g[..., :2 * DM], g[..., 2 * DM:2 * DM + DS], g[..., 2 * DM + DS:]
        dt = F.softplus(dt_raw) * g_dt
        y = scan_reference(xc, dt, -torch.exp(blk.A_log), B * g_B, C * g_C, blk.D).float()
        want = blk.out_proj(y * F.silu(z))
        got = blk(u, alpha)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)


# ---- causality survives steering (perturb the token AND its alpha: nothing before t may move)
@pytest.mark.parametrize("t", [0, 1, 15, 16, 39])
def test_steered_stack_is_causal(t):
    _, steered = pair()
    nudge_gates(steered)
    x, alpha = torch.randn(2, L, DM), rand_alpha()
    x2, alpha2 = x.clone(), alpha.clone()
    x2[:, t] += 5.0
    alpha2[:, t] = rand_alpha(seed=9)[:, t]
    with torch.no_grad():
        a, b = steered(x, alpha), steered(x2, alpha2)
    assert (a[:, :t] - b[:, :t]).abs().max() < 1e-5 if t else True
    assert (a[:, t:] - b[:, t:]).abs().max() > 1e-3


# ---- numerics: bounded positive gates, finite under extreme gate values and fp16-like inputs
def test_gate_range_and_extremes_finite():
    gates = ConceptGates(48, DS)
    with torch.no_grad():
        gates.W.normal_(0, 100.0); gates.b.normal_(0, 100.0)
    for g in gates(rand_alpha(sharp=20.0)):
        assert g.dtype == torch.float32 and (g >= 0).all() and (g <= 2).all()
    _, steered = pair()
    steered.train()
    nudge_gates(steered, scale=50.0)
    out = steered(torch.randn(2, L, DM) * 10, rand_alpha(sharp=20.0))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in steered.parameters())


def test_gates_fp32_under_autocast_and_centering_flag():
    gates = ConceptGates(48, DS)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        g = gates(rand_alpha())
    assert all(t.dtype == torch.float32 for t in g)
    uniform = torch.full((1, 3, K), 1.0 / K)
    with torch.no_grad():
        gates.W.normal_()
        gates.b.zero_()
    assert all(torch.allclose(t, torch.ones_like(t)) for t in gates(uniform))  # uniform alpha -> centred input 0 -> g = 1
    raw = ConceptGates(48, DS, center=False)
    with torch.no_grad():
        raw.W.copy_(gates.W)
    assert not torch.allclose(raw(uniform)[0], torch.ones(1, 3, 48))


def test_alignment_and_concept_count_rejected():
    _, steered = pair()
    with pytest.raises(AssertionError):
        steered(torch.randn(2, L, DM), rand_alpha(n=L - 1))
    with pytest.raises(AssertionError):
        steered(torch.randn(2, L, DM), torch.softmax(torch.randn(2, L, 8), -1))


def test_gate_stats_zero_at_init_and_move_after_nudge():
    _, steered = pair()
    s0 = gate_stats(steered)
    assert len(s0) == 2 and all(v == 0 for d in s0 for k, v in d.items() if k != "layer")
    nudge_gates(steered)
    assert all(d["W_dt_absmean"] > 0 and d["b_C_absmean"] > 0 for d in gate_stats(steered))


def test_plain_files_untouched_by_steering_import():
    """M5 behaviour is unchanged: PlainMamba / MambaBlock are the original classes, no gate params."""
    assert not any("gates" in n for n, _ in PlainMamba(2, **KW).named_parameters())
    assert MambaBlock(**KW).forward.__qualname__ == "MambaBlock.forward"