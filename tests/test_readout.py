import pytest
import torch
import torch.nn.functional as F

from src.fusion import ConceptFrontEnd
from src.readout import ConceptReadout, decomposition_error, readout_diagnostics
from src.steering import SteeredMamba

D, K, SIDE = 24, 16, 6
L = SIDE * SIDE  # small stand-in grid; the real model is d 384, 24x24 = 576, K 16


def rand_alpha(b=2, n=L, seed=0, sharp=1.0):
    g = torch.Generator().manual_seed(seed)
    return F.softmax(sharp * torch.randn(b, n, K, generator=g), dim=-1)


def rand_H(b=2, n=L, d=D, seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(b, n, d, generator=g)


def make(d=D, bias=0.3, seed=0):
    torch.manual_seed(seed)
    r = ConceptReadout(d)
    with torch.no_grad():
        r.w.normal_(0, 0.5)
        r.b.normal_(0, bias)  # non-zero biases so the "+ biases" part of the gate is actually exercised
    return r


# ---- the M7 gate: map sum + biases == logit
@pytest.mark.parametrize("sharp", [0.0, 1.0, 5.0, 30.0])
def test_gate_map_sum_plus_biases_equals_logit_fp32(sharp):
    r = make()
    out = r(rand_H(), rand_alpha(sharp=sharp))
    err = decomposition_error(r, out)
    assert err["map_plus_bias_vs_logit"] < 1e-4, err
    assert err["per_concept"] < 1e-4, err


def test_gate_at_real_sizes():
    r = make(d=384)
    H, alpha = rand_H(b=2, n=576, d=384), rand_alpha(b=2, n=576, sharp=3.0)
    out = r(H, alpha)
    assert out["map"].shape == (2, 24, 24) and out["concept_maps"].shape == (2, 16, 24, 24)
    assert decomposition_error(r, out)["map_plus_bias_vs_logit"] < 1e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gate_with_half_precision_inputs_and_autocast(dtype):
    r = make()
    H, alpha = rand_H().to(dtype), rand_alpha().to(dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = r(H, alpha)
    assert out["logit"].dtype == torch.float32 and out["map"].dtype == torch.float32  # readout always fp32
    err = decomposition_error(r, out)
    assert err["map_plus_bias_vs_logit"] <= 1e-3 * max(1.0, err["logit_absmax"]), err  # fp16 tolerance, with room to spare
    assert err["map_plus_bias_vs_logit"] < 1e-4


def test_gate_is_not_a_tautology_matches_a_loop_oracle():
    """Independent plain-python reference for logit, scores and map (no einsum, no shared code path)."""
    r = make(seed=4)
    H, alpha = rand_H(b=1, seed=5), rand_alpha(b=1, seed=6, sharp=2.0)
    out = r(H, alpha)
    w, b = r.w.detach(), r.b.detach()
    scores, cells = [], torch.zeros(L)
    for k in range(K):
        denom = sum(float(alpha[0, t, k]) for t in range(L))
        h = torch.zeros(D)
        for t in range(L):
            h += (float(alpha[0, t, k]) / denom) * H[0, t]
        scores.append(float(w[k] @ h + b[k]))
        for t in range(L):
            cells[t] += (float(alpha[0, t, k]) / denom) * float(w[k] @ H[0, t])
    assert torch.allclose(out["scores"][0], torch.tensor(scores), atol=1e-4)
    assert torch.allclose(out["logit"][0], torch.tensor(sum(scores)), atol=1e-4)
    assert torch.allclose(out["map"][0].flatten(), cells, atol=1e-4)
    assert abs(float(cells.sum()) + float(b.sum()) - sum(scores)) < 1e-4


# ---- pooling
def test_pooling_weights_are_distributions_over_tokens():
    r = make()
    pi = r.pooling_weights(rand_alpha(sharp=4.0))
    assert torch.allclose(pi.sum(1), torch.ones(2, K), atol=1e-5) and (pi >= 0).all()


def test_uniform_alpha_pools_to_the_mean_for_every_concept():
    r = make()
    H = rand_H()
    out = r(H, torch.full((2, L, K), 1.0 / K))
    for k in range(K):
        assert torch.allclose(out["h"][:, k], H.mean(1), atol=1e-5)


def test_one_hot_alpha_pools_the_assigned_tokens():
    """Concept 3 owns tokens 0..9, concept 5 owns the rest: h_3 = mean of the first ten, h_5 = mean of the others."""
    r = make()
    H = rand_H()
    alpha = torch.zeros(2, L, K)
    alpha[:, :10, 3] = 1.0
    alpha[:, 10:, 5] = 1.0
    out = r(H, alpha)
    assert torch.allclose(out["h"][:, 3], H[:, :10].mean(1), atol=1e-5)
    assert torch.allclose(out["h"][:, 5], H[:, 10:].mean(1), atol=1e-5)
    assert decomposition_error(r, out)["map_plus_bias_vs_logit"] < 1e-4


def test_dead_concept_does_not_nan_and_keeps_the_identity():
    r = make()
    alpha = rand_alpha()
    alpha[..., 7] = 0.0  # concept 7 never fires
    out = r(rand_H(), alpha)
    assert all(torch.isfinite(out[k]).all() for k in ("logit", "scores", "h", "map", "contrib"))
    assert torch.allclose(out["scores"][:, 7], r.b[7].expand(2), atol=1e-6)  # only its bias remains
    assert decomposition_error(r, out)["map_plus_bias_vs_logit"] < 1e-4


# ---- the map
def test_map_is_row_major_like_the_scan():
    """Only token t responds -> only cell (t // SIDE, t % SIDE) is non-zero. Token 0 is top-left."""
    r = make(bias=0.0)
    alpha = torch.full((1, L, K), 1.0 / K)
    for t in (0, 5, 6, 7, L - 1):
        H = torch.zeros(1, L, D)
        H[0, t] = 1.0
        m = r(H, alpha)["map"][0]
        r_, c_ = divmod(t, SIDE)
        assert m[r_, c_].abs() > 1e-6
        m2 = m.clone()
        m2[r_, c_] = 0
        assert m2.abs().max() == 0, t


def test_map_cell_depends_only_on_its_own_state_at_fixed_alpha():
    r = make()
    H, alpha = rand_H(), rand_alpha()
    a = r(H, alpha)["map"]
    H2 = H.clone()
    H2[:, 11] += 3.0
    b = r(H2, alpha)["map"]
    diff = (a - b).abs().flatten(1)
    assert diff[:, 11].min() > 1e-3
    assert torch.cat([diff[:, :11], diff[:, 12:]], 1).max() < 1e-6


def test_concept_maps_sum_to_the_map_and_to_the_scores():
    r = make()
    out = r(rand_H(), rand_alpha(sharp=2.0))
    assert torch.allclose(out["concept_maps"].sum(1), out["map"], atol=1e-5)
    assert torch.allclose(out["concept_maps"].sum((2, 3)) + r.b, out["scores"], atol=1e-4)


def test_map_sign_follows_the_head_direction():
    """With uniform alpha and a single non-zero head w_0 = e_0: a token with H_t[0] > 0 pushes towards fake (map > 0)."""
    r = ConceptReadout(D)
    with torch.no_grad():
        r.w.zero_(); r.b.zero_(); r.w[0, 0] = 1.0
    H = torch.zeros(1, L, D)
    H[0, 2, 0], H[0, 9, 0] = 2.0, -2.0
    m = r(H, torch.full((1, L, K), 1.0 / K))["map"][0].flatten()
    assert m[2] > 0 and m[9] < 0 and m[2] == -m[9]
    assert torch.allclose(m.sum(), r(H, torch.full((1, L, K), 1.0 / K))["logit"][0])


# ---- parameters, init, gradients
def test_param_counts_and_init():
    r = ConceptReadout(384)
    assert sum(p.numel() for p in r.parameters()) == 16 * 384 + 16 == 6_160
    assert r.b.abs().sum() == 0 and r.w.abs().sum() > 0
    out = r(rand_H(d=384, n=576), rand_alpha(n=576))
    assert out["logit"].abs().max() < 1.0  # starts near 0 -> P(fake) near 0.5


def test_gradients_reach_heads_states_and_alpha():
    r = make()
    H = rand_H().requires_grad_(True)
    alpha = rand_alpha().requires_grad_(True)
    out = r(H, alpha)
    F.binary_cross_entropy_with_logits(out["logit"], torch.tensor([1.0, 0.0])).backward()
    for name, t in (("w", r.w), ("b", r.b), ("H", H), ("alpha", alpha)):
        assert t.grad is not None and t.grad.abs().sum() > 0, name
    assert torch.isfinite(alpha.grad).all()


def test_bias_gradient_is_exactly_the_logit_gradient():
    r = make()
    out = r(rand_H(), rand_alpha())
    out["logit"].sum().backward()
    assert torch.allclose(r.b.grad, torch.full((K,), 2.0))  # d(sum over 2 images of z)/d b_k = 2 each


# ---- shapes are enforced
def test_bad_shapes_rejected():
    r = make()
    with pytest.raises(AssertionError):
        r(rand_H(), rand_alpha(n=L - 1))
    with pytest.raises(AssertionError):
        r(rand_H(), torch.softmax(torch.randn(2, L, 8), -1))
    with pytest.raises(AssertionError):
        r(rand_H(d=D + 1), rand_alpha())
    with pytest.raises(AssertionError):  # 35 tokens: not a square grid
        r(rand_H(n=35), rand_alpha(n=35))


# ---- end to end on stand-in sizes: front end -> steered Mamba -> readout, gate holds, loss reaches everything
def test_pipeline_gate_and_gradients():
    torch.manual_seed(0)
    text = F.normalize(torch.randn(K, 16), dim=-1)
    fe = ConceptFrontEnd(text, d_in=32, d=D)
    mamba = SteeredMamba(2, d_model=D, d_state=4)
    head = ConceptReadout(D)
    with torch.no_grad():
        for p in mamba.gate_parameters():
            p.normal_(0, 0.1)  # gates live, so the front end can receive gradient through steering as well
    taps = [torch.randn(2, L, 32), 5 * torch.randn(2, L, 32)]
    o = fe(taps)
    out = head(mamba(o["X"], o["alpha"]), o["alpha"])
    assert out["map"].shape == (2, SIDE, SIDE)
    assert decomposition_error(head, out)["map_plus_bias_vs_logit"] < 1e-4
    F.binary_cross_entropy_with_logits(out["logit"], torch.tensor([1.0, 0.0])).backward()
    for name, p in list(fe.named_parameters()) + list(mamba.named_parameters()) + list(head.named_parameters()):
        if name == "affinity.text_embeds":
            continue
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_diagnostics_run():
    r = make()
    out = r(rand_H(), rand_alpha())
    d = readout_diagnostics(out, polarity=(1,) * 8 + (0,) * 8)
    assert len(d["score_absmean_per_concept"]) == K
    assert d["pooling_entropy_mean"] <= d["pooling_entropy_max_possible"] + 1e-6
    assert set(r.readout_stats()) == {"w_absmean", "b_absmean", "w_norm_per_concept"}