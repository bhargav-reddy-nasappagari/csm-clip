import math

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPVisionConfig, CLIPVisionModel

from src import concepts as K
from src.backbone import ClipLoRABackbone
from src.fusion import ConceptAffinity, ConceptFrontEnd, LayerFusion, affinity_stats

B, N, D, DT, KC = 2, 576, 1024, 768, 16


def taps(b=B, n=N, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, n, d, generator=g) * 3 + 1, torch.randn(b, n, d, generator=g) * 50]


def text(seed=0, dt=DT):
    return torch.nn.functional.normalize(torch.randn(KC, dt, generator=torch.Generator().manual_seed(seed)), dim=-1)


# ---- concept bank
def test_concept_bank_shape_and_polarity():
    assert K.N_CONCEPTS == 16 and len(K.CONCEPTS) == len(set(K.CONCEPTS))
    assert sum(K.CONCEPT_POLARITY) == 8 and K.CONCEPT_POLARITY[:8] == (1,) * 8 and K.CONCEPT_POLARITY[8:] == (0,) * 8
    banned = ("midjourney", "biggan", "adm", "glide", "vqdm", "wukong", "stable diffusion", "dall")  # generator names
    assert not any(b in p.lower() for p in K.prompts() for b in banned)
    assert all(len(p.split()) < 40 for p in K.prompts())  # well inside CLIP's 77-token window


def test_encode_concepts_tiny_text_tower_and_npy_cache(tmp_path):
    cfg = CLIPTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         vocab_size=100, max_position_embeddings=77, projection_dim=24, eos_token_id=2)
    tm = CLIPTextModelWithProjection(cfg)

    def tok(texts, **kw):
        assert len(texts) == 16 and kw["max_length"] == 77
        ids = torch.randint(3, 100, (16, 77)); ids[:, 10] = 2
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    e = K.encode_concepts(tm, tok)
    assert e.shape == (16, 24) and e.dtype == torch.float32
    assert torch.allclose(e.norm(dim=-1), torch.ones(16), atol=1e-5)
    p = str(tmp_path / "assets" / "concepts.npy")
    a = K.load_or_encode(p, tm, tok)
    assert p.endswith(".npy") and np.load(p).shape == (16, 24)
    assert torch.equal(a, K.load_or_encode(p))  # second call reads the cache and needs no text tower


# ---- fusion
def test_fusion_shape_init_and_gradients():
    f = LayerFusion()
    assert torch.allclose(f.mix_weights(), torch.tensor([0.5, 0.5]))
    X = f(taps())
    assert X.shape == (B, N, 384) and torch.isfinite(X).all()
    X.square().mean().backward()
    assert f.mix_logits.grad.abs().sum() > 0 and f.proj.weight.grad.abs().sum() > 0
    assert all(n.weight.grad is not None for n in f.norms)


def test_fusion_insensitive_to_tap_scale():
    f = LayerFusion().eval()
    a, b = taps()
    assert torch.allclose(f([a, b]), f([a, b * 7.0]), atol=1e-4)


def test_fusion_rejects_wrong_tap_count():
    with pytest.raises(AssertionError):
        LayerFusion(n_taps=2)(taps()[:1])


# ---- affinity
def test_affinity_rows_sum_to_one_and_shapes():
    aff = ConceptAffinity(text())
    alpha, logits, C = aff(torch.randn(B, N, 384))
    assert alpha.shape == (B, N, KC) and logits.shape == (B, N, KC) and C.shape == (KC, 384)
    assert torch.allclose(alpha.sum(-1), torch.ones(B, N), atol=1e-5) and (alpha >= 0).all()
    assert aff.text_embeds.requires_grad is False


def test_affinity_scale_init_and_clamp():
    aff = ConceptAffinity(text())
    assert aff.log_scale.exp().item() == pytest.approx(1 / 0.07, rel=1e-5)
    with torch.no_grad():
        aff.log_scale.fill_(10.0)
    _, logits, _ = aff(torch.randn(1, 4, 384))
    assert logits.abs().max() <= 100.0 + 1e-3  # |cos| <= 1 and scale clamped to 100


def test_gradients_reach_both_projections_and_scale():
    fe = ConceptFrontEnd(text())
    out = fe(taps())
    (out["alpha"] * torch.arange(KC, dtype=torch.float32)).sum().backward()  # break the sum(alpha)=1 symmetry
    assert fe.fusion.proj.weight.grad.abs().sum() > 0          # visual projection
    assert fe.affinity.text_proj.weight.grad.abs().sum() > 0   # text projection
    assert fe.affinity.log_scale.grad.abs() > 0
    assert fe.affinity.text_embeds.grad is None                 # frozen CLIP text embeddings


def test_deterministic_and_fp32_under_autocast():
    torch.manual_seed(0); a = ConceptFrontEnd(text())(taps())
    torch.manual_seed(0); b = ConceptFrontEnd(text())(taps())
    assert torch.equal(a["alpha"], b["alpha"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        c = ConceptFrontEnd(text())(taps())
    assert c["alpha"].dtype == torch.float32 and torch.isfinite(c["alpha"]).all()


def test_affinity_stats_and_init_is_soft():
    s = affinity_stats(ConceptFrontEnd(text()).eval()(taps())["alpha"])
    assert s["row_sum_min"] == pytest.approx(1.0, abs=1e-5) and s["row_sum_max"] == pytest.approx(1.0, abs=1e-5)
    # known v1 property (logged in the decision log): at init alpha is near-uniform, i.e. a weak steering signal
    assert 0.8 * s["entropy_max_possible"] < s["entropy_mean"] <= s["entropy_max_possible"]
    one_hot = torch.zeros(1, 4, KC); one_hot[..., 3] = 1
    assert affinity_stats(one_hot)["entropy_mean"] < 1e-3 and affinity_stats(one_hot)["argmax_concepts_used"] == 1


# ---- end to end with the real M3 backbone class (tiny CLIP, no downloads)
def test_end_to_end_gradients_reach_lora_through_front_end():
    cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                           image_size=32, patch_size=8)  # 16 patches
    torch.manual_seed(0)
    bb = ClipLoRABackbone(CLIPVisionModel(cfg).eval(), taps=(2, 4), r=4)
    with torch.no_grad():
        for n, p in bb.named_parameters():
            if "lora_B" in n:
                p.add_(0.02 * torch.randn_like(p))  # B=0 at init would zero A's gradient
    fe = ConceptFrontEnd(text(dt=24), d_in=32, d=8)
    out = fe(bb(torch.randn(2, 3, 32, 32)))
    assert out["X"].shape == (2, 16, 8) and out["alpha"].shape == (2, 16, KC)
    assert torch.allclose(out["alpha"].sum(-1), torch.ones(2, 16), atol=1e-5)
    (out["alpha"] * torch.arange(KC, dtype=torch.float32)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in bb.lora_parameters())
    assert fe.fusion.proj.weight.grad.abs().sum() > 0 and fe.affinity.text_proj.weight.grad.abs().sum() > 0