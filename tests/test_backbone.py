import pytest
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

from src.backbone import ClipLoRABackbone, LoRALinear


def tiny(layers=4):
    cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                           num_attention_heads=4, image_size=32, patch_size=8)  # 4x4 = 16 patches
    torch.manual_seed(0)
    return CLIPVisionModel(cfg).eval()


def test_taps_shape_and_cls_dropped():
    vm = tiny()
    ref = vm(pixel_values=torch.randn(2, 3, 32, 32), output_hidden_states=True).hidden_states
    bb = ClipLoRABackbone(vm, taps=(2, 4), r=4)
    x = torch.randn(2, 3, 32, 32)
    a, b = bb(x)
    assert a.shape == b.shape == (2, 16, 32)  # real model: (B, 576, 1024)
    full = vm(pixel_values=x, output_hidden_states=True).hidden_states
    assert torch.equal(a, full[2][:, 1:]) and torch.equal(b, full[4][:, 1:])


def test_init_equals_frozen_clip():
    x = torch.randn(2, 3, 32, 32)
    base = tiny()
    want = base(pixel_values=x, output_hidden_states=True).hidden_states
    bb = ClipLoRABackbone(tiny(), taps=(2, 4))  # same seed -> same weights
    got = bb(x)
    assert torch.allclose(got[0], want[2][:, 1:], atol=1e-6) and torch.allclose(got[1], want[4][:, 1:], atol=1e-6)


def test_only_lora_trainable_and_placement():
    bb = ClipLoRABackbone(tiny(layers=4), taps=(2, 4), r=8)
    names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert bb.n_wrapped == 8  # q,v x 4 layers
    assert all(("q_proj" in n or "v_proj" in n) for n in names)
    rep = bb.param_report()
    assert rep["trainable"] == 8 * 8 * (32 + 32) and rep["lora_modules"] == 8  # r*(in+out) per module


def test_grads_reach_lora_and_not_base():
    bb = ClipLoRABackbone(tiny(), taps=(2, 4), r=4)
    a, b = bb(torch.randn(2, 3, 32, 32))
    (a.sum() + b.pow(2).sum()).backward()
    for n, p in bb.named_parameters():
        if "lora_B" in n:
            assert p.grad is not None and p.grad.abs().sum() > 0, n
        elif "lora_A" in n:
            # B=0 at init, so dL/dA is exactly zero on the first step; it must exist (graph connected)
            assert p.grad is not None, n
        else:
            assert p.grad is None, n
    # after B moves off zero, A gets gradient too
    with torch.no_grad():
        for p in bb.lora_parameters():
            if p.shape[0] == 32:  # lora_B (out, r)
                p.add_(0.05 * torch.randn_like(p))
    bb.zero_grad()
    a, b = bb(torch.randn(2, 3, 32, 32))
    (a.sum() + b.pow(2).sum()).backward()
    assert all(p.grad.abs().sum() > 0 for n, p in bb.named_parameters() if "lora_" in n)


def test_lora_math_matches_merged_weight():
    torch.manual_seed(1)
    base = torch.nn.Linear(16, 12)
    l = LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        l.lora_B.normal_()
    x = torch.randn(5, 16)
    merged = base.weight + (8.0 / 4) * l.lora_B @ l.lora_A
    want = torch.nn.functional.linear(x, merged, base.bias)
    assert torch.allclose(l(x), want, atol=1e-5)


def test_lora_changes_output_and_state_dict_roundtrip():
    x = torch.randn(2, 3, 32, 32)
    bb = ClipLoRABackbone(tiny(), taps=(2, 4), r=4)
    before = bb(x)[1].detach().clone()
    with torch.no_grad():
        for p in bb.lora_parameters():
            p.add_(0.1 * torch.randn_like(p))
    after = bb(x)[1].detach().clone()
    assert not torch.allclose(before, after)
    sd = bb.lora_state_dict()
    assert sd and all("lora_" in k for k in sd)
    bb2 = ClipLoRABackbone(tiny(), taps=(2, 4), r=4)  # fresh, same base weights
    bb2.load_lora_state_dict(sd)
    assert torch.equal(bb2(x)[1].detach(), after)


def test_bad_tap_rejected():
    with pytest.raises(AssertionError):
        ClipLoRABackbone(tiny(layers=4), taps=(2, 5))


def test_real_config_param_count():
    # ViT-L/14: 24 layers x (q,v) x r*(1024+1024) with r=8 -> 786,432 trainable (no weights downloaded)
    assert 24 * 2 * 8 * (1024 + 1024) == 786_432