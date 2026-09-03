import pytest
import torch
from torch import nn

from ttr.backbone import tiny_backbone
from ttr.config import LoraCfg
from ttr.lora import (
    LoRALinear,
    _layers,
    apply_lora,
    count_params,
    load_lora_state_dict,
    lora_state_dict,
    qkv_out_slices,
    set_trainable,
)


def test_lora_linear_is_identity_at_init():
    base = nn.Linear(8, 12)
    lin = LoRALinear(base, r=2, alpha=4.0, out_slices=None)
    x = torch.randn(3, 8)
    assert torch.allclose(lin(x), base(x))
    assert lin.scale == 2.0


def test_lora_linear_exposes_features():
    base = nn.Linear(8, 12)
    lin = LoRALinear(base, r=2, alpha=2.0, out_slices=None)
    assert lin.in_features == 8
    assert lin.out_features == 12


def test_lora_linear_only_touches_selected_outputs():
    base = nn.Linear(8, 12)
    lin = LoRALinear(base, r=2, alpha=2.0, out_slices=[slice(0, 4), slice(8, 12)])
    with torch.no_grad():
        lin.lora_B.fill_(1.0)
        lin.lora_A.fill_(0.5)
    x = torch.randn(5, 8)
    y, y0 = lin(x), base(x)
    diff = (y - y0).abs().sum(0)
    touched = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
    untouched = torch.tensor([4, 5, 6, 7])
    assert torch.all(diff[touched] > 0)
    assert torch.all(diff[untouched] == 0)


def test_lora_groups_are_independent():
    base = nn.Linear(8, 12)
    lin = LoRALinear(base, r=2, alpha=2.0, out_slices=[slice(0, 4), slice(4, 8)])
    with torch.no_grad():
        lin.lora_A.fill_(0.5)
        lin.lora_B[0].fill_(1.0)
        lin.lora_B[1].zero_()
    x = torch.randn(5, 8)
    y, y0 = lin(x), base(x)
    diff = (y - y0).abs().sum(0)
    assert torch.all(diff[0:4] > 0)
    assert torch.all(diff[4:8] == 0)


def test_lora_linear_rejects_unequal_slices():
    base = nn.Linear(8, 12)
    with pytest.raises(ValueError):
        LoRALinear(base, r=2, alpha=2.0, out_slices=[slice(0, 4), slice(4, 6)])


def test_lora_linear_gradients_flow_to_adapters_not_base():
    base = nn.Linear(8, 12)
    base.weight.requires_grad_(False)
    base.bias.requires_grad_(False)
    lin = LoRALinear(base, r=2, alpha=2.0, out_slices=[slice(0, 2)])
    lin(torch.randn(2, 8)).sum().backward()
    assert lin.lora_A.grad is not None
    assert lin.lora_B.grad is not None  # present; nonzero in general
    assert base.weight.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lora_linear_bf16_autocast():
    base = nn.Linear(8, 12).cuda()
    lin = LoRALinear(base, r=2, alpha=2.0, out_slices=None).cuda()
    x = torch.randn(3, 8, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = lin(x)
    assert y.dtype == torch.bfloat16
    y.sum().backward()
    assert lin.lora_A.grad is not None
    assert lin.lora_A.grad.dtype == torch.float32


def test_qkv_out_slices():
    slices = qkv_out_slices(4, ["q", "v"])
    assert slices == [slice(0, 4), slice(8, 12)]
    slices3 = qkv_out_slices(4, ["q", "k", "v"])
    assert slices3 == [slice(0, 4), slice(4, 8), slice(8, 12)]
    assert qkv_out_slices(4, ["o"]) is None


def test_layers_none_means_all():
    bb = tiny_backbone(depth=3)
    assert _layers(bb, None) == [0, 1, 2]


def test_apply_lora_wraps_qkv_on_all_layers_by_default():
    bb = tiny_backbone(depth=3)
    names = apply_lora(bb, LoraCfg(enabled=True, r=2, targets=["q", "v"]))
    assert names == ["blocks.0.attn.qkv", "blocks.1.attn.qkv", "blocks.2.attn.qkv"]
    for blk in bb.model.blocks:
        assert isinstance(blk.attn.qkv, LoRALinear)
        assert not isinstance(blk.attn.proj, LoRALinear)
    out = bb.forward_features(torch.randn(1, 3, 56, 56))
    assert out.shape == (1, 32, 4, 4)


def test_apply_lora_targets_o_and_layer_subset():
    bb = tiny_backbone(depth=3)
    names = apply_lora(bb, LoraCfg(enabled=True, r=2, targets=["q", "k", "v", "o"], layers=[1]))
    assert names == ["blocks.1.attn.qkv", "blocks.1.attn.proj"]
    assert bb.model.blocks[1].attn.qkv.groups == 3
    assert bb.model.blocks[1].attn.proj.groups == 1
    assert isinstance(bb.model.blocks[1].attn.proj, LoRALinear)
    assert not isinstance(bb.model.blocks[0].attn.qkv, LoRALinear)


def test_apply_lora_rejects_unknown_targets():
    bb = tiny_backbone()
    with pytest.raises(ValueError):
        apply_lora(bb, LoraCfg(enabled=True, targets=["q", "z"]))


def test_apply_lora_is_noop_when_disabled_and_refuses_double_wrap():
    bb = tiny_backbone()
    assert apply_lora(bb, LoraCfg(enabled=False)) == []
    apply_lora(bb, LoraCfg(enabled=True))
    with pytest.raises(RuntimeError):
        apply_lora(bb, LoraCfg(enabled=True))


def test_apply_lora_is_atomic_across_layers():
    bb = tiny_backbone(depth=3)
    apply_lora(bb, LoraCfg(enabled=True, r=2, layers=[1]))
    with pytest.raises(RuntimeError):
        apply_lora(bb, LoraCfg(enabled=True, r=2, layers=[0, 1, 2]))
    assert not isinstance(bb.model.blocks[0].attn.qkv, LoRALinear)
    assert not isinstance(bb.model.blocks[2].attn.qkv, LoRALinear)


def test_apply_lora_preserves_forward_at_init():
    bb = tiny_backbone()
    x = torch.randn(1, 3, 56, 56)
    before = bb.forward_features(x)
    apply_lora(bb, LoraCfg(enabled=True, r=4))
    assert torch.allclose(before, bb.forward_features(x), atol=1e-6)


def test_frozen_mode_freezes_everything():
    bb = tiny_backbone()
    set_trainable(bb, "frozen")
    tr, total = count_params(bb)
    assert tr == 0 and total > 0


def test_lora_mode_trains_only_adapters():
    bb = tiny_backbone(depth=2, embed_dim=32)
    apply_lora(bb, LoraCfg(enabled=True, r=2, targets=["q", "v"]))
    set_trainable(bb, "lora")
    tr, total = count_params(bb)
    # per layer: A 2 groups x (r=2 x in=32) = 128; B 2 groups x (out=32 x r=2) = 128
    # -> 256 per layer; two layers = 512
    assert tr == 512
    assert tr < total
    for n, p in bb.named_parameters():
        assert p.requires_grad == ("lora_" in n)


def test_full_mode_trains_everything():
    bb = tiny_backbone()
    set_trainable(bb, "full")
    tr, total = count_params(bb)
    assert tr == total


def test_lora_mode_without_adapters_is_an_error():
    bb = tiny_backbone()
    with pytest.raises(RuntimeError):
        set_trainable(bb, "lora")


def test_lora_state_dict_roundtrip_changes_outputs():
    bb1 = tiny_backbone(depth=2)
    bb2 = tiny_backbone(depth=2)
    bb2.load_state_dict(bb1.state_dict())  # same base weights, copied BEFORE LoRA renames keys
    apply_lora(bb1, LoraCfg(enabled=True, r=2))
    with torch.no_grad():
        for n, p in bb1.named_parameters():
            if "lora_B" in n:
                p.fill_(0.3)
    sd = lora_state_dict(bb1)
    assert all("lora_" in k or k.endswith(".out_index") for k in sd)
    # 2 layers x (lora_A + lora_B + out_index) for the qkv adapter = 6
    assert len(sd) == 6

    apply_lora(bb2, LoraCfg(enabled=True, r=2))
    x = torch.randn(1, 3, 56, 56)
    assert not torch.allclose(bb1.forward_features(x), bb2.forward_features(x))
    load_lora_state_dict(bb2, sd)
    assert torch.allclose(bb1.forward_features(x), bb2.forward_features(x))


def test_load_lora_state_dict_without_apply_lora_raises_key_error():
    bb1 = tiny_backbone(depth=2)
    apply_lora(bb1, LoraCfg(enabled=True, r=2))
    sd = lora_state_dict(bb1)

    bb2 = tiny_backbone(depth=2)  # no apply_lora called
    with pytest.raises(KeyError):
        load_lora_state_dict(bb2, sd)


def test_load_lora_state_dict_rejects_non_lora_keys():
    bb = tiny_backbone(depth=2)
    apply_lora(bb, LoraCfg(enabled=True, r=2))
    with pytest.raises(ValueError):
        load_lora_state_dict(bb, bb.state_dict())


def test_load_lora_state_dict_rejects_target_mismatch():
    bb1 = tiny_backbone(depth=2)
    apply_lora(bb1, LoraCfg(enabled=True, r=2, targets=["q", "v"]))
    sd = lora_state_dict(bb1)

    bb2 = tiny_backbone(depth=2)
    apply_lora(bb2, LoraCfg(enabled=True, r=2, targets=["q", "k"]))
    with pytest.raises(ValueError):
        load_lora_state_dict(bb2, sd)
