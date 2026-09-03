import torch
from torch import nn

from ttr.backbone import tiny_backbone
from ttr.config import LoraCfg
from ttr.lora import LoRALinear, apply_lora, qkv_out_index


def test_lora_linear_is_identity_at_init():
    base = nn.Linear(8, 12)
    lin = LoRALinear(base, r=2, alpha=4.0, out_index=None)
    x = torch.randn(3, 8)
    assert torch.allclose(lin(x), base(x))
    assert lin.scale == 2.0


def test_lora_linear_only_touches_selected_outputs():
    base = nn.Linear(8, 12)
    idx = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])  # "q" and "v" slices if C=4
    lin = LoRALinear(base, r=2, alpha=2.0, out_index=idx)
    with torch.no_grad():
        lin.lora_B.fill_(1.0)
        lin.lora_A.fill_(0.5)
    x = torch.randn(5, 8)
    y, y0 = lin(x), base(x)
    diff = (y - y0).abs().sum(0)
    assert torch.all(diff[idx] > 0)
    untouched = torch.tensor([4, 5, 6, 7])
    assert torch.all(diff[untouched] == 0)


def test_lora_linear_gradients_flow_to_adapters_not_base():
    base = nn.Linear(8, 12)
    base.weight.requires_grad_(False)
    base.bias.requires_grad_(False)
    lin = LoRALinear(base, r=2, alpha=2.0, out_index=torch.tensor([0, 1]))
    lin(torch.randn(2, 8)).sum().backward()
    assert lin.lora_A.grad is not None
    assert lin.lora_B.grad is not None  # present; nonzero in general
    assert base.weight.grad is None


def test_qkv_out_index_selects_slices():
    idx = qkv_out_index(4, ["q", "v"])
    assert idx.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    assert qkv_out_index(4, ["q", "k", "v"]) is None  # all rows: no indexing needed
    assert qkv_out_index(4, ["o"]) is None  # nothing for qkv


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
    assert bb.model.blocks[1].attn.qkv.out_index is None
    assert isinstance(bb.model.blocks[1].attn.proj, LoRALinear)
    assert not isinstance(bb.model.blocks[0].attn.qkv, LoRALinear)


def test_apply_lora_is_noop_when_disabled_and_refuses_double_wrap():
    import pytest

    bb = tiny_backbone()
    assert apply_lora(bb, LoraCfg(enabled=False)) == []
    apply_lora(bb, LoraCfg(enabled=True))
    with pytest.raises(RuntimeError):
        apply_lora(bb, LoraCfg(enabled=True))


def test_apply_lora_preserves_forward_at_init():
    bb = tiny_backbone()
    x = torch.randn(1, 3, 56, 56)
    before = bb.forward_features(x)
    apply_lora(bb, LoraCfg(enabled=True, r=4))
    assert torch.allclose(before, bb.forward_features(x), atol=1e-6)
