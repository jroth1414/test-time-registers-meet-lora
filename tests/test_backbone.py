import pytest
import torch
from torch import nn

import ttr.backbone as backbone_mod
from ttr.backbone import Backbone, build_backbone, normalization_for, tiny_backbone
from ttr.config import BackboneCfg


def test_tiny_forward_tokens_shape_and_layout():
    bb = tiny_backbone(depth=2, embed_dim=32, heads=2, img=56, patch=14, reg_tokens=0)
    x = torch.randn(2, 3, 56, 56)
    t = bb.forward_tokens(x)
    assert t.shape == (2, 1 + 16, 32)  # cls + 4x4 patches
    assert bb.num_cls == 1 and bb.num_trained_reg == 0 and bb.num_tt_reg == 0
    assert bb.prefix_len() == 1
    assert bb.patch_slice() == slice(1, None)
    assert bb.grid((56, 56)) == (4, 4)


def test_tiny_with_trained_registers_counts_them():
    bb = tiny_backbone(reg_tokens=4)
    t = bb.forward_tokens(torch.randn(1, 3, 56, 56))
    assert t.shape[1] == 1 + 4 + 16
    assert bb.num_trained_reg == 4
    assert bb.prefix_len() == 5


def test_forward_features_returns_patch_grid():
    bb = tiny_backbone()
    f = bb.forward_features(torch.randn(3, 3, 56, 56))
    assert f.shape == (3, 32, 4, 4)


def test_forward_tokens_matches_timm_forward_features_without_tt_registers():
    bb = tiny_backbone()
    x = torch.randn(1, 3, 56, 56)
    ours = bb.forward_tokens(x)
    theirs = bb.model.forward_features(x)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_tt_registers_are_inserted_between_prefix_and_patches():
    bb = tiny_backbone(reg_tokens=2)
    bb.set_tt_registers(3)
    t = bb.forward_tokens(torch.randn(1, 3, 56, 56))
    assert t.shape[1] == 1 + 2 + 3 + 16
    assert bb.tt_reg_slice() == slice(3, 6)
    assert bb.patch_slice() == slice(6, None)
    assert bb.num_patches((56, 56)) == 16


def test_tt_registers_change_patch_outputs_only_through_attention():
    # With zero-initialised registers and a random model, patch outputs differ from the
    # no-register run (attention now has extra keys). This guards against silently
    # dropping the tokens before the blocks.
    bb = tiny_backbone()
    x = torch.randn(1, 3, 56, 56)
    base = bb.forward_features(x)
    bb.set_tt_registers(1)
    with_reg = bb.forward_features(x)
    assert base.shape == with_reg.shape
    assert not torch.allclose(base, with_reg)


def test_set_tt_registers_rejects_negative():
    import pytest

    bb = tiny_backbone()
    with pytest.raises(ValueError):
        bb.set_tt_registers(-1)


def test_capture_resid_per_layer():
    bb = tiny_backbone(depth=3)
    h = bb.capture("resid")
    bb.forward_tokens(torch.randn(2, 3, 56, 56))
    assert sorted(h.data) == [0, 1, 2]
    assert h.data[1].shape == (2, 17, 32)
    h.remove()
    h.clear()
    bb.forward_tokens(torch.randn(1, 3, 56, 56))
    assert h.data == {}


def test_capture_mlp_act_has_hidden_width():
    bb = tiny_backbone(depth=2, embed_dim=32, mlp_ratio=2.0)
    h = bb.capture("mlp_act", layers=[1])
    bb.forward_tokens(torch.randn(1, 3, 56, 56))
    assert list(h.data) == [1]
    assert h.data[1].shape == (1, 17, 64)
    h.remove()


def test_capture_attn_probabilities_sum_to_one():
    bb = tiny_backbone(depth=2, heads=2)
    bb.set_fused_attention(False)
    h = bb.capture("attn", layers=[0])
    bb.forward_tokens(torch.randn(1, 3, 56, 56))
    a = h.data[0]
    assert a.shape == (1, 2, 17, 17)
    assert torch.allclose(a.sum(-1), torch.ones(1, 2, 17), atol=1e-5)
    h.remove()


def test_add_mlp_hook_can_modify_activations():
    bb = tiny_backbone(depth=2)
    seen = {}

    def zero_neuron_3(module, inp, out):
        out = out.clone()
        out[..., 3] = 0
        seen["ok"] = True
        return out

    hd = bb.add_mlp_hook(0, zero_neuron_3)
    cap = bb.capture("mlp_act", layers=[0])
    bb.forward_tokens(torch.randn(1, 3, 56, 56))
    assert seen["ok"]
    assert torch.all(cap.data[0][..., 3] == 0)
    hd.remove()
    cap.remove()


def test_build_backbone_tiny_applies_test_time_registers():
    cfg = BackboneCfg(
        name="tiny",
        registers="test_time",
        num_test_time_registers=2,
        pretrained=False,
    )
    bb = build_backbone(cfg)
    assert bb.num_tt_reg == 2
    assert bb.forward_tokens(torch.randn(1, 3, 56, 56)).shape[1] == 1 + 2 + 16


def test_build_backbone_rejects_trained_registers_on_model_without_them():
    import pytest

    with pytest.raises(ValueError):
        build_backbone(BackboneCfg(name="tiny", registers="trained", pretrained=False))


def test_build_backbone_enables_attention_capture_flag():
    bb = build_backbone(BackboneCfg(name="tiny", pretrained=False, capture_attention=True))
    assert all(not blk.attn.fused_attn for blk in bb.model.blocks)


def test_normalization_for_tiny_falls_back_to_imagenet():
    bb = tiny_backbone()
    mean, std = normalization_for(bb)
    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]


def test_forward_tokens_uses_checkpoint_seq_when_grad_checkpointing_enabled(monkeypatch):
    bb = tiny_backbone(depth=2, embed_dim=32, heads=2, img=56, patch=14)
    calls = {"n": 0}
    real_checkpoint_seq = backbone_mod.checkpoint_seq

    def wrapper(functions, x, *args, **kwargs):
        calls["n"] += 1
        return real_checkpoint_seq(functions, x, *args, **kwargs)

    monkeypatch.setattr(backbone_mod, "checkpoint_seq", wrapper)

    bb.model.set_grad_checkpointing(True)
    x = torch.randn(2, 3, 56, 56, requires_grad=True)
    with torch.enable_grad():
        out = bb.forward_tokens(x)
    assert calls["n"] == 1
    assert out.shape == (2, 1 + 16, 32)


def test_forward_tokens_does_not_use_checkpoint_seq_when_flag_off(monkeypatch):
    bb = tiny_backbone(depth=2, embed_dim=32, heads=2, img=56, patch=14)
    calls = {"n": 0}
    real_checkpoint_seq = backbone_mod.checkpoint_seq

    def wrapper(functions, x, *args, **kwargs):
        calls["n"] += 1
        return real_checkpoint_seq(functions, x, *args, **kwargs)

    monkeypatch.setattr(backbone_mod, "checkpoint_seq", wrapper)

    x = torch.randn(2, 3, 56, 56, requires_grad=True)
    with torch.enable_grad():
        bb.forward_tokens(x)
    assert calls["n"] == 0


def test_capture_rejects_unknown_target_before_installing_hooks():
    bb = tiny_backbone(depth=2)
    with pytest.raises(ValueError):
        bb.capture("bogus", layers=[])


def test_capture_attn_with_fused_attention_raises_before_installing_hooks():
    bb = tiny_backbone(depth=2, heads=2)
    assert all(blk.attn.fused_attn for blk in bb.model.blocks)
    with pytest.raises(RuntimeError):
        bb.capture("attn")
    for blk in bb.model.blocks:
        assert len(blk.attn.attn_drop._forward_hooks) == 0


def test_backbone_rejects_non_vision_transformer():
    with pytest.raises(TypeError):
        Backbone(nn.Linear(2, 2))


def test_build_backbone_rejects_unknown_registers_mode():
    with pytest.raises(ValueError):
        build_backbone(BackboneCfg(name="tiny", pretrained=False, registers="bogus"))
