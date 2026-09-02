import torch

from ttr.backbone import tiny_backbone


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
