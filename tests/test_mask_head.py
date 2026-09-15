import pytest
import torch
import torch.nn.functional as F

from ttr.config import HeadCfg
from ttr.heads import build_head
from ttr.mask_head import MaskHead


def test_mask_head_shapes():
    head = MaskHead(32, 5, hidden=16, num_layers=1, heads=2)
    out = head(torch.randn(2, 32, 4, 4), (56, 56))
    assert out.shape == (2, 5, 56, 56)


def test_mask_head_nonsquare_grid_shapes():
    head = MaskHead(32, 5, hidden=16, num_layers=1, heads=2)
    out = head(torch.randn(2, 32, 4, 6), (56, 84))
    assert out.shape == (2, 5, 56, 84)


def test_mask_head_rejects_hidden_not_divisible_by_heads():
    with pytest.raises(ValueError):
        MaskHead(8, 3, hidden=15, heads=2)


def test_mask_head_trains_on_toy():
    torch.manual_seed(0)
    head = MaskHead(8, 3, hidden=16, num_layers=1, heads=2)
    feat = torch.randn(4, 8, 2, 2)
    target = torch.tensor([0, 1, 2, 1]).view(4, 1, 1).expand(4, 8, 8).contiguous()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(head(feat, (8, 8)), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]


def test_mask_head_queries_and_pos_receive_gradients():
    torch.manual_seed(0)
    head = MaskHead(8, 3, hidden=16, num_layers=1, heads=2)
    feat = torch.randn(2, 8, 4, 4)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(head(feat, (8, 8)), target)
    loss.backward()
    assert head.queries.grad.abs().max() > 0
    assert head.pos.grad.abs().max() > 0


def test_mask_head_positional_embedding_breaks_permutation_invariance():
    torch.manual_seed(0)
    head = MaskHead(8, 3, hidden=16, num_layers=1, heads=2)
    b, h, w = 2, 4, 4
    feat = torch.randn(b, 8, h, w)
    tokens = feat.flatten(2).transpose(1, 2)
    pos_hw = F.interpolate(head.pos, size=(h, w), mode="bilinear", align_corners=False)
    pos_tokens = pos_hw.flatten(2).transpose(1, 2)
    q = head.queries.unsqueeze(0).expand(b, -1, -1)

    mem = head.mem_proj(tokens) + pos_tokens
    out1 = head.decoder(q, mem)

    perm = torch.randperm(tokens.shape[1])
    mem_perm = head.mem_proj(tokens[:, perm, :]) + pos_tokens
    out2 = head.decoder(q, mem_perm)

    assert not torch.allclose(out1, out2)


def test_build_head_dispatches_mask():
    h = build_head(HeadCfg(type="mask", hidden=16), 32, 10)
    assert isinstance(h, MaskHead)
