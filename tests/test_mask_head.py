import torch

from ttr.config import HeadCfg
from ttr.heads import build_head
from ttr.mask_head import MaskHead


def test_mask_head_shapes():
    head = MaskHead(32, 5, hidden=16, num_layers=1, heads=2)
    out = head(torch.randn(2, 32, 4, 4), (56, 56))
    assert out.shape == (2, 5, 56, 56)


def test_mask_head_trains_on_toy():
    torch.manual_seed(0)
    head = MaskHead(8, 3, hidden=16, num_layers=1, heads=2)
    feat = torch.randn(4, 8, 2, 2)
    target = torch.randint(0, 3, (4, 8, 8))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(head(feat, (8, 8)), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]


def test_build_head_dispatches_mask():
    h = build_head(HeadCfg(type="mask", hidden=16), 32, 10)
    assert isinstance(h, MaskHead)
