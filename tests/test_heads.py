import torch

from ttr.config import HeadCfg
from ttr.heads import LinearHead, build_head


def test_linear_head_upsamples_to_label_size():
    head = LinearHead(32, 5)
    feat = torch.randn(2, 32, 4, 4)
    out = head(feat, (56, 56))
    assert out.shape == (2, 5, 56, 56)


def test_linear_head_trains():
    head = LinearHead(8, 3)
    feat = torch.randn(4, 8, 2, 2)
    target = torch.randint(0, 3, (4, 8, 8))
    opt = torch.optim.SGD(head.parameters(), lr=0.5)
    losses = []
    for _ in range(20):
        loss = torch.nn.functional.cross_entropy(head(feat, (8, 8)), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_build_head_linear():
    h = build_head(HeadCfg(type="linear"), 32, 150)
    assert isinstance(h, LinearHead)


def test_build_head_unknown():
    import pytest

    with pytest.raises(ValueError):
        build_head(HeadCfg(type="nope"), 32, 150)
