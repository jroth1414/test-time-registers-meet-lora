import torch
from torch import nn

from ttr.lora import LoRALinear


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
    assert lin.lora_B.grad is not None  # zero-valued but present
    assert base.weight.grad is None
