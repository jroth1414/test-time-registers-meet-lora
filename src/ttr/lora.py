"""Minimal LoRA for timm ViT attention linears, with output-slice targeting.

y = W x + b + (alpha / r) * B A x, where B rows are scattered onto `out_index` output
features. With out_index = q and v rows of the fused qkv projection this is exactly
"LoRA on W_Q and W_V" while leaving W_K untouched.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ttr.backbone import Backbone  # noqa: F401
from ttr.config import LoraCfg  # noqa: F401


class LoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, r: int, alpha: float, out_index: Tensor | None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if r < 1:
            raise ValueError("rank must be >= 1")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        n_out = base.out_features if out_index is None else int(out_index.numel())
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(n_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if out_index is None:
            self.register_buffer("out_index", None)
        else:
            self.register_buffer("out_index", out_index.long())

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scale
        if self.out_index is None:
            return y + delta
        y = y.clone()
        y[..., self.out_index] = y[..., self.out_index] + delta
        return y

    def extra_repr(self) -> str:
        sel = "all" if self.out_index is None else int(self.out_index.numel())
        return f"r={self.r}, alpha={self.alpha}, out={sel}/{self.base.out_features}"
