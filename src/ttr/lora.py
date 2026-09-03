"""Minimal LoRA for timm ViT attention linears, with output-slice targeting.

y = W x + b + (alpha / r) * B A x, where B rows are scattered onto `out_index` output
features. With out_index = q and v rows of the fused qkv projection this is exactly
"LoRA on W_Q and W_V" while leaving W_K untouched.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ttr.backbone import Backbone
from ttr.config import LoraCfg


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


_SLICES = {"q": 0, "k": 1, "v": 2}


def qkv_out_index(embed_dim: int, targets: list[str]) -> Tensor | None:
    """Row indices of the fused qkv output for the requested q/k/v slices. None if all
    three are requested (plain LoRA) or none are."""
    want = [t for t in ("q", "k", "v") if t in targets]
    if not want or len(want) == 3:
        return None
    rows = [torch.arange(_SLICES[t] * embed_dim, (_SLICES[t] + 1) * embed_dim) for t in want]
    return torch.cat(rows)


def _layers(bb: Backbone, layers) -> list[int]:
    return list(range(bb.depth)) if layers == "all" else [int(i) for i in layers]


def apply_lora(bb: Backbone, cfg: LoraCfg) -> list[str]:
    if not cfg.enabled:
        return []
    bad = set(cfg.targets) - {"q", "k", "v", "o"}
    if bad:
        raise ValueError(f"unknown LoRA targets {sorted(bad)}")
    wrapped: list[str] = []
    for i in _layers(bb, cfg.layers):
        attn = bb.model.blocks[i].attn
        if isinstance(attn.qkv, LoRALinear) or isinstance(attn.proj, LoRALinear):
            raise RuntimeError(f"blocks.{i}.attn already has LoRA")
        if any(t in cfg.targets for t in ("q", "k", "v")):
            idx = qkv_out_index(bb.embed_dim, cfg.targets)
            attn.qkv = LoRALinear(attn.qkv, cfg.r, cfg.alpha, idx, cfg.dropout)
            wrapped.append(f"blocks.{i}.attn.qkv")
        if "o" in cfg.targets:
            attn.proj = LoRALinear(attn.proj, cfg.r, cfg.alpha, None, cfg.dropout)
            wrapped.append(f"blocks.{i}.attn.proj")
    return wrapped


def count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def set_trainable(bb: Backbone, mode: str) -> None:
    if mode == "frozen":
        for p in bb.parameters():
            p.requires_grad_(False)
    elif mode == "full":
        for p in bb.parameters():
            p.requires_grad_(True)
    elif mode == "lora":
        n_lora = 0
        for name, p in bb.named_parameters():
            is_lora = "lora_" in name
            p.requires_grad_(is_lora)
            n_lora += int(is_lora)
        if n_lora == 0:
            raise RuntimeError("mode='lora' but apply_lora was not called")
    else:
        raise ValueError(f"unknown train mode {mode!r}")
