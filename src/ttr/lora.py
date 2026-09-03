"""Standard per-weight-matrix LoRA for timm ViT attention linears.

Each `LoRALinear` wraps one base `nn.Linear` and adds one independent (A, B) adapter pair
per targeted output slice ("group"): y = W x + b + (alpha / r) * sum_g scatter_g(B_g A_g x).
A group is a contiguous span of output features -- for example the q or v rows of a fused
qkv projection -- each with its own r-dimensional input subspace. Passing `out_slices=None`
yields a single group covering the whole output (plain LoRA on this linear). This is
standard LoRA applied independently to each targeted weight matrix (W_Q, W_K, W_V, W_O),
not a single adapter shared across matrices.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ttr.backbone import Backbone
from ttr.config import LoraCfg


class LoRALinear(nn.Module):
    """y = W x + b + (alpha/r) * sum_g scatter_g(B_g A_g x)

    Each group g is one contiguous output slice (e.g. the q rows of a fused qkv) with its own
    independent (A_g, B_g), which is standard LoRA applied per weight matrix. With
    out_slices=None there is a single group covering every output (plain LoRA on this linear).
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        out_slices: list[slice] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r < 1:
            raise ValueError("rank must be >= 1")
        self.base, self.r, self.alpha, self.scale = base, r, alpha, alpha / r
        if out_slices is None:
            out_slices = [slice(0, base.out_features)]
        widths = {s.stop - s.start for s in out_slices}
        if len(widths) != 1:
            raise ValueError("all output slices must have the same width")
        self.width = widths.pop()
        self.groups = len(out_slices)
        self.lora_A = nn.Parameter(torch.empty(self.groups, r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.groups, self.width, r))
        for g in range(self.groups):
            nn.init.kaiming_uniform_(self.lora_A[g], a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        idx = torch.cat([torch.arange(s.start, s.stop) for s in out_slices])
        self.register_buffer("out_index", idx.long())
        self.out_slices = [(s.start, s.stop) for s in out_slices]  # plain ints, for repr/state

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        h = torch.einsum("...i,gri->...gr", self.dropout(x), self.lora_A)
        d = torch.einsum("...gr,gwr->...gw", h, self.lora_B) * self.scale
        d = d.reshape(*d.shape[:-2], self.groups * self.width)
        return y.index_add(-1, self.out_index, d.to(y.dtype))

    def extra_repr(self) -> str:
        return (
            f"r={self.r}, alpha={self.alpha}, groups={self.groups}, "
            f"width={self.width}/{self.base.out_features}"
        )


_SLICES = {"q": 0, "k": 1, "v": 2}


def qkv_out_slices(embed_dim: int, targets: list[str]) -> list[slice] | None:
    """Output-row slices of the fused qkv projection for the requested q/k/v targets, in
    q, k, v order. Each requested target becomes its own group, i.e. independent LoRA on
    that weight matrix. Requesting all of q, k, v yields three groups (standard LoRA on
    each of W_Q, W_K, W_V), not a single merged group. None if none of q/k/v is requested."""
    want = [t for t in ("q", "k", "v") if t in targets]
    if not want:
        return None
    return [slice(_SLICES[t] * embed_dim, (_SLICES[t] + 1) * embed_dim) for t in want]


def _layers(bb: Backbone, layers) -> list[int]:
    if layers is None or layers == "all":
        return list(range(bb.depth))
    return [int(i) for i in layers]


def apply_lora(bb: Backbone, cfg: LoraCfg) -> list[str]:
    if not cfg.enabled:
        return []
    bad = set(cfg.targets) - {"q", "k", "v", "o"}
    if bad:
        raise ValueError(f"unknown LoRA targets {sorted(bad)}")
    layer_ids = _layers(bb, cfg.layers)
    # Validate every targeted block before wrapping any, so a conflict on a later layer
    # doesn't leave earlier layers partially wrapped.
    for i in layer_ids:
        attn = bb.model.blocks[i].attn
        if isinstance(attn.qkv, LoRALinear) or isinstance(attn.proj, LoRALinear):
            raise RuntimeError(f"blocks.{i}.attn already has LoRA")
    wrapped: list[str] = []
    for i in layer_ids:
        attn = bb.model.blocks[i].attn
        if any(t in cfg.targets for t in ("q", "k", "v")):
            slices = qkv_out_slices(bb.embed_dim, cfg.targets)
            attn.qkv = LoRALinear(attn.qkv, cfg.r, cfg.alpha, slices, cfg.dropout)
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


def lora_state_dict(bb: Backbone) -> dict[str, Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in bb.state_dict().items()
        if "lora_" in k or k.endswith(".out_index")
    }


def load_lora_state_dict(bb: Backbone, state: dict[str, Tensor]) -> None:
    model_sd = bb.state_dict()
    missing = [k for k in state if k not in model_sd]
    if missing:
        raise KeyError(f"LoRA keys not present in model (call apply_lora first): {missing[:3]}")
    bad = [k for k in state if "lora_" not in k and not k.endswith(".out_index")]
    if bad:
        raise ValueError(f"refusing to load non-LoRA keys: {bad[:3]}")
    mismatched = [
        k for k in state if k.endswith(".out_index") and not torch.equal(state[k], model_sd[k])
    ]
    if mismatched:
        raise ValueError(f"LoRA targets mismatch (out_index differs) for: {mismatched[:3]}")
    bb.load_state_dict(state, strict=False)
