"""Backbone: a thin wrapper over timm VisionTransformer with an explicit token layout.

Layout of the token axis, always: [cls] [trained registers] [test-time registers] [patches].
"""

from __future__ import annotations

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn


class CaptureHandle:
    def __init__(self) -> None:
        self.data: dict[int, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.data.clear()


class Backbone(nn.Module):
    def __init__(self, model: VisionTransformer) -> None:
        super().__init__()
        if not isinstance(model, VisionTransformer):
            raise TypeError("Backbone only wraps timm VisionTransformer")
        self.model = model
        self.embed_dim = model.embed_dim
        self.depth = len(model.blocks)
        ps = model.patch_embed.patch_size
        self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        self.num_cls = 1 if model.cls_token is not None else 0
        self.num_trained_reg = int(getattr(model, "num_reg_tokens", 0) or 0)
        self.num_tt_reg = 0
        self.tt_reg_init = "zeros"

    # ---- layout -----------------------------------------------------------------
    def grid(self, img_hw: tuple[int, int]) -> tuple[int, int]:
        return img_hw[0] // self.patch_size, img_hw[1] // self.patch_size

    def num_patches(self, img_hw: tuple[int, int]) -> int:
        h, w = self.grid(img_hw)
        return h * w

    def prefix_len(self) -> int:
        return self.num_cls + self.num_trained_reg + self.num_tt_reg

    def patch_slice(self) -> slice:
        return slice(self.prefix_len(), None)

    def tt_reg_slice(self) -> slice:
        start = self.num_cls + self.num_trained_reg
        return slice(start, start + self.num_tt_reg)

    def set_tt_registers(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be >= 0")
        self.num_tt_reg = n

    # ---- forward ----------------------------------------------------------------
    def forward_tokens(self, x: Tensor) -> Tensor:
        m = self.model
        x = m.patch_embed(x)
        x = m._pos_embed(x)  # prepends cls + trained registers, adds pos embed
        if self.num_tt_reg > 0:
            b, _, c = x.shape
            reg = x.new_zeros(b, self.num_tt_reg, c)
            p = self.num_cls + self.num_trained_reg
            x = torch.cat([x[:, :p], reg, x[:, p:]], dim=1)
        x = m.patch_drop(x)
        x = m.norm_pre(x)
        for blk in m.blocks:
            x = blk(x)
        return m.norm(x)

    def forward_features(self, x: Tensor) -> Tensor:
        h, w = self.grid(tuple(x.shape[-2:]))
        t = self.forward_tokens(x)[:, self.patch_slice()]
        return t.transpose(1, 2).reshape(t.shape[0], self.embed_dim, h, w)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)


def tiny_backbone(
    depth: int = 2,
    embed_dim: int = 32,
    heads: int = 2,
    img: int = 56,
    patch: int = 14,
    reg_tokens: int = 0,
    mlp_ratio: float = 2.0,
) -> Backbone:
    """Random tiny ViT for CPU tests. Same code path as the real models."""
    m = VisionTransformer(
        img_size=img,
        patch_size=patch,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=heads,
        num_classes=0,
        reg_tokens=reg_tokens,
        mlp_ratio=mlp_ratio,
        dynamic_img_size=True,
    )
    m.eval()
    return Backbone(m)
