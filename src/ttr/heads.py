"""Segmentation heads on top of Backbone patch grids."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ttr.config import HeadCfg


class LinearHead(nn.Module):
    """1x1 conv on the patch grid, bilinear upsample to label resolution."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.cls = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, feat: Tensor, out_hw: tuple[int, int]) -> Tensor:
        return F.interpolate(self.cls(feat), size=out_hw, mode="bilinear", align_corners=False)


def build_head(cfg: HeadCfg, in_dim: int, num_classes: int) -> nn.Module:
    if cfg.type == "linear":
        return LinearHead(in_dim, num_classes)
    if cfg.type == "mask":
        from ttr.mask_head import MaskHead  # chunk 8

        return MaskHead(in_dim, num_classes, hidden=cfg.hidden)
    raise ValueError(f"unknown head type {cfg.type!r}")
