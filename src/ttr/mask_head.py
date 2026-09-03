"""MaskFormer-lite: one per-class query, transformer decoder, dot-product masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MaskHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden: int = 256,
        num_layers: int = 2,
        heads: int = 8,
    ):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.pixel = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.mem_proj = nn.Linear(in_dim, hidden)
        self.queries = nn.Embedding(num_classes, hidden)
        layer = nn.TransformerDecoderLayer(
            hidden,
            heads,
            dim_feedforward=hidden * 2,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )

    def forward(self, feat: Tensor, out_hw: tuple[int, int]) -> Tensor:
        b = feat.shape[0]
        pix = F.interpolate(self.pixel(feat), scale_factor=4, mode="bilinear", align_corners=False)
        mem = self.mem_proj(feat.flatten(2).transpose(1, 2))  # (B, hw, hidden)
        q = self.queries.weight.unsqueeze(0).expand(b, -1, -1)
        q = self.decoder(q, mem)  # (B, K, hidden)
        emb = self.mask_embed(q)
        logits = torch.einsum("bkc,bchw->bkhw", emb, pix)
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
