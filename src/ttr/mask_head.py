"""MaskFormer-lite: image-conditioned per-class dynamic kernels from a small transformer
decoder over positionally encoded patch tokens, fixed query-to-class assignment, per-pixel
cross-entropy. Omits Hungarian matching, per-query classification, dice/focal mask losses,
a multi-scale pixel decoder, masked attention, and deep supervision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_


class MaskHead(nn.Module):
    """MaskFormer-lite: image-conditioned per-class dynamic kernels from a small transformer
    decoder over positionally encoded patch tokens, fixed query-to-class assignment,
    per-pixel cross-entropy. Omits Hungarian matching, per-query classification, dice/focal
    mask losses, a multi-scale pixel decoder, masked attention, and deep supervision.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden: int = 256,
        num_layers: int = 2,
        heads: int = 8,
        upsample: int = 4,
    ):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.upsample = upsample
        self.pixel_in = nn.Sequential(nn.Conv2d(in_dim, hidden, 1), nn.GELU())
        self.pixel_out = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.mem_proj = nn.Linear(in_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, hidden, 32, 32))
        trunc_normal_(self.pos, std=0.02)
        self.queries = nn.Parameter(torch.randn(num_classes, hidden) * hidden**-0.5)
        layer = nn.TransformerDecoderLayer(
            hidden,
            heads,
            dim_feedforward=hidden * 2,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(hidden))
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=False),
        )

    def forward(self, feat: Tensor, out_hw: tuple[int, int]) -> Tensor:
        b = feat.shape[0]
        h, w = feat.shape[-2:]
        pix = self.pixel_out(
            F.interpolate(
                self.pixel_in(feat),
                scale_factor=self.upsample,
                mode="bilinear",
                align_corners=False,
            )
        )
        tokens = feat.flatten(2).transpose(1, 2)  # (B, hw, in_dim)
        pos_hw = F.interpolate(self.pos, size=(h, w), mode="bilinear", align_corners=False)
        mem = self.mem_proj(tokens) + pos_hw.flatten(2).transpose(1, 2)  # (B, hw, hidden)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        q = self.decoder(q, mem)  # (B, K, hidden)
        emb = self.mask_embed(q)
        logits = torch.einsum("bkc,bchw->bkhw", emb, pix)
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
