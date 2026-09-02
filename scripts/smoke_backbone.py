# scripts/smoke_backbone.py  (manual: downloads ~90 MB per model on first run)
"""Print token layout and feature shapes for the real checkpoints used in the study."""

import torch

from ttr.backbone import build_backbone, normalization_for
from ttr.config import BackboneCfg

NAMES = [
    "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch16_clip_224.openai",
]

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name in NAMES:
        bb = build_backbone(BackboneCfg(name=name, img_size=224)).to(dev)
        x = torch.randn(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            t = bb.forward_tokens(x)
            f = bb.forward_features(x)
        print(
            f"{name}: tokens {tuple(t.shape)} grid {tuple(f.shape[-2:])} "
            f"cls={bb.num_cls} trained_reg={bb.num_trained_reg} mean={normalization_for(bb)[0]}"
        )
