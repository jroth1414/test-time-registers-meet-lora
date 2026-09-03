"""Deterministic toy segmentation data: axis-aligned rectangles, colour = class + noise."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

SYNTHETIC_NUM_CLASSES = 4
_PALETTE = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9], [0.8, 0.8, 0.1]])


class SyntheticSegDataset(Dataset):
    def __init__(
        self, n: int, img_size: int, num_classes: int = SYNTHETIC_NUM_CLASSES, seed: int = 0
    ):
        if num_classes > len(_PALETTE):
            raise ValueError("at most 4 synthetic classes")
        self.n, self.s, self.k, self.seed = n, img_size, num_classes, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 100_003 + i)
        s = self.s
        label = torch.zeros(s, s, dtype=torch.long)
        for _ in range(3):
            c = int(torch.randint(1, self.k, (1,), generator=g))
            y0, x0 = torch.randint(0, s // 2, (2,), generator=g).tolist()
            h, w = torch.randint(s // 4, s // 2, (2,), generator=g).tolist()
            label[y0 : y0 + h, x0 : x0 + w] = c
        img = _PALETTE[label].permute(2, 0, 1)  # (3, s, s)
        img = img + 0.05 * torch.randn(3, s, s, generator=g)
        img = (img - 0.5) / 0.5
        return img.float(), label
