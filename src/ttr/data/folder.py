from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegFolderDataset(Dataset):
    """Images + label PNGs, joined by index. label_fn maps raw PNG values to class ids."""

    def __init__(
        self,
        image_paths: list[Path],
        label_paths: list[Path],
        transform: Callable,
        label_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        if len(image_paths) != len(label_paths):
            raise ValueError("image/label count mismatch")
        self.image_paths = list(image_paths)
        self.label_paths = list(label_paths)
        self.transform = transform
        self.label_fn = label_fn

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_paths[i]).convert("RGB")
        raw = np.array(Image.open(self.label_paths[i]))
        if raw.ndim == 3:
            raw = raw[..., 0]
        lab = self.label_fn(raw).astype(np.int64)
        return self.transform(img, lab)
