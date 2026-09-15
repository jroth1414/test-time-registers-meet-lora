"""Cityscapes fine annotations: 34 label ids -> 19 train ids (Cordts et al. 2016)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_BACKGROUND_IDS = [0, 3, 10]  # road, wall, sky (train ids)
_TRAIN_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
_LUT = np.full(256, 255, dtype=np.int64)
for _t, _raw in enumerate(_TRAIN_IDS):
    _LUT[_raw] = _t


def cityscapes_label_fn(arr: np.ndarray) -> np.ndarray:
    return _LUT[arr.astype(np.int64)]


def cityscapes_pairs(root: str | Path, split: str) -> tuple[list[Path], list[Path]]:
    root = Path(root)
    imgs = sorted((root / "leftImg8bit" / split).rglob("*_leftImg8bit.png"))
    if not imgs:
        raise FileNotFoundError(
            f"no Cityscapes images under {root / 'leftImg8bit' / split}; see docs/DATA.md"
        )
    labs = [
        root
        / "gtFine"
        / split
        / p.parent.name
        / p.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        for p in imgs
    ]
    missing = [lab for lab in labs if not lab.exists()]
    if missing:
        raise ValueError(f"missing gtFine labels, e.g. {missing[0]}")
    return imgs, labs
