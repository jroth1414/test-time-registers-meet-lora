"""ADE20K SceneParsing (ADEChallengeData2016). Raw PNG: 0 = other/ignore, 1..150 = classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ADE20K_NUM_CLASSES = 150
# Homogeneous-background classes for H3 (0-indexed ADE150 ids): wall, sky, floor, ceiling,
# road, water, sea. Verify against objectInfo150.csv in the download and adjust if needed.
ADE20K_BACKGROUND_IDS = [0, 2, 3, 5, 6, 21, 26]

_SPLIT = {"train": "training", "val": "validation"}


def ade20k_label_fn(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.int64) - 1
    out[arr == 0] = 255
    return out


def ade20k_pairs(root: str | Path, split: str) -> tuple[list[Path], list[Path]]:
    base = Path(root) / "ADEChallengeData2016"
    sub = _SPLIT[split]
    imgs = sorted((base / "images" / sub).glob("*.jpg"))
    labs = sorted((base / "annotations" / sub).glob("*.png"))
    if not imgs:
        raise FileNotFoundError(f"no ADE20K images under {base / 'images' / sub}; see docs/DATA.md")
    stems_match = all(img.stem == lab.stem for img, lab in zip(imgs, labs, strict=True))
    if len(imgs) != len(labs) or not stems_match:
        raise ValueError("ADE20K images and annotations do not line up")
    return imgs, labs
