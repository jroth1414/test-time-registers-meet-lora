"""ADE20K SceneParsing (ADEChallengeData2016). Raw PNG: 0 = other/ignore, 1..150 = classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ADE20K_NUM_CLASSES = 150
# Homogeneous-background classes for H3 (0-indexed ADE150 ids): wall, sky, floor, ceiling,
# road, water, sea. Verified against objectInfo150.txt (see docs/DATA.md).
ADE20K_BACKGROUND_IDS = [0, 2, 3, 5, 6, 21, 26]

_SPLIT = {"train": "training", "val": "validation"}


def ade20k_label_fn(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.int64) - 1
    out[arr == 0] = 255
    return out


def ade20k_pairs(root: str | Path, split: str) -> tuple[list[Path], list[Path]]:
    base = Path(root) / "ADEChallengeData2016"
    if split not in _SPLIT:
        raise ValueError(f"split must be one of {sorted(_SPLIT)}, got {split!r}")
    sub = _SPLIT[split]
    img_dir = base / "images" / sub
    ann_dir = base / "annotations" / sub
    imgs = sorted(img_dir.glob("*.jpg"))
    labs = sorted(ann_dir.glob("*.png"))
    if not imgs:
        raise FileNotFoundError(f"no ADE20K images under {img_dir}; see docs/DATA.md")
    if len(imgs) != len(labs):
        raise ValueError(
            f"ADE20K images and annotations do not line up: {len(imgs)} images under "
            f"{img_dir} vs {len(labs)} annotations under {ann_dir}; see docs/DATA.md"
        )
    for img, lab in zip(imgs, labs, strict=True):
        if img.stem != lab.stem:
            raise ValueError(
                f"ADE20K images and annotations do not line up: {img.name} is paired "
                f"with {lab.name}; see docs/DATA.md"
            )
    return imgs, labs
