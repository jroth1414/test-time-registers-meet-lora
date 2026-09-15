"""LaRS maritime obstacle segmentation (Zust, Pers, Kristan, ICCV 2023), semantic masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

LARS_NUM_CLASSES = 3
LARS_RAW = {"obstacle": 0, "water": 1, "sky": 2}  # verify with scripts/inspect_lars.py
LARS_BACKGROUND_IDS = [LARS_RAW["water"], LARS_RAW["sky"]]
_LUT = np.full(256, 255, dtype=np.int64)
for _name, _v in LARS_RAW.items():
    _LUT[_v] = _v


def lars_label_fn(arr: np.ndarray) -> np.ndarray:
    return _LUT[arr.astype(np.int64)]


def lars_pairs(root: str | Path, split: str) -> tuple[list[Path], list[Path]]:
    root = Path(root) / split
    imgs = sorted((root / "images").glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"no LaRS images under {root / 'images'}; see docs/DATA.md")
    labs = [root / "semantic_masks" / f"{p.stem}.png" for p in imgs]
    missing = [lab for lab in labs if not lab.exists()]
    if missing:
        raise ValueError(f"missing LaRS masks, e.g. {missing[0]}")
    return imgs, labs
