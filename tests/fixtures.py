from pathlib import Path

import numpy as np
from PIL import Image


def write_fake_ade20k(root: Path, n: int = 3, size=(64, 96)) -> Path:
    """Creates root/ADEChallengeData2016/{images,annotations}/{training,validation}/..."""
    base = root / "ADEChallengeData2016"
    rng = np.random.default_rng(0)
    for split in ("training", "validation"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "annotations" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
            lab = rng.integers(0, 5, size, dtype=np.uint8)  # 0 = "other" in ADE raw labels
            Image.fromarray(img).save(base / "images" / split / f"ADE_{split}_{i:08d}.jpg")
            Image.fromarray(lab).save(base / "annotations" / split / f"ADE_{split}_{i:08d}.png")
    return root


def write_fake_cityscapes(root: Path, n: int = 2, size=(64, 128)) -> Path:
    rng = np.random.default_rng(1)
    for split in ("train", "val"):
        for city in ("aachen",):
            (root / "leftImg8bit" / split / city).mkdir(parents=True)
            (root / "gtFine" / split / city).mkdir(parents=True)
            for i in range(n):
                img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
                lab = rng.choice([0, 7, 8, 23, 26, 33], size=size).astype(np.uint8)
                stem = f"{city}_{i:06d}_000019"
                Image.fromarray(img).save(
                    root / "leftImg8bit" / split / city / f"{stem}_leftImg8bit.png"
                )
                Image.fromarray(lab).save(
                    root / "gtFine" / split / city / f"{stem}_gtFine_labelIds.png"
                )
    return root


def write_fake_lars(root: Path, n: int = 2, size=(64, 96)) -> Path:
    rng = np.random.default_rng(2)
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "semantic_masks").mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
            lab = np.full(size, 1, dtype=np.uint8)  # water
            lab[: size[0] // 3] = 2  # sky on top
            lab[40:50, 30:40] = 0  # an obstacle
            Image.fromarray(img).save(root / split / "images" / f"lars_{i:04d}.jpg")
            Image.fromarray(lab).save(root / split / "semantic_masks" / f"lars_{i:04d}.png")
    return root
