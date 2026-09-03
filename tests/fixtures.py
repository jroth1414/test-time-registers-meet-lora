from pathlib import Path

import numpy as np
from PIL import Image


def write_fake_ade20k(root: Path, n: int = 3, size=(64, 96)) -> Path:
    """Creates root/ADEChallengeData2016/{images,annotations}/{training,validation}/..."""
    base = root / "ADEChallengeData2016"
    rng = np.random.default_rng(0)
    for split in ("training", "validation"):
        (base / "images" / split).mkdir(parents=True)
        (base / "annotations" / split).mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
            lab = rng.integers(0, 5, size, dtype=np.uint8)  # 0 = "other" in ADE raw labels
            Image.fromarray(img).save(base / "images" / split / f"ADE_{split}_{i:08d}.jpg")
            Image.fromarray(lab).save(base / "annotations" / split / f"ADE_{split}_{i:08d}.png")
    return root
