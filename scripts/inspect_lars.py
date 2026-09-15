"""Print the unique mask values and folder layout of a LaRS download, to confirm LARS_RAW."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
for p in sorted(root.rglob("*"))[:20]:
    print(p.relative_to(root))
masks = sorted(root.rglob("*.png"))[:50]
c = Counter()
for m in masks:
    c.update(np.unique(np.array(Image.open(m))).tolist())
print("mask values seen (value: files):", dict(c))
