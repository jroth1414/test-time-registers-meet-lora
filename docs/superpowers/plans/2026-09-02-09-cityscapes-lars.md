# Chunk 9: Cityscapes, LaRS, and Maritime Metrics

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Read `2026-09-02-00-master-plan.md` first.

**Goal:** The second and third datasets of the study plus the two LaRS-specific metrics the proposal names (water-edge accuracy, obstacle F1), and the background-class lists that drive H3.

**Architecture:** Both datasets reuse `SegFolderDataset` from chunk 5 with a pair-lister and a `label_fn`. LaRS metrics are boundary-F1 and pixel-F1 proxies computed from the same `(pred, target)` tensors the evaluator already has, so the runner gains one optional hook: `extra_metrics(pred, target)` looked up by dataset name.

**Tech Stack:** torch, numpy.

**Spec:** CLAUDE.md "Datasets": Cityscapes 19 classes, LaRS maritime panoptic with water-edge accuracy and obstacle F1; H3 needs "fraction of homogeneous background".

## Global Constraints (restated)

- Cityscapes and LaRS both need registration; loaders raise `FileNotFoundError` pointing to `docs/DATA.md`.
- LaRS official numbers for the paper come from the LaRS evaluation toolkit; ours are proxies for model selection and per-image H3 analysis. Say so in `DATA.md`.

## Interfaces

- Consumes (chunk 5): `SegFolderDataset`, `train_transform/eval_transform`, registry dicts in `ttr/data/__init__.py`. (chunk 7): `evaluate()`.
- Produces:

```python
# ttr/data/cityscapes.py
CITYSCAPES_NUM_CLASSES = 19; CITYSCAPES_BACKGROUND_IDS = [0, 3, 10]   # road, wall, sky
def cityscapes_pairs(root, split) -> tuple[list[Path], list[Path]]
def cityscapes_label_fn(arr) -> np.ndarray                                # 34 ids -> 19 trainIds, else 255

# ttr/data/lars.py
LARS_NUM_CLASSES = 3; LARS_BACKGROUND_IDS = [1, 2]                        # water, sky
LARS_RAW = {"obstacle": 0, "water": 1, "sky": 2}                          # verify against download
def lars_pairs(root, split); def lars_label_fn(arr)

# ttr/metrics_lars.py
def boundary_f1(pred, target, class_id, tol=10, ignore_index=255) -> float
def pixel_f1(pred, target, class_id, ignore_index=255) -> float
def lars_extra_metrics(pred, target) -> dict[str, float]      # {"water_edge_f1", "obstacle_f1"}

# ttr/data/__init__.py additions
def extra_metrics_fn(name: str) -> Callable | None
```

---

### Task 1: Cityscapes

**Files:**
- Create: `src/ttr/data/cityscapes.py`
- Modify: `src/ttr/data/__init__.py`
- Modify: `tests/fixtures.py` (add `write_fake_cityscapes`)
- Test: `tests/test_data_cityscapes.py`

- [ ] **Step 1: Fixture and failing tests**

```python
# tests/fixtures.py (append)
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
                Image.fromarray(img).save(root / "leftImg8bit" / split / city / f"{stem}_leftImg8bit.png")
                Image.fromarray(lab).save(root / "gtFine" / split / city / f"{stem}_gtFine_labelIds.png")
    return root
```

```python
# tests/test_data_cityscapes.py
from pathlib import Path

import numpy as np

from tests.fixtures import write_fake_cityscapes
from ttr.config import DataCfg
from ttr.data import background_class_ids, build_dataset, num_classes
from ttr.data.cityscapes import cityscapes_label_fn, cityscapes_pairs


def test_label_fn_maps_to_train_ids():
    raw = np.array([[0, 7, 8], [23, 26, 33]], dtype=np.uint8)
    out = cityscapes_label_fn(raw)
    assert out.tolist() == [[255, 0, 1], [10, 13, 18]]


def test_pairs_and_registry(tmp_path: Path):
    root = write_fake_cityscapes(tmp_path, n=2)
    imgs, labs = cityscapes_pairs(root, "train")
    assert len(imgs) == 2 and labs[0].name.endswith("_gtFine_labelIds.png")
    ds = build_dataset(DataCfg(name="cityscapes", root=str(root), img_size=64), "val")
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and set(y.unique().tolist()) <= {0, 1, 10, 13, 18, 255}
    assert num_classes("cityscapes") == 19 and background_class_ids("cityscapes") == [0, 3, 10]
```

- [ ] **Step 2: Implement `cityscapes.py`**

```python
# src/ttr/data/cityscapes.py
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
        raise FileNotFoundError(f"no Cityscapes images under {root / 'leftImg8bit' / split}; see docs/DATA.md")
    labs = [root / "gtFine" / split / p.parent.name / p.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            for p in imgs]
    missing = [l for l in labs if not l.exists()]
    if missing:
        raise ValueError(f"missing gtFine labels, e.g. {missing[0]}")
    return imgs, labs
```

Registry edits in `ttr/data/__init__.py`: import the three names, add `"cityscapes": CITYSCAPES_NUM_CLASSES` and `"cityscapes": CITYSCAPES_BACKGROUND_IDS`, and the branch

```python
    if cfg.name == "cityscapes":
        imgs, labs = cityscapes_pairs(cfg.root, split)
        return SegFolderDataset(imgs, labs, tf, cityscapes_label_fn)
```

- [ ] **Step 3: Run and commit**

Run: `pytest tests/test_data_cityscapes.py -v` (2 passed).

```powershell
git add src/ttr/data tests/fixtures.py tests/test_data_cityscapes.py
git commit -m "feat: Cityscapes dataset with 19-class train ids"
```

---

### Task 2: LaRS

**Files:**
- Create: `src/ttr/data/lars.py`
- Modify: `src/ttr/data/__init__.py`
- Create: `scripts/inspect_lars.py`
- Modify: `tests/fixtures.py`, `docs/DATA.md`
- Test: `tests/test_data_lars.py`

LaRS layout assumed (verify with `scripts/inspect_lars.py` after download and fix the constants):
`root/{train,val}/images/*.jpg` and `root/{train,val}/semantic_masks/*.png` with pixel values obstacle=0, water=1, sky=2, 255 = ignore/void.

- [ ] **Step 1: Fixture and failing tests**

```python
# tests/fixtures.py (append)
def write_fake_lars(root: Path, n: int = 2, size=(64, 96)) -> Path:
    rng = np.random.default_rng(2)
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "semantic_masks").mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
            lab = np.full(size, 1, dtype=np.uint8)          # water
            lab[: size[0] // 3] = 2                          # sky on top
            lab[40:50, 30:40] = 0                            # an obstacle
            Image.fromarray(img).save(root / split / "images" / f"lars_{i:04d}.jpg")
            Image.fromarray(lab).save(root / split / "semantic_masks" / f"lars_{i:04d}.png")
    return root
```

```python
# tests/test_data_lars.py
from pathlib import Path

import numpy as np

from tests.fixtures import write_fake_lars
from ttr.config import DataCfg
from ttr.data import background_class_ids, build_dataset, num_classes
from ttr.data.lars import lars_label_fn, lars_pairs


def test_label_fn_identity_with_ignore():
    raw = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    assert lars_label_fn(raw).tolist() == [[0, 1], [2, 255]]


def test_pairs_and_registry(tmp_path: Path):
    root = write_fake_lars(tmp_path)
    imgs, labs = lars_pairs(root, "val")
    assert len(imgs) == 2 and imgs[0].stem == labs[0].stem
    ds = build_dataset(DataCfg(name="lars", root=str(root), img_size=64), "val")
    _, y = ds[0]
    assert set(y.unique().tolist()) <= {0, 1, 2}
    assert num_classes("lars") == 3 and background_class_ids("lars") == [1, 2]
```

- [ ] **Step 2: Implement `lars.py`, registry, inspector**

```python
# src/ttr/data/lars.py
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
    missing = [l for l in labs if not l.exists()]
    if missing:
        raise ValueError(f"missing LaRS masks, e.g. {missing[0]}")
    return imgs, labs
```

```python
# scripts/inspect_lars.py
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
```

Registry: add `"lars": LARS_NUM_CLASSES`, `"lars": LARS_BACKGROUND_IDS`, and the `lars_pairs`/`lars_label_fn` branch.

- [ ] **Step 3: Update `docs/DATA.md`**

Append:

```markdown
## Cityscapes
Register at cityscapes-dataset.com, download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`,
extract to `data/cityscapes/` so that `data/cityscapes/leftImg8bit/train/<city>/*.png` exists.
19 train ids; everything else is 255. Background classes for H3: road, wall, sky.

## LaRS
Register at lojzezust.github.io/lars-dataset, download images and semantic masks, place as
`data/lars/{train,val}/{images,semantic_masks}/`. Run `python scripts/inspect_lars.py data/lars`
and confirm the mask values match `LARS_RAW` in `ttr/data/lars.py`; edit the dict if not.
Classes: obstacle 0, water 1, sky 2. Background classes for H3: water, sky.
Our water-edge F1 and obstacle F1 are proxies; report official numbers from the LaRS toolkit
in the paper.
```

- [ ] **Step 4: Run and commit**

Run: `pytest tests/test_data_lars.py -v` (2 passed).

```powershell
git add src/ttr/data tests scripts/inspect_lars.py docs/DATA.md
git commit -m "feat: LaRS dataset loader and inspector"
```

---

### Task 3: LaRS metrics and the evaluator hook

**Files:**
- Create: `src/ttr/metrics_lars.py`
- Modify: `src/ttr/data/__init__.py` (`extra_metrics_fn`)
- Modify: `src/ttr/run.py` (`evaluate` gains `extra_fn=None`; per-image CSV gains extra columns)
- Test: `tests/test_metrics_lars.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_metrics_lars.py
import math

import torch

from ttr.metrics_lars import boundary_f1, lars_extra_metrics, pixel_f1


def _scene(edge_row):
    t = torch.full((32, 32), 1)          # water
    t[:edge_row] = 2                      # sky above the edge
    return t


def test_boundary_f1_perfect_and_shifted():
    t = _scene(10)
    assert math.isclose(boundary_f1(t, t, class_id=1, tol=2), 1.0)
    assert boundary_f1(_scene(11), t, class_id=1, tol=2) == 1.0      # within tolerance
    assert boundary_f1(_scene(16), t, class_id=1, tol=2) == 0.0      # outside tolerance


def test_pixel_f1_obstacle():
    t = torch.full((8, 8), 1)
    t[2:4, 2:4] = 0
    p = t.clone()
    p[2:4, 2:6] = 0                   # 4 tp, 4 fp, 0 fn -> precision .5 recall 1 -> f1 .667
    assert math.isclose(pixel_f1(p, t, 0), 2 / 3)
    assert math.isnan(pixel_f1(torch.ones(4, 4), torch.ones(4, 4), 0))


def test_lars_extra_metrics_keys():
    t = _scene(10)
    out = lars_extra_metrics(t, t)
    assert set(out) == {"water_edge_f1", "obstacle_f1"}
```

- [ ] **Step 2: Implement `metrics_lars.py`**

```python
# src/ttr/metrics_lars.py
"""Proxy maritime metrics: boundary F1 of the water region and pixel F1 of obstacles."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _boundary(mask: Tensor) -> Tensor:
    m = mask.float()[None, None]
    eroded = -F.max_pool2d(-m, 3, stride=1, padding=1)
    return (m - eroded)[0, 0] > 0


def _dilate(mask: Tensor, tol: int) -> Tensor:
    if tol <= 0:
        return mask
    return F.max_pool2d(mask.float()[None, None], 2 * tol + 1, stride=1, padding=tol)[0, 0] > 0


def boundary_f1(pred: Tensor, target: Tensor, class_id: int, tol: int = 10, ignore_index: int = 255) -> float:
    valid = target != ignore_index
    pb = _boundary(pred == class_id) & valid
    tb = _boundary(target == class_id) & valid
    if tb.sum() == 0 and pb.sum() == 0:
        return math.nan
    precision = (pb & _dilate(tb, tol)).sum() / max(int(pb.sum()), 1)
    recall = (tb & _dilate(pb, tol)).sum() / max(int(tb.sum()), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def pixel_f1(pred: Tensor, target: Tensor, class_id: int, ignore_index: int = 255) -> float:
    valid = target != ignore_index
    p, t = (pred == class_id) & valid, (target == class_id) & valid
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    if tp + fp + fn == 0:
        return math.nan
    return 2 * tp / (2 * tp + fp + fn)


def lars_extra_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    return {"water_edge_f1": boundary_f1(pred, target, class_id=1, tol=10),
            "obstacle_f1": pixel_f1(pred, target, class_id=0)}
```

Registry addition:

```python
def extra_metrics_fn(name: str):
    if name == "lars":
        from ttr.metrics_lars import lars_extra_metrics
        return lars_extra_metrics
    return None
```

- [ ] **Step 3: Wire into `evaluate`**

In `ttr/run.py`, add parameter `extra_fn=None` to `evaluate`. Inside the per-image loop, compute `extras = extra_fn(p, t) if extra_fn else {}`, include them in the CSV row, and accumulate per-key running means (ignoring NaN) that are added to the returned dict as `extra_<key>`. In `run()`, pass `extra_fn=extra_metrics_fn(cfg.data.name)` to both `evaluate` calls and copy `extra_*` keys into `metrics`. Extend `tests/test_run.py`:

```python
def test_evaluate_extra_metrics_are_reported(tmp_results: Path):
    cfg = _cfg()
    dev = torch.device("cpu")
    bb, head, _ = build_model(cfg, dev, _calib_loader(), tmp_results)
    va = torch.utils.data.DataLoader(build_dataset(cfg.data, "val"), batch_size=8)
    out = evaluate(bb, head, va, 4, [0], dev, amp=False, extra_fn=lambda p, t: {"dummy": 1.0})
    assert out["extra_dummy"] == 1.0
```

- [ ] **Step 4: Run, generate configs, commit**

Run: `pytest -q` (all green). Then:

```powershell
python scripts/make_factorial.py --dataset cityscapes --root data/cityscapes
python scripts/make_factorial.py --dataset lars --root data/lars
git add src/ttr tests configs/cityscapes configs/lars
git commit -m "feat: LaRS proxy metrics and evaluator hook; Cityscapes and LaRS factorials"
```

Append the chunk line to `PROGRESS.md`. Cityscapes images are 2048x1024; at `img_size=224` the centre crop discards most of the frame, so for the paper set `data.img_size=448` for Cityscapes and LaRS in `make_factorial.py` (patch-14 backbones accept 448) and note the batch size drop to 8.
