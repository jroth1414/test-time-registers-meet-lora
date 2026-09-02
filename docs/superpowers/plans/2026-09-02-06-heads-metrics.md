# Chunk 6: Linear Head, Confusion-Matrix Metrics, Throughput

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Read `2026-09-02-00-master-plan.md` first.

**Goal:** The linear-probe segmentation head, an mIoU meter that ignores 255, per-image IoU for H3, and an images-per-second timer.

**Architecture:** `LinearHead` is a 1x1 conv on the patch grid followed by bilinear upsampling to label resolution. `ConfusionMeter` accumulates a KxK matrix with `torch.bincount`; per-image IoU reuses the same code on a single sample.

**Tech Stack:** torch.

**Spec:** proposal "linear probe (isolates feature quality)"; metrics mIoU and throughput.

## Interfaces

- Consumes (chunk 2): `Backbone.forward_features(x) -> (B, C, h, w)`, `Backbone.embed_dim`.
- Produces:

```python
# src/ttr/heads.py
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int)
    def forward(self, feat: Tensor, out_hw: tuple[int, int]) -> Tensor      # (B, K, H, W)
def build_head(cfg: HeadCfg, in_dim: int, num_classes: int) -> nn.Module    # chunk 8 adds "mask"

# src/ttr/metrics.py
class ConfusionMeter:
    def __init__(self, num_classes: int, ignore_index: int = 255)
    def update(self, pred: Tensor, target: Tensor) -> None                    # (B,H,W) long each
    def miou(self) -> float; def pixel_acc(self) -> float; def per_class_iou(self) -> list[float]
    def reset(self)
def image_miou(pred: Tensor, target: Tensor, num_classes: int, ignore_index=255) -> float   # one image
def background_fraction(target: Tensor, bg_ids: list[int], ignore_index=255) -> float
def measure_throughput(fn, x: Tensor, iters: int = 10, warmup: int = 2) -> float          # images/s
```

---

### Task 1: LinearHead and build_head

**Files:**
- Create: `src/ttr/heads.py`
- Test: `tests/test_heads.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_heads.py
import torch

from ttr.config import HeadCfg
from ttr.heads import LinearHead, build_head


def test_linear_head_upsamples_to_label_size():
    head = LinearHead(32, 5)
    feat = torch.randn(2, 32, 4, 4)
    out = head(feat, (56, 56))
    assert out.shape == (2, 5, 56, 56)


def test_linear_head_trains():
    head = LinearHead(8, 3)
    feat = torch.randn(4, 8, 2, 2)
    target = torch.randint(0, 3, (4, 8, 8))
    opt = torch.optim.SGD(head.parameters(), lr=0.5)
    losses = []
    for _ in range(20):
        loss = torch.nn.functional.cross_entropy(head(feat, (8, 8)), target)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_build_head_linear():
    h = build_head(HeadCfg(type="linear"), 32, 150)
    assert isinstance(h, LinearHead)


def test_build_head_unknown():
    import pytest

    with pytest.raises(ValueError):
        build_head(HeadCfg(type="nope"), 32, 150)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_heads.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# src/ttr/heads.py
"""Segmentation heads on top of Backbone patch grids."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ttr.config import HeadCfg


class LinearHead(nn.Module):
    """1x1 conv on the patch grid, bilinear upsample to label resolution."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.cls = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, feat: Tensor, out_hw: tuple[int, int]) -> Tensor:
        return F.interpolate(self.cls(feat), size=out_hw, mode="bilinear", align_corners=False)


def build_head(cfg: HeadCfg, in_dim: int, num_classes: int) -> nn.Module:
    if cfg.type == "linear":
        return LinearHead(in_dim, num_classes)
    if cfg.type == "mask":
        from ttr.mask_head import MaskHead  # chunk 8

        return MaskHead(in_dim, num_classes, hidden=cfg.hidden)
    raise ValueError(f"unknown head type {cfg.type!r}")
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_heads.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/ttr/heads.py tests/test_heads.py
git commit -m "feat: linear probe segmentation head"
```

---

### Task 2: ConfusionMeter, per-image IoU, background fraction

**Files:**
- Create: `src/ttr/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import math

import torch

from ttr.metrics import ConfusionMeter, background_fraction, image_miou


def test_confusion_meter_hand_example():
    m = ConfusionMeter(num_classes=3)
    target = torch.tensor([[[0, 0, 1, 1], [2, 2, 255, 255]]])
    pred = torch.tensor([[[0, 1, 1, 1], [2, 0, 0, 0]]])
    m.update(pred, target)
    # class0: tp=1, fp=1(pred0,tgt2), fn=1(pred1,tgt0) -> 1/3
    # class1: tp=2, fp=1, fn=0 -> 2/3 ; class2: tp=1, fp=0, fn=1 -> 1/2
    ious = m.per_class_iou()
    assert math.isclose(ious[0], 1 / 3) and math.isclose(ious[1], 2 / 3) and math.isclose(ious[2], 0.5)
    assert math.isclose(m.miou(), (1 / 3 + 2 / 3 + 0.5) / 3)
    assert math.isclose(m.pixel_acc(), 4 / 6)


def test_confusion_meter_skips_absent_classes_and_resets():
    m = ConfusionMeter(num_classes=4)
    m.update(torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]))
    assert m.miou() == 1.0
    assert math.isnan(m.per_class_iou()[3])
    m.reset()
    assert math.isnan(m.miou())


def test_image_miou_matches_meter():
    pred = torch.tensor([[0, 1], [1, 1]])
    target = torch.tensor([[0, 0], [1, 255]])
    assert math.isclose(image_miou(pred, target, 2), (0.5 + 0.5) / 2)


def test_background_fraction():
    target = torch.tensor([[0, 0, 5], [255, 2, 2]])
    assert math.isclose(background_fraction(target, [0, 2]), 4 / 5)
    assert math.isnan(background_fraction(torch.full((2, 2), 255), [0]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_metrics.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# src/ttr/metrics.py
"""Segmentation metrics from a confusion matrix, plus throughput."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from ttr.utils import Timer


class ConfusionMeter:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.k = num_classes
        self.ignore = ignore_index
        self.reset()

    def reset(self) -> None:
        self.mat = torch.zeros(self.k, self.k, dtype=torch.long)

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor) -> None:
        keep = target != self.ignore
        t = target[keep].long().cpu()
        p = pred[keep].long().cpu()
        idx = t * self.k + p
        self.mat += torch.bincount(idx, minlength=self.k * self.k).reshape(self.k, self.k)

    def per_class_iou(self) -> list[float]:
        tp = self.mat.diag().double()
        union = self.mat.sum(0).double() + self.mat.sum(1).double() - tp
        iou = tp / union
        return [float(v) if u > 0 else math.nan for v, u in zip(iou, union)]

    def miou(self) -> float:
        vals = [v for v in self.per_class_iou() if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else math.nan

    def pixel_acc(self) -> float:
        total = self.mat.sum().item()
        return self.mat.diag().sum().item() / total if total else math.nan


def image_miou(pred: Tensor, target: Tensor, num_classes: int, ignore_index: int = 255) -> float:
    m = ConfusionMeter(num_classes, ignore_index)
    m.update(pred.unsqueeze(0), target.unsqueeze(0))
    return m.miou()


def background_fraction(target: Tensor, bg_ids: list[int], ignore_index: int = 255) -> float:
    valid = target != ignore_index
    n = int(valid.sum())
    if n == 0:
        return math.nan
    bg = torch.zeros_like(target, dtype=torch.bool)
    for c in bg_ids:
        bg |= target == c
    return int((bg & valid).sum()) / n


@torch.no_grad()
def measure_throughput(fn: Callable[[Tensor], Tensor], x: Tensor, iters: int = 10, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn(x)
    with Timer() as t:
        for _ in range(iters):
            fn(x)
    return iters * x.shape[0] / max(t.seconds, 1e-9)
```

- [ ] **Step 4: Add a throughput test to `tests/test_metrics.py` and run**

```python
def test_measure_throughput_positive():
    from ttr.metrics import measure_throughput

    ips = measure_throughput(lambda x: x * 2, torch.zeros(4, 3), iters=3, warmup=1)
    assert ips > 0
```

Run: `pytest tests/test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/ttr/metrics.py tests/test_metrics.py
git commit -m "feat: confusion-matrix mIoU, per-image IoU, background fraction, throughput"
```

Append the chunk line to `docs/superpowers/plans/PROGRESS.md`.
