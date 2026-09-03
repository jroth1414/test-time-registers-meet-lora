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
        return [float(v) if u > 0 else math.nan for v, u in zip(iou, union, strict=True)]

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
def measure_throughput(
    fn: Callable[[Tensor], Tensor], x: Tensor, iters: int = 10, warmup: int = 2
) -> float:
    for _ in range(warmup):
        fn(x)
    with Timer() as t:
        for _ in range(iters):
            fn(x)
    return iters * x.shape[0] / max(t.seconds, 1e-9)
