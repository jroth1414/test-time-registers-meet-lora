"""Proxy maritime metrics: boundary F1 of the water region and pixel F1 of obstacles.

Boundary pixels within one pixel of a void (ignore_index) label are dropped from both the
prediction and target boundary sets, on both sides of the comparison: a void blob otherwise
draws a closed ring in the target boundary that no prediction can match, and would also count
a prediction that fills the void differently from its surroundings as a boundary error.
"""

from __future__ import annotations

import math

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


def boundary_f1(
    pred: Tensor, target: Tensor, class_id: int, tol: int = 10, ignore_index: int = 255
) -> float:
    valid = target != ignore_index
    near_ignore = _dilate(target == ignore_index, 1)
    pb = _boundary(pred == class_id) & valid & ~near_ignore
    tb = _boundary(target == class_id) & valid & ~near_ignore
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
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    if tp + fp + fn == 0:
        return math.nan
    return 2 * tp / (2 * tp + fp + fn)


def lars_extra_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    return {
        "water_edge_f1": boundary_f1(pred, target, class_id=1, tol=10),
        "obstacle_f1": pixel_f1(pred, target, class_id=0),
    }
