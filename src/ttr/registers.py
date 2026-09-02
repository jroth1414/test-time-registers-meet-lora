"""Outlier tokens, register neurons, and test-time registers (Jiang et al., 2025).

Pipeline:
  1. calibrate_outlier_threshold: robust norm threshold tau at one residual layer.
  2. find_register_neurons: MLP neurons whose activation concentrates on outlier tokens.
  3. install_test_time_registers: zero those neurons on patch tokens, write their max
     activation into the appended test-time register tokens.
Diagnostics: outlier_fraction (H1) and attention_entropy.
"""

from __future__ import annotations

import json  # noqa: F401
import math  # noqa: F401
from collections.abc import Iterable
from dataclasses import asdict, dataclass  # noqa: F401
from pathlib import Path  # noqa: F401

import torch
from torch import Tensor

from ttr.backbone import Backbone


def _images(batch) -> Tensor:
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def _resolve_layer(bb: Backbone, layer: int) -> int:
    return layer % bb.depth


@dataclass
class OutlierStats:
    layer: int
    tau: float
    median: float
    mad: float
    k: float


@torch.no_grad()
def patch_norms(bb: Backbone, x: Tensor, layer: int = -1) -> Tensor:
    """L2 norm of each patch token in the residual stream after block `layer`. (B, P)"""
    layer = _resolve_layer(bb, layer)
    h = bb.capture("resid", layers=[layer])
    try:
        bb.forward_tokens(x)
        resid = h.data[layer]
    finally:
        h.remove()
    return resid[:, bb.patch_slice()].norm(dim=-1)


@torch.no_grad()
def calibrate_outlier_threshold(
    bb: Backbone, loader: Iterable, layer: int = -1, k: float = 4.0, max_images: int = 64
) -> OutlierStats:
    """tau = median + k * 1.4826 * MAD over patch-token norms of up to max_images images."""
    layer = _resolve_layer(bb, layer)
    norms, seen = [], 0
    dev = next(bb.parameters()).device
    for batch in loader:
        x = _images(batch).to(dev)
        norms.append(patch_norms(bb, x, layer).flatten().cpu())
        seen += x.shape[0]
        if seen >= max_images:
            break
    n = torch.cat(norms)
    med = n.median().item()
    mad = (n - med).abs().median().item()
    tau = med + k * 1.4826 * mad
    return OutlierStats(layer=layer, tau=tau, median=med, mad=mad, k=k)


@torch.no_grad()
def outlier_mask(bb: Backbone, x: Tensor, stats: OutlierStats) -> Tensor:
    return patch_norms(bb, x, stats.layer) > stats.tau


@torch.no_grad()
def outlier_fraction(
    bb: Backbone, loader: Iterable, stats: OutlierStats, max_images: int | None = None
) -> float:
    dev = next(bb.parameters()).device
    total, hits, seen = 0, 0, 0
    for batch in loader:
        x = _images(batch).to(dev)
        m = outlier_mask(bb, x, stats)
        hits += int(m.sum())
        total += m.numel()
        seen += x.shape[0]
        if max_images is not None and seen >= max_images:
            break
    return hits / max(total, 1)
