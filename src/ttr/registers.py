"""Outlier tokens, register neurons, and test-time registers (Jiang et al., 2025).

Pipeline:
  1. calibrate_outlier_threshold: robust norm threshold tau at one residual layer.
  2. find_register_neurons: MLP neurons whose activation concentrates on outlier tokens.
  3. install_test_time_registers: zero those neurons on patch tokens, write their max
     activation into the appended test-time register tokens.
Diagnostics: outlier_fraction (H1) and attention_entropy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

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


@dataclass
class RegisterNeurons:
    layer_to_neurons: dict[int, list[int]]
    stats: OutlierStats
    scores: dict[int, list[float]] | None = None


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


def score_register_neurons(acts: dict[int, Tensor], outlier: Tensor) -> dict[int, Tensor]:
    """Per-layer, per-neuron score: mean |activation| on outlier tokens minus on normal tokens,
    divided by the overall std so layers are comparable. acts[l] is (B, P, H) patch-only.
    Returns zeros for every layer when there are no outlier tokens.
    """
    out: dict[int, Tensor] = {}
    flat_mask = outlier.reshape(-1)
    for layer, a in acts.items():
        a2 = a.reshape(-1, a.shape[-1]).abs()
        if flat_mask.sum() == 0 or (~flat_mask).sum() == 0:
            out[layer] = torch.zeros(a.shape[-1])
            continue
        on = a2[flat_mask].mean(0)
        off = a2[~flat_mask].mean(0)
        std = a2.std(0) + 1e-6
        out[layer] = ((on - off) / std).cpu()
    return out


def select_register_neurons(
    scores: dict[int, Tensor], quantile: float = 0.999, max_neurons: int = 64
) -> dict[int, list[int]]:
    """Keep (layer, neuron) pairs with score >= global quantile, at most max_neurons, best first."""
    entries = [
        (float(s), layer, n) for layer, v in scores.items() for n, s in enumerate(v.tolist())
    ]
    if not entries:
        return {}
    all_scores = torch.tensor([e[0] for e in entries])
    thresh = torch.quantile(all_scores, quantile).item() if quantile > 0 else -math.inf
    kept = sorted([e for e in entries if e[0] >= thresh], key=lambda e: -e[0])[:max_neurons]
    sel: dict[int, list[int]] = {}
    for _, layer, n in kept:
        sel.setdefault(layer, []).append(n)
    return {layer: sorted(ns) for layer, ns in sorted(sel.items())}


@torch.no_grad()
def find_register_neurons(
    bb: Backbone,
    loader: Iterable,
    stats: OutlierStats,
    quantile: float = 0.999,
    max_neurons: int = 64,
    max_images: int = 64,
) -> RegisterNeurons:
    dev = next(bb.parameters()).device
    acts_by_layer: dict[int, list[Tensor]] = {layer: [] for layer in range(bb.depth)}
    masks: list[Tensor] = []
    seen = 0
    for batch in loader:
        x = _images(batch).to(dev)
        cap_act = bb.capture("mlp_act")
        cap_res = bb.capture("resid", layers=[stats.layer])
        try:
            bb.forward_tokens(x)
            for layer in range(bb.depth):
                acts_by_layer[layer].append(cap_act.data[layer][:, bb.patch_slice()].cpu())
            norms = cap_res.data[stats.layer][:, bb.patch_slice()].norm(dim=-1)
            masks.append((norms > stats.tau).cpu())
        finally:
            cap_act.remove()
            cap_res.remove()
        seen += x.shape[0]
        if seen >= max_images:
            break
    acts = {layer: torch.cat(v) for layer, v in acts_by_layer.items()}
    outlier = torch.cat(masks)
    scores = score_register_neurons(acts, outlier)
    # No outlier tokens means nothing to redirect: an empty map, not 64 arbitrary neurons
    # (a quantile over all-zero scores would otherwise keep everything).
    if int(outlier.sum()) == 0:
        sel: dict[int, list[int]] = {}
    else:
        sel = select_register_neurons(scores, quantile=quantile, max_neurons=max_neurons)
    return RegisterNeurons(
        layer_to_neurons=sel,
        stats=stats,
        scores={layer: s.tolist() for layer, s in scores.items()},
    )


def save_register_neurons(rn: RegisterNeurons, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer_to_neurons": {str(k): v for k, v in rn.layer_to_neurons.items()},
        "stats": asdict(rn.stats),
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_register_neurons(path: str | Path) -> RegisterNeurons:
    d = json.loads(Path(path).read_text())
    return RegisterNeurons(
        layer_to_neurons={int(k): list(v) for k, v in d["layer_to_neurons"].items()},
        stats=OutlierStats(**d["stats"]),
    )


def install_test_time_registers(bb: Backbone, rn: RegisterNeurons):
    """For each (layer, neuron): zero the neuron on patch tokens and write the max patch
    activation into a test-time register token (neurons round-robin over registers).
    Returns the hook handles; remove them to restore the plain model.
    """
    if bb.num_tt_reg < 1:
        raise RuntimeError("Backbone has no test-time registers; call set_tt_registers(n>=1) first")
    handles = []
    for layer, neurons in rn.layer_to_neurons.items():
        if not neurons:
            continue
        idx = torch.tensor(sorted(neurons))
        reg_of = torch.arange(len(idx)) % bb.num_tt_reg

        def hook(module, inp, out, idx=idx, reg_of=reg_of):
            out = out.clone()
            ps, rs = bb.patch_slice(), bb.tt_reg_slice()
            idx_d = idx.to(out.device)
            peak = out[:, ps][..., idx_d].max(dim=1).values  # (B, n)
            out[:, ps, idx_d] = 0.0
            reg_rows = rs.start + reg_of.to(out.device)  # (n,)
            out[:, reg_rows, idx_d] = peak
            return out

        handles.append(bb.add_mlp_hook(layer, hook))
    return handles
