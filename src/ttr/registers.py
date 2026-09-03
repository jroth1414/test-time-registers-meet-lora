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
    if not (-bb.depth <= layer < bb.depth):
        raise ValueError(f"layer {layer} out of range for depth {bb.depth}")
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
    if not norms:
        raise ValueError("calibration loader yielded no batches")
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
        std = a2.std(0, unbiased=False) + 1e-6
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
    """Streaming version of score_register_neurons: accumulates per-layer sums instead of
    buffering every image's activations, so host memory stays O(hidden_width) per layer
    instead of O(images * tokens * hidden_width).
    """
    dev = next(bb.parameters()).device
    sum_on: dict[int, Tensor] = {}
    sum_off: dict[int, Tensor] = {}
    sum_all: dict[int, Tensor] = {}
    sumsq_all: dict[int, Tensor] = {}
    count_on = 0
    count_off = 0
    count_all = 0
    seen = 0
    for batch in loader:
        x = _images(batch).to(dev)
        cap_act = bb.capture("mlp_act")
        cap_res = bb.capture("resid", layers=[stats.layer])
        try:
            bb.forward_tokens(x)
            norms = cap_res.data[stats.layer][:, bb.patch_slice()].norm(dim=-1)
            mask = (norms > stats.tau).cpu()
            flat_mask = mask.reshape(-1)
            n_on = int(flat_mask.sum())
            n_off = int((~flat_mask).sum())
            count_on += n_on
            count_off += n_off
            count_all += flat_mask.numel()
            for layer in range(bb.depth):
                a = cap_act.data[layer][:, bb.patch_slice()].float().cpu()
                hidden = a.shape[-1]
                if layer not in sum_all:
                    sum_on[layer] = torch.zeros(hidden)
                    sum_off[layer] = torch.zeros(hidden)
                    sum_all[layer] = torch.zeros(hidden)
                    sumsq_all[layer] = torch.zeros(hidden)
                a2 = a.reshape(-1, hidden).abs()
                sum_all[layer] += a2.sum(0)
                sumsq_all[layer] += (a2 * a2).sum(0)
                if n_on:
                    sum_on[layer] += a2[flat_mask].sum(0)
                if n_off:
                    sum_off[layer] += a2[~flat_mask].sum(0)
        finally:
            cap_act.remove()
            cap_res.remove()
        seen += x.shape[0]
        if seen >= max_images:
            break

    # No outlier tokens, or no normal tokens (tau too low): nothing meaningful to redirect,
    # so return an empty map instead of a quantile over degenerate (all-zero, or undefined) scores.
    if count_on == 0 or count_off == 0:
        sel: dict[int, list[int]] = {}
        scores = {layer: torch.zeros_like(s) for layer, s in sum_all.items()}
    else:
        scores = {}
        for layer in sum_all:
            on = sum_on[layer] / count_on
            off = sum_off[layer] / count_off
            mean_all = sum_all[layer] / count_all
            var = (sumsq_all[layer] / count_all - mean_all**2).clamp_min(0)
            std = var.sqrt() + 1e-6
            scores[layer] = (on - off) / std
        sel = select_register_neurons(scores, quantile=quantile, max_neurons=max_neurons)

    return RegisterNeurons(
        layer_to_neurons=sel,
        stats=stats,
        scores={layer: s.tolist() for layer, s in scores.items()},
    )


def save_register_neurons(rn: RegisterNeurons, path: str | Path, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer_to_neurons": {str(k): v for k, v in rn.layer_to_neurons.items()},
        "stats": asdict(rn.stats),
    }
    if meta is not None:
        payload["meta"] = meta
    Path(path).write_text(json.dumps(payload, indent=2))


def load_register_neurons(path: str | Path) -> RegisterNeurons:
    d = json.loads(Path(path).read_text())
    return RegisterNeurons(
        layer_to_neurons={int(k): list(v) for k, v in d["layer_to_neurons"].items()},
        stats=OutlierStats(**d["stats"]),
    )


def install_test_time_registers(
    bb: Backbone, rn: RegisterNeurons
) -> list[torch.utils.hooks.RemovableHandle]:
    """For each (layer, neuron): zero the neuron on patch tokens and write the max patch
    activation into a test-time register token (neurons round-robin over registers).
    Returns the hook handles; remove them to restore the plain model.
    """
    if bb.num_tt_reg < 1:
        raise RuntimeError("Backbone has no test-time registers; call set_tt_registers(n>=1) first")
    for layer, neurons in rn.layer_to_neurons.items():
        if not (0 <= layer < bb.depth):
            raise ValueError(f"layer {layer} is out of range [0, {bb.depth})")
        hidden = bb.model.blocks[layer].mlp.fc1.out_features
        for n in neurons:
            if not (0 <= n < hidden):
                raise ValueError(f"neuron {n} in layer {layer} is out of range [0, {hidden})")
    handles = []
    for layer, neurons in rn.layer_to_neurons.items():
        if not neurons:
            continue
        idx = torch.tensor(sorted(neurons))
        reg_of = torch.arange(len(idx)) % bb.num_tt_reg

        def hook(module, inp, out, idx=idx, reg_of=reg_of):
            if bb.num_tt_reg < 1:
                raise RuntimeError(
                    "test-time registers were removed after install; "
                    "call remove() on the handles first"
                )
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


@torch.no_grad()
def attention_entropy(
    bb: Backbone, loader: Iterable, layers: list[int] | None = None, max_images: int = 64
) -> dict[str, float]:
    """Mean Shannon entropy (nats) of attention rows, averaged over heads, layers and images,
    reported separately for the cls query, the test-time register queries, and patch queries.
    Trained register rows, when the backbone has them, are deliberately excluded from all
    three groups: they are neither cls, test-time register, nor patch queries.
    Leaves fused attention disabled on the backbone.
    """
    bb.set_fused_attention(False)
    dev = next(bb.parameters()).device
    sums = {"cls": 0.0, "tt_reg": 0.0, "patch": 0.0}
    counts = {k: 0 for k in sums}
    seen = 0
    for batch in loader:
        x = _images(batch).to(dev)
        cap = bb.capture("attn", layers=layers)
        try:
            bb.forward_tokens(x)
            for a in cap.data.values():  # (B, heads, T, T)
                ent = -(a * (a + 1e-12).log()).sum(-1)  # (B, heads, T)
                ent = ent.mean(1)  # (B, T)
                groups = {
                    "cls": ent[:, : bb.num_cls],
                    "tt_reg": ent[:, bb.tt_reg_slice()],
                    "patch": ent[:, bb.patch_slice()],
                }
                for k, v in groups.items():
                    if v.numel():
                        sums[k] += float(v.sum())
                        counts[k] += v.numel()
        finally:
            cap.remove()
        seen += x.shape[0]
        if seen >= max_images:
            break
    return {k: (sums[k] / counts[k] if counts[k] else float("nan")) for k in sums}
