"""One factorial cell: build, train, evaluate, diagnose, write results."""

from __future__ import annotations

import argparse  # noqa: F401
import math  # noqa: F401
import sys  # noqa: F401
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: F401
from torch import nn  # noqa: F401
from torch.utils.data import DataLoader  # noqa: F401

from ttr.backbone import Backbone, build_backbone, normalization_for  # noqa: F401
from ttr.config import RunCfg, load_config, save_config  # noqa: F401
from ttr.data import background_class_ids, build_dataset, num_classes  # noqa: F401
from ttr.heads import build_head
from ttr.lora import apply_lora, count_params, lora_state_dict, set_trainable  # noqa: F401
from ttr.metrics import (  # noqa: F401
    ConfusionMeter,
    background_fraction,
    image_miou,
    measure_throughput,
)
from ttr.registers import (
    attention_entropy,  # noqa: F401
    calibrate_outlier_threshold,
    find_register_neurons,
    install_test_time_registers,
    load_register_neurons,
    outlier_fraction,  # noqa: F401
    save_register_neurons,
)
from ttr.utils import (  # noqa: F401
    Timer,
    append_csv_row,
    get_device,
    make_run_dir,
    seed_everything,
    write_json,
)


def build_model(cfg: RunCfg, device: torch.device, calib_loader, run_dir: Path):
    """Backbone -> outlier calibration (frozen) -> registers -> LoRA -> freeze policy -> head."""
    bb = build_backbone(cfg.backbone).to(device)
    info: dict = {}

    # 1. Calibrate tau on the untouched model; diagnostics reuse it after training (H1).
    #    Two thresholds: the model-specific outlier layer, and the last layer the head consumes.
    stats = calibrate_outlier_threshold(
        bb, calib_loader, layer=cfg.backbone.outlier_layer, max_images=cfg.data.calib_images
    )
    stats_last = calibrate_outlier_threshold(
        bb, calib_loader, layer=-1, max_images=cfg.data.calib_images
    )
    write_json(
        {"outlier_layer": stats.__dict__, "last_layer": stats_last.__dict__},
        run_dir / "outlier_stats.json",
    )
    info["outlier_stats"] = stats.__dict__
    info["stats"] = stats  # OutlierStats object reused by run() diagnostics
    info["stats_last"] = stats_last

    # 2. Test-time registers.
    info["num_register_neurons"] = 0
    if cfg.backbone.registers == "test_time":
        if cfg.backbone.register_neuron_path:
            rn = load_register_neurons(cfg.backbone.register_neuron_path)
        else:
            rn = find_register_neurons(bb, calib_loader, stats, max_images=cfg.data.calib_images)
        save_register_neurons(rn, run_dir / "register_neurons.json")
        install_test_time_registers(bb, rn)
        info["num_register_neurons"] = sum(len(v) for v in rn.layer_to_neurons.values())

    # 3. LoRA and freeze policy.
    if cfg.train.mode == "lora" and not cfg.lora.enabled:
        raise RuntimeError("train.mode=lora requires lora.enabled=true")
    if cfg.train.mode != "lora" and cfg.lora.enabled:
        raise RuntimeError(
            "lora.enabled=true is only valid with train.mode=lora "
            "(full mode would train adapters too)"
        )
    info["lora_modules"] = apply_lora(bb, cfg.lora)
    bb.to(device)
    set_trainable(bb, cfg.train.mode)
    if cfg.train.grad_checkpoint and cfg.train.mode != "frozen":
        bb.model.set_grad_checkpointing(True)
    info["trainable_params"], info["total_params"] = count_params(bb)

    # 4. Head.
    head = build_head(cfg.head, bb.embed_dim, num_classes(cfg.data.name)).to(device)
    info["head_params"] = count_params(head)[0]
    return bb, head, info
