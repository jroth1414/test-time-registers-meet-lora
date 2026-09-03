"""One factorial cell: build, train, evaluate, diagnose, write results."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ttr.backbone import Backbone, build_backbone, normalization_for
from ttr.config import RunCfg, load_config, save_config
from ttr.data import background_class_ids, build_dataset, num_classes
from ttr.heads import build_head
from ttr.lora import apply_lora, count_params, lora_state_dict, set_trainable
from ttr.metrics import (
    ConfusionMeter,
    background_fraction,
    image_miou,
    measure_throughput,
)
from ttr.registers import (
    attention_entropy,
    calibrate_outlier_threshold,
    find_register_neurons,
    install_test_time_registers,
    load_register_neurons,
    outlier_fraction,
    save_register_neurons,
)
from ttr.utils import (
    Timer,
    append_csv_row,
    get_device,
    make_run_dir,
    read_json,
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


def _autocast(device: torch.device, amp: bool):
    enabled = amp and device.type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def train_one_epoch(
    bb: Backbone, head: nn.Module, loader, opt, sched, device, amp: bool, mode: str
) -> float:
    head.train()
    bb.train(mode != "frozen")
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device, amp):
            if mode == "frozen":
                with torch.no_grad():
                    feat = bb.forward_features(x)
            else:
                feat = bb.forward_features(x)
            logits = head(feat, tuple(y.shape[-2:]))
            loss = F.cross_entropy(logits.float(), y, ignore_index=255)
        if not torch.isfinite(loss):
            # All-ignore batch (255) yields NaN; skip rather than poison AdamW.
            sched.step()
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    bb, head, loader, num_cls: int, bg_ids: list[int], device, amp: bool, per_image_path=None
) -> dict:
    head.eval()
    bb.eval()
    meter = ConfusionMeter(num_cls)
    if per_image_path is not None and Path(per_image_path).exists():
        Path(per_image_path).unlink()
    idx = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with _autocast(device, amp):
            pred = head(bb.forward_features(x), tuple(y.shape[-2:])).argmax(1)
        meter.update(pred, y)
        if per_image_path is not None:
            for p, t in zip(pred, y, strict=True):
                append_csv_row(
                    per_image_path,
                    {
                        "index": idx,
                        "miou": image_miou(p, t, num_cls),
                        "bg_fraction": background_fraction(t, bg_ids),
                    },
                )
                idx += 1
    return {
        "miou": meter.miou(),
        "pixel_acc": meter.pixel_acc(),
        "per_class_iou": meter.per_class_iou(),
    }


def _loaders(cfg: RunCfg):
    tr = build_dataset(cfg.data, "train")
    va = build_dataset(cfg.data, "val")
    kw = dict(num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(tr, cfg.data.batch_size, shuffle=True, drop_last=True, **kw),
        DataLoader(va, cfg.data.batch_size, shuffle=False, **kw),
    )


def run(cfg: RunCfg, force: bool = False) -> dict:
    run_dir = make_run_dir(cfg.out_dir, cfg.run_id)
    if (run_dir / "metrics.json").exists() and not force:
        return {**read_json_safe(run_dir / "metrics.json"), "skipped": True}
    seed_everything(cfg.train.seed)
    device = get_device()
    if cfg.backbone.name != "tiny" and cfg.backbone.img_size != cfg.data.img_size:
        raise ValueError(
            f"backbone.img_size={cfg.backbone.img_size} != data.img_size={cfg.data.img_size}"
        )

    # Normalisation must match the checkpoint; resolve it before datasets are built.
    probe = build_backbone(cfg.backbone)
    cfg.data.mean, cfg.data.std = normalization_for(probe)
    del probe
    save_config(cfg, run_dir / "config.yaml")

    train_loader, val_loader = _loaders(cfg)
    bb, head, info = build_model(cfg, device, val_loader, run_dir)
    k = num_classes(cfg.data.name)
    bg = background_class_ids(cfg.data.name)

    groups = [{"params": head.parameters(), "lr": cfg.train.lr}]
    bb_params = [p for p in bb.parameters() if p.requires_grad]
    if bb_params:
        lr_bb = cfg.train.lr if cfg.train.mode == "lora" else cfg.train.lr_backbone
        groups.append({"params": bb_params, "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    steps = max(cfg.train.epochs * len(train_loader), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )

    best, best_state = -1.0, None
    with Timer() as wall:
        for epoch in range(cfg.train.epochs):
            with Timer() as t:
                loss = train_one_epoch(
                    bb, head, train_loader, opt, sched, device, cfg.train.amp, cfg.train.mode
                )
            ev = evaluate(bb, head, val_loader, k, bg, device, cfg.train.amp)
            append_csv_row(
                run_dir / "log.csv",
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_miou": ev["miou"],
                    "val_pixel_acc": ev["pixel_acc"],
                    "lr": opt.param_groups[0]["lr"],
                    "epoch_seconds": t.seconds,
                },
            )
            if ev["miou"] > best:
                best = ev["miou"]
                best_state = {
                    "head": {k_: v.cpu() for k_, v in head.state_dict().items()},
                    "lora": lora_state_dict(bb),
                    # full fine-tune has no adapter to save; keep the whole backbone
                    # (gitignored .pt)
                    "backbone": (
                        {k_: v.cpu() for k_, v in bb.state_dict().items()}
                        if cfg.train.mode == "full"
                        else None
                    ),
                }
            print(
                f"[{cfg.run_id}] epoch {epoch} loss {loss:.4f} val mIoU {ev['miou']:.4f}",
                flush=True,
            )

    torch.save(best_state, run_dir / "head_lora.pt")
    head.load_state_dict(best_state["head"])
    if best_state["backbone"] is not None:
        bb.load_state_dict(best_state["backbone"])
    elif best_state["lora"]:
        bb.load_state_dict(best_state["lora"], strict=False)
    final = evaluate(
        bb, head, val_loader, k, bg, device, cfg.train.amp, per_image_path=run_dir / "per_image.csv"
    )

    metrics = {
        "skipped": False,
        "best_miou": best,
        "final_miou": final["miou"],
        "pixel_acc": final["pixel_acc"],
        "per_class_iou": final["per_class_iou"],
        "epochs": cfg.train.epochs,
        "trainable_params": info["trainable_params"],
        "total_params": info["total_params"],
        "head_params": info["head_params"],
        "wall_seconds": wall.seconds,
    }
    write_json(metrics, run_dir / "metrics.json")

    if cfg.diagnostics:
        bb.eval()
        x, _ = next(iter(val_loader))
        diag = {
            "num_register_neurons": info["num_register_neurons"],
            "outlier_stats": info["outlier_stats"],
        }

        # Throughput first: attention_entropy disables fused attention and would slow the backbone.
        # Measured under the same autocast as evaluation so images/s describes what actually ran.
        def _fwd(t):
            with _autocast(device, cfg.train.amp):
                return bb.forward_features(t)

        diag["images_per_s"] = measure_throughput(_fwd, x.to(device))
        diag["throughput_dtype"] = "bf16" if (cfg.train.amp and device.type == "cuda") else "fp32"
        n = cfg.data.calib_images
        # H1 primary: fixed tau from the frozen model, at the model-specific outlier layer.
        diag["outlier_fraction"] = outlier_fraction(bb, val_loader, info["stats"], max_images=n)
        diag["outlier_fraction_last_layer"] = outlier_fraction(
            bb, val_loader, info["stats_last"], max_images=n
        )
        # H1 robustness: recalibrate tau on the trained model; if LoRA only rescaled the residual
        # stream, this fraction stays put while the fixed-tau one moves.
        recal = calibrate_outlier_threshold(
            bb, val_loader, layer=cfg.backbone.outlier_layer, max_images=n
        )
        diag["outlier_stats_post"] = recal.__dict__
        diag["outlier_fraction_recalibrated"] = outlier_fraction(
            bb, val_loader, recal, max_images=n
        )
        diag["attn_entropy"] = attention_entropy(
            bb, val_loader, layers=[bb.depth - 1], max_images=16
        )
        write_json(diag, run_dir / "diagnostics.json")
    return metrics


def read_json_safe(p: Path) -> dict:
    try:
        return read_json(p)
    except Exception:
        return {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run one factorial cell")
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value OmegaConf dotlist overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    m = run(cfg, force=args.force)
    print({k: v for k, v in m.items() if k != "per_class_iou"})


if __name__ == "__main__":
    main(sys.argv[1:])
