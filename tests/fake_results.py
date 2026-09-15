"""Fabricate a results/ tree with controllable numbers so analysis maths can be unit-tested."""

from pathlib import Path

import yaml

from ttr.utils import write_json


def make_run(
    root: Path,
    dataset,
    fam,
    mode,
    reg,
    seed,
    miou,
    outlier,
    per_image=None,
    head="linear",
    metrics_extra: dict | None = None,
    diag_extra: dict | None = None,
):
    run_id = f"{dataset}__{fam}__{mode}__{reg}__s{seed}" + ("__mask" if head == "mask" else "")
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "run_id": run_id,
        "backbone": {"name": f"{fam}-ckpt", "registers": reg},
        "lora": {"enabled": mode == "lora"},
        "head": {"type": head},
        "data": {"name": dataset},
        "train": {"mode": mode, "seed": seed},
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    metrics = {
        "best_miou": miou + 0.005,
        "final_miou": miou,
        "pixel_acc": 0.9,
        "trainable_params": 100,
        "total_params": 1000,
        "head_params": 10,
        "epochs": 1,
        "wall_seconds": 1.0,
    }
    if metrics_extra:
        metrics.update(metrics_extra)
    write_json(metrics, d / "metrics.json")
    diag = {
        "outlier_fraction": outlier,
        "attn_entropy": {"cls": 2.0, "tt_reg": 1.0, "patch": 3.0},
        "images_per_s": 50.0,
        "num_register_neurons": 3,
    }
    if diag_extra:
        diag.update(diag_extra)
    write_json(diag, d / "diagnostics.json")
    if per_image is not None:
        lines = ["index,miou,bg_fraction"] + [f"{i},{m},{b}" for i, (m, b) in enumerate(per_image)]
        (d / "per_image.csv").write_text("\n".join(lines))
    return d


def standard_tree(root: Path):
    """DINOv2-S on ade20k, 2 seeds. Numbers chosen so H1 and H2 hold.

    The registered LoRA arms carry a per-seed jitter of +/- 0.001 so the paired
    differences against `lora/none` have non-zero variance and scipy returns a real
    p value. Cell means stay 0.305 / 0.345 / 0.375 / 0.395.
    """
    for s in (0, 1):
        j = 0.001 if s else -0.001
        make_run(root, "ade20k", "vits", "frozen", "none", s, 0.30 + 0.01 * s, 0.050)
        make_run(
            root, "ade20k", "vits", "lora", "none", s, 0.34 + 0.01 * s, 0.048
        )  # outliers persist
        make_run(
            root, "ade20k", "vits", "lora", "test_time", s, 0.37 + 0.01 * s + j, 0.004
        )  # closes 60% of gap
        make_run(root, "ade20k", "vits", "lora", "trained", s, 0.39 + 0.01 * s + j, 0.002)
    return root
