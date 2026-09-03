from pathlib import Path

import pandas as pd
import torch

from ttr.config import load_config
from ttr.data import build_dataset
from ttr.run import build_model, evaluate, train_one_epoch


def _cfg(*overrides):
    base = [
        "backbone.name=tiny",
        "backbone.pretrained=false",
        "backbone.img_size=56",
        "data.name=synthetic",
        "data.img_size=56",
        "data.batch_size=4",
        "data.num_workers=0",
        "data.calib_images=8",
        "train.epochs=1",
        "train.amp=false",
    ]
    return load_config(overrides=base + list(overrides))


def _calib_loader():
    return [torch.randn(4, 3, 56, 56) for _ in range(2)]


def test_build_model_frozen(tmp_results: Path):
    bb, head, info = build_model(_cfg(), torch.device("cpu"), _calib_loader(), tmp_results)
    assert info["trainable_params"] == 0
    assert (tmp_results / "outlier_stats.json").exists()
    assert head(bb.forward_features(torch.randn(1, 3, 56, 56)), (56, 56)).shape == (1, 4, 56, 56)


def test_build_model_lora_test_time_registers(tmp_results: Path):
    cfg = _cfg(
        "train.mode=lora",
        "lora.enabled=true",
        "lora.r=2",
        "backbone.registers=test_time",
        "backbone.num_test_time_registers=1",
    )
    bb, head, info = build_model(cfg, torch.device("cpu"), _calib_loader(), tmp_results)
    assert bb.num_tt_reg == 1
    assert info["trainable_params"] > 0 and info["num_register_neurons"] >= 0
    assert (tmp_results / "register_neurons.json").exists()


def test_build_model_lora_mode_requires_lora_enabled(tmp_results: Path):
    import pytest

    with pytest.raises(RuntimeError):
        build_model(_cfg("train.mode=lora"), torch.device("cpu"), _calib_loader(), tmp_results)
    with pytest.raises(RuntimeError):
        build_model(
            _cfg("train.mode=full", "lora.enabled=true"),
            torch.device("cpu"),
            _calib_loader(),
            tmp_results,
        )


def test_train_skips_all_ignore_batches(tmp_results: Path):
    cfg = _cfg()
    dev = torch.device("cpu")
    bb, head, _ = build_model(cfg, dev, _calib_loader(), tmp_results)
    x = torch.randn(2, 3, 56, 56)
    y = torch.full((2, 56, 56), 255, dtype=torch.long)  # every pixel ignored
    loader = [(x, y)]
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    before = [p.detach().clone() for p in head.parameters()]
    loss = train_one_epoch(bb, head, loader, opt, sched, dev, amp=False, mode="frozen")
    assert loss == 0.0  # nothing counted
    assert all(torch.equal(a, b) for a, b in zip(before, head.parameters(), strict=True))


def test_train_reduces_loss_and_eval_writes_per_image(tmp_results: Path):
    cfg = _cfg()
    dev = torch.device("cpu")
    bb, head, _ = build_model(cfg, dev, _calib_loader(), tmp_results)
    tr = torch.utils.data.DataLoader(build_dataset(cfg.data, "train"), batch_size=8, shuffle=True)
    va = torch.utils.data.DataLoader(build_dataset(cfg.data, "val"), batch_size=8)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    l0 = train_one_epoch(bb, head, tr, opt, sched, dev, amp=False, mode="frozen")
    for _ in range(4):
        l1 = train_one_epoch(bb, head, tr, opt, sched, dev, amp=False, mode="frozen")
    assert l1 < l0
    out = evaluate(
        bb, head, va, 4, [0], dev, amp=False, per_image_path=tmp_results / "per_image.csv"
    )
    assert set(out) >= {"miou", "pixel_acc", "per_class_iou"}
    df = pd.read_csv(tmp_results / "per_image.csv")
    assert list(df.columns) == ["index", "miou", "bg_fraction"] and len(df) == 8
