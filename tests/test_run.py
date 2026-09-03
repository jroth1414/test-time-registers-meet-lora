from pathlib import Path

import pandas as pd
import pytest
import torch

import ttr.run as run_mod
from ttr.config import load_config
from ttr.data import build_dataset
from ttr.registers import OutlierStats, RegisterNeurons, save_register_neurons
from ttr.run import build_model, evaluate, main, run, train_one_epoch
from ttr.utils import read_json, write_json


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


def test_build_model_loads_register_neurons_from_path(tmp_results: Path, tmp_path: Path):
    """When register_neuron_path is set, build_model loads it instead of running detection."""
    rn_path = tmp_path / "neurons.json"
    rn = RegisterNeurons(
        layer_to_neurons={0: [0, 1]},
        stats=OutlierStats(layer=0, tau=1.0, median=0.5, mad=0.1, k=4.0),
    )
    save_register_neurons(rn, rn_path)
    cfg = _cfg(
        "backbone.registers=test_time",
        "backbone.num_test_time_registers=1",
        f"backbone.register_neuron_path={rn_path.as_posix()}",
    )
    bb, head, info = build_model(cfg, torch.device("cpu"), _calib_loader(), tmp_results)
    assert info["num_register_neurons"] == 2


def test_build_model_test_time_without_neuron_path_raises_on_real_checkpoint(
    tmp_results: Path,
):
    cfg = _cfg(
        "backbone.name=vit_small_patch14_dinov2.lvd142m",
        "backbone.pretrained=false",
        "backbone.registers=test_time",
    )
    with pytest.raises(ValueError, match="register_neuron_path"):
        build_model(cfg, torch.device("cpu"), _calib_loader(), tmp_results)


def test_build_model_lora_mode_requires_lora_enabled(tmp_results: Path):
    with pytest.raises(RuntimeError):
        build_model(_cfg("train.mode=lora"), torch.device("cpu"), _calib_loader(), tmp_results)
    with pytest.raises(RuntimeError):
        build_model(
            _cfg("train.mode=full", "lora.enabled=true"),
            torch.device("cpu"),
            _calib_loader(),
            tmp_results,
        )


def test_build_model_grad_checkpoint_full_mode(tmp_results: Path):
    cfg = _cfg("train.mode=full", "train.grad_checkpoint=true")
    bb, head, info = build_model(cfg, torch.device("cpu"), _calib_loader(), tmp_results)
    assert bb.model.grad_checkpointing is True
    assert info["trainable_params"] > 0


@pytest.mark.filterwarnings("ignore:Detected call of .lr_scheduler.step.*:UserWarning")
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
    loss, n_skipped = train_one_epoch(bb, head, loader, opt, sched, dev, amp=False, mode="frozen")
    assert loss == 0.0  # nothing counted
    assert n_skipped == 1
    assert all(torch.equal(a, b) for a, b in zip(before, head.parameters(), strict=True))


def test_train_reduces_loss_and_eval_writes_per_image(tmp_results: Path):
    cfg = _cfg()
    dev = torch.device("cpu")
    bb, head, _ = build_model(cfg, dev, _calib_loader(), tmp_results)
    tr = torch.utils.data.DataLoader(build_dataset(cfg.data, "train"), batch_size=8, shuffle=True)
    va = torch.utils.data.DataLoader(build_dataset(cfg.data, "val"), batch_size=8)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    l0, _ = train_one_epoch(bb, head, tr, opt, sched, dev, amp=False, mode="frozen")
    for _ in range(4):
        l1, _ = train_one_epoch(bb, head, tr, opt, sched, dev, amp=False, mode="frozen")
    assert l1 < l0
    out = evaluate(
        bb, head, va, 4, [0], dev, amp=False, per_image_path=tmp_results / "per_image.csv"
    )
    assert set(out) >= {"miou", "pixel_acc", "per_class_iou"}
    df = pd.read_csv(tmp_results / "per_image.csv")
    assert list(df.columns) == ["index", "miou", "bg_fraction"] and len(df) == 8


def test_run_end_to_end_writes_all_artifacts(tmp_results: Path):
    cfg = _cfg(f"out_dir={tmp_results.as_posix()}", "run_id=e2e", "train.epochs=2")
    m = run(cfg)
    d = tmp_results / "e2e"
    for f in (
        "config.yaml",
        "log.csv",
        "metrics.json",
        "diagnostics.json",
        "per_image.csv",
        "head_lora.pt",
    ):
        assert (d / f).exists(), f
    assert m["best_miou"] >= 0 and m["epochs"] == 2
    assert "final_miou" in m
    diag = read_json(d / "diagnostics.json")
    assert {
        "outlier_fraction",
        "outlier_fraction_last_layer",
        "outlier_fraction_recalibrated",
        "outlier_stats_post",
        "attn_entropy",
        "images_per_s",
        "recalibrated",
    } <= set(diag)
    per_image = pd.read_csv(d / "per_image.csv")
    assert len(per_image) == 8
    log_lines = (d / "log.csv").read_text().strip().splitlines()
    assert "n_skipped" in log_lines[0]
    # second call is skipped
    assert run(cfg)["skipped"] is True
    assert run(cfg, force=True)["skipped"] is False


def test_run_reruns_when_diagnostics_missing(tmp_results: Path):
    """A metrics.json without diagnostics.json is not a complete run when diagnostics=True."""
    cfg = _cfg(f"out_dir={tmp_results.as_posix()}", "run_id=partial", "train.epochs=1")
    d = tmp_results / "partial"
    d.mkdir()
    write_json({"final_miou": 0.5}, d / "metrics.json")
    m = run(cfg)
    assert m["skipped"] is False


def test_run_reruns_when_metrics_truncated(tmp_results: Path):
    cfg = _cfg(f"out_dir={tmp_results.as_posix()}", "run_id=trunc", "train.epochs=1")
    d = tmp_results / "trunc"
    d.mkdir()
    (d / "metrics.json").write_text("{")
    m = run(cfg)
    assert m["skipped"] is False


def test_force_rerun_truncates_log_csv(tmp_results: Path):
    cfg = _cfg(f"out_dir={tmp_results.as_posix()}", "run_id=trunclog", "train.epochs=2")
    run(cfg)
    run(cfg, force=True)
    lines = (tmp_results / "trunclog" / "log.csv").read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 epochs, not doubled


def test_run_headline_metric_is_last_epoch_not_best(tmp_results: Path, monkeypatch):
    """metrics.json's final_miou must come from the last-epoch weights, not a best-epoch
    restore; best_miou is still reported, but only as the max over the per-epoch curve."""
    cfg = _cfg(f"out_dir={tmp_results.as_posix()}", "run_id=lastep", "train.epochs=3")
    real_evaluate = run_mod.evaluate
    seen = []

    def fake_evaluate(*args, **kwargs):
        out = real_evaluate(*args, **kwargs)
        if "per_image_path" not in kwargs:
            # per-epoch validation call: fake a strictly descending curve
            seen.append(1)
            out["miou"] = 1.0 - 0.1 * len(seen)
        return out

    monkeypatch.setattr(run_mod, "evaluate", fake_evaluate)
    m = run(cfg)
    assert m["best_miou"] == pytest.approx(0.9)  # epoch 0's faked value, the max
    assert m["final_miou"] != pytest.approx(0.9)  # real last-epoch eval, unfaked


def test_run_lora_mode_saves_lora_state(tmp_results: Path):
    cfg = _cfg(
        f"out_dir={tmp_results.as_posix()}",
        "run_id=lora_e2e",
        "train.mode=lora",
        "lora.enabled=true",
        "lora.r=2",
        "train.epochs=1",
    )
    m = run(cfg)
    assert m["skipped"] is False
    ckpt = torch.load(tmp_results / "lora_e2e" / "head_lora.pt", weights_only=False)
    assert ckpt["backbone"] is None
    assert len(ckpt["lora"]) > 0


def test_diagnostics_skips_recalibration_when_frozen(tmp_results: Path):
    cfg = _cfg(f"out_dir={tmp_results.as_posix()}", "run_id=frozen_diag", "train.mode=frozen")
    run(cfg)
    diag = read_json(tmp_results / "frozen_diag" / "diagnostics.json")
    assert diag["recalibrated"] is False
    assert diag["outlier_fraction_recalibrated"] == diag["outlier_fraction"]


def test_cli_accepts_overrides(tmp_results: Path):
    main(
        [
            "--config",
            "configs/debug_synthetic.yaml",
            f"out_dir={tmp_results.as_posix()}",
            "run_id=cli",
            "train.epochs=1",
        ]
    )
    assert (tmp_results / "cli" / "metrics.json").exists()


def test_cli_force_flag_reruns(tmp_results: Path, capsys):
    argv = [
        "--config",
        "configs/debug_synthetic.yaml",
        f"out_dir={tmp_results.as_posix()}",
        "run_id=cliforce",
        "train.epochs=1",
    ]
    main(argv)
    capsys.readouterr()
    main(argv)
    assert "'skipped': True" in capsys.readouterr().out
    main(argv + ["--force"])
    assert "'skipped': False" in capsys.readouterr().out


def test_run_full_mode_saves_backbone(tmp_results: Path):
    cfg = _cfg(
        f"out_dir={tmp_results.as_posix()}",
        "run_id=full",
        "train.mode=full",
        "train.epochs=1",
    )
    m = run(cfg)
    assert m["skipped"] is False and m["trainable_params"] > 0
    ckpt = torch.load(tmp_results / "full" / "head_lora.pt", weights_only=False)
    assert ckpt["backbone"] is not None and ckpt["lora"] == {}
