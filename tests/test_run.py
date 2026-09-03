from pathlib import Path

import torch

from ttr.config import load_config
from ttr.run import build_model


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
