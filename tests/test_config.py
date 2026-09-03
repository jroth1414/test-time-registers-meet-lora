from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from ttr.config import DataCfg, LoraCfg, RunCfg, load_config, save_config


def test_defaults_are_the_documented_ones():
    cfg = load_config()
    assert isinstance(cfg, RunCfg)
    assert cfg.backbone.registers == "none"
    assert cfg.lora.targets == ["q", "v"]
    assert cfg.lora.layers == "all"
    assert cfg.train.mode == "frozen"
    assert cfg.data.mean == [0.485, 0.456, 0.406]
    assert cfg.backbone.outlier_layer == -1


def test_dotlist_overrides_nested_fields():
    cfg = load_config(
        overrides=["lora.enabled=true", "lora.r=4", "train.seed=3", "lora.layers=[0,1]"]
    )
    assert cfg.lora.enabled is True
    assert cfg.lora.r == 4
    assert cfg.train.seed == 3
    assert list(cfg.lora.layers) == [0, 1]


def test_yaml_roundtrip(tmp_path: Path):
    cfg = load_config(overrides=["run_id=abc", "backbone.registers=test_time"])
    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    back = load_config(str(p))
    assert back.run_id == "abc"
    assert back.backbone.registers == "test_time"
    assert back == cfg


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigKeyError):
        load_config(overrides=["backbone.nonsense=1"])


def test_sub_configs_are_plain_dataclasses():
    assert LoraCfg().r == 8
    assert DataCfg().name == "ade20k"
