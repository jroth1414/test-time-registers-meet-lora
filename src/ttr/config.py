"""Run configuration: dataclasses + OmegaConf loader.

Every experiment is one RunCfg. YAML files and `key=value` overrides merge onto the
dataclass defaults; unknown keys raise, so typos fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class BackboneCfg:
    name: str = "vit_small_patch14_dinov2.lvd142m"  # timm VisionTransformer name, or "tiny"
    img_size: int = 224
    pretrained: bool = True
    registers: str = "none"  # "none" | "test_time" | "trained"
    num_test_time_registers: int = 1
    register_neuron_path: str | None = None  # JSON written by ttr.registers
    capture_attention: bool = False


@dataclass
class LoraCfg:
    enabled: bool = False
    r: int = 8
    alpha: float = 16.0
    targets: list[str] = field(default_factory=lambda: ["q", "v"])  # subset of q,k,v,o
    layers: Any = "all"  # "all" or list[int]
    dropout: float = 0.0


@dataclass
class HeadCfg:
    type: str = "linear"  # "linear" | "mask"
    hidden: int = 256


@dataclass
class DataCfg:
    name: str = "ade20k"  # "ade20k" | "cityscapes" | "lars" | "synthetic"
    root: str = "data/ade20k"
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    calib_images: int = 64
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TrainCfg:
    mode: str = "frozen"  # "frozen" | "lora" | "full"
    epochs: int = 10
    lr: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 0.01
    amp: bool = True
    grad_checkpoint: bool = False
    seed: int = 0


@dataclass
class RunCfg:
    run_id: str = "debug"
    out_dir: str = "results"
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    head: HeadCfg = field(default_factory=HeadCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    diagnostics: bool = True


def load_config(path: str | None = None, overrides: list[str] | None = None) -> RunCfg:
    """Merge defaults <- YAML file <- dotlist overrides, and return a plain RunCfg."""
    conf = OmegaConf.structured(RunCfg)
    if path is not None:
        conf = OmegaConf.merge(conf, OmegaConf.load(path))
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(conf)


def save_config(cfg: RunCfg, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), str(path))
