"""Emit one YAML per factorial cell x seed. Cells: backbone x mode x registers."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

BACKBONES = {
    "vits": {
        "none": "vit_small_patch14_dinov2.lvd142m",
        "trained": "vit_small_patch14_reg4_dinov2.lvd142m",
    },
    "vitb": {
        "none": "vit_base_patch14_dinov2.lvd142m",
        "trained": "vit_base_patch14_reg4_dinov2.lvd142m",
    },
    "clipb": {"none": "vit_base_patch16_clip_224.openai", "trained": None},
}
# Residual layer where each family's outlier tokens live (artifacts/register_neurons/README.md).
OUTLIER_LAYER = {"vits": 10, "vitb": 8, "clipb": 11}
MODES = ["frozen", "lora", "full"]
REGISTERS = ["none", "test_time", "trained"]
CELLS = [
    (b, m, r)
    for b, m, r in itertools.product(BACKBONES, MODES, REGISTERS)
    if not (r == "trained" and BACKBONES[b]["trained"] is None)
]


def cell_config(
    dataset: str, fam: str, mode: str, reg: str, seed: int, epochs: int, root: str
) -> dict:
    name = BACKBONES[fam]["trained" if reg == "trained" else "none"]
    reg_path = None
    if reg == "test_time":
        reg_path = f"artifacts/register_neurons/{name.replace('.', '_')}.json"
    return {
        "run_id": f"{dataset}__{fam}__{mode}__{reg}__s{seed}",
        "backbone": {
            "name": name,
            "img_size": 224,
            "registers": reg,
            "num_test_time_registers": 1,
            "outlier_layer": OUTLIER_LAYER[fam],
            "register_neuron_path": reg_path,
        },
        "lora": {
            "enabled": mode == "lora",
            "r": 8,
            "alpha": 16.0,
            "targets": ["q", "v"],
        },
        "head": {"type": "linear"},
        "data": {
            "name": dataset,
            "root": root,
            "img_size": 224,
            "batch_size": 16,
            "num_workers": 4,
            "calib_images": 64,
        },
        "train": {
            "mode": mode,
            "epochs": epochs,
            "lr": 1e-3,
            "lr_backbone": 1e-5,
            "amp": True,
            "grad_checkpoint": fam == "vitb" and mode == "full",
            "seed": seed,
        },
        "diagnostics": True,
    }


def write_configs(
    out: Path, dataset="ade20k", seeds=(0, 1, 2), epochs=10, root="data/ade20k"
) -> list[Path]:
    out = Path(out) / dataset
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for (fam, mode, reg), seed in itertools.product(CELLS, seeds):
        cfg = cell_config(dataset, fam, mode, reg, seed, epochs, root)
        p = out / f"{cfg['run_id']}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs")
    ap.add_argument("--dataset", default="ade20k")
    ap.add_argument("--root", default="data/ade20k")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()
    ps = write_configs(a.out, a.dataset, a.seeds, a.epochs, a.root)
    print(f"wrote {len(ps)} configs under {Path(a.out) / a.dataset}")
