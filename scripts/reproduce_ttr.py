# scripts/reproduce_ttr.py
"""Detect register neurons on a real checkpoint and show that test-time registers remove
outlier tokens. Prints before/after outlier fraction, saves the neuron JSON and an attention
figure. Acceptance: outlier fraction after < 0.2 * before, and the cls attention map looks
object-centred instead of speckled.
"""

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from ttr.backbone import build_backbone, normalization_for
from ttr.config import BackboneCfg
from ttr.registers import (
    attention_entropy,
    calibrate_outlier_threshold,
    find_register_neurons,
    install_test_time_registers,
    outlier_fraction,
    save_register_neurons,
)


class Folder(Dataset):
    def __init__(self, root, size, mean, std):
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.tf = T.Compose(
            [T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--images", required=True)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--quantile", type=float, default=0.999)
    ap.add_argument("--max-neurons", type=int, default=64)
    ap.add_argument("--num-registers", type=int, default=1)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = build_backbone(
        BackboneCfg(name=args.model, img_size=args.img_size, capture_attention=True)
    ).to(dev)
    mean, std = normalization_for(bb)
    ds = Folder(args.images, args.img_size, mean, std)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    stats = calibrate_outlier_threshold(
        bb, loader, layer=args.layer, k=args.k, max_images=args.calib
    )
    before = outlier_fraction(bb, loader, stats, max_images=args.calib)
    ent_before = attention_entropy(bb, loader, layers=[bb.depth - 1], max_images=16)
    rn = find_register_neurons(bb, loader, stats, args.quantile, args.max_neurons, args.calib)
    print(f"tau={stats.tau:.2f} (median {stats.median:.2f}) outlier fraction before: {before:.4f}")
    print(
        "register neurons:",
        {layer_idx: len(v) for layer_idx, v in rn.layer_to_neurons.items()},
    )

    bb.set_tt_registers(args.num_registers)
    handles = install_test_time_registers(bb, rn)
    after = outlier_fraction(bb, loader, stats, max_images=args.calib)
    ent_after = attention_entropy(bb, loader, layers=[bb.depth - 1], max_images=16)
    print(f"outlier fraction after: {after:.4f}  (ratio {after / max(before, 1e-9):.3f})")
    print("cls attention entropy before/after:", ent_before["cls"], ent_after["cls"])

    meta = {
        "model": args.model,
        "img_size": args.img_size,
        "layer": stats.layer,
        "k": args.k,
        "quantile": args.quantile,
        "max_neurons": args.max_neurons,
        "num_registers": args.num_registers,
        "n_calib_images": args.calib,
        "images_dir": args.images,
        "git_sha": _git_sha(),
    }
    tag = args.model.replace(".", "_")
    save_register_neurons(rn, Path(args.out) / "register_neurons" / f"{tag}.json", meta=meta)

    # Attention figure: cls -> patch attention, first 4 images, before/after, at two layers:
    # the last block and the deepest layer that got an intervention (if any).
    x = torch.stack([ds[i] for i in range(4)]).to(dev)
    last_layer = bb.depth - 1
    intervention_layer = max(rn.layer_to_neurons) if rn.layer_to_neurons else last_layer
    fig_layers = sorted({last_layer, intervention_layer})

    def cls_maps(layers):
        cap = bb.capture("attn", layers=layers)
        bb.forward_tokens(x)
        h, w = bb.grid((args.img_size, args.img_size))
        out = {
            layer_idx: cap.data[layer_idx].mean(1)[:, 0, bb.patch_slice()].reshape(4, h, w).cpu()
            for layer_idx in layers
        }
        cap.remove()
        return out

    maps_after = cls_maps(fig_layers)
    for hd in handles:
        hd.remove()
    bb.set_tt_registers(0)
    maps_before = cls_maps(fig_layers)

    n_rows = 2 * len(fig_layers)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
    for row, layer_idx in enumerate(fig_layers):
        b_maps, a_maps = maps_before[layer_idx], maps_after[layer_idx]
        r_before, r_after = 2 * row, 2 * row + 1
        for i in range(4):
            vmin = min(b_maps[i].min().item(), a_maps[i].min().item())
            vmax = max(b_maps[i].max().item(), a_maps[i].max().item())
            axes[r_before, i].imshow(b_maps[i], vmin=vmin, vmax=vmax)
            axes[r_before, i].set_title(f"before L{layer_idx}")
            axes[r_before, i].axis("off")
            axes[r_after, i].imshow(a_maps[i], vmin=vmin, vmax=vmax)
            axes[r_after, i].set_title(f"after L{layer_idx}")
            axes[r_after, i].axis("off")
    Path(args.out, "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(args.out) / "figures" / f"attn_{tag}.png", dpi=120, bbox_inches="tight")
    print("wrote", Path(args.out) / "figures" / f"attn_{tag}.png")
    if before < 1e-6:
        print("N/A (no outliers to remove)")
    elif after < 0.2 * before:
        print("PASS")
    else:
        print("FAIL: tune --k, --quantile, --max-neurons, --layer")


if __name__ == "__main__":
    main()
