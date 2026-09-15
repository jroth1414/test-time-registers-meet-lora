"""Aggregate results/ into tables, paired tests, and hypothesis verdicts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ttr.utils import read_json

KEYS = ["dataset", "fam", "mode", "registers", "head"]

# Optional top-level metrics.json keys forwarded verbatim when present.
_OPTIONAL_METRIC_KEYS = ("head_params", "total_params", "epochs")

# Optional diagnostics.json keys forwarded verbatim when present.
_OPTIONAL_DIAG_KEYS = (
    "outlier_fraction_last_layer",
    "outlier_fraction_recalibrated",
    "norm_ratio_p999",
    "norm_ratio_max",
    "throughput_dtype",
    "recalibrated",
)


def _parse_run_id(run_id: str) -> dict:
    parts = run_id.split("__")
    dataset, fam, mode, reg, seed = parts[:5]
    return {
        "dataset": dataset,
        "fam": fam,
        "mode": mode,
        "registers": reg,
        "seed": int(seed.lstrip("s")),
        "head": "mask" if len(parts) > 5 and parts[5] == "mask" else "linear",
    }


def collect(results_root: str | Path) -> pd.DataFrame:
    """Flatten a results/ tree (one metrics.json per run directory) into a DataFrame.

    Skips non-directory entries and any directory without a metrics.json at its own
    level (this excludes `sanity/` and `sweep_logs/`, which nest runs or hold logs
    rather than being runs themselves). Does not recurse.
    """
    rows = []
    for d in sorted(Path(results_root).iterdir()):
        if not d.is_dir():
            continue
        if not (d / "metrics.json").exists():
            continue
        try:
            parsed = _parse_run_id(d.name)
        except (ValueError, IndexError):
            print(f"skipping {d.name}: not a factorial run id")
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        m = read_json(d / "metrics.json")
        diag = read_json(d / "diagnostics.json") if (d / "diagnostics.json").exists() else {}
        row = {
            "run_id": d.name,
            **parsed,
            "backbone_name": cfg["backbone"]["name"],
            "best_miou": m.get("best_miou"),
            "final_miou": m.get("final_miou"),
            "pixel_acc": m.get("pixel_acc"),
            "trainable_params": m.get("trainable_params"),
            "wall_seconds": m.get("wall_seconds"),
            "outlier_fraction": diag.get("outlier_fraction"),
            "attn_entropy_cls": (diag.get("attn_entropy") or {}).get("cls"),
            "images_per_s": diag.get("images_per_s"),
            "num_register_neurons": diag.get("num_register_neurons"),
        }
        for k in _OPTIONAL_METRIC_KEYS:
            row[k] = m.get(k)
        for k in _OPTIONAL_DIAG_KEYS:
            row[k] = diag.get(k)
        row.update({k: v for k, v in m.items() if k.startswith("extra_")})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    num = [
        c
        for c in (
            "best_miou",
            "final_miou",
            "pixel_acc",
            "outlier_fraction",
            "attn_entropy_cls",
            "images_per_s",
        )
        if c in df.columns
    ] + [c for c in df.columns if c.startswith("extra_")]
    g = df.groupby(KEYS)
    out = g[num].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n"] = g.size()
    out["trainable_params"] = g["trainable_params"].first()
    if "head_params" in df.columns:
        out["head_params"] = g["head_params"].first()
    return out.reset_index()


def to_markdown(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(df.to_markdown(index=False, floatfmt=".4f"))
