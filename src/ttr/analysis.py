"""Aggregate results/ into tables, paired tests, and hypothesis verdicts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

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

# Columns summarize aggregates as mean/std over seeds, when the tree carries them.
_SUMMARY_NUMERIC = (
    "final_miou",
    "best_miou",
    "pixel_acc",
    "outlier_fraction",
    "outlier_fraction_last_layer",
    "outlier_fraction_recalibrated",
    "norm_ratio_p999",
    "norm_ratio_max",
    "attn_entropy_cls",
    "images_per_s",
)

# Columns summarize carries through from the first run of each cell.
_SUMMARY_FIRST = ("head_params", "throughput_dtype", "recalibrated")

_SUMMARY_NOTE = (
    "Frozen arms share one backbone, so their diagnostics have zero std across seeds by "
    "construction. backbone_trainable_params counts backbone parameters only and is 0 on "
    "every frozen arm; head_params is listed separately. std uses ddof = 1 over seeds."
)

_TEST_NOTE = (
    "p is a two-sided paired t-test over seed-matched final_miou; cells with zero variance "
    "show p = n/a. Tests are uncorrected across cells; the paper reports one primary "
    "comparison per hypothesis."
)

_H3_NOTE = (
    "Spearman over distinct validation images with the gain averaged over seeds; rank and "
    "lars_is_max verdicts need at least two datasets."
)

_H3_BACKBONE_NOTE = (
    "Each gain is the seed-matched mean of test-time registers minus no registers on "
    "final_miou. clip_gain_larger compares CLIP against the best DINOv2 family present."
)


def _parse_run_id(run_id: str) -> dict:
    parts = run_id.split("__")
    dataset, fam, mode, reg, seed = parts[:5]
    tail = parts[5:]
    if tail not in ([], ["mask"]):
        # Sweep variants such as __r16 name a different configuration. Parsing one as a
        # factorial cell would average two designs into the same row.
        raise ValueError(f"{run_id}: unknown run id suffix {'__'.join(tail)}")
    return {
        "dataset": dataset,
        "fam": fam,
        "mode": mode,
        "registers": reg,
        "seed": int(seed.lstrip("s")),
        "head": "mask" if tail == ["mask"] else "linear",
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
    candidates = [c for c in _SUMMARY_NUMERIC if c in df.columns]
    candidates += [c for c in df.columns if c.startswith("extra_")]
    # A diagnostic absent from every run arrives as an all-None object column; aggregating it
    # would raise, so keep the columns that actually hold numbers.
    num = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(KEYS)
    out = g[num].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n"] = g.size()
    out["n_final_miou"] = g["final_miou"].count()
    if "trainable_params" in df.columns:
        out["backbone_trainable_params"] = g["trainable_params"].first()
    for c in _SUMMARY_FIRST:
        if c in df.columns:
            out[c] = g[c].first()
    return out.reset_index()


def _for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing verdicts and missing statistics with "n/a" for the markdown tables."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = ["n/a" if pd.isna(v) else f"{v:.4f}" for v in d[c]]
        elif d[c].dtype == object:
            d[c] = [
                "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else v for v in d[c]
            ]
    return d


def to_markdown(df: pd.DataFrame, path: str | Path, note: str | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = _for_display(df).to_markdown(index=False, floatfmt=".4f")
    if note:
        text += "\n\n" + note + "\n"
    Path(path).write_text(text)


def _cell(df: pd.DataFrame, sel: dict) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for k, v in sel.items():
        m &= df[k] == v
    return df[m].sort_values("seed")


def paired(df: pd.DataFrame, key: str, a: dict, b: dict) -> dict:
    """Paired t-test over matched seeds between cell `a` and cell `b`.

    The headline metric is `final_miou`; callers should pass that as `key` unless
    they explicitly want the reference `best_miou` curve.
    """
    ca, cb = _cell(df, a), _cell(df, b)
    merged = ca.merge(cb, on="seed", suffixes=("_a", "_b"))
    diff = (merged[f"{key}_a"] - merged[f"{key}_b"]).to_numpy()
    n = len(diff)
    out = {
        "mean_diff": float(diff.mean()) if n else math.nan,
        "t": math.nan,
        "p": math.nan,
        "n": n,
        "zero_variance": False,
        "ci95_low": math.nan,
        "ci95_high": math.nan,
    }
    if n < 2:
        return out
    if diff.std() == 0:
        # Zero-variance differences make scipy's moment calculation emit a spurious
        # "precision loss" RuntimeWarning. The t statistic diverges and the p value is
        # undefined, so report neither p = 0 nor a confidence interval: the seeds agree
        # to the last digit, which is a property of the fixture or of a shared backbone
        # rather than evidence about the population.
        out["zero_variance"] = True
        out["t"] = math.inf if diff[0] != 0 else math.nan
        return out
    t, p = stats.ttest_1samp(diff, 0.0)
    half = float(stats.t.ppf(0.975, n - 1)) * float(diff.std(ddof=1)) / math.sqrt(n)
    out["t"] = float(t)
    out["p"] = float(p)
    out["ci95_low"] = out["mean_diff"] - half
    out["ci95_high"] = out["mean_diff"] + half
    return out


def _groups(df: pd.DataFrame):
    return df.groupby(["dataset", "fam", "head"])


def _mean(col: pd.Series) -> float:
    """Mean of a column that may be empty or all-None (collect leaves gaps as None)."""
    return float(pd.to_numeric(col, errors="coerce").mean())


_H1_COLUMNS = [
    "dataset",
    "fam",
    "head",
    "outlier_ratio_lora_over_frozen",
    "H1_outliers_persist",
    "miou_gap_lora_none_vs_trained",
    "ci95_low",
    "ci95_high",
    "p",
    "n",
    "zero_variance",
    "H1_gap_significant",
    "H1",
]

_H2_COLUMNS = [
    "dataset",
    "fam",
    "head",
    "miou_lora_none",
    "miou_lora_tt",
    "miou_lora_trained",
    "gap_exists",
    "gap_closure",
    "tt_minus_none",
    "p",
    "n",
    "ci95_low",
    "ci95_high",
    "zero_variance",
    "tt_minus_trained",
    "p_tt_vs_trained",
    "extra_params_tt",
    "H2",
]


_H3_BACKBONE_COLUMNS = [
    "dataset",
    "head",
    "gain_clipb",
    "gain_vits",
    "gain_vitb",
    "clip_gain_larger",
]


def h1(df: pd.DataFrame, tol: float = 0.2, alpha: float = 0.05) -> pd.DataFrame:
    """Outliers persist under LoRA (ratio within 1 +- tol) and LoRA-none loses to LoRA-trained.

    Both halves of H1 carry their own verdict. The mIoU half needs a significant paired
    difference, not a negative mean: a gap of 0.001 mIoU with p = 0.7 supports nothing.
    """
    if df.empty:
        return pd.DataFrame(columns=_H1_COLUMNS)
    rows = []
    for (ds, fam, head), g in _groups(df):
        fr = _mean(_cell(g, {"mode": "frozen", "registers": "none"})["outlier_fraction"])
        lo = _mean(_cell(g, {"mode": "lora", "registers": "none"})["outlier_fraction"])
        ratio = lo / fr if fr > 0 else math.nan
        gap = paired(
            g,
            "final_miou",
            {"mode": "lora", "registers": "none"},
            {"mode": "lora", "registers": "trained"},
        )
        persist = None if math.isnan(ratio) else bool(abs(ratio - 1) <= tol)
        if gap["n"] < 2 or math.isnan(gap["p"]):
            significant = None
        else:
            significant = bool(gap["mean_diff"] < 0 and gap["p"] < alpha)
        verdict = None if persist is None or significant is None else bool(persist and significant)
        rows.append(
            {
                "dataset": ds,
                "fam": fam,
                "head": head,
                "outlier_ratio_lora_over_frozen": ratio,
                "H1_outliers_persist": persist,
                "miou_gap_lora_none_vs_trained": gap["mean_diff"],
                "ci95_low": gap["ci95_low"],
                "ci95_high": gap["ci95_high"],
                "p": gap["p"],
                "n": gap["n"],
                "zero_variance": gap["zero_variance"],
                "H1_gap_significant": significant,
                "H1": verdict,
            }
        )
    out = pd.DataFrame(rows, columns=_H1_COLUMNS)
    return out.astype({"H1_outliers_persist": object, "H1_gap_significant": object, "H1": object})


def h2(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Test-time registers close >= threshold of the LoRA-none -> LoRA-trained mIoU gap.

    The closure fraction only means something when the trained arm wins. When the trained
    arm ties or loses, the denominator is zero or negative and a ratio would flip sign, so
    `gap_exists` is False and the closure stays NaN.
    """
    if df.empty:
        return pd.DataFrame(columns=_H2_COLUMNS)
    rows = []
    for (ds, fam, head), g in _groups(df):
        c_none = _cell(g, {"mode": "lora", "registers": "none"})
        c_tt = _cell(g, {"mode": "lora", "registers": "test_time"})
        c_tr = _cell(g, {"mode": "lora", "registers": "trained"})
        none = _mean(c_none["final_miou"])
        tt = _mean(c_tt["final_miou"])
        tr = _mean(c_tr["final_miou"])
        gap = tr - none
        gap_exists = None if math.isnan(gap) else bool(gap > 0)
        closure = (tt - none) / gap if gap > 0 else math.nan
        test = paired(
            g,
            "final_miou",
            {"mode": "lora", "registers": "test_time"},
            {"mode": "lora", "registers": "none"},
        )
        vs_trained = paired(
            g,
            "final_miou",
            {"mode": "lora", "registers": "test_time"},
            {"mode": "lora", "registers": "trained"},
        )
        # Test-time registers are training-free: the tt arm must train the same parameter
        # count as the plain arm, so this column is the zero the method claims.
        extra = _mean(c_tt["trainable_params"]) - _mean(c_none["trainable_params"])
        verdict = (
            None if math.isnan(closure) else bool(closure >= threshold and test["mean_diff"] > 0)
        )
        rows.append(
            {
                "dataset": ds,
                "fam": fam,
                "head": head,
                "miou_lora_none": none,
                "miou_lora_tt": tt,
                "miou_lora_trained": tr,
                "gap_exists": gap_exists,
                "gap_closure": closure,
                "tt_minus_none": test["mean_diff"],
                "p": test["p"],
                "n": test["n"],
                "ci95_low": test["ci95_low"],
                "ci95_high": test["ci95_high"],
                "zero_variance": test["zero_variance"],
                "tt_minus_trained": vs_trained["mean_diff"],
                "p_tt_vs_trained": vs_trained["p"],
                "extra_params_tt": float(extra),
                "H2": verdict,
            }
        )
    out = pd.DataFrame(rows, columns=_H2_COLUMNS)
    return out.astype({"gap_exists": object, "H2": object})


def _per_image_gains(root: Path, g: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, int]:
    """Seed-averaged per-image gain of the test-time-register arm over the plain arm.

    Returns one row per distinct validation image plus the number of seeds that contributed.
    Averaging first keeps Spearman on distinct images: pooling the image-seed pairs would
    feed the correlation the same image several times and shrink its p value.
    """
    frames = []
    for seed in sorted(g["seed"].unique()):
        a = _cell(g, {"mode": mode, "registers": "test_time", "seed": seed})
        b = _cell(g, {"mode": mode, "registers": "none", "seed": seed})
        if a.empty or b.empty:
            continue
        run_a, run_b = a.iloc[0]["run_id"], b.iloc[0]["run_id"]
        pa, pb = root / run_a / "per_image.csv", root / run_b / "per_image.csv"
        if not (pa.exists() and pb.exists()):
            continue
        m = pd.read_csv(pa).merge(pd.read_csv(pb), on="index", suffixes=("_tt", "_none"))
        if not np.allclose(
            m["bg_fraction_tt"].to_numpy(dtype=float),
            m["bg_fraction_none"].to_numpy(dtype=float),
            atol=1e-6,
            equal_nan=True,
        ):
            raise ValueError(f"per-image records of {run_a} and {run_b} are not aligned")
        # Drop only rows missing the columns h3 actually uses: LaRS per-image CSVs also
        # carry water_edge_f1/obstacle_f1, which metrics_lars.py legitimately leaves NaN for
        # images with no water boundary or no obstacle pixels (not bad data), and a blanket
        # dropna() would silently drop those images from the Spearman sample.
        m = m.dropna(subset=["miou_tt", "miou_none", "bg_fraction_tt"])
        frames.append(
            pd.DataFrame(
                {
                    "index": m["index"],
                    "gain": m["miou_tt"] - m["miou_none"],
                    "bg": m["bg_fraction_tt"],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["gain", "bg"]), 0
    pooled = pd.concat(frames, ignore_index=True)
    per_image = pooled.groupby("index").agg(gain=("gain", "mean"), bg=("bg", "mean"))
    return per_image.reset_index(drop=True), len(frames)


def h3(results_root: str | Path, df: pd.DataFrame, mode: str = "lora") -> pd.DataFrame:
    """Per (dataset, fam, head): Spearman correlation between per-image background fraction and the
    per-image IoU gain of test-time registers over the same adaptation mode without them.
    """
    rows = []
    root = Path(results_root)
    for (ds, fam, head), g in _groups(df):
        per_image, n_seeds = _per_image_gains(root, g, mode)
        gains = per_image["gain"].tolist()
        bgs = per_image["bg"].tolist()
        if len(gains) >= 3 and pd.Series(bgs).nunique() >= 2 and pd.Series(gains).nunique() >= 2:
            rho, p = stats.spearmanr(bgs, gains)
        else:
            # Fewer than 3 images, or constant background/gain: scipy would emit a
            # ConstantInputWarning (a RuntimeWarning subclass) for the latter, so skip the call.
            rho, p = math.nan, math.nan
        rows.append(
            {
                "dataset": ds,
                "fam": fam,
                "head": head,
                "n_images": len(gains),
                "n_seeds": n_seeds,
                "mean_gain": float(pd.Series(gains).mean()) if gains else math.nan,
                "mean_bg_fraction": float(pd.Series(bgs).mean()) if bgs else math.nan,
                "spearman_rho": float(rho),
                "p": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return _h3_ordering_verdicts(out)


def _h3_ordering_verdicts(out: pd.DataFrame) -> pd.DataFrame:
    """Add the cross-dataset claims of H3: rank agreement and the maritime maximum.

    Both need at least two datasets with per-image evidence inside one (fam, head) group.
    One dataset carries no ordering, so its verdict stays None rather than True.
    """
    evidence = out["mean_gain"].notna() & out["mean_bg_fraction"].notna()
    gain_rank = out.groupby(["fam", "head"])["mean_gain"].rank(ascending=False, method="min")
    bg_rank = out.groupby(["fam", "head"])["mean_bg_fraction"].rank(ascending=False, method="min")
    rank_ok = pd.Series([None] * len(out), index=out.index, dtype=object)
    lars_max = pd.Series([None] * len(out), index=out.index, dtype=object)
    for _, group in out.groupby(["fam", "head"]):
        ev = group.index[evidence.loc[group.index]]
        if len(ev) < 2:
            continue
        for i in ev:
            rank_ok.loc[i] = bool(gain_rank.loc[i] == bg_rank.loc[i])
        lars = [i for i in ev if out.loc[i, "dataset"] == "lars"]
        if lars:
            lars_max.loc[group.index] = bool(
                out.loc[lars, "mean_gain"].max() >= out.loc[ev, "mean_gain"].max()
            )
    out["H3_rank_ok"] = rank_ok
    out["lars_is_max"] = lars_max
    return out


def h3_backbone_ordering(df: pd.DataFrame, mode: str = "lora") -> pd.DataFrame:
    """H3's backbone claim: the test-time-register gain is larger for CLIP than for DINOv2."""
    if df.empty:
        return pd.DataFrame(columns=_H3_BACKBONE_COLUMNS)
    rows = []
    for (ds, head), g in df.groupby(["dataset", "head"]):
        gains = {}
        for fam in ("clipb", "vits", "vitb"):
            r = paired(
                g[g["fam"] == fam],
                "final_miou",
                {"mode": mode, "registers": "test_time"},
                {"mode": mode, "registers": "none"},
            )
            gains[fam] = r["mean_diff"]
        dino = [gains[f] for f in ("vits", "vitb") if not math.isnan(gains[f])]
        larger = (
            None if math.isnan(gains["clipb"]) or not dino else bool(gains["clipb"] > max(dino))
        )
        rows.append(
            {
                "dataset": ds,
                "head": head,
                "gain_clipb": gains["clipb"],
                "gain_vits": gains["vits"],
                "gain_vitb": gains["vitb"],
                "clip_gain_larger": larger,
            }
        )
    out = pd.DataFrame(rows, columns=_H3_BACKBONE_COLUMNS)
    return out.astype({"clip_gain_larger": object})


_MODE_ORDER = ("frozen", "lora", "full")
_REGISTER_ORDER = ("none", "test_time", "trained")


def _plot_miou(summary: pd.DataFrame, path: Path) -> None:
    """One panel per (head, dataset): mIoU by adaptation mode, grouped by register arm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heads = [h for h in ("linear", "mask") if h in set(summary["head"])]
    datasets = sorted(summary["dataset"].unique())
    modes = [m for m in _MODE_ORDER if m in set(summary["mode"])]
    registers = [r for r in _REGISTER_ORDER if r in set(summary["registers"])]
    fig, axes = plt.subplots(
        max(len(heads), 1),
        max(len(datasets), 1),
        figsize=(2.8 * max(len(datasets), 1) + 1.6, 2.4 * max(len(heads), 1) + 0.6),
        squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors
    width = 0.8 / max(len(registers), 1)
    handles: dict[str, object] = {}
    for i, head in enumerate(heads):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            panel = summary[(summary["head"] == head) & (summary["dataset"] == dataset)]
            for k, reg in enumerate(registers):
                xs, ys, errs = [], [], []
                for x, mode in enumerate(modes):
                    cell = panel[(panel["mode"] == mode) & (panel["registers"] == reg)]
                    if cell.empty:
                        continue
                    xs.append(x + (k - (len(registers) - 1) / 2) * width)
                    ys.append(cell.iloc[0]["final_miou_mean"])
                    # A NaN std means one seed, so matplotlib draws no whisker. Filling it with
                    # 0 would draw a flat cap that reads as a measured zero spread.
                    errs.append(cell.iloc[0]["final_miou_std"])
                if not xs:
                    continue
                bars = ax.bar(xs, ys, width=width, yerr=errs, capsize=2, color=colors[k % 10])
                handles.setdefault(reg, bars)
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, fontsize=7)
            ax.set_title(f"{dataset} / {head}", fontsize=8)
            if j == 0:
                ax.set_ylabel("mIoU (mean over seeds, bar = 1 sample std)", fontsize=7)
    if handles:
        fig.legend(
            list(handles.values()),
            list(handles.keys()),
            loc="lower center",
            ncol=len(handles),
            fontsize=7,
        )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_table(df: pd.DataFrame, out: Path, stem: str, note: str | None = None) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    to_markdown(df, out / f"{stem}.md", note=note)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate results/<run_id>/ into the paper's tables, hypothesis verdicts and figure."
        )
    )
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = collect(a.results)
    if df.empty:
        print("no finished runs under", a.results)
        return
    s = summarize(df)
    _write_table(s, out, "summary", _SUMMARY_NOTE)
    _write_table(h1(df), out, "h1", _TEST_NOTE)
    _write_table(h2(df), out, "h2", _TEST_NOTE)
    _write_table(h3(a.results, df), out, "h3", _H3_NOTE)
    _write_table(h3_backbone_ordering(df), out, "h3_backbones", _H3_BACKBONE_NOTE)
    _plot_miou(s, out / "miou_by_cell.png")
    print(f"{len(df)} runs -> {out}")


if __name__ == "__main__":
    main()
