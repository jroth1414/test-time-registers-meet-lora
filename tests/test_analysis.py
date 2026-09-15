import math
from pathlib import Path

from tests.fake_results import make_run, standard_tree
from ttr.analysis import collect, h1, h2, paired, summarize


def test_collect_flattens_runs(tmp_path: Path):
    df = collect(standard_tree(tmp_path))
    assert len(df) == 8
    assert {
        "dataset",
        "fam",
        "mode",
        "registers",
        "seed",
        "best_miou",
        "outlier_fraction",
        "attn_entropy_cls",
    } <= set(df.columns)
    row = df[(df["mode"] == "lora") & (df["registers"] == "test_time") & (df["seed"] == 1)].iloc[0]
    assert abs(row["final_miou"] - 0.38) < 1e-9 and row["head"] == "linear"
    assert abs(row["best_miou"] - 0.385) < 1e-9


def test_summarize_mean_std_n(tmp_path: Path):
    s = summarize(collect(standard_tree(tmp_path)))
    r = s[(s["mode"] == "lora") & (s["registers"] == "none")].iloc[0]
    assert abs(r["final_miou_mean"] - 0.345) < 1e-9 and r["n"] == 2 and r["best_miou_std"] > 0
    assert abs(r["best_miou_mean"] - 0.350) < 1e-9
    assert r["head_params"] == 10


def test_collect_forwards_extra_diagnostics_and_metrics(tmp_path: Path):
    make_run(
        tmp_path,
        "lars",
        "vits",
        "lora",
        "test_time",
        0,
        0.5,
        0.01,
        diag_extra={
            "norm_ratio_p999": 1.8,
            "norm_ratio_max": 2.9,
            "outlier_fraction_last_layer": 0.0,
            "outlier_fraction_recalibrated": 0.05,
            "throughput_dtype": "bf16",
        },
        metrics_extra={"extra_water_edge_f1": 0.7},
    )
    df = collect(tmp_path)
    row = df.iloc[0]
    assert row["norm_ratio_p999"] == 1.8
    assert row["norm_ratio_max"] == 2.9
    assert row["outlier_fraction_last_layer"] == 0.0
    assert row["outlier_fraction_recalibrated"] == 0.05
    assert row["throughput_dtype"] == "bf16"
    assert row["extra_water_edge_f1"] == 0.7

    s = summarize(df)
    assert "extra_water_edge_f1_mean" in s.columns


def test_collect_skips_stray_files_and_sanity_dir(tmp_path: Path):
    standard_tree(tmp_path)
    (tmp_path / "notes.txt").write_text("not a run")
    sanity = tmp_path / "sanity"
    make_run(sanity, "ade20k", "vits", "frozen", "none", 0, 0.1, 0.5)
    df = collect(tmp_path)
    assert len(df) == 8
    assert not (df["run_id"] == "sanity").any()


def test_paired_over_seeds(tmp_path: Path):
    df = collect(standard_tree(tmp_path))
    r = paired(
        df,
        "final_miou",
        {"mode": "lora", "registers": "test_time"},
        {"mode": "lora", "registers": "none"},
    )
    assert r["n"] == 2 and abs(r["mean_diff"] - 0.03) < 1e-9 and 0 <= r["p"] <= 1


def test_paired_empty_cell_returns_zero_n_and_nans(tmp_path: Path):
    df = collect(standard_tree(tmp_path))
    r = paired(
        df,
        "final_miou",
        {"mode": "lora", "registers": "none"},
        {"mode": "lora", "registers": "does_not_exist"},
    )
    assert r["n"] == 0
    assert math.isnan(r["mean_diff"])
    assert math.isnan(r["t"])
    assert math.isnan(r["p"])


def test_h1_and_h2_verdicts(tmp_path: Path):
    df = collect(standard_tree(tmp_path))
    r1 = h1(df).iloc[0]
    assert abs(r1["outlier_ratio_lora_over_frozen"] - 0.96) < 1e-9
    assert r1["miou_gap_lora_none_vs_trained"] < 0 and r1["H1"] is True
    assert type(r1["H1"]) is bool
    r2 = h2(df).iloc[0]
    assert abs(r2["gap_closure"] - 0.6) < 1e-9 and r2["H2"] is True
    assert type(r2["H2"]) is bool


def test_h1_and_h2_clip_no_trained_arm(tmp_path: Path):
    for s in (0, 1):
        make_run(tmp_path, "ade20k", "clipb", "lora", "none", s, 0.30 + 0.01 * s, 0.048)
        make_run(tmp_path, "ade20k", "clipb", "lora", "test_time", s, 0.34 + 0.01 * s, 0.004)
    df = collect(tmp_path)

    r1 = h1(df)
    assert len(r1) == 1
    row1 = r1.iloc[0]
    assert row1["H1"] is None
    assert math.isnan(row1["miou_gap_lora_none_vs_trained"])

    r2 = h2(df)
    assert len(r2) == 1
    row2 = r2.iloc[0]
    assert row2["H2"] is None
    assert math.isnan(row2["gap_closure"])
    assert not math.isnan(row2["tt_minus_none"])
    assert abs(row2["tt_minus_none"] - 0.04) < 1e-9
