import math
import warnings
from pathlib import Path

from tests.fake_results import make_run, standard_tree
from ttr.analysis import collect, h1, h2, h3, main, paired, summarize


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


def test_h1_and_h2_verdicts_are_plain_python_types_across_mixed_groups(tmp_path: Path):
    # Three groups in one table so pandas cannot fall back on a homogeneous-dtype column:
    # vits gets True (ratio near 1, gap negative), vitb gets False (ratio far from 1),
    # clipb gets None (no frozen or trained arm at all).
    standard_tree(tmp_path)
    for s in (0, 1):
        make_run(tmp_path, "ade20k", "vitb", "frozen", "none", s, 0.30 + 0.01 * s, 0.050)
        make_run(tmp_path, "ade20k", "vitb", "lora", "none", s, 0.34 + 0.01 * s, 0.010)
        make_run(tmp_path, "ade20k", "vitb", "lora", "trained", s, 0.39 + 0.01 * s, 0.002)
        make_run(tmp_path, "ade20k", "clipb", "lora", "none", s, 0.30 + 0.01 * s, 0.048)
        make_run(tmp_path, "ade20k", "clipb", "lora", "test_time", s, 0.34 + 0.01 * s, 0.004)
    df = collect(tmp_path)

    r1 = h1(df)
    for _, row in r1.iterrows():
        assert type(row["H1"]) in (bool, type(None))
    r1 = r1.set_index("fam")
    assert r1.loc["vits", "H1"] is True
    assert abs(r1.loc["vitb", "outlier_ratio_lora_over_frozen"] - 0.20) < 1e-9
    assert r1.loc["vitb", "H1"] is False
    assert r1.loc["clipb", "H1"] is None

    r2 = h2(df)
    for _, row in r2.iterrows():
        assert type(row["H2"]) in (bool, type(None))


def test_paired_zero_variance_no_scipy_warning(tmp_path: Path):
    df = collect(standard_tree(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = paired(
            df,
            "final_miou",
            {"mode": "lora", "registers": "test_time"},
            {"mode": "lora", "registers": "none"},
        )
    assert r["t"] == math.inf
    assert r["p"] == 0.0
    assert r["n"] == 2


def test_h3_spearman_positive_when_gain_tracks_background(tmp_path: Path):
    per_none = [(0.5, 0.1), (0.5, 0.5), (0.5, 0.9)]
    per_tt = [(0.51, 0.1), (0.55, 0.5), (0.60, 0.9)]  # gain grows with bg fraction
    make_run(tmp_path, "lars", "vits", "lora", "none", 0, 0.5, 0.05, per_image=per_none)
    make_run(tmp_path, "lars", "vits", "lora", "test_time", 0, 0.55, 0.01, per_image=per_tt)
    df = collect(tmp_path)
    r = h3(tmp_path, df).iloc[0]
    assert r["spearman_rho"] > 0.99 and r["mean_gain"] > 0
    assert math.isclose(r["mean_bg_fraction"], 0.5)


def test_cli_writes_tables_and_figures(tmp_path: Path):
    standard_tree(tmp_path / "results")
    main(["--results", str(tmp_path / "results"), "--out", str(tmp_path / "out")])
    for f in ("summary.md", "summary.csv", "h1.md", "h2.md", "h3.md", "miou_by_cell.png"):
        assert (tmp_path / "out" / f).exists(), f


def test_cli_no_finished_runs_returns_without_writing_tables(tmp_path: Path):
    empty_results = tmp_path / "results"
    empty_results.mkdir()
    out = tmp_path / "out"
    main(["--results", str(empty_results), "--out", str(out)])
    assert not (out / "summary.md").exists()


def test_h3_keeps_images_with_nan_maritime_metrics(tmp_path: Path):
    # LaRS per_image.csv carries water_edge_f1/obstacle_f1, which metrics_lars.py leaves NaN
    # for an image with no water boundary or no obstacle pixels (a normal image, not bad data).
    # A blanket dropna() must not drop those rows from the Spearman sample.
    d_none = make_run(tmp_path, "lars", "vits", "lora", "none", 0, 0.5, 0.05)
    d_tt = make_run(tmp_path, "lars", "vits", "lora", "test_time", 0, 0.55, 0.01)
    (d_none / "per_image.csv").write_text(
        "index,miou,bg_fraction,water_edge_f1,obstacle_f1\n"
        "0,0.5,0.1,0.8,\n"
        "1,0.5,0.5,,0.6\n"
        "2,0.5,0.9,0.75,0.65\n"
    )
    (d_tt / "per_image.csv").write_text(
        "index,miou,bg_fraction,water_edge_f1,obstacle_f1\n"
        "0,0.51,0.1,0.8,\n"
        "1,0.55,0.5,,0.6\n"
        "2,0.60,0.9,0.75,0.65\n"
    )
    df = collect(tmp_path)
    r = h3(tmp_path, df).iloc[0]
    assert r["n_images"] == 3
    assert not math.isnan(r["spearman_rho"])


def test_h3_constant_gain_no_scipy_warning(tmp_path: Path):
    per_none = [(0.5, 0.1), (0.5, 0.5), (0.5, 0.9)]
    per_tt = [(0.55, 0.1), (0.55, 0.5), (0.55, 0.9)]  # constant gain: 0.05 everywhere
    make_run(tmp_path, "lars", "vits", "lora", "none", 0, 0.5, 0.05, per_image=per_none)
    make_run(tmp_path, "lars", "vits", "lora", "test_time", 0, 0.55, 0.01, per_image=per_tt)
    df = collect(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = h3(tmp_path, df).iloc[0]
    assert math.isnan(r["spearman_rho"])


def test_h3_rank_ok_none_without_per_image_evidence(tmp_path: Path):
    # standard_tree writes no per_image.csv anywhere, so its h3 row has no gains/backgrounds
    # to correlate; H3_rank_ok must be None (not False) rather than asserting a spurious verdict.
    df = collect(standard_tree(tmp_path))
    r = h3(tmp_path, df).iloc[0]
    assert math.isnan(r["mean_gain"]) and math.isnan(r["mean_bg_fraction"])
    assert r["H3_rank_ok"] is None
    assert type(r["H3_rank_ok"]) in (bool, type(None))
