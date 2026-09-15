import math
import warnings
from pathlib import Path

import pytest

from tests.fake_results import make_run, standard_tree
from ttr.analysis import (
    _plot_miou,
    collect,
    h1,
    h2,
    h3,
    h3_backbone_ordering,
    main,
    paired,
    summarize,
    to_markdown,
)


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
    assert abs(row["final_miou"] - 0.381) < 1e-9 and row["head"] == "linear"
    assert abs(row["best_miou"] - 0.386) < 1e-9


def test_summarize_mean_std_n(tmp_path: Path):
    s = summarize(collect(standard_tree(tmp_path)))
    r = s[(s["mode"] == "lora") & (s["registers"] == "none")].iloc[0]
    assert abs(r["final_miou_mean"] - 0.345) < 1e-9 and r["n"] == 2 and r["best_miou_std"] > 0
    assert abs(r["best_miou_mean"] - 0.350) < 1e-9
    assert r["head_params"] == 10
    # trainable_params counts backbone parameters only, so the summary says so by name.
    assert "trainable_params" not in s.columns
    assert r["backbone_trainable_params"] == 100
    assert r["n_final_miou"] == 2


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
    assert "norm_ratio_p999_mean" in s.columns
    assert abs(s.iloc[0]["norm_ratio_p999_mean"] - 1.8) < 1e-9
    assert "outlier_fraction_recalibrated_mean" in s.columns
    assert s.iloc[0]["throughput_dtype"] == "bf16"


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
    # diffs 0.029 and 0.031: mean 0.03, sample std 0.001414, t 30, p 0.021213 (df = 1).
    assert r["n"] == 2 and abs(r["mean_diff"] - 0.03) < 1e-9
    assert r["zero_variance"] is False
    assert abs(r["t"] - 30.0) < 1e-6
    assert abs(r["p"] - 0.0212) < 5e-4
    assert r["ci95_low"] < 0.03 < r["ci95_high"]
    assert abs(r["ci95_low"] - 0.017294) < 1e-5 and abs(r["ci95_high"] - 0.042706) < 1e-5


def test_paired_negative_difference_keeps_sign_and_p(tmp_path: Path):
    df = collect(standard_tree(tmp_path))
    r = paired(
        df,
        "final_miou",
        {"mode": "lora", "registers": "none"},
        {"mode": "lora", "registers": "trained"},
    )
    # diffs -0.049 and -0.051: mean -0.05, t -50, p 0.012731 (df = 1).
    assert abs(r["mean_diff"] + 0.05) < 1e-9
    assert abs(r["t"] + 50.0) < 1e-6
    assert abs(r["p"] - 0.0127) < 5e-4
    assert r["zero_variance"] is False
    assert r["ci95_high"] < 0


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
    assert math.isnan(r["ci95_low"]) and math.isnan(r["ci95_high"])
    assert r["zero_variance"] is False


def test_h1_and_h2_verdicts(tmp_path: Path):
    df = collect(standard_tree(tmp_path))
    r1 = h1(df).iloc[0]
    assert abs(r1["outlier_ratio_lora_over_frozen"] - 0.96) < 1e-9
    assert r1["H1_outliers_persist"] is True
    assert abs(r1["miou_gap_lora_none_vs_trained"] + 0.05) < 1e-9
    assert abs(r1["p"] - 0.0127) < 5e-4
    assert not r1["zero_variance"]
    assert r1["ci95_high"] < 0
    assert r1["H1_gap_significant"] is True and r1["H1"] is True
    assert type(r1["H1"]) is bool
    r2 = h2(df).iloc[0]
    assert abs(r2["gap_closure"] - 0.6) < 1e-9 and r2["H2"] is True
    assert type(r2["H2"]) is bool
    assert r2["gap_exists"] is True
    assert r2["extra_params_tt"] == 0
    assert abs(r2["tt_minus_none"] - 0.03) < 1e-9
    assert abs(r2["tt_minus_trained"] + 0.02) < 1e-9
    # Both registered arms carry the same per-seed jitter, so their difference is constant
    # and the paired p value stays undefined rather than collapsing to 0.
    assert math.isnan(r2["p_tt_vs_trained"])


def test_h1_gap_not_significant_is_false_not_true(tmp_path: Path):
    # The trained arm beats LoRA-none by 0.001 mIoU, well inside the seed spread (p about 0.7).
    # H1 claims LoRA stays below a trained-register backbone, so an insignificant gap fails it.
    none = [0.340, 0.350, 0.360]
    trained = [0.345, 0.351, 0.357]
    for s in (0, 1, 2):
        make_run(tmp_path, "ade20k", "vits", "frozen", "none", s, 0.30 + 0.01 * s, 0.050)
        make_run(tmp_path, "ade20k", "vits", "lora", "none", s, none[s], 0.048)
        make_run(tmp_path, "ade20k", "vits", "lora", "trained", s, trained[s], 0.002)
    r = h1(collect(tmp_path)).iloc[0]
    assert r["H1_outliers_persist"] is True
    assert abs(r["miou_gap_lora_none_vs_trained"] + 0.001) < 1e-6
    assert abs(r["p"] - 0.707) < 1e-3
    assert r["n"] == 3
    assert r["ci95_low"] < 0 < r["ci95_high"]
    assert r["H1_gap_significant"] is False
    assert r["H1"] is False


def test_h2_no_gap_to_close_when_trained_arm_loses(tmp_path: Path):
    # The trained arm sits below LoRA-none, so the denominator of the closure fraction is
    # negative. A ratio there would report a "closure" above 1 for a test-time arm that also
    # lost mIoU, so h2 must refuse the verdict instead.
    for s in (0, 1):
        j = 0.001 if s else -0.001
        make_run(tmp_path, "ade20k", "vits", "lora", "none", s, 0.345 + j, 0.048)
        make_run(tmp_path, "ade20k", "vits", "lora", "test_time", s, 0.335 + j, 0.004)
        make_run(tmp_path, "ade20k", "vits", "lora", "trained", s, 0.335 + 2 * j, 0.002)
    r = h2(collect(tmp_path)).iloc[0]
    assert abs(r["miou_lora_none"] - 0.345) < 1e-9
    assert abs(r["miou_lora_tt"] - 0.335) < 1e-9
    assert abs(r["miou_lora_trained"] - 0.335) < 1e-9
    assert r["gap_exists"] is False
    assert math.isnan(r["gap_closure"])
    assert r["tt_minus_none"] < 0
    assert r["H2"] is None


def test_h1_and_h2_clip_no_trained_arm(tmp_path: Path):
    for s in (0, 1):
        make_run(tmp_path, "ade20k", "clipb", "lora", "none", s, 0.30 + 0.01 * s, 0.048)
        make_run(tmp_path, "ade20k", "clipb", "lora", "test_time", s, 0.34 + 0.01 * s, 0.004)
    df = collect(tmp_path)

    r1 = h1(df)
    assert len(r1) == 1
    row1 = r1.iloc[0]
    assert row1["H1"] is None
    assert row1["H1_outliers_persist"] is None
    assert row1["H1_gap_significant"] is None
    assert math.isnan(row1["miou_gap_lora_none_vs_trained"])

    r2 = h2(df)
    assert len(r2) == 1
    row2 = r2.iloc[0]
    assert row2["H2"] is None
    assert row2["gap_exists"] is None
    assert math.isnan(row2["gap_closure"])
    assert not math.isnan(row2["tt_minus_none"])
    assert abs(row2["tt_minus_none"] - 0.04) < 1e-9


def test_h1_and_h2_verdicts_are_plain_python_types_across_mixed_groups(tmp_path: Path):
    # Three groups in one table so pandas cannot fall back on a homogeneous-dtype column:
    # vits gets True (ratio near 1, gap negative), vitb gets False (ratio far from 1),
    # clipb gets None (no frozen or trained arm at all).
    standard_tree(tmp_path)
    for s in (0, 1):
        j = 0.001 if s else -0.001
        make_run(tmp_path, "ade20k", "vitb", "frozen", "none", s, 0.30 + 0.01 * s, 0.050)
        make_run(tmp_path, "ade20k", "vitb", "lora", "none", s, 0.34 + 0.01 * s, 0.010)
        make_run(tmp_path, "ade20k", "vitb", "lora", "trained", s, 0.39 + 0.01 * s + j, 0.002)
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
    # lora/none minus frozen/none is exactly 0.04 on both seeds: the t statistic diverges and
    # the p value is undefined. Reporting p = 0.0 would claim certainty the data cannot support.
    df = collect(standard_tree(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = paired(
            df,
            "final_miou",
            {"mode": "lora", "registers": "none"},
            {"mode": "frozen", "registers": "none"},
        )
    assert r["n"] == 2
    assert r["zero_variance"] is True
    assert r["t"] == math.inf
    assert math.isnan(r["p"])
    assert math.isnan(r["ci95_low"]) and math.isnan(r["ci95_high"])
    assert abs(r["mean_diff"] - 0.04) < 1e-9


def test_h3_spearman_positive_when_gain_tracks_background(tmp_path: Path):
    per_none = [(0.5, 0.1), (0.5, 0.5), (0.5, 0.9)]
    per_tt = [(0.51, 0.1), (0.55, 0.5), (0.60, 0.9)]  # gain grows with bg fraction
    make_run(tmp_path, "lars", "vits", "lora", "none", 0, 0.5, 0.05, per_image=per_none)
    make_run(tmp_path, "lars", "vits", "lora", "test_time", 0, 0.55, 0.01, per_image=per_tt)
    df = collect(tmp_path)
    r = h3(tmp_path, df).iloc[0]
    assert r["spearman_rho"] > 0.99 and r["mean_gain"] > 0
    assert math.isclose(r["mean_bg_fraction"], 0.5)


def test_collect_skips_run_ids_with_an_unknown_suffix(tmp_path: Path):
    # A sweep variant such as __r16 is a different configuration, not a factorial cell. Folding
    # it into the (dataset, fam, mode, registers, head) cell would average two designs together.
    standard_tree(tmp_path)
    stray = make_run(tmp_path, "ade20k", "vits", "lora", "none", 9, 0.40, 0.048)
    stray.rename(stray.with_name(stray.name + "__r16"))
    df = collect(tmp_path)
    assert len(df) == 8
    assert not df["run_id"].str.endswith("__r16").any()
    s = summarize(df)
    assert s[(s["mode"] == "lora") & (s["registers"] == "none")].iloc[0]["n"] == 2


def test_to_markdown_renders_missing_verdicts_and_appends_a_note(tmp_path: Path):
    for s in (0, 1):
        make_run(tmp_path, "ade20k", "clipb", "lora", "none", s, 0.30 + 0.01 * s, 0.048)
        make_run(tmp_path, "ade20k", "clipb", "lora", "test_time", s, 0.34 + 0.01 * s, 0.004)
    path = tmp_path / "out" / "h1.md"
    to_markdown(h1(collect(tmp_path)), path, note="Note text.")
    text = path.read_text()
    assert "n/a" in text
    assert "None" not in text and "nan" not in text
    assert text.rstrip().endswith("Note text.")


def test_cli_writes_tables_and_figures(tmp_path: Path):
    standard_tree(tmp_path / "results")
    main(["--results", str(tmp_path / "results"), "--out", str(tmp_path / "out")])
    for f in (
        "summary.md",
        "summary.csv",
        "h1.md",
        "h1.csv",
        "h2.md",
        "h2.csv",
        "h3.md",
        "h3.csv",
        "h3_backbones.md",
        "h3_backbones.csv",
        "miou_by_cell.png",
    ):
        assert (tmp_path / "out" / f).exists(), f
    assert "ddof" in (tmp_path / "out" / "summary.md").read_text()
    assert "two-sided paired t-test" in (tmp_path / "out" / "h1.md").read_text()


def test_plot_handles_a_single_seed_cell_without_a_zero_error_bar(tmp_path: Path):
    # One seed leaves final_miou_std NaN. Filling it with 0 would draw a flat cap that reads
    # as a measured zero spread, so the bar carries the NaN through to matplotlib.
    make_run(tmp_path, "ade20k", "vits", "frozen", "none", 0, 0.30, 0.050)
    make_run(tmp_path, "ade20k", "vits", "lora", "test_time", 0, 0.37, 0.004)
    s = summarize(collect(tmp_path))
    assert math.isnan(s.iloc[0]["final_miou_std"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _plot_miou(s, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 0


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


def test_h3_averages_gain_per_image_across_seeds(tmp_path: Path):
    # Two seeds over the same three validation images. Pooling the six image-seed pairs would
    # feed Spearman duplicated images and shrink its p value; h3 averages each image first.
    bg = (0.1, 0.5, 0.9)
    tt_by_seed = {0: (0.51, 0.55, 0.60), 1: (0.53, 0.57, 0.62)}
    for s in (0, 1):
        make_run(
            tmp_path,
            "lars",
            "vits",
            "lora",
            "none",
            s,
            0.5,
            0.05,
            per_image=[(0.5, b) for b in bg],
        )
        make_run(
            tmp_path,
            "lars",
            "vits",
            "lora",
            "test_time",
            s,
            0.55,
            0.01,
            per_image=list(zip(tt_by_seed[s], bg, strict=True)),
        )
    r = h3(tmp_path, collect(tmp_path)).iloc[0]
    assert r["n_images"] == 3
    assert r["n_seeds"] == 2
    # Seed-averaged gains 0.02, 0.06, 0.11 still grow with the background fraction.
    assert abs(r["mean_gain"] - (0.02 + 0.06 + 0.11) / 3) < 1e-9
    assert math.isclose(r["mean_bg_fraction"], 0.5)
    assert r["spearman_rho"] > 0.99


def test_h3_raises_when_per_image_records_disagree(tmp_path: Path):
    d_none = make_run(tmp_path, "lars", "vits", "lora", "none", 0, 0.5, 0.05)
    d_tt = make_run(tmp_path, "lars", "vits", "lora", "test_time", 0, 0.55, 0.01)
    (d_none / "per_image.csv").write_text("index,miou,bg_fraction\n0,0.5,0.1\n1,0.5,0.5\n")
    (d_tt / "per_image.csv").write_text("index,miou,bg_fraction\n0,0.51,0.1\n1,0.55,0.7\n")
    with pytest.raises(ValueError, match="not aligned"):
        h3(tmp_path, collect(tmp_path))


def test_h3_rank_ok_none_for_a_single_dataset(tmp_path: Path):
    # One dataset carries no ordering claim, however strong its own correlation is.
    per_none = [(0.5, 0.1), (0.5, 0.5), (0.5, 0.9)]
    per_tt = [(0.60, 0.1), (0.55, 0.5), (0.51, 0.9)]  # gain falls with bg
    make_run(tmp_path, "lars", "vits", "lora", "none", 0, 0.5, 0.05, per_image=per_none)
    make_run(tmp_path, "lars", "vits", "lora", "test_time", 0, 0.55, 0.01, per_image=per_tt)
    r = h3(tmp_path, collect(tmp_path)).iloc[0]
    assert r["spearman_rho"] < -0.99
    assert r["H3_rank_ok"] is None
    assert r["lars_is_max"] is None


def test_h3_rank_and_lars_max_across_two_datasets(tmp_path: Path):
    for ds, bgs, tts in (
        ("ade20k", (0.1, 0.2, 0.3), (0.51, 0.52, 0.53)),
        ("lars", (0.6, 0.7, 0.8), (0.55, 0.56, 0.57)),
    ):
        make_run(
            tmp_path,
            ds,
            "vits",
            "lora",
            "none",
            0,
            0.5,
            0.05,
            per_image=[(0.5, b) for b in bgs],
        )
        make_run(
            tmp_path,
            ds,
            "vits",
            "lora",
            "test_time",
            0,
            0.55,
            0.01,
            per_image=list(zip(tts, bgs, strict=True)),
        )
    out = h3(tmp_path, collect(tmp_path)).set_index("dataset")
    assert abs(out.loc["ade20k", "mean_gain"] - 0.02) < 1e-9
    assert abs(out.loc["lars", "mean_gain"] - 0.06) < 1e-9
    assert out.loc["ade20k", "H3_rank_ok"] is True
    assert out.loc["lars", "H3_rank_ok"] is True
    assert out.loc["ade20k", "lars_is_max"] is True
    assert out.loc["lars", "lars_is_max"] is True


def test_h3_backbone_ordering_compares_clip_against_dinov2(tmp_path: Path):
    for s in (0, 1):
        make_run(tmp_path, "ade20k", "clipb", "lora", "none", s, 0.30 + 0.01 * s, 0.048)
        make_run(tmp_path, "ade20k", "clipb", "lora", "test_time", s, 0.34 + 0.01 * s, 0.004)
        make_run(tmp_path, "ade20k", "vits", "lora", "none", s, 0.30 + 0.01 * s, 0.048)
        make_run(tmp_path, "ade20k", "vits", "lora", "test_time", s, 0.32 + 0.01 * s, 0.004)
    r = h3_backbone_ordering(collect(tmp_path)).iloc[0]
    assert r["dataset"] == "ade20k" and r["head"] == "linear"
    assert abs(r["gain_clipb"] - 0.04) < 1e-9
    assert abs(r["gain_vits"] - 0.02) < 1e-9
    assert math.isnan(r["gain_vitb"])
    assert r["clip_gain_larger"] is True


def test_h3_rank_ok_none_without_per_image_evidence(tmp_path: Path):
    # standard_tree writes no per_image.csv anywhere, so its h3 row has no gains/backgrounds
    # to correlate; H3_rank_ok must be None (not False) rather than asserting a spurious verdict.
    df = collect(standard_tree(tmp_path))
    r = h3(tmp_path, df).iloc[0]
    assert math.isnan(r["mean_gain"]) and math.isnan(r["mean_bg_fraction"])
    assert r["H3_rank_ok"] is None
    assert type(r["H3_rank_ok"]) in (bool, type(None))
