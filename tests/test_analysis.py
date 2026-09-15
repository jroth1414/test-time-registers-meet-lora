from pathlib import Path

from tests.fake_results import make_run, standard_tree
from ttr.analysis import collect, summarize


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
    assert abs(row["best_miou"] - 0.38) < 1e-9 and row["head"] == "linear"


def test_summarize_mean_std_n(tmp_path: Path):
    s = summarize(collect(standard_tree(tmp_path)))
    r = s[(s["mode"] == "lora") & (s["registers"] == "none")].iloc[0]
    assert abs(r["best_miou_mean"] - 0.345) < 1e-9 and r["n"] == 2 and r["best_miou_std"] > 0
    assert abs(r["final_miou_mean"] - 0.345) < 1e-9


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
