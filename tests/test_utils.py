import json
from pathlib import Path

import numpy as np
import torch

from ttr.utils import (
    Timer,
    append_csv_row,
    get_device,
    make_run_dir,
    read_json,
    seed_everything,
    write_json,
)


def test_seed_makes_torch_and_numpy_deterministic():
    seed_everything(123)
    a = torch.randn(3)
    n1 = np.random.rand(2)
    seed_everything(123)
    b = torch.randn(3)
    n2 = np.random.rand(2)
    assert torch.equal(a, b)
    assert (n1 == n2).all()


def test_get_device_returns_a_device():
    d = get_device()
    assert d.type in {"cuda", "cpu"}


def test_make_run_dir_creates_nested_and_is_idempotent(tmp_results: Path):
    p = make_run_dir(tmp_results, "exp/one")
    assert p.is_dir()
    assert make_run_dir(tmp_results, "exp/one") == p


def test_json_roundtrip(tmp_path: Path):
    p = tmp_path / "m.json"
    write_json({"miou": 0.5, "k": [1, 2]}, p)
    assert read_json(p) == {"miou": 0.5, "k": [1, 2]}
    assert json.loads(p.read_text())["miou"] == 0.5


def test_write_json_maps_non_finite_floats_to_null(tmp_path: Path):
    p = tmp_path / "nan.json"
    write_json({"a": float("nan"), "b": [1.0, float("inf")]}, p)
    assert read_json(p) == {"a": None, "b": [1.0, None]}
    assert "NaN" not in p.read_text() and "Infinity" not in p.read_text()


def test_append_csv_row_writes_header_once(tmp_path: Path):
    p = tmp_path / "log.csv"
    append_csv_row(p, {"epoch": 0, "loss": 1.5})
    append_csv_row(p, {"epoch": 1, "loss": 1.0})
    lines = p.read_text().strip().splitlines()
    assert lines[0] == "epoch,loss"
    assert len(lines) == 3


def test_timer_measures_positive_seconds():
    with Timer() as t:
        sum(range(1000))
    assert t.seconds >= 0.0
