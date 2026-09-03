"""Seeding, device selection, result-directory helpers, and a wall-clock timer."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_dir(out_dir: str | Path, run_id: str) -> Path:
    p = Path(out_dir) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize_non_finite(obj: Any) -> Any:
    """Recursively map NaN/Inf/-Inf floats to None so no bare non-finite value reaches disk."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_non_finite(obj)
    Path(path).write_text(json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


class Timer:
    """`with Timer() as t: ...; t.seconds`. Synchronises CUDA so GPU work is counted."""

    def __enter__(self) -> Timer:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._t0
