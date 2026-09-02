from pathlib import Path

import pytest


@pytest.fixture
def tmp_results(tmp_path: Path) -> Path:
    """An empty results root for tests that write run directories."""
    d = tmp_path / "results"
    d.mkdir()
    return d
