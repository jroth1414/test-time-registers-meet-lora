from pathlib import Path

import numpy as np

from tests.fixtures import write_fake_lars
from ttr.config import DataCfg
from ttr.data import background_class_ids, build_dataset, num_classes
from ttr.data.lars import lars_label_fn, lars_pairs


def test_label_fn_identity_with_ignore():
    raw = np.array([[0, 1, 4], [2, 255, 200]], dtype=np.uint8)
    assert lars_label_fn(raw).tolist() == [[0, 1, 255], [2, 255, 255]]


def test_label_fn_raises_on_non_2d_array():
    import pytest

    raw = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="2-D"):
        lars_label_fn(raw)


def test_pairs_and_registry(tmp_path: Path):
    root = write_fake_lars(tmp_path)
    imgs, labs = lars_pairs(root, "val")
    assert len(imgs) == 2 and imgs[0].stem == labs[0].stem
    ds = build_dataset(DataCfg(name="lars", root=str(root), img_size=64), "val")
    _, y = ds[0]
    assert set(y.unique().tolist()) <= {0, 1, 2}
    assert num_classes("lars") == 3 and background_class_ids("lars") == [1, 2]
