from pathlib import Path

import torch

from tests.fixtures import write_fake_ade20k
from ttr.config import DataCfg
from ttr.data import background_class_ids, build_dataset, num_classes
from ttr.data.synthetic import SYNTHETIC_NUM_CLASSES, SyntheticSegDataset


def test_synthetic_is_deterministic_and_learnable_shape():
    a = SyntheticSegDataset(n=5, img_size=56, num_classes=4, seed=1)
    b = SyntheticSegDataset(n=5, img_size=56, num_classes=4, seed=1)
    x, y = a[2]
    assert x.shape == (3, 56, 56) and y.shape == (56, 56)
    assert torch.equal(x, b[2][0]) and torch.equal(y, b[2][1])
    assert y.min() >= 0 and y.max() < 4
    # image colour encodes the label so a 1x1 head can fit it
    for c in y.unique().tolist():
        assert x[:, y == c].std(dim=1).max() < 0.3


def test_registry_synthetic_and_num_classes():
    cfg = DataCfg(name="synthetic", img_size=56)
    tr, va = build_dataset(cfg, "train"), build_dataset(cfg, "val")
    assert len(tr) == 32 and len(va) == 8
    assert num_classes("synthetic") == SYNTHETIC_NUM_CLASSES == 4
    assert background_class_ids("synthetic") == [0]


def test_registry_ade20k(tmp_path: Path):
    root = write_fake_ade20k(tmp_path, n=2)
    cfg = DataCfg(name="ade20k", root=str(root), img_size=56)
    ds = build_dataset(cfg, "val")
    assert len(ds) == 2 and num_classes("ade20k") == 150
    assert 2 in background_class_ids("ade20k")  # sky


def test_registry_unknown_name():
    import pytest

    with pytest.raises(KeyError):
        build_dataset(DataCfg(name="nope"), "train")
