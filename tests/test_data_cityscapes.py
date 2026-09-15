from pathlib import Path

import numpy as np

from tests.fixtures import write_fake_cityscapes
from ttr.config import DataCfg
from ttr.data import background_class_ids, build_dataset, num_classes
from ttr.data.cityscapes import cityscapes_label_fn, cityscapes_pairs


def test_label_fn_maps_to_train_ids():
    raw = np.array([[0, 7, 8], [23, 26, 33]], dtype=np.uint8)
    out = cityscapes_label_fn(raw)
    assert out.tolist() == [[255, 0, 1], [10, 13, 18]]


def test_pairs_and_registry(tmp_path: Path):
    root = write_fake_cityscapes(tmp_path, n=2)
    imgs, labs = cityscapes_pairs(root, "train")
    assert len(imgs) == 2 and labs[0].name.endswith("_gtFine_labelIds.png")
    ds = build_dataset(DataCfg(name="cityscapes", root=str(root), img_size=64), "val")
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and set(y.unique().tolist()) <= {0, 1, 10, 13, 18, 255}
    assert num_classes("cityscapes") == 19 and background_class_ids("cityscapes") == [0, 3, 10]
