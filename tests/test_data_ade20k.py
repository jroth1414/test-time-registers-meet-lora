from pathlib import Path

import numpy as np
import torch

from tests.fixtures import write_fake_ade20k
from ttr.data.ade20k import ADE20K_NUM_CLASSES, ade20k_label_fn, ade20k_pairs
from ttr.data.folder import SegFolderDataset
from ttr.data.transforms import eval_transform


def test_label_fn_shifts_and_ignores_zero():
    raw = np.array([[0, 1], [150, 7]], dtype=np.uint8)
    out = ade20k_label_fn(raw)
    assert out.tolist() == [[255, 0], [149, 6]]
    assert ADE20K_NUM_CLASSES == 150


def test_pairs_are_sorted_and_aligned(tmp_path: Path):
    root = write_fake_ade20k(tmp_path, n=3)
    imgs, labs = ade20k_pairs(root, "train")
    assert len(imgs) == len(labs) == 3
    assert all(img.stem == lab.stem for img, lab in zip(imgs, labs, strict=True))
    vi, vl = ade20k_pairs(root, "val")
    assert len(vi) == 3 and "validation" in str(vi[0])


def test_dataset_returns_normalised_image_and_mapped_label(tmp_path: Path):
    root = write_fake_ade20k(tmp_path, n=2)
    imgs, labs = ade20k_pairs(root, "train")
    ds = SegFolderDataset(imgs, labs, eval_transform(56, [0.5] * 3, [0.5] * 3), ade20k_label_fn)
    x, y = ds[0]
    assert x.shape == (3, 56, 56) and y.shape == (56, 56)
    assert y.dtype == torch.int64
    assert y.max() <= 255 and set(y.unique().tolist()) <= {0, 1, 2, 3, 255}
    assert len(ds) == 2


def test_pairs_mismatch_raises(tmp_path: Path):
    import pytest

    root = write_fake_ade20k(tmp_path, n=2)
    extra = root / "ADEChallengeData2016" / "images" / "training" / "ADE_training_99999999.jpg"
    src = root / "ADEChallengeData2016" / "images" / "training" / "ADE_training_00000000.jpg"
    extra.write_bytes(src.read_bytes())
    with pytest.raises(ValueError):
        ade20k_pairs(root, "train")
