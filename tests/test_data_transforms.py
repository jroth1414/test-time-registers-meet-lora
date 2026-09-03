import numpy as np
import torch
from PIL import Image

from ttr.data.transforms import eval_transform, train_transform

MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def _sample(h=80, w=120):
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    lab = np.zeros((h, w), dtype=np.int64)
    lab[:, w // 2 :] = 3
    lab[:5, :5] = 255
    return img, lab


def test_train_transform_shapes_and_label_values():
    tf = train_transform(56, MEAN, STD)
    img, lab = _sample()
    x, y = tf(img, lab)
    assert x.shape == (3, 56, 56) and x.dtype == torch.float32
    assert y.shape == (56, 56) and y.dtype == torch.int64
    assert set(y.unique().tolist()) <= {0, 3, 255}


def test_eval_transform_is_deterministic_and_centre_crops():
    tf = eval_transform(56, MEAN, STD)
    img, lab = _sample()
    x1, y1 = tf(img, lab)
    x2, y2 = tf(img, lab)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert x1.shape == (3, 56, 56) and y1.shape == (56, 56)
    # left half is class 0, right half class 3 after resize+centre crop
    assert (y1[:, :20] == 0).float().mean() > 0.9
    assert (y1[:, -20:] == 3).float().mean() > 0.9


def test_normalisation_applied():
    tf = eval_transform(56, MEAN, STD)
    img = Image.fromarray(np.full((60, 60, 3), 255, dtype=np.uint8))
    x, _ = tf(img, np.zeros((60, 60), dtype=np.int64))
    assert torch.allclose(x, torch.ones_like(x))  # (1 - 0.5) / 0.5
