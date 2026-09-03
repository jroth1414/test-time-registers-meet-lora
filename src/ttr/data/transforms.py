"""Joint image/label transforms built on torchvision transforms.v2."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2


class _Joint:
    def __init__(self, pipeline: v2.Compose) -> None:
        self.pipeline = pipeline

    def __call__(self, img: Image.Image, label: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        im = tv_tensors.Image(v2.functional.pil_to_tensor(img.convert("RGB")))
        lb = tv_tensors.Mask(torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0))
        im, lb = self.pipeline(im, lb)
        return im.as_subclass(torch.Tensor), lb.squeeze(0).long().as_subclass(torch.Tensor)


def _finish(mean, std) -> list:
    return [
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None},
                   scale=True),
        v2.Normalize(mean, std),
    ]


def train_transform(img_size: int, mean, std) -> _Joint:
    return _Joint(v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(0.25, 1.0), ratio=(0.75, 1.333), antialias=True),
        v2.RandomHorizontalFlip(),
        *_finish(mean, std),
    ]))


def eval_transform(img_size: int, mean, std) -> _Joint:
    return _Joint(v2.Compose([
        v2.Resize(img_size, antialias=True),        # shorter side -> img_size
        v2.CenterCrop(img_size),
        *_finish(mean, std),
    ]))
