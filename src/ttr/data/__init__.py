"""Dataset registry. build_dataset(cfg, split) is the only entry point the runner uses."""

from __future__ import annotations

from torch.utils.data import Dataset

from ttr.config import DataCfg
from ttr.data.ade20k import ADE20K_BACKGROUND_IDS, ADE20K_NUM_CLASSES, ade20k_label_fn, ade20k_pairs
from ttr.data.folder import SegFolderDataset
from ttr.data.synthetic import SYNTHETIC_NUM_CLASSES, SyntheticSegDataset
from ttr.data.transforms import eval_transform, train_transform

_NUM_CLASSES = {"ade20k": ADE20K_NUM_CLASSES, "synthetic": SYNTHETIC_NUM_CLASSES}
_BACKGROUND = {"ade20k": ADE20K_BACKGROUND_IDS, "synthetic": [0]}


def num_classes(name: str) -> int:
    return _NUM_CLASSES[name]


def background_class_ids(name: str) -> list[int]:
    return list(_BACKGROUND[name])


def build_dataset(cfg: DataCfg, split: str) -> Dataset:
    if split not in ("train", "val"):
        raise ValueError(split)
    tf = (train_transform if split == "train" else eval_transform)(cfg.img_size, cfg.mean, cfg.std)
    if cfg.name == "synthetic":
        n = 32 if split == "train" else 8
        seed = 0 if split == "train" else 1
        return SyntheticSegDataset(n, cfg.img_size, seed=seed)
    if cfg.name == "ade20k":
        imgs, labs = ade20k_pairs(cfg.root, split)
        return SegFolderDataset(imgs, labs, tf, ade20k_label_fn)
    raise KeyError(f"unknown dataset {cfg.name!r}")
