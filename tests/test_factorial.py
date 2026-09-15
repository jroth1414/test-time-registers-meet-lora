from pathlib import Path

import yaml

from scripts.make_factorial import CELLS, write_configs
from ttr.config import load_config


def test_factorial_cells_and_files(tmp_path: Path):
    paths = write_configs(tmp_path, dataset="ade20k", seeds=[0, 1, 2], epochs=10)
    # 3 backbones x 3 modes x 3 register conditions, minus CLIP trained
    # (3 cells) = 24 cells x 3 seeds
    assert len(paths) == 72 and len(CELLS) == 24
    cfg = yaml.safe_load(paths[0].read_text())
    assert cfg["run_id"].startswith("ade20k__")
    assert not any("__mask" in p.name for p in paths)
    trained = [p for p in paths if "__trained__" in p.name]
    assert all("reg4" in yaml.safe_load(p.read_text())["backbone"]["name"] for p in trained)
    assert not any("clip" in p.name and "__trained__" in p.name for p in paths)
    vits_path = next(p for p in paths if "__vits__frozen__none__s0" in p.name)
    vits = yaml.safe_load(vits_path.read_text())
    assert vits["backbone"]["outlier_layer"] == 9

    tt_paths = [p for p in paths if "__test_time__" in p.name]
    assert tt_paths
    for p in tt_paths:
        reg_path = yaml.safe_load(p.read_text())["backbone"]["register_neuron_path"]
        assert reg_path.startswith("artifacts/res224/register_neurons/")
        assert Path(reg_path).exists(), reg_path


def test_factorial_mask_head_suffix(tmp_path):
    paths = write_configs(tmp_path, dataset="ade20k", seeds=[0], epochs=1, head="mask")
    assert all(p.name.endswith("__mask.yaml") for p in paths)
    cfg = load_config(str(paths[0]))
    assert cfg.head.type == "mask"


def test_factorial_high_res_datasets_use_448(tmp_path):
    p = write_configs(tmp_path, dataset="lars", seeds=[0], epochs=1, root="data/lars")[0]
    cfg = yaml.safe_load(p.read_text())
    assert cfg["data"]["img_size"] == 448 and cfg["backbone"]["img_size"] == 448
    assert cfg["data"]["batch_size"] == 8


def test_factorial_lars_test_time_uses_res448_neuron_maps(tmp_path):
    paths = write_configs(tmp_path, dataset="lars", seeds=[0], epochs=1, root="data/lars")
    tt_paths = [p for p in paths if "__test_time__" in p.name]
    assert tt_paths
    for p in tt_paths:
        reg_path = yaml.safe_load(p.read_text())["backbone"]["register_neuron_path"]
        assert "artifacts/res448/" in reg_path
        assert Path(reg_path).exists(), reg_path
