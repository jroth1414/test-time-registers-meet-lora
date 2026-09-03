from pathlib import Path

import yaml

from scripts.make_factorial import CELLS, write_configs


def test_factorial_cells_and_files(tmp_path: Path):
    paths = write_configs(tmp_path, dataset="ade20k", seeds=[0, 1, 2], epochs=10)
    # 3 backbones x 3 modes x 3 register conditions, minus CLIP trained
    # (3 cells) = 24 cells x 3 seeds
    assert len(paths) == 72 and len(CELLS) == 24
    cfg = yaml.safe_load(paths[0].read_text())
    assert cfg["run_id"].startswith("ade20k__")
    trained = [p for p in paths if "__trained__" in p.name]
    assert all("reg4" in yaml.safe_load(p.read_text())["backbone"]["name"] for p in trained)
    assert not any("clip" in p.name and "__trained__" in p.name for p in paths)
    vits_path = next(p for p in paths if "__vits__frozen__none__s0" in p.name)
    vits = yaml.safe_load(vits_path.read_text())
    assert vits["backbone"]["outlier_layer"] == 10

    tt_paths = [p for p in paths if "__test_time__" in p.name]
    assert tt_paths
    for p in tt_paths:
        reg_path = yaml.safe_load(p.read_text())["backbone"]["register_neuron_path"]
        assert reg_path.startswith("artifacts/res224/register_neurons/")
        assert Path(reg_path).exists(), reg_path
