# Data

All datasets live under `data/` (gitignored). Layouts the loaders expect:

## ADE20K (SceneParsing 2016)
`scripts/download_ade20k.ps1` -> `data/ade20k/ADEChallengeData2016/{images,annotations}/{training,validation}/`.
150 classes; raw label 0 is ignored (mapped to 255), classes 1..150 become 0..149.
Background classes used for the H3 homogeneity fraction: wall, sky, floor, ceiling, road, water, sea.
The background class ids were verified against `objectInfo150.txt` (wall, sky, floor, ceiling, road, water, sea).
The `objectInfo150.txt` and `sceneCategories.txt` files sit next to `images/` in the extracted archive.
Run `scripts/download_ade20k.ps1` from the repository root; paths are relative.

## Calibration images
`data/calib/`: any 64+ natural JPEGs used by `scripts/reproduce_ttr.py`. Copy 200 ADE20K
validation images there once ADE20K is downloaded.

## Evaluation protocol (all arms, all datasets)
Resize the shorter side to `data.img_size`, centre crop to a square, no test-time augmentation.
Training uses RandomResizedCrop(scale 0.25-1.0) and horizontal flip. `DataCfg.mean`/`std` default
to ImageNet statistics; the runner (`ttr.run`) overrides them from the backbone's timm
`pretrained_cfg` via `normalization_for` before building any dataset and records the resolved
values in the run's `config.yaml`, so CLIP runs use CLIP's own statistics.

## Cityscapes
Register at cityscapes-dataset.com, download `leftImg8bit_trainvaltest.zip` and
`gtFine_trainvaltest.zip`, extract to `data/cityscapes/` so that
`data/cityscapes/leftImg8bit/train/<city>/*.png` exists. 19 train ids; everything else is 255.
Background classes for H3: road, wall, sky.

## LaRS
Register at lojzezust.github.io/lars-dataset, download images and semantic masks, place as
`data/lars/{train,val}/{images,semantic_masks}/`. Run `python scripts/inspect_lars.py data/lars`
and confirm the mask values match `LARS_RAW` in `ttr/data/lars.py`; edit the dict if not.
Classes: obstacle 0, water 1, sky 2. Background classes for H3: water, sky.
Our water-edge F1 and obstacle F1 are proxies; report official numbers from the LaRS toolkit
in the paper.
