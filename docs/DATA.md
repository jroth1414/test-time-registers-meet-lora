# Data

All datasets live under `data/` (gitignored). Layouts the loaders expect:

## ADE20K (SceneParsing 2016)
`scripts/download_ade20k.ps1` -> `data/ade20k/ADEChallengeData2016/{images,annotations}/{training,validation}/`.
150 classes; raw label 0 is ignored (mapped to 255), classes 1..150 become 0..149.
Background classes used for the H3 homogeneity fraction: wall, sky, floor, ceiling, road, water, sea.
The background class ids were verified against `objectInfo150.txt` (wall, sky, floor, ceiling, road, water, sea).
The `objectInfo150.txt` and `sceneCategories.txt` files sit next to `images/` in the extracted archive.

## Calibration images
`data/calib/`: any 64+ natural JPEGs used by `scripts/reproduce_ttr.py`. Copy 200 ADE20K
validation images there once ADE20K is downloaded.

## Evaluation protocol (all arms, all datasets)
Resize the shorter side to `data.img_size`, centre crop to a square, no test-time augmentation.
Training uses RandomResizedCrop(scale 0.25-1.0) and horizontal flip. Mean/std come from the
backbone's timm `pretrained_cfg` (ImageNet for DINOv2, CLIP's own for CLIP).

## Cityscapes and LaRS
See chunk 9 of the plan; both need registration on the dataset websites.
