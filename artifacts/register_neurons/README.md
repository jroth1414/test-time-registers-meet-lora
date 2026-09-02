# Register neurons

Produced by `scripts/reproduce_ttr.py`. One JSON per checkpoint: outlier threshold stats and
the (layer -> neuron indices) map. Reused by every `registers=test_time` run so that all seeds
and adaptation modes share one intervention.

Calibration and outlier-fraction measurement both use 64 images from `data/calib` (200 ADE20K
validation JPEGs), batch size 8, on one RTX 5070 Ti. Acceptance: outlier fraction after
`< 0.2 * before`.

| checkpoint | flags | outlier frac before | after | ratio | neurons |
|---|---|---|---|---|---|
| vit_small_patch14_dinov2.lvd142m | `--layer -2` | 0.0147 | 0.0013 | 0.090 | 19 |
| vit_base_patch14_dinov2.lvd142m | `--layer 8` | 0.0175 | 0.0035 | 0.199 | 37 |
| vit_base_patch16_clip_224.openai | `--img-size 224 --quantile 0.995 --max-neurons 200` | 0.0390 | 0.0002 | 0.004 | 185 |
| vit_small_patch14_reg4_dinov2.lvd142m (baseline) | (default) | 0.0000 | 0.0000 | n/a | n/a |

## Notes

- **Default flags (`--layer -1`, `--k 4.0`, `--quantile 0.999`) under-detect on this model
  family.** At the final residual layer the patch-norm distribution for DINOv2 is not
  heavy-tailed relative to its own median (layer 11 for ViT-S: median 19.72, p999 25.62, max
  27.05), so the MAD threshold at `k=4` finds ~0 outliers. Norm histograms per layer show the
  real outlier signature one to three blocks earlier: DINOv2-S at layer 10 (`--layer -2`) has
  median 6.74 vs max 28.81; DINOv2-B at layer 8 (`--layer 8`, i.e. `--layer -4` of 12) has
  median 8.43 vs max 428.91. Moving `--layer` earlier, as the task brief anticipates, was
  sufficient for both DINOv2 checkpoints.
- **CLIP ViT-B/16 needed the opposite move: `--layer -1` (the default) but a looser quantile.**
  Its outlier signature is sharpest at the last layer (layer 11: median 14.48, p99 80.33), but
  the default `--quantile 0.999, --max-neurons 64` only captured 37 of the relevant neurons,
  leaving `after/before = 0.249` (just above the 0.2 bar). Loosening to
  `--quantile 0.995 --max-neurons 200` picked up 185 neurons across layers 3-11 and drove the
  ratio to 0.004.
- **`vit_small_patch14_reg4_dinov2.lvd142m` (Step 3 baseline) is a reference, not a PASS/FAIL
  case.** Its trained register tokens already absorb the outlier patches: outlier fraction before
  is 0.0000 at default flags, so `find_register_neurons` returns an empty neuron map and there is
  nothing for test-time registers to redirect. This is the H1 comparison point: vanilla LoRA on a
  non-`_reg` backbone is expected to stay well above this near-zero baseline.
- `src/ttr/registers.py` was not modified; every model above passes (or, for the reg4 baseline,
  behaves as expected) using only the CLI knobs the task brief allows
  (`--k`, `--quantile`, `--max-neurons`, `--layer`).
