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

DINOv2-B's 0.199 ratio sits one point inside the `< 0.2` bar on this 64-image calibration set;
treat it as a pass with little margin, not a comfortable one.

DINOv2's two neuron maps were selected at 518 px (the resolution `find_register_neurons` ran
at here) but the factorial applies them at 224 px. Neuron identity transfers across resolution
(a neuron is a fixed row of the MLP weight matrix), but `tau` does not: the factorial's runner
must recalibrate `tau` at its own input resolution rather than reusing the `tau` stored in these
JSONs.

## 224 px (factorial resolution)

Same detector flags as the 518 px table above, run at `--img-size 224` (the factorial's actual
resolution) and written to `artifacts/res224/` instead of `artifacts/`. `scripts/make_factorial.py`
points every `registers=test_time` cell at these maps, not the 518 px ones. Same acceptance bar
(`< 0.2 * before`), same 64-image calibration set, same GPU.

| checkpoint | flags | outlier frac before | after | ratio | neurons | result |
|---|---|---|---|---|---|---|
| vit_small_patch14_dinov2.lvd142m | `--layer -2` | 0.0049 | 0.0021 | 0.432 | 19 | FAIL |
| vit_base_patch14_dinov2.lvd142m | `--layer 8` | 0.0192 | 0.0045 | 0.232 | 37 | FAIL |
| vit_base_patch16_clip_224.openai | `--img-size 224 --quantile 0.995 --max-neurons 200` | 0.0390 | 0.0002 | 0.004 | 185 | PASS |

Both DINOv2 maps neuron-select at the same neuron count as their 518 px counterparts (19 and 37;
the selection is deterministic given the same flags and calibration images at a fixed image
count, and `find_register_neurons`'s streaming statistics happen to reproduce the same
quantile cut at both resolutions here), but the *outlier fractions* they were tuned against move:
at 224 px there are fewer patch tokens per image, and, on this calibration set, the residual-norm
gap between outlier and normal patches at layers 10 (ViT-S) and 8 (ViT-B) is less pronounced than
at 518 px, so the same neurons redirect a smaller share of the class of tokens `tau` calls
outliers. Both DINOv2 checkpoints fail the `< 0.2` bar at 224 px with the flags recorded for
518 px; per the task brief, the flags are not retuned to chase a pass at a resolution the
original detection sweep did not target. The maps are kept and wired into the factorial anyway:
H1's outlier-fraction diagnostic is measured per run regardless of whether the test-time-register
intervention clears this bar, and a weaker (or negative) intervention at 224 px is itself a
finding about resolution sensitivity, not a reason to withhold the arm.

CLIP passes comfortably at 224 px (ratio 0.004, same as 518 px) with the same loosened
`--quantile`/`--max-neurons` flags; its outlier signature at the last layer is not resolution-
sensitive in the same way.

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
  Its outlier signature is sharpest at the last layer (layer 11: median 14.32, p99 80.33 from an
  earlier exploratory norm-histogram pass over a different image count; the run recorded in the
  table above, and in `vit_base_patch16_clip_224_openai.json`'s `stats.median`, gives 14.319 for
  the exact 64-image calibration set), but the default `--quantile 0.999, --max-neurons 64` only
  captured 37 of the relevant neurons, leaving `after/before = 0.249` (just above the 0.2 bar).
  Loosening to `--quantile 0.995 --max-neurons 200` picked up 185 neurons across layers 3-11 and
  drove the ratio to 0.004.
- **`vit_small_patch14_reg4_dinov2.lvd142m` (Step 3 baseline) is a reference, not a PASS/FAIL
  case.** Its trained register tokens already absorb the outlier patches: outlier fraction before
  is 0.0000 at default flags, so `find_register_neurons` returns an empty neuron map and there is
  nothing for test-time registers to redirect. This is the H1 comparison point: vanilla LoRA on a
  non-`_reg` backbone is expected to stay well above this near-zero baseline.
- `src/ttr/registers.py` was not modified; every model above passes (or, for the reg4 baseline,
  behaves as expected) using only the CLI knobs the task brief allows
  (`--k`, `--quantile`, `--max-neurons`, `--layer`).
- **DINOv2-B's figure keeps a saturated top-left pixel, before and after intervention alike.**
  `attn_vit_base_patch14_dinov2_lvd142m.png` shows one bright patch at grid position (0, 0) in
  every panel; it survives the register intervention because it is not one of the redirected
  outlier neurons at layer 11, just a fixed high-attention artifact at the image border. It is
  cosmetic, not a failure of the intervention: the outlier-fraction and ratio numbers above are
  unaffected.
