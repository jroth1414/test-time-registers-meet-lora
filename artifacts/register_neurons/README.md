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
| vit_small_patch14_dinov2.lvd142m | `--layer -3 --quantile 0.99 --max-neurons 200 --k 3` | 0.0098 | 0.0014 | 0.144 | 185 | PASS |
| vit_base_patch14_dinov2.lvd142m | `--layer 8 --quantile 0.995 --max-neurons 200 --k 4` | 0.0192 | 0.0005 | 0.025 | 185 | PASS |
| vit_base_patch16_clip_224.openai | `--img-size 224 --quantile 0.995 --max-neurons 200` | 0.0390 | 0.0002 | 0.004 | 185 | PASS |

Both DINOv2 rows were originally run at the flags carried over from 518 px (`--layer -2` for
ViT-S, `--layer 8` for ViT-B, both at the default `--quantile 0.999 --max-neurons 64 --k 4.0`)
and both failed the `< 0.2` bar at 224 px (ratios 0.432 and 0.232): at 224 px there are fewer
patch tokens per image, and the residual-norm gap between outlier and normal patches that those
flags were tuned to find at 518 px is less pronounced, so the same neurons redirect a smaller
share of the tokens `tau` calls outliers. The controller authorised one bounded round of
224 px-specific detector tuning to resolve this; see "224 px detector tuning" below. Both
DINOv2 checkpoints now pass at 224 px with resolution-specific flags: ViT-S needed an earlier
layer (`-3` instead of `-2`) as well as a looser quantile and stricter `k`; ViT-B needed only a
looser quantile at its original layer.

CLIP passes comfortably at 224 px (ratio 0.004, same as 518 px) with the same loosened
`--quantile`/`--max-neurons` flags; its outlier signature at the last layer is not resolution-
sensitive in the same way, so it was not part of this tuning round.

## 224 px detector tuning

One bounded round of hyper-parameter tuning, run on `data/calib` (200 ADE20K validation JPEGs,
64-image calibration subset, batch size 8, RTX 5070 Ti), scoped to the two DINOv2 checkpoints
that failed at 224 px with their 518 px flags. Grid: `--quantile` in {0.999, 0.995, 0.99} x
`--max-neurons` in {64, 128, 200} at the default layer and `k=4` (9 runs), then `--layer` and
`--k` varied around the best 2 quantile/max-neurons combos by ratio (6 more runs). 15 trials per
model, 30 total. Selection rule: lowest `after/before` ratio subject to `before >= 0.003` (so
the threshold is not trivially loose) and neurons `<= 200`; ties broken by fewer neurons. Chosen
rows are marked **bold**.

### vit_small_patch14_dinov2.lvd142m (default layer `-2`, default `k=4` unless noted)

| layer | quantile | max-neurons | k | before | after | ratio | neurons |
|---|---|---|---|---|---|---|---|
| -2 | 0.999 | 64 | 4 | 0.0049 | 0.0021 | 0.432 | 19 |
| -2 | 0.999 | 128 | 4 | 0.0049 | 0.0021 | 0.432 | 19 |
| -2 | 0.999 | 200 | 4 | 0.0049 | 0.0021 | 0.432 | 19 |
| -2 | 0.995 | 64 | 4 | 0.0049 | 0.0026 | 0.531 | 64 |
| -2 | 0.995 | 128 | 4 | 0.0049 | 0.0020 | 0.395 | 93 |
| -2 | 0.995 | 200 | 4 | 0.0049 | 0.0020 | 0.395 | 93 |
| -2 | 0.99 | 64 | 4 | 0.0049 | 0.0026 | 0.531 | 64 |
| -2 | 0.99 | 128 | 4 | 0.0049 | 0.0018 | 0.358 | 128 |
| -2 | 0.99 | 200 | 4 | 0.0049 | 0.0018 | 0.370 | 185 |
| -2 | 0.99 | 128 | 3 | 0.0127 | 0.0139 | 1.091 | 128 |
| -3 | 0.99 | 128 | 4 | 0.0028 | 0.0006 | 0.217 | 128 (before < 0.003, excluded) |
| -3 | 0.99 | 128 | 3 | 0.0098 | 0.0048 | 0.494 | 128 |
| -2 | 0.99 | 200 | 3 | 0.0127 | 0.0142 | 1.120 | 185 |
| -3 | 0.99 | 200 | 4 | 0.0028 | 0.0005 | 0.174 | 185 (before < 0.003, excluded) |
| **-3** | **0.99** | **200** | **3** | **0.0098** | **0.0014** | **0.144** | **185** |

### vit_base_patch14_dinov2.lvd142m (default layer `8`, default `k=4` unless noted)

| layer | quantile | max-neurons | k | before | after | ratio | neurons |
|---|---|---|---|---|---|---|---|
| 8 | 0.999 | 64 | 4 | 0.0192 | 0.0045 | 0.232 | 37 |
| 8 | 0.999 | 128 | 4 | 0.0192 | 0.0045 | 0.232 | 37 |
| 8 | 0.999 | 200 | 4 | 0.0192 | 0.0045 | 0.232 | 37 |
| 8 | 0.995 | 64 | 4 | 0.0192 | 0.0038 | 0.197 | 64 |
| 8 | 0.995 | 128 | 4 | 0.0192 | 0.0005 | 0.029 | 128 |
| **8** | **0.995** | **200** | **4** | **0.0192** | **0.0005** | **0.025** | **185** |
| 8 | 0.99 | 64 | 4 | 0.0192 | 0.0038 | 0.197 | 64 |
| 8 | 0.99 | 128 | 4 | 0.0192 | 0.0005 | 0.029 | 128 |
| 8 | 0.99 | 200 | 4 | 0.0192 | 0.0005 | 0.029 | 200 |
| 8 | 0.995 | 200 | 3 | 0.0240 | 0.0068 | 0.285 | 185 |
| 7 | 0.995 | 200 | 4 | 0.0127 | 0.0006 | 0.048 | 185 |
| 7 | 0.995 | 200 | 3 | 0.0201 | 0.0047 | 0.233 | 185 |
| 8 | 0.995 | 128 | 3 | 0.0240 | 0.0068 | 0.282 | 128 |
| 7 | 0.995 | 128 | 4 | 0.0127 | 0.0005 | 0.043 | 128 |
| 7 | 0.995 | 128 | 3 | 0.0201 | 0.0046 | 0.230 | 128 |

Both DINOv2 checkpoints now clear the `< 0.2` bar at 224 px: ViT-S at ratio 0.144 (previously
0.432) and ViT-B at ratio 0.025 (previously 0.232). Two ViT-S trials at `--layer -3` beat the
chosen ratio (0.217 and 0.174) but were excluded because their `before` (0.0028) falls under the
0.003 floor set to keep the threshold non-trivial; the chosen `k=3` variant at the same layer and
quantile clears the floor (`before=0.0098`) while still passing.

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
