# Plan progress

One line per finished chunk: date, chunk, commit, deviations.

- 2026-09-02 chunk 1 scaffold: 9b0f880 — torch 2.11.0+cu128 installed, CUDA confirmed on the RTX 5070 Ti; 11 tests pass; `tabulate` added to dependencies early (chunk 10 needs it); test for unknown config keys asserts `ConfigKeyError` instead of bare `Exception` to satisfy ruff B017.
- 2026-09-02 chunk 2 backbone: f37d758 (branch sdd/02-backbone) — subagent-driven; 21 backbone tests, 32 total. Deviations from the plan: `forward_tokens` now honours `model.grad_checkpointing` via timm `checkpoint_seq` (plan gap found in final review; plan text updated); `capture()` validates the target and fused-attention state before registering any hook via a module-level `_CAPTURE_TARGETS` map; dead `tt_reg_init` removed; extra guard tests. Real checkpoints verified by `scripts/smoke_backbone.py` (DINOv2-S/B with and without registers, CLIP ViT-B/16) and are now in the timm cache.
