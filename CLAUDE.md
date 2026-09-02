# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Semester research project for JHU EN.705.744 *Deep Learning using Transformers* (Fall 2026), by John Roth.
The project is a single empirical study, not a product: it must produce a rubric-graded paper by
2026-12-07 and a workshop-ready manuscript on the same experiments.

**Research question.** Pretrained ViTs (DINOv2, CLIP) grow high-norm "outlier" patch tokens in
low-information regions (sky, water, walls) that corrupt dense features. Darcet et al. fix this by
retraining with register tokens; Jiang et al. (NeurIPS 2025, arXiv:2506.08010) fix it training-free
by moving the activations of a few "register neurons" into appended *test-time registers*. Nobody
has checked what happens under LoRA fine-tuning. We test three hypotheses:

| | Hypothesis | Falsified if |
|---|---|---|
| H1 | LoRA leaves outlier tokens in place; vanilla LoRA stays below a trained-register backbone on mIoU | outlier-token fraction after LoRA moves more than 20% from the frozen model, or LoRA matches the registered backbone |
| H2 | Test-time registers + LoRA close at least 50% of that mIoU gap at zero extra parameters | gap closure under 50% |
| H3 | Gain grows with homogeneous-background fraction; largest on maritime imagery (LaRS), larger for CLIP than DINOv2 | no correlation with background fraction, or ordering reversed |

The authoritative statement of the design is `proposal/proposal.tex`. The reasoning behind the
choice, and the alternatives rejected, live in `research_ideas.md` (Ideas A-G) and
`research_ideas_applied.md` (Ideas H-N). This project is Idea A with LaRS added as the third dataset.

## Hard dates

| Date | Milestone |
|---|---|
| 2026-09-07 week | Topic submitted (the one-page proposal) |
| 2026-10-19 week | Paper draft due: needs the ADE20K factorial done |
| late Oct / early Nov 2026 | MaCVi workshop at WACV 2027 deadline (LaRS slice only); not yet posted at macvi.org/workshop/macvi27 |
| 2026-12-07 | Final course paper due |
| Feb-Mar 2027 | CVPR 2027 workshop deadlines (full study) |
| ~June 2027 | WACV 2028 round 1 (fallback for the full study) |

WACV 2027 main track is closed (round 2 was 2026-08-28). Do not plan around it.

## Repository layout

```
CourseOutline_705.744.8X.FA26.pdf   course calendar (read-only input, do not edit)
project_rubric_transformers.png     grading rubric (read-only input)
research_ideas.md                   idea shortlist A-G with venue/timing analysis
research_ideas_applied.md           applied ideas H-N tied to CVPR/WACV workshops
proposal/proposal.tex               one-page proposal (LaTeX + TikZ block diagram)
proposal/proposal.pdf               compiled proposal, the file that gets uploaded
```

There is no experiment code yet. When it is added, keep this shape so the paper's factorial maps
onto the tree: `src/` (backbone wrappers, register-neuron detection, LoRA injection, seg heads,
metrics), `configs/` (one YAML per factorial cell), `scripts/` (train/eval entry points),
`results/` (committed CSVs and plots only), `data/` (gitignored). Record the exact commands here
once they exist; do not leave future sessions to guess them.

## Commands

**Build the proposal** (from `proposal/`):

```
pdflatex -interaction=nonstopmode -halt-on-error proposal.tex
rm -f proposal.aux proposal.out proposal.log
```

The proposal must stay **exactly one page**. Check the log line `Output written on proposal.pdf (1 page`
after every edit. If it spills, trim prose before touching the layout: the geometry, `\small`
body, and two-column references are already at their limits. pandoc, LibreOffice and wkhtmltopdf
are not installed; pdflatex/xelatex and Chrome are.

**Environment facts:** Windows 11, PowerShell primary with Git Bash available, Python 3.11.9,
git 2.55, one NVIDIA RTX 5070 Ti (16 GB). The proposal says "one 24 GB GPU"; the local card is
16 GB, so ViT-B experiments need gradient checkpointing or cloud bursts, and ViT-L is out of scope
locally.

## Experimental design (what the code must implement)

The paper is a 3 x 3 factorial per backbone, run on three datasets, three seeds per cell:

- **Adaptation axis:** frozen backbone / LoRA on `W_Q, W_V` / full fine-tune.
- **Register axis:** none / test-time registers (Jiang et al.) / trained registers (only DINOv2
  ships `*_reg` checkpoints, which is why DINOv2 is the anchor backbone; CLIP has no trained-register
  arm, so its upper bound is missing by design).
- **Backbones:** DINOv2 ViT-S/14 and ViT-B/14 (`dinov2_vits14`, `dinov2_vitb14`, and the `_reg`
  variants), CLIP ViT-B/16. DINOv3 ViT-B/16 is a stretch goal; its checkpoints are gated behind a
  Meta license form on Hugging Face.
- **Heads:** linear probe (isolates feature quality) and a light Mask2Former-style head.
- **Datasets:** ADE20K (150 classes), Cityscapes (19 classes, registration required for download),
  LaRS (maritime panoptic, ~4k frames, registration required). LaRS is the H3 stress test; report
  its water-edge accuracy and obstacle F1 alongside mIoU.
- **Diagnostics that must be logged for every run:** fraction of patch tokens with norm above the
  calibration threshold tau, attention entropy of CLS/register tokens, images/s. H1 is decided by
  the outlier fraction, not by mIoU alone.
- **Sweeps:** LoRA rank {4, 8, 16}, target modules ({q,v} vs {q,k,v,o}), number of test-time
  registers {1, 4, 8}, LoRA layer subset, learning rate.

Sequence the work so the draft deadline is safe: reproduce Jiang et al. and the outlier diagnostics
first (weeks 1-2), then the full ADE20K factorial (weeks 3-6), then Cityscapes and LaRS.
The test-time-register step is training-free, so a frozen-backbone result on ADE20K should exist
before any LoRA training starts. External code to build on: `nickjiang2378/test-time-registers`
(official implementation), Hugging Face `peft` for LoRA, `timm`/`torch.hub` for DINOv2, `open_clip`
for CLIP.

## Rubric requirements the paper must visibly satisfy

The rubric (`project_rubric_transformers.png`) weights Intro and Hypotheses/Method at 26 points
each, and Related Work, Application, Conclusions at 16 each. Things that cost points if missing:

- A **block diagram** of the method (the TikZ figure in the proposal is the seed; keep it updated).
- The **math** for attention, the LoRA update `W + (alpha/r) B A`, and the register-neuron
  intervention, written out.
- **Hyper-parameter search and multi-seed statistics** (mean, std, paired tests) in the
  Application section. Single-seed tables will be marked down.
- Conclusions that answer each hypothesis by name and state why the study was worth running.
- References formatted properly; anchor papers [1] Darcet 2024, [2] Jiang 2025, [3] DINOv2,
  [4] DINOv3, [8] LaRS were verified online on 2026-09-02; the rest are standard citations.

## Writing conventions

Documents in this repo follow the `stop-slop` skill: active voice with a named actor, no adverbs,
no em dashes, no "not X but Y" contrasts, no announcement phrases. The proposal already went
through that pass; apply the same pass to the paper and to any README before committing.

## Git attribution

John Roth is the sole author of this repository. Never add a `Co-Authored-By: Claude` trailer, a
`Claude-Session` trailer, or a "Generated with Claude Code" line to any commit message or pull
request. `.claude/settings.json` sets `attribution.commit` and `attribution.pr` to empty strings
and `attribution.sessionUrl` to false so the harness does not append them; do not remove that
block. Commit only when asked.

## Tooling gotchas seen in this repo

- The Bash tool on this Windows setup mangles backslashes inside heredocs. Write or patch LaTeX
  with the Write/Edit tools, not with `cat <<EOF` or inline Python that embeds `\usepackage`.
- Reading `proposal.pdf` with the Read tool renders the page, which is the fastest way to confirm
  the one-page constraint and diagram placement after a rebuild.
