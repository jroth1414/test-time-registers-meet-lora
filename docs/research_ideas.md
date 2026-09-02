# Candidate Research Topics — EN.705.744 Deep Learning using Transformers (Fall 2026)

Goal: a vision-transformer project that (a) scores well on the course rubric and (b) is a credible
first draft of a WACV-style paper. Each idea below is written to slot directly into the required
proposal sections: problem / candidate solution / hypotheses, method + dataset, references.

## 0. Reality check on venue timing

| Venue | Deadline | Status on 2026-09-02 |
|---|---|---|
| WACV 2027 main track, round 1 | 2026-06-26 | passed |
| WACV 2027 main track, round 2 | 2026-08-28 | passed (decisions 2026-10-09) |
| WACV 2027 workshops (Jan 4-5, 2027) | set per workshop, e.g. CV4EO: 2026-10-02 | mostly too early; a few may run into Nov |
| CVPR 2027 main | ~2026-11-13 (unconfirmed) | before course paper is due (12/07) |
| CVPR 2027 workshops | ~Feb-Mar 2027 | realistic |
| ICCV 2027 main | ~Mar 2027 | realistic |
| WACV 2028 main, round 1 | ~late June 2027 | realistic and the natural "WACV" target |

Practical plan: treat the 12/07 course paper as a complete WACV-format draft (8 pages, two-column,
WACV template), then submit it to a CVPR 2027 workshop or WACV 2028 round 1 after one polish pass.
Design the experiments so the course version is already the paper: multiple seeds, ablations,
a real baseline, and a block diagram of the method.

Course milestones that shape scope: topic due week of 09/07, draft due week of 10/19 (six weeks
in), final due 12/07. That is ~13 weeks total with realistically ~8 weeks of compute. Every idea
below is sized for a single 16-24 GB GPU plus occasional cloud bursts.

## 1. Selection criteria used

- **Transformer-centric**: the contribution is about attention/tokens/adaptation, not just "apply ViT to dataset X".
- **Feasible**: frozen or PEFT-tuned public checkpoints (DINOv2/v3, CLIP, VideoMAE, RT-DETR, Point-MAE). No pretraining from scratch.
- **Falsifiable hypotheses**: each idea has a clear null result that would still be a publishable negative finding.
- **Rubric coverage**: math to write down (attention, low-rank updates, merging operators), a block diagram, hyperparameter sweeps, and statistics across seeds.
- **Maps to course modules** so you can reuse assignment code.

---

## Idea A — Register artifacts vs. parameter-efficient fine-tuning for dense prediction  (RECOMMENDED)

**Course modules:** 3 (ViT), 9 (DINO), 12 (LoRA), 6 (segmentation)

**Problem.** Pretrained ViTs (DINOv2, CLIP, OpenCLIP) develop a sparse set of high-norm "outlier"
patch tokens that act as global memory and corrupt attention/feature maps. Darcet et al. (ICLR 2024)
fix it by retraining with register tokens; Jiang et al. (NeurIPS 2025) show a *training-free*
fix by relocating the outlier neurons' activations into "test-time registers". Nobody has
studied what happens to these artifacts when the model is then adapted with LoRA/adapters for
a dense task, which is how practitioners actually use these backbones.

**Candidate solution.** Register-aware PEFT: (i) detect the outlier-neuron set once, (ii) inject
test-time registers, (iii) apply LoRA to attention projections for a segmentation head
(linear probe and a light Mask2Former-style head). Compare against LoRA without registers,
full fine-tuning, and DINOv2-with-trained-registers.

**Hypotheses.**
- H1: Artifacts survive LoRA fine-tuning (the outlier neurons are not in the low-rank update path), so mIoU with vanilla LoRA is bounded below the registered model.
- H2: Test-time registers + LoRA close at least 50% of the mIoU gap to the trained-register model at zero extra parameters.
- H3: The benefit grows with input resolution / patch count and is larger for CLIP than DINOv2.
- H4 (bonus): DINOv3 (trained with registers + Gram anchoring) has residual artifacts under PEFT that the same fix reduces.

**Method & data.** Backbones: DINOv2 ViT-S/14 and ViT-B/14 (with and without registers), CLIP
ViT-B/16, optionally DINOv3 ViT-B/16. Datasets: Pascal VOC 2012, ADE20K (150 cls), Cityscapes.
Metrics: mIoU, attention-map entropy, fraction of high-norm tokens, throughput. Sweeps: LoRA rank
{4,8,16}, target modules {q,v} vs {q,k,v,o}, number of registers {1,4,8}. 3 seeds.
Compute: light (frozen backbone, small heads). Code: github.com/nickjiang2378/test-time-registers + HF PEFT.

**Why it can publish.** Clean mechanistic question, training-free method, strong recent anchors,
cheap to reproduce. A negative result on H1 is still informative.

**Key references.**
- Darcet, Oquab, Mairal, Bojanowski. "Vision Transformers Need Registers." ICLR 2024. arXiv:2309.16588
- Jiang, Dravid, Efros, Darrell. "Vision Transformers Don't Need Trained Registers." NeurIPS 2025. arXiv:2506.08010
- Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision." TMLR 2024. arXiv:2304.07193
- Siméoni et al. "DINOv3." arXiv:2508.10104 (2025)
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685
- Cheng et al. "Masked-attention Mask Transformer for Universal Image Segmentation." CVPR 2022 (Mask2Former)

---

## Idea B — Where should LoRA go in a detection transformer? Cross-domain PEFT for DETR-family models

**Course modules:** 5 (DETR / deformable attention), 12 (LoRA), 4 (hierarchical backbones)

**Problem.** DETR-style detectors (Deformable DETR, DINO-DETR, RT-DETR) transfer poorly to
aerial/UAV or adverse-weather domains, and full fine-tuning per domain is expensive. PEFT for
detection transformers is under-studied compared to classification; the few existing works
(LoRA on DiffusionDet for aerial few-shot; LoRA-RT-DETR for ultrasound) pick a placement ad hoc.

**Candidate solution.** A systematic *placement and rank-allocation* study: LoRA in
(a) backbone attention, (b) encoder (deformable) attention, (c) decoder self-attention,
(d) decoder cross-attention + object queries + reference-point heads, and combinations, at a
fixed trainable-parameter budget. Add a simple budget-allocation rule (e.g., allocate rank by
per-module gradient norm at step 0) and test whether it beats uniform rank.

**Hypotheses.**
- H1: For domain shift with the same object categories (Cityscapes to Foggy Cityscapes), adapting only encoder attention recovers at least 90% of full-fine-tune AP at under 5% params.
- H2: For category-and-scale shift (COCO to VisDrone / DIOR), decoder cross-attention + query adaptation matters most; backbone-only LoRA underperforms.
- H3: Gradient-norm rank allocation beats uniform rank at equal budget by at least 1 AP.

**Method & data.** RT-DETR-R18/R50 (COCO-pretrained), Deformable DETR as second architecture.
Targets: VisDrone2019-DET, DIOR (remote sensing), Cityscapes to Foggy Cityscapes. Metrics:
AP, AP_small, params trained, GPU-hours, convergence epochs. 3 seeds for headline numbers.
Compute: medium. VisDrone is ~10k images; 12-epoch runs on one GPU are feasible.

**Risk.** UAV small-object detection is crowded with "improved RT-DETR" papers; keep the
contribution about *adaptation*, not a new detector.

**Key references.**
- Carion et al. "End-to-End Object Detection with Transformers." ECCV 2020 (DETR)
- Zhu et al. "Deformable DETR." ICLR 2021. arXiv:2010.04159
- Zhao et al. "DETRs Beat YOLOs on Real-time Object Detection." CVPR 2024 (RT-DETR). arXiv:2304.08069
- Zhang et al. "DINO: DETR with Improved DeNoising Anchor Boxes." ICLR 2023
- Hu et al. LoRA, ICLR 2022
- "Analyzing the Impact of Low-Rank Adaptation for Cross-Domain Few-Shot Object Detection in Aerial Images." arXiv:2504.06330 (2025)
- Xin et al. "Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey and Benchmark." IJCV 2026 / arXiv:2402.02242

---

## Idea C — Token merging inside the backbone of detection transformers with spatial un-merging

**Course modules:** 3, 4, 5

**Problem.** Token Merging (ToMe) gives ~2x ViT throughput for classification with no retraining,
and ALGM extends it to semantic segmentation. Detection is harder: the multi-scale neck and
deformable attention need a dense spatial grid, so tokens must be *un-merged* back onto the
grid, and merging small objects away is catastrophic.

**Candidate solution.** Training-free merging in a ViT backbone (ViTDet or a ViT-backbone
RT-DETR variant) with (i) a bipartite merge schedule that protects high-frequency /
high-attention-entropy tokens, (ii) an exact un-merge (scatter with source-index tracking) before
the FPN/neck, and (iii) an optional 1-epoch LoRA "repair" fine-tune.

**Hypotheses.**
- H1: Naive ToMe + un-merge costs more than 3 AP on COCO at 2x speedup; AP_small suffers most.
- H2: Attention-entropy-protected merging cuts that loss to under 1 AP at the same speedup, training-free.
- H3: A 1-epoch LoRA repair recovers the rest; the gap is object-size dependent.

**Method & data.** ViTDet-B (MAE-pretrained, COCO), evaluation on COCO val2017 and VisDrone
(small objects). Metrics: AP/AP_s/AP_m/AP_l, images/s, FLOPs. Sweeps: tokens merged per layer r,
protection threshold, merge similarity feature (keys vs. values). Compute: light for the training-free
part (inference only), medium for the repair fine-tune.

**Key references.**
- Bolya et al. "Token Merging: Your ViT But Faster." ICLR 2023. arXiv:2210.09461
- Norouzi et al. "ALGM: Adaptive Local-then-Global Token Merging for Efficient Semantic Segmentation with Plain Vision Transformers." CVPR 2024. arXiv:2406.09936
- Li et al. "Exploring Plain Vision Transformer Backbones for Object Detection." ECCV 2022 (ViTDet)
- "CubistMerge: Spatial-Preserving Token Merging for Diverse ViT Backbones." arXiv:2509.21764 (2025)
- Bonnaerens & Dambre. "Learned Thresholds Token Merging and Pruning for Vision Transformers." arXiv:2307.10780

---

## Idea D — Low-rank attention updates for continual test-time adaptation of ViTs

**Course modules:** 3, 12

**Problem.** Test-time adaptation (TENT, CoTTA, SAR) updates LayerNorm affine parameters with an
entropy objective. On ViTs this is unstable under continual/mixed shifts and small batches
(error accumulation, collapse). WACV 2026 accepted several TTA papers, so the venue fit is strong.

**Candidate solution.** Restrict the adaptable parameter set to a low-rank update on Q/V
projections (LoRA-TTA) with a periodic reset to identity and an entropy-plus-consistency
objective. Compare parameter subsets (LN vs Q/V-LoRA vs MLP-LoRA vs all) as the main axis.

**Hypotheses.**
- H1: Q/V low-rank updates match or beat LN-only adaptation on ImageNet-C at severity 5 with batch size 16.
- H2: They are more stable under continual shift (10 corruption types in sequence): lower variance across orderings and no collapse.
- H3: Rank 4 or lower suffices; gains saturate.

**Method & data.** ViT-B/16 (DeiT and AugReg weights), CIFAR-10-C/100-C for sweeps, ImageNet-C
for headline, ACDC for a segmentation extension if time permits. Metrics: error, ECE, forgetting
across the sequence. 5 seeds / orderings. Compute: light.

**Risk.** Crowded area; novelty must come from the *analysis of which ViT parameters to adapt*
and stability under continual shift, with honest comparison to 3-4 strong baselines.

**Key references.**
- Wang et al. "Tent: Fully Test-Time Adaptation by Entropy Minimization." ICLR 2021
- Wang et al. "Continual Test-Time Domain Adaptation." CVPR 2022 (CoTTA)
- Niu et al. "Towards Stable Test-Time Adaptation in Dynamic Wild World." ICLR 2023 (SAR)
- Hendrycks & Dietterich. "Benchmarking Neural Network Robustness to Common Corruptions." ICLR 2019
- Hu et al. LoRA, ICLR 2022
- "Continual Test-Time Adaptation in Computer Vision: Methods, Benchmarks, and Future Directions." arXiv:2607.08164 (2026 survey)

---

## Idea E — Motion-aware token budgeting for video transformers

**Course modules:** 7 (ViViT), 8 (VideoMAE / masked modeling)

**Problem.** Video ViTs tokenize uniformly in space-time, so static background costs as much as
moving foreground. Learned token pruning exists but needs retraining and ignores the cheap
motion signal already present in frame differences.

**Candidate solution.** Allocate a fixed token budget per clip using a frame-difference (or
tiny optical-flow) saliency prior: keep more tubelets where motion is high, merge/drop elsewhere,
and un-merge for dense heads if needed. Plug into a VideoMAE-pretrained ViT-B, fine-tuned with LoRA.

**Hypotheses.**
- H1: At a 50% token budget, motion-aware selection loses under 1% top-1 on UCF101/HMDB51 vs. 3% or more for uniform subsampling.
- H2: Gains are largest on motion-defined datasets (Something-Something v2 subset) and smallest on appearance-biased ones (UCF101).

**Method & data.** VideoMAE ViT-B (K400-pretrained), UCF101, HMDB51, SSv2-mini (20 classes).
Metrics: top-1/top-5, GFLOPs, throughput. Compute: medium-high (video I/O dominates). Use short
clips (16 frames at 224) and pre-decoded tensors.

**Key references.**
- Arnab et al. "ViViT: A Video Vision Transformer." ICCV 2021
- Tong et al. "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." NeurIPS 2022
- Bolya et al. ToMe, ICLR 2023; Bolya & Hoffman. "Token Merging for Fast Stable Diffusion." CVPRW 2023
- Wang et al. "Efficient Video Transformers with Spatial-Temporal Token Selection." ECCV 2022 (STTS)
- Feichtenhofer et al. "Masked Autoencoders As Spatiotemporal Learners." NeurIPS 2022

---

## Idea F — Dense-feature distillation from DINOv3 into frozen CLIP for open-vocabulary segmentation

**Course modules:** 9 (DINO), 10 (CLIP), 6 (segmentation)

**Problem.** CLIP patch tokens are semantically aligned to text but spatially noisy, so
zero-shot / open-vocabulary segmentation (MaskCLIP, SCLIP, ProxyCLIP) is weak. DINOv2/v3 have
sharp, spatially coherent dense features but no language alignment. CLIP-DINOiser already
distills DINOv2 correlations into CLIP; DINOv3's Gram-anchored features and the Gram-matrix loss
formulation are new and untested for this purpose.

**Candidate solution.** Freeze CLIP ViT-B/16; train a light adapter (LoRA on the last k blocks)
to match the *Gram matrix* (patch-pair similarity) of DINOv3 features on unlabeled images
(COCO train images, no labels). Evaluate training-free segmentation with text prompts.

**Hypotheses.**
- H1: Gram-matrix distillation from DINOv3 improves mIoU over CLIP-DINOiser (DINOv2 correlation) on VOC/ADE/Cityscapes zero-shot segmentation.
- H2: Distilling structure (Gram) rather than features preserves CLIP's text alignment (zero-shot classification on ImageNet drops under 0.5%).
- H3: Improvement is largest at high resolution where CLIP artifacts dominate; pairs well with Idea A's registers.

**Method & data.** Unlabeled COCO images for distillation; VOC-20, ADE-150, Cityscapes,
COCO-Stuff for eval. Metrics: mIoU (training-free protocol of SCLIP/ProxyCLIP), ImageNet
zero-shot top-1. Compute: medium (one forward of a 7B DINOv3 teacher is expensive; use the
distilled ViT-B/L DINOv3 checkpoints).

**Key references.**
- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021 (CLIP)
- Zhou, Loy, Dai. "Extract Free Dense Labels from CLIP." ECCV 2022 (MaskCLIP)
- Wysoczańska et al. "CLIP-DINOiser: Teaching CLIP a few DINO tricks for open-vocabulary semantic segmentation." ECCV 2024
- Lan et al. "ProxyCLIP: Proxy Attention Improves CLIP for Open-Vocabulary Segmentation." ECCV 2024
- Siméoni et al. DINOv3, arXiv:2508.10104
- Naeem et al. "SILC: Improving Vision Language Pretraining with Self-Distillation." ECCV 2024

---

## Idea G — Parameter-efficient and test-time adaptation for point-cloud transformers

**Course modules:** 11 (point transformers), 12 (LoRA)

**Problem.** Point-MAE / PointGPT / ReCon pretrained point transformers are fine-tuned fully per
dataset; PEFT for point clouds (IDPT, Point-PEFT, DAPT) is young, and real-scan datasets
(ScanObjectNN) show a sim-to-real gap. WACV 2026 had "Revisiting LayerNorm for Point Cloud TTA".

**Candidate solution.** Unified study of (i) LoRA on attention vs. token prompts vs. adapters for
Point-MAE/ReCon backbones at equal budget, and (ii) whether the same low-rank module can be
reused for test-time adaptation under ModelNet-C style corruptions.

**Hypotheses.**
- H1: Attention LoRA (r=8) matches full fine-tuning on ScanObjectNN PB-T50-RS within 0.5% OA at under 3% params.
- H2: Prompt-based methods degrade more than LoRA under geometric corruptions.
- H3: Reusing the LoRA slot for entropy-minimization TTA improves corrupted-set OA by 3 or more points without collapse.

**Method & data.** Point-MAE, ReCon (ShapeNet-pretrained), ModelNet40, ScanObjectNN,
ModelNet-C. Metrics: OA, mAcc, params, corruption error. Compute: light (point clouds are cheap).

**Key references.**
- Zhao et al. "Point Transformer." ICCV 2021; Wu et al. "Point Transformer V3." CVPR 2024
- Pang et al. "Masked Autoencoders for Point Cloud Self-supervised Learning." ECCV 2022 (Point-MAE)
- Zha et al. "Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models." ICCV 2023 (IDPT)
- Zhou et al. "Dynamic Adapter Meets Prompt Tuning: Parameter-Efficient Transfer Learning for Point Cloud Analysis." CVPR 2024 (DAPT)
- Ren et al. "ModelNet-C: Benchmarking and Analyzing Point Cloud Classification under Corruptions." ICML 2022

---

## Comparison

| Idea | Novelty | Compute | Risk of being scooped | Rubric fit (math/diagram/ablations) | Venue fit |
|---|---|---|---|---|---|
| A registers x PEFT | high | light | low-med | excellent | WACV / CVPR workshop |
| B LoRA placement in DETR | med-high | medium | medium | excellent | WACV |
| C token merging for detection | medium | light-med | medium | very good | WACV |
| D LoRA-TTA for ViTs | medium | light | high (crowded) | very good | WACV (TTA-friendly) |
| E motion-aware video tokens | medium | med-high | medium | good | WACV |
| F DINOv3 to CLIP Gram distillation | high | medium | medium (fast-moving) | very good | CVPR/ICCV-level if it works |
| G point-cloud PEFT + TTA | medium | light | medium | good | WACV / 3DV |

**Recommendation.** Pick **A** as the primary topic: it has the best novelty-to-compute ratio,
concrete anchor papers from the last 12 months, a training-free component that guarantees results
by the 10/19 draft, and a natural extension (DINOv3, CLIP, Idea F) if things go well. **B** is
the best alternative if you want detection experience; **D** is the safest if you want a low-risk
empirical paper with a clear venue precedent.

---

## Proposal skeleton (one page) mapped to the rubric

1. **Title**
2. **Outline (rubric sections)**: Intro & goals; Hypotheses & method; Related work; Application/experiments; Conclusions / what is learned.
3. **Problem, candidate solution, hypotheses**: one paragraph problem, one paragraph solution with a *block diagram*, three numbered falsifiable hypotheses (copy from the idea above).
4. **Applied ML method & dataset**: backbone checkpoints, the equations you will write (attention, LoRA update dW = BA, merging/un-merge operator or distillation loss), datasets with splits, metrics, hyperparameter sweep list, seeds, compute budget.
5. **References**: 6-8, at least 3 from 2024-2026.

Rubric hooks to hit explicitly in the paper later: block diagram (Hypotheses & Method, 26 pts),
math for the method, hyperparameter search + statistics over seeds (Application, 16 pts),
and a "why did we do this" conclusion tied back to the hypotheses (Conclusions, 16 pts).

## Verification notes

Venue dates were checked on 2026-09-02 against wacv.thecvf.com and deadline trackers; CVPR 2027
dates are projections. Anchor papers for A, B, C, F were confirmed to exist via arXiv search;
the remaining references are well-known works cited from memory and should be spot-checked
(title/venue/year) before going into the proposal.
