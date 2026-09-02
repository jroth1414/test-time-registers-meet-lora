# Applied Vision Topics with a CVPR (main or workshop) Path

Companion to `research_ideas.md` (Ideas A-G). These are *applied* projects: a real domain, a real
dataset with an active community, and a specific CVPR workshop that publishes proceedings (CVPRW
papers appear in IEEE Xplore / CVF open access). Each is still transformer-centric so it satisfies
the course.

## How realistic is CVPR main vs. workshop for a one-semester, one-GPU project?

- **CVPR main** (deadline ~2026-11-13 for CVPR 2027): acceptance ~22%, and reviewers expect a
  method-level contribution with broad benchmarks. For a semester project this is a long shot
  unless the finding is sharp and surprising (analysis papers like Idea A or Idea M below are the
  most plausible route). Your 12/07 course paper lands after the deadline anyway, so the realistic
  main-track target is ICCV 2027 (~March 2027) or WACV 2028.
- **CVPR workshops** (June 2027, deadlines ~Feb-March 2027): acceptance is typically 40-60%, the
  audience is domain experts, and a clean adaptation study with strong baselines is enough. Many
  workshops also run a *challenge* whose technical reports are published; winning or placing is a
  concrete, dated result.
- Strategy that works: pick a domain workshop with a standing dataset/challenge, make the paper a
  rigorous *adaptation study* of a current foundation model on that domain, and keep one
  method-level twist so it is not "we fine-tuned model X".

Already-taken angles to avoid (checked 2026-09-02): DINOv3 for building change detection
(ChangeDINO, SemDINO), zero-shot SAM 3 evaluation on remote sensing (arXiv:2607.09583,
SegEarth-OV3), Depth Anything LoRA for endoscopy (DARES, EndoDAC, Endo3DAC, ColonAdapter, EndoGMDE),
VGGT under fisheye geometry (RayTun3R), open-vocabulary thermal detection via distillation
(Thermal-Det). The ideas below are positioned around these.

---

## Idea H — Sensor-resolution shift in Earth observation: web-DINOv3 vs. satellite-DINOv3 on Sentinel-2  (EarthVision @ CVPR)

**Modules:** 3, 9, 12

**Problem.** DINOv3 ships two backbones: web-pretrained (LVD-1689M) and satellite-pretrained
(SAT-493M, 0.6 m Maxar imagery). Most operational EO work uses free Sentinel-2 at 10-60 m and
multispectral bands, a 20-100x resolution gap from SAT-493M. Which backbone should a practitioner
start from, and how do you bridge the resolution and band gap parameter-efficiently? Nobody has
published this comparison.

**Candidate solution.** Controlled study of web-DINOv3 vs. SAT-DINOv3 (ViT-B and ViT-L) under
(i) linear probe, (ii) LoRA, (iii) LoRA + a learned multispectral patch-embedding adapter that
maps 12 Sentinel-2 bands to the RGB embedding space, (iv) a scale adapter that upsamples tokens to
mimic sub-metre GSD. Tasks: scene classification, multi-label land cover, semantic segmentation.

**Hypotheses.**
- H1: SAT-DINOv3 loses its advantage below ~3 m GSD; on 10 m Sentinel-2 the web backbone is equal or better under linear probing.
- H2: A multispectral patch-embedding adapter trained with LoRA recovers more than half of the gap to fully supervised Sentinel-2 models (e.g., Prithvi, SatMAE) at under 3% trainable parameters.
- H3: Gains from the adapter are largest for bands outside RGB (SWIR, red-edge) on vegetation/water classes.

**Method & data.** EuroSAT (10 cls, 13 bands), BigEarthNet-S2 (43-label), SEN12MS or
DFC2020 for segmentation; optionally fMoW-Sentinel. Metrics: top-1, mAP, mIoU, params, GPU-hours.
Compute: light-medium (Sentinel-2 chips are 64-120 px).

**Why it can publish.** EarthVision loves benchmark-driven adaptation studies with clear practical
guidance; a "which backbone to use at which GSD" table is directly actionable. Plausible ICCV/WACV
main-track if extended with 3+ backbones (DINOv3-SAT, Prithvi-2, SatMAE, Clay).

**Key references.**
- Siméoni et al. "DINOv3." arXiv:2508.10104 (2025), incl. SAT-493M backbone
- Cong et al. "SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery." NeurIPS 2022
- Jakubik et al. "Foundation Models for Generalist Geospatial AI (Prithvi)." arXiv:2310.18660
- Sumbul et al. "BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding." IGARSS 2019
- Helber et al. "EuroSAT." IEEE JSTARS 2019
- Reed et al. "Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning." ICCV 2023

---

## Idea I — Site-level test-time adaptation for camera-trap species recognition  (CV4Animals / FGVC @ CVPR)

**Modules:** 3, 9, 12 (+ TTA concepts)

**Problem.** Camera-trap classifiers fail at *new camera sites*: iWildCam deliberately splits train
and test by location, and background/illumination bias dominates errors. Sites produce thousands
of unlabeled images (mostly empty frames and bursts of the same animal), which is free adaptation
data that current pipelines (MegaDetector crop → classifier) ignore.

**Candidate solution.** Per-site unsupervised adaptation of a DINOv3 ViT-B/16 classifier:
LoRA on attention, trained on the site's unlabeled images with (i) burst-consistency (images in a
burst share a label), (ii) background-token suppression using empty frames from the same camera
as negatives, (iii) entropy minimization with class-prior correction for the long tail. No labels
from the new site.

**Hypotheses.**
- H1: Site-level adaptation with zero labels improves macro-F1 on unseen iWildCam locations by more than 5 points over a frozen DINOv3 linear probe.
- H2: Burst consistency is the dominant signal; removing it loses most of the gain.
- H3: The empty-frame negative mining specifically reduces "background species" confusions (tail classes at a site).

**Method & data.** iWildCam 2021/2022 (WILDS split), NACTI for long-tail; MegaDetector v5/v6 for
crops. Metrics: macro-F1, per-site accuracy, tail-class recall, calibration. Compute: light (crops
at 224). Baselines: frozen linear probe, full fine-tune, TENT/SAR on LN, BioCLIP zero-shot.

**Why it can publish.** CV4Animals and the FGVC workshop both publish; the "new site, no labels"
framing is exactly what ecologists ask for, and there is a recent unified study
(arXiv:2603.20509) that flags this as open.

**Key references.**
- Beery et al. "The iWildCam 2021 Competition Dataset." arXiv:2105.03494
- Koh et al. "WILDS: A Benchmark of in-the-Wild Distribution Shifts." ICML 2021
- Beery, Morris, Yang. "Efficient Pipeline for Camera Trap Image Review (MegaDetector)." arXiv:1907.06772
- Stevens et al. "BioCLIP: A Vision Foundation Model for the Tree of Life." CVPR 2024
- Wang et al. "Tent." ICLR 2021; Niu et al. "SAR." ICLR 2023
- "Lessons and Open Questions from a Unified Study of Camera-Trap Species Recognition Over Time." arXiv:2603.20509 (2026)

---

## Idea J — Player-centric ball action spotting with query-based temporal transformers  (CVSports @ CVPR, SoccerNet challenge)

**Modules:** 5 (DETR-style queries), 7 (video transformers)

**Problem.** SoccerNet 2026 Task 2 requires localizing ball actions in time *and* attributing each
to a player (team + jersey number). Current spotting models (FAANTRA-style query decoders) treat
attribution as a separate stage; joint models are underexplored, and the 2027 edition of the
challenge will reuse the task.

**Candidate solution.** A DETR-style temporal decoder where each action query attends jointly to
(a) precomputed clip features (Baidu / VideoMAE features shipped by SoccerNet) and (b) per-frame
player tokens from a tracker (jersey OCR + team colour embeddings). Hungarian matching on
(time, class, player) triplets. LoRA-adapt the VideoMAE feature extractor if compute allows.

**Hypotheses.**
- H1: Joint decoding beats a two-stage spot-then-assign pipeline on the official player-centric mAP metric.
- H2: Cross-attention to player tokens improves temporal precision too (tighter action timing), not just attribution.
- H3: Attribution accuracy is bottlenecked by jersey OCR; an uncertainty-aware assignment head recovers part of the loss.

**Method & data.** SoccerNet Ball Action Spotting + Player-Centric annotations (public, free with
NDA), SoccerNet GSR/jersey-number data. Metrics: official tight/loose mAP, attribution accuracy.
Compute: light-medium if you use the provided features; medium if you fine-tune a video backbone.

**Why it can publish.** CVSports publishes challenge reports and full papers; a top-5 finish on the
2027 leaderboard is a dated, verifiable result. Risk: the challenge deadline (typically ~May) is
after the course, so plan the course paper on the 2026 test split.

**Key references.**
- Giancola et al. "SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos." CVPRW 2018
- "SoccerNet 2026 Challenges Results." arXiv:2607.07320 (2026)
- Carion et al. DETR, ECCV 2020
- Soares et al. "Temporally Precise Action Spotting in Soccer Videos Using Dense Detection Anchors." ICIP 2022
- Tong et al. VideoMAE, NeurIPS 2022
- Cioppa et al. "SoccerNet-Tracking / GSR: Game State Reconstruction." CVPRW 2024

---

## Idea K — Promptable concept segmentation for dense, repetitive agricultural instances  (Agriculture-Vision @ CVPR)

**Modules:** 6 (segmentation), 10 (vision-language), 12 (LoRA)

**Problem.** SAM 3 (Nov 2025) introduced *promptable concept segmentation*: give a noun phrase,
get every instance. Agriculture is the stress test: hundreds of near-identical, occluded instances
per image (wheat heads, apples, grape berries) with domain-specific concepts SAM 3's training
vocabulary barely covers. Zero-shot evaluations exist for remote sensing but not for in-field
crop counting.

**Candidate solution.** (i) Benchmark SAM 3 zero-shot and one-shot on GlobalWheat, MinneApple,
and Agriculture-Vision; (ii) LoRA on SAM 3's text-conditioned decoder + a learned concept-token
per crop type, trained on a few hundred labelled images; (iii) a density-aware presence head that
handles >100 instances per image. Compare against Grounding DINO + SAM 2 and a supervised
RT-DETR/Mask2Former.

**Hypotheses.**
- H1: Zero-shot SAM 3 recall collapses when instance count per image exceeds ~50; the failure is in recognition (presence head), not mask quality.
- H2: LoRA on the decoder plus a learned concept token, trained on 200 images, matches a fully supervised detector on counting MAE while keeping open-vocabulary ability on unseen crops.
- H3: Adaptation transfers across crops of similar morphology (wheat → barley) better than the supervised baseline.

**Method & data.** GlobalWheat 2021 (head detection, multi-country), MinneApple (detection +
counting), Agriculture-Vision (aerial patterns). Metrics: AP, counting MAE/RMSE, mask IoU,
open-vocab recall on a held-out crop. Compute: medium (SAM 3 is large; freeze the image encoder).

**Key references.**
- Ravi et al. "SAM 2: Segment Anything in Images and Videos." arXiv:2408.00714
- Carion et al. (Meta) "SAM 3: Segment Anything with Concepts." 2025 (check exact citation)
- Liu et al. "Grounding DINO." ECCV 2024
- David et al. "Global Wheat Head Detection 2021." Plant Phenomics 2021
- Häni, Roy, Isler. "MinneApple: A Benchmark Dataset for Apple Detection and Segmentation." IEEE RA-L 2020
- Chiu et al. "Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis." CVPR 2020

---

## Idea L — Cross-spectral few-shot adaptation of detection transformers to thermal imagery  (PBVS @ CVPR)

**Modules:** 5, 12 (this is Idea B applied to a modality shift)

**Problem.** RGB-pretrained detectors degrade on long-wave infrared; thermal labels are scarce.
Current fixes either translate RGB→IR with generative models (heavy) or train multimodal fusion
models that need paired RGB-T at inference. A practitioner with a thermal camera and ~100 labelled
frames needs a cheaper recipe.

**Candidate solution.** Few-shot PEFT of RT-DETR / Deformable DETR to thermal: LoRA on attention
plus a re-initialised, trainable patch-embedding ("modality stem"), with the placement study from
Idea B used to find where thermal adaptation actually happens. Optional: self-training on unlabelled
thermal video with pseudo-labels.

**Hypotheses.**
- H1: With 100 labelled thermal frames, LoRA + trainable stem beats full fine-tuning (which overfits) by more than 3 AP on FLIR ADAS v2 and LLVIP.
- H2: Most adaptation happens in the stem and the first two encoder stages; decoder LoRA adds under 0.5 AP.
- H3: Unlabelled thermal video self-training closes half the remaining gap to the fully supervised thermal model.

**Method & data.** FLIR ADAS v2 (thermal + RGB, aligned subset), LLVIP (paired, pedestrians),
KAIST multispectral. Metrics: AP50, AP, params, label-efficiency curve (10/50/100/500 labels).
Compute: medium.

**Key references.**
- Zhao et al. RT-DETR, CVPR 2024; Zhu et al. Deformable DETR, ICLR 2021
- Jia et al. "LLVIP: A Visible-infrared Paired Dataset for Low-light Vision." ICCVW 2021
- FLIR ADAS Dataset v2 (Teledyne FLIR), 2022
- "Few-Shot LoRA Adaptation of a Flow-Matching Foundation Model for Cross-Spectral Object Detection." arXiv:2601.04381 (2026)
- "Thermal-Det: Language-Guided Cross-Modal Distillation for Open-Vocabulary Thermal Object Detection." arXiv:2605.10130 (2026)
- Hu et al. LoRA, ICLR 2022

---

## Idea M — Feed-forward 3D foundation models (VGGT / MapAnything) adapted to surgical video  (medical CV or 3D workshops @ CVPR; MICCAI as fallback)

**Modules:** 3, 11 (3D), 12

**Problem.** VGGT (CVPR 2025 best paper) predicts cameras, depth and point maps from a handful of
frames in one forward pass, but it was trained on rigid, textured, pinhole scenes. Endoscopy is
non-rigid, specular, low-texture and has wide-angle optics. Depth-Anything-style *monocular*
adaptations to endoscopy are crowded; multi-view feed-forward adaptation is not.

**Candidate solution.** LoRA-adapt VGGT's alternating frame/global attention on endoscopic
sequences using self-supervised photometric + multi-view consistency losses (no GT depth), with
a learned intrinsics/positional-encoding adapter for the wide-angle optics. Evaluate against GT
depth and poses on SCARED and Hamlyn.

**Hypotheses.**
- H1: Zero-shot VGGT depth on SCARED is worse than adapted monocular Depth Anything; after LoRA + self-supervision it is better on multi-frame metrics (pose ATE, multi-view consistency).
- H2: Adapting positional encodings / camera head matters more than adapting the frame attention (echoing RayTun3R's fisheye finding).
- H3: Improvements survive at 4-8 input frames on a 24 GB GPU (feasibility claim).

**Method & data.** SCARED (GT depth via structured light), Hamlyn, EndoSLAM; StereoMIS if
available. Metrics: AbsRel, δ1, pose ATE/RPE, chamfer to GT point clouds. Compute: medium-high;
VGGT-1B inference is fine on 24 GB, LoRA training needs gradient checkpointing.

**Risk.** Highest technical risk of this list; highest ceiling (plausible main-track story).

**Key references.**
- Wang et al. "VGGT: Visual Geometry Grounded Transformer." CVPR 2025 (best paper)
- Wang et al. "DUSt3R: Geometric 3D Vision Made Easy." CVPR 2024
- Keetha et al. "MapAnything: Universal Feed-Forward Metric 3D Reconstruction." arXiv 2025
- Allan et al. "Stereo Correspondence and Reconstruction of Endoscopic Data (SCARED) Challenge." arXiv:2101.01133
- "DARES: Depth Anything in Robotic Endoscopic Surgery with Self-supervised Vector-LoRA." arXiv:2408.17433 (position against)
- "RayTun3R" (fisheye adaptation of 3D foundation models), 2026 (verify citation)

---

## Idea N — Maritime obstacle detection & segmentation for USVs with PEFT-adapted transformers  (MaCVi @ CVPR)

**Modules:** 5, 6, 12

**Problem.** Unmanned surface vehicles need reliable obstacle segmentation under glare, waves and
horizon ambiguity; the MaCVi workshop runs yearly challenges (MODS, LaRS, SeaDronesSee) and is far
less crowded than driving. Most entries are heavily tuned CNNs; foundation-model PEFT baselines
are missing.

**Candidate solution.** DINOv3 / Mask2Former with LoRA plus a *horizon-aware positional prior*
(a learned token that encodes estimated horizon line, injected into attention as a bias), and an
uncertainty head for water-edge false positives. Evaluate on LaRS panoptic obstacle segmentation
and SeaDronesSee detection.

**Hypotheses.**
- H1: PEFT-adapted DINOv3 + Mask2Former beats the best published LaRS baseline on obstacle Q-metrics at under 5% trainable params.
- H2: The horizon prior reduces false positives at the water-sky boundary by more than 20% relative.
- H3: Adaptation from LaRS to MODS (different boats/cameras) generalises better than the CNN baselines.

**Method & data.** LaRS (panoptic, 4k images), MODS (obstacle detection benchmark), SeaDronesSee
(aerial maritime). Metrics: LaRS PQ / water-edge accuracy / obstacle F1, MODS detection F1 by
danger zone. Compute: light-medium.

**Key references.**
- Žust, Perš, Kristan. "LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark." ICCV 2023
- Bovcon et al. "MODS: A USV-Oriented Object Detection and Obstacle Segmentation Benchmark." IEEE T-ITS 2022
- Varga et al. "SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water." WACV 2022
- "4th Workshop on Maritime Computer Vision (MaCVi): Challenge Overview." arXiv:2604.13244 (2026)
- Cheng et al. Mask2Former, CVPR 2022; Siméoni et al. DINOv3

---

## Comparison (applied set)

| Idea | Workshop | Novelty | Compute | Crowdedness | Main-track ceiling |
|---|---|---|---|---|---|
| H Sentinel-2 vs DINOv3-SAT | EarthVision | high (unstudied comparison) | light-med | low | medium (ICCV/WACV analysis paper) |
| I site-level TTA camera traps | CV4Animals / FGVC | high | light | low | medium |
| J SoccerNet player-centric spotting | CVSports | medium | light-med | medium (challenge) | low, but dated leaderboard result |
| K SAM 3 agriculture | Agriculture-Vision | med-high | medium | low-med | medium |
| L thermal few-shot DETR PEFT | PBVS | medium | medium | medium | low-medium |
| M VGGT surgical | medical / 3D | high | med-high | low (multi-view) | high if it works |
| N maritime PEFT + horizon prior | MaCVi | medium | light-med | low | low-medium |

## How these combine with A and B

- **A + H**: register artifacts and test-time registers on DINOv3-SAT for EO segmentation. One
  mechanistic paper (A) and one applied paper (H) from the same codebase.
- **B + L** or **B + N**: the LoRA placement study (B) becomes the method; thermal or maritime is
  the applied venue. Same experiments, two audiences.
- **A + K**: CLIP/SAM 3 artifacts at high resolution on dense agricultural instances.

Recommended applied picks: **H** (cheapest, clearest gap, EarthVision is a strong venue) and **I**
(zero-label site adaptation is a real need with a clean hypothesis). **M** if you want a
high-ceiling gamble and can borrow GPU time.

## Verification notes

Workshop existence (EarthVision, CV4Animals, CVSports, PBVS, MaCVi at CVPR 2026), the SoccerNet 2026
task list, SAM 3's release and its remote-sensing evaluations, DINOv3's SAT-493M backbone, the
crowded state of Depth-Anything-for-endoscopy, and the thermal LoRA/distillation papers were all
checked online on 2026-09-02. Older references are cited from memory; spot-check titles and venues
before the proposal. Workshop deadlines for CVPR 2027 are not announced yet; expect Feb-Mar 2027.
