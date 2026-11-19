# Roadmap

The v1 model predicts a plausible thermal appearance and disclaims thermometry.
Everything below keeps that positioning. Each phase is independently shippable,
CI-tested, and justified by published evidence rather than hype; where a claim
was tested locally, the measured number is stated.

## What limits v1 (error decomposition)

1. **Regression blur.** L1 objectives predict the conditional mean E[T | RGB];
   wherever several thermal readings are plausible, the mean is a smear.
2. **Night.** LLVIP's own paper measures visible-spectrum pedestrian detection
   missing 22.59% at night vs 10.66% from infrared (Jia et al., ICCVW 2021) —
   the signal is partly absent from the input, which no loss can conjure.
3. **Domain coverage.** Training pairs are surveillance/driving scenes; the
   fixed validation split cannot see out-of-domain error at all.
4. **Train/infer mismatch.** Trained on 256 crops, infers at 384 short side.
5. **Bolted-on prior.** The DeepLabV3 heat prior runs only at inference and is
   fused heat-only-raises; training never sees it.
6. **Temporal blindness.** Single-frame training plus EMA display smoothing.

## Phase 1 — training-quality wins (implemented, ablation pending)

- **Focal Frequency Loss** (`--frequency-alpha`): adaptively weights
  hard-to-synthesize frequency components. Jiang et al., "Focal Frequency Loss
  for Image Reconstruction and Synthesis", ICCV 2021 (pix2pix FID 5.70 → 4.97
  on CelebA-HQ). Opt-in; zero new dependencies.
- **Weight EMA** (`--ema-decay`): Polyak-averaged weights validated and
  checkpointed; export/evaluate prefer EMA automatically.
- Measured already: fp16 autocast on MPS gains only ~2.5% at this model size;
  channels-last stays ~23% slower than contiguous (see TRAINING.md). Defaults
  unchanged — benchmarks beat assumptions.

## Phase 2 — occlusion-masked temporal filtering

Replace the EMA `TemporalSmoother` with flow-based consistency: forward/back
optical flow (cv2, low resolution), forward-backward consistency occlusion
masks, edge-aware propagation of the previous heat map. Pattern from Bonneel
et al., "Practical Temporal Consistency for Image-Based Graphics Applications",
SIGGRAPH Asia 2015. Expected: visibly steadier video, no new dependencies.
RAFT-in-the-loop was considered and deferred: per-frame network cost, warping
tears at flow errors, and no published metric supporting the claimed gains.

## Phase 3 — FLIR video pair rescue (+3,747 real pairs)

The FLIR ADAS video recordings are excluded from training because the RGB and
thermal streams are spatially unaligned. Recover them with the
mono-modality registration pattern from Tang et al., "Unsupervised Misaligned
Infrared and Visible Image Fusion via Cross-Modality Generation and
Registration", IJCAI 2022: translate RGB to pseudo-thermal with our own model,
register real thermal to pseudo-thermal (much easier than cross-modality),
warp, and gate by registration residual. Delivers the only video-domain
training data available and unlocks temporal training. Requires visual QA of
warped pairs before training on them.

## Phase 4 — style conditioning

ThermalGen (Xiao et al., NeurIPS 2025) shows day/night/sensor/viewpoint
conditions are separable and that conditioning on them beats averaging across
them. Lightweight version for this codebase: a small conditioning embedding
(FiLM on the encoder) for lighting/scene regime. Directly targets buckets 2
and 3 without generative machinery.

## Phase 5 — one-step generative head

The quality ceiling of regression is the conditional mean; the generative
literature shows the gain is perceptual (ThermalGen tables: GAN/flow methods
tie on PSNR/SSIM for LLVIP/FLIR but dominate FID/LPIPS). One-step generative
translation is now practical without slow sampling:

- MeanFlow — "Mean Flows for One-step Generative Modeling",
  arXiv:2505.13447 (NeurIPS 2025): one-step generation in a single training
  phase via average velocity.
- Shortcut Models — Frans et al., ICLR 2025 (oral): one network, one training
  phase, variable step budgets.
- TReFT — "Taming Rectified Flow Models For One-Step Image Translation",
  arXiv:2511.20307: one-step translation specifically.

Plan: conditional flow-matching head on the same ResNet18 U-Net backbone,
bootstrapped from the v1 regression model as teacher; single forward pass at
inference (CLI latency unchanged); sampling replaces averaging, so night
ambiguity yields plausible samples rather than a smear.

## Phase 6 (research) — domain expansion

Pseudo-pairs for out-of-domain scenes (nature, interiors) generated offline by
open generative image models, consistency-filtered, mixed into training.
ThermalGen itself is positioned as a training-data synthesis engine; the same
trick with permissively licensed inputs would address bucket 3 and the weights
license restriction together. Requires licensing review of teacher models.

## Considered and rejected

- **Hand-built physics renderer (Stefan–Boltzmann).** Targets are display
  normalized (per-image p1/p99), so radiometric output is normalized away;
  emissivity/temperature are unobservable from RGB, so "physics by
  construction" yields physics-consistent, not physics-correct, heat; and it
  invites thermometry claims this project explicitly disclaims. Synthetic
  pretraining remains possible without the physics claims (Phase 6).
- **Heat-equation loss terms.** Cheap to ablate against the existing
  L1+SSIM+gradient objective; no supporting evidence yet, so not scheduled
  ahead of Phases 2–5.
