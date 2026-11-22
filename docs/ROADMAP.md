# Roadmap

## Direction after 20 Nov 2026

Blind tests (docs/EXPERIMENTS.md) settled the main question: no regression
U-Net, with perceptual, adversarial or sensor-noise additions, survives a
side-by-side look at real thermal, and none passes as real on its own. A large
pretrained image model fine-tuned for one-step translation (pix2pix-Turbo on
SD-Turbo, `--engine turbo`) is the first thing that does: on single frames,
judges called it generated about as often as real footage. The U-Net stays as
the fast engine; the turbo engine is the photographic one.

Next, in order of expected value:

1. **Physics tells.** Judges still catch turbo frames side by side because
   they copy visible-light detail: see-through windscreens (glass is opaque in
   LWIR), road paint and logos, glare from lit lamps and headlights, and
   people whose heat follows clothing brightness. Levers: the RGB-conditioned
   critic (`--pair-gan-weight`), a people/animals-only semantic prior, and
   training data where these cases are common. An opt-in
   [visible-detail consistency loss and audit](TRAINING.md#visible-detail-consistency-experimental)
   now test these failure modes without changing the shipped checkpoint.
2. **Data diversity.** Training data is driving, surveillance and night
   streets from three cameras. Larger aligned RGB-thermal sets exist, e.g.
   LasHeR (1,224 aligned visible/thermal video pairs, 730K+ frame pairs, far
   more varied scenes); check each licence before use.
3. **Video.** The [real-footage tests](EXPERIMENTS.md#turbo-video-stride-and-adapter-fusion-22-nov-2026)
   support the default motion-compensated smoothing. The optional accelerated
   stride refreshes predictions when RGB motion does not track. Similar-looking
   cuts, fast motion and temporal training remain open problems.
4. **Speed.** Inference folds the LoRA adapters into the base weights; previous
   M1 Pro tests took about 2.5 s per vertical frame before this change.
   The [turbo draft preset](EXPERIMENTS.md#two-keyframe-turbo-draft-preview-22-nov-2026)
   reduces the model resolution and interpolates between keyframes for
   editorial passes. Distilling into a smaller student or running on a GPU
   are still options for full-rate exports.

The phases below are the earlier plan (19-20 Nov), kept for the record.

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

## Phase 1 — training-quality wins (delivered)

- **Focal Frequency Loss** (`--frequency-alpha`): adaptively weights
  hard-to-synthesize frequency components. Jiang et al., "Focal Frequency Loss
  for Image Reconstruction and Synthesis", ICCV 2021 (pix2pix FID 5.70 → 4.97
  on CelebA-HQ). Opt-in; zero new dependencies.
- **Weight EMA** (`--ema-decay`): Polyak-averaged weights validated and
  checkpointed; export/evaluate prefer EMA automatically.
- Measured already: fp16 autocast on MPS gains only ~2.5% at this model size;
  channels-last stays ~23% slower than contiguous (see TRAINING.md). Defaults
  unchanged — benchmarks beat assumptions.
- **Ablation (3k iterations, fp16, batch 16):** frequency loss −1.9% (best
  arm, improves every source), weight EMA −1.0%, combined within noise of
  EMA alone. Both are recommended defaults for future runs; numbers and
  artifacts in TRAINING.md and `outputs/ablation/`.

## Phase 2 — occlusion-masked temporal filtering (delivered)

Replace the EMA `TemporalSmoother` with flow-based consistency: forward/back
optical flow (cv2, low resolution), forward-backward consistency occlusion
masks, edge-aware propagation of the previous heat map. Pattern from Bonneel
et al., "Practical Temporal Consistency for Image-Based Graphics Applications",
SIGGRAPH Asia 2015. Static shots take a single-flow shortcut, keeping the
end-to-end cost within ~2% of plain EMA at 720p. Measured on a controlled
6 px/frame pan of real footage: motion-compensated output error 5.73 vs 6.05
gray levels for plain EMA (5.7% more temporally consistent), with zero-motion
behavior bit-identical to EMA. `--no-flow-smoothing` restores the previous
blend. RAFT-in-the-loop was considered and deferred: per-frame network cost,
warping tears at flow errors, and no published metric supporting the claimed
gains.

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

**Round-one result (honest):** a plain-CFM FlowUNet warmstarted from the v1
teacher reached one-step validation L1 0.34 against the teacher's 0.17 after
11.5k iterations at batch 4, with the validation curve flat after ~7k.
Multi-step Euler sampling does not close the gap (1-step 0.343, 2-step 0.345,
4-step 0.349, 8-step 0.352): integration granularity is not the bottleneck —
the velocity field itself is underfit at this compute budget, consistent
with MeanFlow's own finding that plain flow matching degrades under single-
step sampling. Remaining paths: the MeanFlow JVP consistency term
(arXiv:2505.13447), which is the actual mechanism behind working one-step
sampling, with a longer schedule and the v2 regression model as warmstart
teacher; or a shortcut-style teacher-to-data interpolation. Checkpoints and
log: `outputs_flow/`.

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
