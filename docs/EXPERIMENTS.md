# Experiment log

A living lab notebook for the photorealism campaign: what was tried, what it
measured, and what runs *right now*. Newest first. Conclusions get folded
into TRAINING.md/ROADMAP.md; this file is the running record.

Metric of record: the real-vs-fake patch classifier
(`scripts/evaluate_realism.py`, chance .500; deployed pipeline = model +
guided edge transfer + sensor simulation) plus LPIPS on the 512px validation
split. Validation L1 is tracked but explicitly NOT the campaign goal — L1
rewards blur.

## In flight (2026-11-20)

**v5 perceptual-dose sweep** — three parallel arms on rented RTX 4090s
(Vast.ai), launched ~09:20 NZDT, ~$1.20-1.90 per arm:

| Arm | Recipe | Tests | ETA (NZDT) |
| --- | --- | --- | --- |
| lpips100 | LPIPS weight 1.0, 20k iters, crop 448, 512px data | strong-dose ceiling | ~12:10 |
| lpips050-b16 | LPIPS 0.5, 20k (the killed Mac control, re-homed) | dose-response midpoint | ~13:10 |
| lpips050-long | LPIPS 0.5, 50k iters | does structure compound with schedule? | ~13:55 |

All arms: batch 16, conditioning, FFL, weight EMA. A fourth identical Mac
arm was killed at iter 2,370 (12%) and re-homed to a 4090 — local Mac is
relegated to dev/eval duty; training >1 GPU-hour goes to the cloud
(`scripts/cloud_run.py`, `scripts/fire_arm.sh`).

**Baseline being attacked** (v4, LPIPS 0.25): LPIPS 0.479, classifier 95.5%,
Luna: "visibly better than v3 but still clearly distinguishable; missing
scene-aligned structural detail".

## Completed

**2026-11-20 v4 (LPIPS 0.25, 20k, 448-crop):** LPIPS 0.773→0.479 (−38%),
model output texture 40×, classifier flat at 95.5%. Mechanism validated,
dose insufficient. EMA and guided filtering ruled out as bottlenecks.

**2026-11-20 v3 (512px-native data, crop 448, no LPIPS):** best val L1
0.1646 (all sources improved) but *smoother* than v1 — exposed the L1-blur
trap. Became the packaged model (best L1 available at the time).

**2026-11-20 v2 (all ingredients stacked, 288px):** 0.1711 L1 / 0.6688
LPIPS vs incumbent 0.1696 / 0.6535 — short-run ablation gains do not stack
at full length. Negative result, documented.

**2026-11-20 Flow head round one (plain CFM, 11.5k):** one-step L1 0.34 vs
teacher 0.17; multi-step Euler does not close it. Field underfit; MeanFlow
JVP term is the only remaining Phase-5 path. Parked.

**2026-11-19/20 Sensor simulation:** ThermalSensor (column banding with
height drift, FPN, read noise, optics blur, vignette) calibrated against
real footage and Luna-verified as plausible ("banding complaint fixed, noise
about right"). Classifier: 96.1%→85.7% on 288px val. Default-on in CLI/UI.

**2026-11-19 Ablations (3k iters):** FFL −1.9%, EMA −1.0%, combined no
additive gain. Both adopted as defaults for subsequent runs.

## Standing decisions

- Mac = dev/eval/CI only; >1 GPU-hour goes to rented 4090s (~$1-2/experiment)
- Realism benchmark + Luna verdicts judge photorealism, never validation L1
- Instances are destroyed the moment their run finishes; no idle billing
- Every result — positive or negative — is recorded here and in the docs
