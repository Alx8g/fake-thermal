# CPU training

## Device selection and overrides

The historical CPU benchmark below remains unchanged. New runs accept
`--device auto|cpu|mps|cuda` (auto: CUDA > MPS > CPU). Batches and both thermal
targets/RGB inputs move to the selected device; validation grids return to CPU.
Arithmetic remains fp32 by default; opt into fp16 autocast on CUDA/MPS with
`--precision fp16` (see below). Entry points set
`PYTORCH_ENABLE_MPS_FALLBACK=1` unless already configured. MPS uses contiguous
memory format; CPU/CUDA use channels-last; `--layout` overrides either.
Benchmark timings synchronize the
accelerator. CUDA/MPS have not been hardware-tested on this CPU-only host.

`--batch-size`, `--iters` and `--crop` override benchmark batch/iteration values
and the default 256 crop. Supply both batch and iterations to start a run
without a benchmark. Otherwise a benchmark file must supply missing values.
Use a separate output directory for each device/run; do not reuse an active
training directory. The original running process is not restarted or modified.

```sh
python scripts/train.py --device cuda --batch-size 16 --iters 50000 --crop 384 --output outputs/cuda-run
python scripts/train.py --device mps --benchmark --batch-size 8 --crop 320 --output outputs/mps-run
python scripts/train.py --device mps --batch-size 8 --crop 320 --output outputs/mps-run
python scripts/train.py --device mps --batch-size 16 --iters 50000 --precision fp16 --output outputs/mps-run
python scripts/train.py --device mps --batch-size 16 --iters 50000 --output outputs/mps-run --resume
```

## Mixed precision, memory layout and resume

`--precision fp16` opts into float16 autocast on CUDA/MPS (default fp32):
convolutions run fp16 while normalization, the loss and checkpointed weights
stay fp32, with predictions cast back before the loss. Apple GPUs execute fp16
at twice the fp32 rate, so MPS runs typically train faster; measure locally with
`--benchmark --precision fp16` rather than assuming. No gradient scaler is
used: the L1-scale loss keeps gradients well inside fp16 range, and the
non-finite-loss guard aborts a diverging run. fp16 is rejected on CPU.

`--layout contiguous|channels_last` overrides the default tensor memory
format. PyTorch 2.15 moved channels-last conv backward on MPS onto a dedicated
Metal path, so re-benchmarking the override is worthwhile after major PyTorch
upgrades; the historical default (contiguous on MPS) is unchanged.

`--frequency-alpha` (default 0, off) adds the Focal Frequency Loss
(Jiang et al., ICCV 2021) to the training objective: an adaptive weight on
hard-to-synthesize frequency components of the prediction. This targets the
regression-to-mean blur of L1 objectives; `--frequency-alpha 1.0` is the
recommended setting. The FFT runs on CPU under the MPS fallback.

`--ema-decay` (default 0, off) maintains an exponential moving average of the
weights and validates/checkpoints the averaged weights, retaining the raw
weights for the optimizer. `--ema-decay 0.999` is the recommended setting;
`scripts/export_weights.py` and `scripts/evaluate.py` automatically prefer the
EMA weights when a checkpoint contains them. Both flags participate in resume
matching.

`--resume` continues the run saved at the output's `ckpt/last.pt`: model,
optimizer, scheduler, iteration, best L1 and all RNG states are restored, and
the seeded sampling stream is replayed from the checkpointed position so the
remaining sample sequence matches an uninterrupted run. Iterations may be
extended (a longer cosine horizon), never shortened; batch size, crop, seed,
precision and layout must match or the resume is rejected with the differing
keys. Older checkpoints without precision/layout keys resume as fp32/auto.
Resuming a completed run is a safe no-op; rerunning without `--resume` starts a
new run and overwrites prior artifacts.

### Realism training options

`--gan-weight` and `--fm-weight` (default 0, off) add a conditional
multi-scale PatchGAN critic (`fake_thermal/discriminator.py`, hinge loss,
spectral norm) and pix2pixHD feature matching on its activations. The critic
and its optimizer are checkpointed, so adversarial runs resume. Feature-matching
values are small (~0.01), so useful weights are around 10; `--d-lr` sets the
critic learning rate (default 2e-4).

`--fullres` adds a small full-resolution RGB feature path into the head.
Without it the finest features the head sees are a bilinear upsample of the
stride-2 decoder output, which caps pixel-level detail. `--init` warm-starts
from a checkpoint (EMA weights preferred) or an exported state dict; a model
without the full-resolution path loads into one with it and reproduces its
output exactly, because the new head inputs start at zero.

`--precision bf16` enables bfloat16 autocast on CUDA (safer than fp16 for
adversarial training). On CUDA, cuDNN autotuning and TF32 are enabled.

`--workers` defaults to the CPU count minus two (at most 16). Augmenting one
448px sample costs ~19 ms of CPU on an M1 Pro core and more on rented vCPUs;
with the old default of two workers, rented RTX 4090s sat at roughly a third
of their speed waiting for images. Training targets are dequantized (uniform
+-0.5 grey level) before the percentile stretch so a critic cannot tell real
from generated thermal by the 8-bit value lattice.

`--snapshot-every N` also saves fp16 copies of the evaluated weights to
`snapshots/iter_N.pt`, so mid-run models survive and can be compared.

### pix2pix-Turbo (SD-Turbo, one step)

`scripts/train_turbo.py` fine-tunes pix2pix-Turbo (Parmar et al. 2024; adapted
in `fake_thermal/turbo.py`) on the same paired data: LoRA adapters on SD-Turbo's
UNet and VAE, VAE skip connections, L2 + 5 x LPIPS(VGG) reconstruction and a
CLIP-feature (vision-aided) discriminator at weight 0.5. It needs
`diffusers transformers peft vision-aided-loss "setuptools<81"` and OpenAI's
`clip` package, and downloads SD-Turbo (~5 GB). At batch 2 and 512px it needs
about 15 GB of GPU memory with activation checkpointing (`--no-checkpointing`
overflows 24 GB) and runs ~1.05 s/step on an RTX 4090. `--init` continues from
saved weights; `--ids outputs/realism/ids.json` renders the held-out realism
frames when training ends.

#### Visible-detail consistency (experimental)

Fine-tune a separate checkpoint with `--appearance-weight .3` and compare
with a run from the same initialization and seed at `--appearance-weight 0`.
On every fourth optimizer step (`--appearance-every`), a second pass gets the same RGB crop with one
visible-only edit. The extra loss is weighted toward changed pixels and
compares against the unchanged thermal target and, more lightly, the
detached original prediction. Unedited RGB/thermal pairs still train the
base objective. No extra model, checkpoint format or inference pass is needed.
For a small L2-only smoke run, use `--lambda-lpips 0 --lambda-gan 0` to
skip loading both feature networks; this is not the production training recipe.

Glare and print (text, logos or road markings) use synthetic edits out of the
box. For glass and clothing, provide aligned 8-bit grayscale masks with white
pixels on the glass or fabric only:
`pairs/{train,val}/physics/{glass,clothing}/<stem>.png`. Mark the material
interior; leave frames with no reliable mask unannotated. Glass edits soften
visible interior detail, and fabric edits lighten or darken it without editing
the thermal target. RGB, mask and thermal share the geometric augmentation.
Neither edit assumes that real materials have one fixed thermal appearance.

Use the same validation pairs, size and seed for the control and candidate:

```sh
python scripts/appearance_report.py --data data512 --weights path/to/checkpoint.pt \
  --size 384 --seed 42 --output outputs/appearance-report.json
```

The report measures the mean change in raw heat in edited regions and in
untouched context, plus error to aligned thermal and the clean output's
intensity range. Lower edit drift alone is not a win: a flatter model can
score well. Check clean error and range, and run the existing blind preference
test on **untouched** held-out frames before changing the shipped checkpoint.
Without annotated validation masks, glass and clothing are reported as
unmeasured, not as passes.

### Rented GPUs

`scripts/vast_fleet.py` runs on the machine holding the Vast API key and the
prepared data:

```sh
python scripts/vast_fleet.py up --offer <id> --label run1          # rent, wait for ssh
python scripts/vast_fleet.py push --label run1                     # code + data512, 6 rsync streams
python scripts/vast_fleet.py launch --label run1 --arm gan -- --precision bf16 ...
python scripts/vast_fleet.py launch --label run1 --arm next --after gan -- ...
python scripts/vast_fleet.py watch --label run1 --floor 0.5 --max-hours 5
```

`watch` pulls snapshots, grids, logs and rendered frames as runs progress,
pulls final checkpoints when a run ends, destroys the box once every run on it
has ended (unless `outputs/vast/<label>.hold` exists), and destroys every box
when account credit reaches `--floor`. One stream through Vast's ssh relay
moved ~3 MB/s from the Mac; six parallel rsyncs reached ~11 MB/s. Do not push
with macOS `tar`: it adds `._` AppleDouble files that the loader reads as an
extra data source.

With a benchmark batch override, all 50 steps use that batch; otherwise 25
steps each at 2 and 4 are compared. Install the appropriate platform PyTorch
wheel for accelerators. Larger crops do not add detail to existing 288px-short-
side prepared data: training resizes crops, validation center-crops (padding
if necessary). Reprepare at higher resolution if genuine extra detail is needed.
Saved config records the effective device, crop, batch and iterations.

## Architecture and objective

`fake_thermal/model.py`: ImageNet-pretrained torchvision ResNet18 encoder,
skip connections after the stem and all residual stages, lightweight bilinear
upsampling/conv decoder (128/64/32/24 channels), one sigmoid output channel.
Input is RGB float `[0,1]`; ImageNet normalization happens inside the model.
Forward pads on the right/bottom to multiples of 32 (replicate padding), then
unpads predictions to the exact input H/W, including odd sizes. Decoder uses
GroupNorm; pretrained encoder BatchNorm is fine-tuned.

`fake_thermal/data.py`: each full thermal target gets a clipped per-image
1st/99th percentile stretch to `[0,1]` before augmentation. Constant images map
to zero. This harmonizes display contrast, not physical temperature calibration.
Random resized crop (area .6–1.0, near-square aspect ratio) to 256×256 and hflip
are shared between RGB/target. Only RGB receives strong color jitter,
gamma/exposure, occasional grayscale, Gaussian blur, noise and JPEG degradation.
The weighted replacement sampler gives LLVIP/KAIST/FLIR equal expected source
probability, independently of source size.

`fake_thermal/losses.py`: `L1 + 0.5*(1-SSIM) + 0.1*gradient_L1`, using local
7×7-window SSIM. AdamW uses decoder LR 3e-4 and encoder LR 3e-5, weight decay
.01, with cosine decay to zero. No adversarial training or GAN.

## Commands

From the checkout with the prepared data present:

```sh
.venv/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python -u scripts/train.py --benchmark > outputs/benchmark.log 2>&1
nohup env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python -u scripts/train.py > outputs/train.log 2>&1 < /dev/null &
```

The benchmark is **50 actual training steps**: 25 at batch 2, 25 at batch 4.
The first five per candidate warm up kernels; selection maximizes measured
images/second over the remaining 20. Timing includes data loading, forward,
loss, backward and optimizer updates. A full 300-image validation pass estimates
checkpoint evaluation overhead. Six Torch intra-op threads, one inter-op thread,
two persistent loading workers and channels-last tensors fit the eight-core CPU.
Channels-last is used; no separate layout A/B speed claim is made.

Measured on this machine:

| Batch | Seconds/iteration | Images/second |
| --- | ---: | ---: |
| 2 | 0.1946 | 10.28 |
| **4** | **0.3025** | **13.22** |

Validation benchmark: 5.15 seconds. Planned run: **43,734 iterations**, batch 4,
estimated **3.80 hours**, allowing 5% headroom within the requested four hours.
Disk write/visualization overhead and sustained machine load can change ETA.
Benchmark weights are discarded; full training starts fresh from ImageNet.

## Validation and artifacts

The fixed 300-image subset has **100 images per source**, seed 42; exact IDs
are saved to `outputs/val_ids.json`. Validation uses deterministic 256×256 center
crops and target normalization, no random augmentation. L1 averages pixels and
images equally; balanced counts make it equally source-weighted. This measures
relative thermal intensity, not Celsius accuracy. Existing dataset caveats apply:
FLIR scene IDs are inferred, LLVIP is low-light, and KAIST may revisit routes.

Every 500 iterations and at the final step:

- Validate all 300 images, log aggregate and per-source L1.
- Save `outputs/val_NNNNNN.jpg`: four examples per source, RGB | GT | prediction.
- Atomically replace `outputs/ckpt/last.pt`; replace `best.pt` on improved L1.
- Keep only last/best checkpoints, retaining the visual grids across the run.

Checkpoints include model, optimizer, scheduler, iteration, best/current L1,
configuration, val IDs and RNG states. Resume with `--resume` (see above);
rerunning the training command without it starts a new run and overwrites its
artifacts. Load only trusted locally generated checkpoints. `outputs/config.json`,
`benchmark.json`, `benchmark.log` and `train.log` retain the run configuration
and evidence. All outputs are gitignored. The initial run below used CPU only;
current runs support CUDA/MPS (see above).

## Initial checkpoint

The nohup run started as PID 15293. At iteration 500, measured training time was
0.305 s/iteration before validation. The fixed-subset L1 was **0.196300**
(FLIR 0.189511, KAIST 0.166515, LLVIP 0.232874). The saved visual grid is
`outputs/val_000500.jpg`; it was opened and visually inspected. At this early
stage predictions are smooth/coarse and do not yet reproduce fine structure.
Both `outputs/ckpt/last.pt` and `best.pt` exist. These are initial metrics, not
the final result; the background run continues after this commit.

## Final M1 Pro run (MPS, PyTorch 2.15)

Full 30,000-iteration run: `scripts/train.py --device mps --batch-size 16
--iters 30000`, ~0.242–0.286 s/iter, **2.02 hours wall clock**, batch 16 at
256 crop in fp32/contiguous. Best checkpoint **iteration 17,500, validation L1
0.169593**; the fully annealed final checkpoint measured 0.171567, so the best
checkpoint was exported. Validation splits are identical across runs
(`outputs/eval.json` holds recomputed per-source numbers; recomputation matched
stored checkpoint values exactly).

| Variant | Validation L1 | Notes |
| --- | ---: | --- |
| Trained model (best.pt, iter 17,500) | **0.169593** | shipped weights |
| Trained model (last.pt, iter 30,000) | 0.171567 | |
| Model + semantic prior fused (CLI default) | 0.175363 | prior costs L1 on this split; kept for out-of-distribution robustness |
| Semantic prior only (raw) | 0.252625 | baseline |
| Semantic prior only (per-image p1/p99 stretch) | 0.384504 | baseline |

MPS benchmark matrix (batch 16, 50 steps, warmed):

| Precision / layout | s/iter | images/s |
| --- | ---: | ---: |
| fp32 / contiguous (default) | 0.218 | 73.3 |
| fp16 / contiguous | 0.213 | 75.2 |
| fp32 / channels-last | 0.284 | 56.3 |
| fp16 / channels-last | 0.283 | 56.5 |

Measured conclusions: fp16 autocast gains only ~2.5% on this workload
(bandwidth/launch-bound at this model size), and channels-last remains ~23%
slower than contiguous on MPS even with PyTorch 2.15's Metal backward path —
the defaults (fp32/contiguous on MPS) stay. End-to-end CLI throughput on the
same machine (640×512 input, working size 384, x264 crf 18): **12.3 fps** with
the semantic prior, **21.3 fps** without (`--no-semantic`).

### Loss-configuration ablation (3,000 iterations, fp16, batch 16)

| Configuration | Best validation L1 | Δ vs base |
| --- | ---: | ---: |
| base (L1 + SSIM + gradient) | 0.187308 | — |
| + focal frequency (`--frequency-alpha 1`) | **0.183800** | −1.9% |
| + weight EMA (`--ema-decay 0.999`) | 0.185505 | −1.0% |
| + both | 0.185551 | −0.9% |

The frequency loss is the clear short-run win, improving every source; EMA
also improves checkpointed weights within 3k iterations, though its benefit
compounds on full schedules (the shipped 30k run peaked at iteration 17,500 —
EMA checkpointing exists to smooth exactly that). The combined arm shows no
additive gain at this length, within run-to-run noise of EMA alone. Future
runs use `--frequency-alpha 1 --ema-decay 0.999`. Artifacts:
`outputs/ablation/{base,ffl,ema,both}/`.

### Style-conditioning A/B (same protocol, FFL+EMA both on)

| Configuration | Best validation L1 |
| --- | ---: |
| FFL + EMA | 0.185551 |
| + `--conditioning` | **0.183667** (−1.0%) |

FLIR improved most (0.1863 → 0.1752, −6%) while night-time LLVIP slightly
regressed (0.2265 → 0.2321): consistent with the night analysis — the FiLM
head helps where RGB carries recoverable signal and cannot add what dark
frames lack. Conditioned runs are recommended for day-domain accuracy;
unconditioned remains reasonable where night performance dominates.
Artifacts: `outputs/cond/`.

### v3: 512px-native data, crop 448 (the resolution run)

Prepared the datasets at 512px short side (native thermal resolution for the
FLIR video recordings; `data512/`), re-registered the video pairs at native
640x512, and trained 20,000 iterations at crop 448 with focal frequency loss,
weight EMA and conditioning. **Best validation L1 0.164636** (EMA, iteration
14,500), improving every source against the incumbent (FLIR 0.1418, KAIST
0.1430, LLVIP 0.2095) — though the different crop/resolution regime makes
this not directly comparable to the 256-crop numbers above.

**The photorealism diagnosis** (Luna-verified side-by-sides plus a CNN patch
classifier, `scripts/evaluate_realism.py`): validation L1 rewards blur —
L1-optimal prediction is the conditional mean — and v3 is measurably
*smoother* than v1 (output texture 0.2-0.4 vs real ~779-2857 Laplacian
variance). The deployed pipeline (guided edge transfer + sensor simulation)
reaches 584 vs real 779 texture on FLIR/LLVIP samples; the classifier still
separates real from fake at 94.2% on 512-native patches. Luna's verdict:
past the "blobs" stage, scene layout surviving, but still distinguishable by
missing scene-aligned high-frequency structure. **v4 adds the perceptual
loss (`--lpips-alpha 0.25`)** to break the L1-blur equilibrium; the realism
benchmark, not validation L1, is the metric of record for this campaign.
Artifacts: `outputs/realism_v3.json`, `outputs/morning_v3.png`, `outputs/v3/`.

### v4: perceptual training loss (the realism campaign's metric of record)

20,000 iterations at 448-crop on 512px data, identical to v3 plus
`--lpips-alpha 0.25` (LPIPS-AlexNet added to the loss). Results:

| Metric | v3 | v4 |
| --- | ---: | ---: |
| LPIPS on the 512px val split | 0.7727 | **0.4794** (−38%) |
| Raw model output texture (Laplacian var) | ~0.4 | **15.4** |
| Validation L1 | 0.1646 | 0.1666 |
| Real-vs-fake classifier (deployed pipeline) | 94.2% | 95.5% |

The perceptual loss works — LPIPS dropped 38% and the model's own output
gained ~40x texture — but at this weight and schedule it does not yet close
the visual gap (Luna: "visibly better, still clearly distinguishable; the
missing piece is scene-aligned structural detail"). EMA checkpointing does
not blur the gain (EMA 15.0 vs raw 15.4), and the guided edge-transfer stage
is not the bottleneck (classifier identical with or without it). Remaining
levers: perceptual weight (0.5/1.0) and schedule length (50k), packaged as
the v5 arms in `scripts/cloud_run.py`. Artifacts: `outputs/eval_v4.json`,
`outputs/realism_v4.json`, `outputs/morning_v4.png`, `outputs/v4/`.

### Full-length recipe comparison (v2, 30k iterations)

The v2 run stacked every validated ingredient — focal frequency loss, weight
EMA, style conditioning and the 3,193 rescued video pairs — on the same
256-crop/288px protocol as the shipped incumbent:

| Model | Validation L1 | LPIPS (AlexNet) |
| --- | ---: | ---: |
| incumbent (plain L1+SSIM+gradient) | **0.169593** | **0.6535** |
| v2 (FFL + EMA + conditioning + video) | 0.171092 | 0.6688 |

The short-run ablation gains did not stack at full length: EMA checkpointing
lags late-run improvements, conditioning trades KAIST accuracy for FLIR/LLVIP,
and the daytime-heavy video pairs shift the source balance. **The incumbent
remains the packaged model at this scale.** The decisive next lever is
resolution: v3 trains on 512px-native data (including the video pairs
re-registered at native 640x512 thermal) at crop 448, evaluated at its
deployment resolution via the realism benchmark (scripts/evaluate_realism.py).
Artifacts: `outputs/eval_v2.json`, `outputs/v2/`.

### FLIR video pair rescue (Phase 3 full pass)

3,405 of the 3,747 excluded unaligned video frames (90.8%) registered
successfully under the 0.30 residual gate and were saved to
`data/pairs-video/` (residuals p50=0.16, p90=0.29). **Two-round visual QA**
(full-resolution overlay crops judged independently): the clean recordings
pass 12/12 judgeable crops with no systematic offset, scale or rotation
residue; the recording `video-msNEBxJE5PPDqenBM` failed throughout (0/6,
49% gate pass) and was **excluded** (moved to `data/pairs-video-rejected/`).
The remaining **3,193 pairs are integrated into `data/pairs/train/`** with
the `video_` source prefix — the balanced sampler weights them as one
source alongside FLIR/KAIST/LLVIP. QA artifacts:
`outputs/rescue_overlay_blink.mp4`, `outputs/rescue_zoom.png`,
`outputs/rescue_fringe.png`.
