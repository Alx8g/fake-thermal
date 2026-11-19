# CPU training

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
configuration, val IDs and RNG states. Automatic resume is not implemented in
this step; rerunning the training command starts a new run and overwrites its
artifacts. Load only trusted locally generated checkpoints. `outputs/config.json`,
`benchmark.json`, `benchmark.log` and `train.log` retain the run configuration
and evidence. All outputs are gitignored. Training uses CPU only.

## Initial checkpoint

The nohup run started as PID 15293. At iteration 500, measured training time was
0.305 s/iteration before validation. The fixed-subset L1 was **0.196300**
(FLIR 0.189511, KAIST 0.166515, LLVIP 0.232874). The saved visual grid is
`outputs/val_000500.jpg`; it was opened and visually inspected. At this early
stage predictions are smooth/coarse and do not yet reproduce fine structure.
Both `outputs/ckpt/last.pt` and `best.pt` exist. These are initial metrics, not
the final result; the background run continues after this commit.
