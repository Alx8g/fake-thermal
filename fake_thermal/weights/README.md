# Bundled weights — not MIT licensed

`fake_thermal.pt` (U-Net engine, ~25 MB fp16): ResNet18 U-Net with style
conditioning and the full-resolution detail path, trained 20k iterations with
L1/SSIM/gradient, focal frequency and LPIPS losses (`lpips100`), then
fine-tuned 8k iterations with a conditional multi-scale PatchGAN critic
(weight 0.5) and feature matching (weight 10) on 512px data (`gan050`, EMA
weights at iteration 8,000). Held-out realism: KID x1000 141 (104 with sensor
simulation), DISTS 0.317; blind judges still spot it next to real thermal
every time.

`turbo.pt` (turbo engine, ~19 MB fp16): pix2pix-Turbo LoRA, input-conv and
VAE-skip weights for SD-Turbo, trained 11k steps at batch 2 on 512px crops
(`turbo2`). SD-Turbo itself is downloaded from Hugging Face on first use and
is under its own licence. Held-out realism: KID x1000 40, DISTS 0.281; with
the people-and-animals semantic prior, blind judges spotted it next to the
real frame 83% of the time (50% on KAIST street scenes) and called single
frames generated 63% of the time (real frames: 30%).

Both are plausible thermal-appearance models, not calibrated temperature
models. Full measurements: [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md).

Training sources: LLVIP, KAIST Multispectral Pedestrian, FLIR ADAS aligned
pairs and re-registered FLIR video pairs. These carry research/non-commercial
terms or other restrictions. **These weights are for non-commercial/research
use only until retrained on permissively licensed data.** The repository's MIT
license applies to code, not these weights, the datasets or SD-Turbo. Upstream
terms apply and this notice does not grant additional rights; verify
permissions before use or redistribution.

Export a U-Net checkpoint with `python scripts/export_weights.py <ckpt>`;
`snapshots/iter_N.pt` files written by `--snapshot-every` are already in the
packaged format. Score any model with `scripts/realism_report.py`.
