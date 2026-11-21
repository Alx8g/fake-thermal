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
(`turbo2`), then 24k more with photo-only light, print and night-darkening
augmentation and a cosine learning-rate decay (mean of the 20k-24k
snapshots, `turbo4`), then 3k more at batch 1 with cast-light augmentation:
headlight beams, street-light pools and wall spill multiplied into the photo
while the thermal target stays as it was. The shipped file is the mean of
the 2k-3k snapshots of that last run (`turbo4-cast`). SD-Turbo itself is
downloaded from Hugging Face on first use and is under its own licence.
Held-out realism with the semantic prior: KID x1000 31.7, DISTS 0.266
(turbo4: 32.6, 0.268; turbo2: 38.9, 0.283). Blind judges spotted it next to
the real frame 71% of the time against turbo4's 73% in the same round (KAIST
street scenes: 16 of 32, chance), and called single frames generated 54% of
the time (real frames: 46%, turbo4: 63%, 24 frames each). Shown beside
turbo4's render of the same frame with no real panel, it was preferred on 79
of 96 sheets of the user's own footage (lit pillars, floors and tables no
longer read warm) and tied on held-out frames: ahead at night, behind on
daytime street scenes, where it warms the sky and darkens the near road.

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
