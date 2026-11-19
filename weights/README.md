# Bundled weights — not MIT licensed

`fake_thermal.pt` is an approximately 24 MB fp16 state dict from iteration 2,000
of the ResNet18 U-Net run (validation L1 0.191197). This is an early checkpoint,
not a calibrated temperature model or the final training result.

Training sources: LLVIP, KAIST Multispectral Pedestrian, and FLIR ADAS aligned
pairs. These carry research/non-commercial terms or other restrictions.
**These weights are for non-commercial/research use only until retrained on
permissively licensed data.** The repository's MIT license applies to code,
not these weights or datasets. Upstream terms apply and this notice does not
grant additional rights; verify permissions before use or redistribution.

See the [README weights license and dataset credits](../README.md#weights-license)
and [dataset report](../docs/DATA_REPORT.md). Export newer trusted checkpoints
with `python scripts/export_weights.py outputs/ckpt/best.pt` from the repo root.
