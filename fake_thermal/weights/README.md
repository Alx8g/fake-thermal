# Bundled weights — not MIT licensed

`fake_thermal.pt` is an approximately 24 MB fp16 state dict from iteration
17,500 of a 30,000-iteration ResNet18 U-Net run on Apple M1 Pro (MPS). The best
checkpoint by validation L1 was exported: **0.169593** on the fixed 300-image
split (FLIR 0.148, KAIST 0.146, LLVIP 0.221 at that iteration), about 33% below
the semantic-prior-only baseline (raw prior L1 0.2526). This is a plausible
thermal-appearance model, not a calibrated temperature model.

Training sources: LLVIP, KAIST Multispectral Pedestrian, and FLIR ADAS aligned
pairs. These carry research/non-commercial terms or other restrictions.
**These weights are for non-commercial/research use only until retrained on
permissively licensed data.** The repository's MIT license applies to code,
not these weights or datasets. Upstream terms apply and this notice does not
grant additional rights; verify permissions before use or redistribution.

See the [README weights license and dataset credits](../../README.md#weights-license)
and [dataset report](../../docs/DATA_REPORT.md). Evaluate checkpoints against
the baseline with `python scripts/evaluate.py outputs/ckpt/last.pt outputs/ckpt/best.pt`
and export newer trusted checkpoints with `python scripts/export_weights.py outputs/ckpt/best.pt`
from the repo root.
