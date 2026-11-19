# fake-thermal

Turn RGB videos and photos into a synthetic thermal-camera look, using a
CPU-friendly ResNet18 U-Net trained on paired RGB/thermal images.

**This predicts typical heat appearance, not measured temperature.** It cannot
recover real thermal information from phone RGB sensors. Colors are relative
predicted intensity, not Celsius/Fahrenheit. Predictions may be wrong: do not
use them for medical, safety, surveillance decisions, or temperature measurement.
The bundled weights are an early checkpoint and outputs can be blurry/coarse.

## Demo

<!-- Placeholder: add RGB | predicted thermal demo images after training. -->
*Demo images coming soon.* Local demo generation and video paths are documented
in [Inference and demos](docs/INFERENCE.md).

## Install (CPU, Python 3.10+)

Install system **ffmpeg** (including ffprobe), then clone and install:

```sh
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
git clone https://github.com/Alx8g/fake-thermal.git
cd fake-thermal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

No GPU or dataset download is needed for inference. The checkout includes the
approximately 24 MB `weights/fake_thermal.pt` fp16 state dict; the CPU model runs
in fp32. See the separate **weights license** below before using it.

## CLI

```sh
fake-thermal input.mov -o out.mp4 --palette ironbow
fake-thermal input.mov -o comparison.mp4 --side-by-side --size 512 --smoothing .65 --raw
fake-thermal photo.jpg -o thermal.png --palette arctic
fake-thermal input.mov -o out.mp4 --weights weights/fake_thermal.pt
```

Palettes: **ironbow**, **white-hot**, **black-hot**, **rainbow**, **arctic**.
Default working short side is 384; output returns to source resolution.
Inference defaults to two CPU threads (`--threads 2`). EMA smoothing defaults
to .65 previous-frame weight and resets on detected hard cuts; `--smoothing 0`
disables it. `--raw` adds a grayscale `_raw.mp4` or `_raw.png` alongside output.

ffmpeg decodes MOV/HEVC and rotation metadata, including 4K/60fps inputs. Output
is H.264 MP4 at source average fps; audio is copied when MP4-compatible, with
AAC fallback otherwise. Variable-frame-rate timestamps become constant-rate
timing. Processing is not real-time: early measurements with training running
were roughly 5 fps at default working size. Real iPhone HDR color handling has
not been validated. See [media behavior and limitations](docs/INFERENCE.md).

## Web UI

```sh
python app.py
```

Open **http://localhost:7860**, upload a video or image, choose palette,
side-by-side view and smoothing, then click **Run** to preview and download.
Optional grayscale exports are available too. The UI binds all interfaces;
use it on a trusted local network. No public share tunnel is enabled.

## Reproduce data and training

Review upstream dataset terms first. Data are not bundled. Preparation downloads
several large archives (including inspected-but-excluded sources); allow at
least 15 GB for data and environment, with additional room for training outputs.

```sh
python -m pip install huggingface_hub pyarrow
python scripts/prepare_data.py --download
# Existing populated data/pairs is protected against overwrite.
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python scripts/train.py --benchmark
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python scripts/train.py
python scripts/export_weights.py outputs/ckpt/best.pt
```

Training uses ImageNet-pretrained ResNet18 (downloaded by torchvision), paired
geometric augmentation, per-image percentile-normalized targets, balanced
LLVIP/KAIST/FLIR sampling, L1 + SSIM + gradient loss, and AdamW; no GAN.
Training commands start a new run and overwrite training outputs: do not run
them against an existing active run. Export only trusted local checkpoints.

- [Training architecture, benchmark and validation](docs/TRAINING.md)
- [Dataset preparation, counts, splits and alignment caveats](docs/DATA_REPORT.md)
- [Inference, demos and performance](docs/INFERENCE.md)

## Dataset credits

| Dataset | Credits/source | Use and caveats |
| --- | --- | --- |
| LLVIP | [Original LLVIP project](https://github.com/bupt-ai-cz/LLVIP); [jsonhash mirror](https://huggingface.co/datasets/jsonhash/LLVIP) | Paired low-light images; official train/test split; non-commercial terms. |
| KAIST Multispectral Pedestrian | [KAIST benchmark](https://soonminhwang.github.io/rgbt-ped-detection/); [AmineMarnissi mirror](https://huggingface.co/datasets/AmineMarnissi/ThermalKaist) | Day/night sequences; sparse mirror subsampled by sequence; held-out whole sets. Mirror declares `other`, not a permissive grant. |
| FLIR ADAS | [FLIR dataset and terms](https://www.flir.com/oem/adas/adas-dataset-form/); [jsonhash aligned mirror](https://huggingface.co/datasets/jsonhash/FLIR_aligned) | Approximately registered pairs with residual mismatch; inferred split does not certify scene-disjointness. |

Please cite the original dataset publications when using them and consult each
upstream license. Public mirrors do not grant additional rights. No training
images are redistributed in this repository.

## Code license

Source code is licensed under the [MIT License](LICENSE). **MIT applies to code,
not the bundled weights or upstream datasets.** Third-party dependencies retain
their own licenses.

## Weights license

The bundled `weights/fake_thermal.pt` was trained on **LLVIP, KAIST and FLIR**,
which carry research/non-commercial terms or other restrictions. **Weights are
for non-commercial/research use only until retrained on permissively licensed
data.** They are not MIT-licensed. Upstream terms still apply; this notice does
not grant rights beyond those terms, and unrestricted redistribution or
commercial use is not established. Verify upstream permissions for your use.

The included early checkpoint is iteration 2,000 (validation L1 0.191197), not
the final training result. See [weights notice](weights/README.md).

## Tests

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s scripts -p 'test_*.py' -v
```

Tests run without `data/`: preparation tests use synthetic images/keys, sampler
tests use synthetic source lists, and CLI tests generate random model weights
and a one-second video with audio. No pretrained encoder download is required
by tests. GitHub Actions runs both suites with CPU Torch and ffmpeg on Python 3.10.
