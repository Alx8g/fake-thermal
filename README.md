# fake-thermal

Turn RGB videos and photos into a synthetic thermal-camera look, using a
CPU-friendly ResNet18 U-Net trained on paired RGB/thermal images.

Premiere Pro and DaVinci Resolve plugins are planned; this version provides
the CLI and web UI only.

**This predicts typical heat appearance, not measured temperature.** It cannot
recover real thermal information from phone RGB sensors. Colors are relative
predicted intensity, not Celsius/Fahrenheit. Predictions may be wrong: do not
use them for medical, safety, surveillance decisions, or temperature measurement.
The bundled weights are the best checkpoint of a 30,000-iteration run
(validation L1 0.1696, a third below the semantic-prior-only baseline);
outputs are plausible thermal appearance, not thermometry.

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
approximately 24 MB `fake_thermal/weights/fake_thermal.pt` fp16 state dict; the CPU model runs
in fp32. See the separate **weights license** below before using it.

## CLI

```sh
fake-thermal input.mov -o out.mp4 --palette ironbow
fake-thermal input.mov -o comparison.mp4 --side-by-side --size 512 --smoothing .65 --raw
fake-thermal photo.jpg -o thermal.png --palette arctic
fake-thermal input.mov -o out.mp4 --weights fake_thermal/weights/fake_thermal.pt
fake-thermal input.mov -o model-only.mp4 --no-semantic
fake-thermal input.mov -o stronger.mp4 --semantic-strength .9
fake-thermal input.mov -o out.mp4 --device auto --auto-contrast
fake-thermal input.mov -o fixed-display.mp4 --device cpu --no-auto-contrast
```

Palettes: **ironbow**, **white-hot**, **black-hot**, **rainbow**, **arctic**.
Default working short side is 384; output returns to source resolution.
Inference defaults to two CPU threads (`--threads 2`). Temporal smoothing
defaults to motion-compensated propagation: the previous heat map is carried
along estimated motion with occlusion masking (Bonneel et al., 2015 pattern),
falls back to plain previous-frame weighting on static shots (`--no-flow-smoothing`)
and resets on detected hard cuts; `--smoothing 0` disables it entirely.
`--raw` adds a grayscale `_raw.mp4` or `_raw.png` alongside output.

**Semantic heat priors are on by default** (`--semantic`, strength .7). A
pretrained torchvision DeepLabV3 MobileNetV3 model runs at 320px short side,
using soft probabilities to suggest typical heat for people/animals/vehicles.
Orange/yellow and very bright white color heuristics suggest flames/lights.
The prior only raises model heat, before temporal smoothing. Luminance-guided
upsampling sharpens boundaries at source resolution. These are assumptions,
not evidence of heat: orange clothing, white walls, reflections and cold
vehicles can be false positives; actual flames or people can be missed.

First semantic use downloads torchvision's COCO-with-VOC-labels checkpoint
(about 42 MB) to the Torch cache and requires internet. `--no-semantic` or
`--semantic-strength 0` avoids that download and the segmentation cost.

`--device auto` (default) uses **CUDA > Apple MPS > CPU**, for both the thermal
and segmentation models. Explicit `cpu`, `mps`, and `cuda` are supported;
unavailable explicit devices produce an error. Models stay fp32, including on
MPS. Entry points enable PyTorch's MPS CPU fallback for unsupported operations
unless `PYTORCH_ENABLE_MPS_FALLBACK` is already set. GPU/MPS require a compatible
PyTorch installation; the CPU wheel command above is for CPU-only machines.

**Auto contrast is on by default**: the displayed heat is stretched between
2nd/99.5th percentiles with an EMA of those bounds, reset on scene cuts.
`--no-auto-contrast` preserves fixed intensity-to-color mapping. Raw exports
remain fused/smoothed heat without this display stretch. Contrast is purely
visual and can exaggerate tiny temperature-independent differences.

ffmpeg decodes MOV/HEVC and rotation metadata, including 4K/60fps inputs. Output
is H.264 MP4 at source average fps; audio is copied when MP4-compatible, with
AAC fallback otherwise. Variable-frame-rate timestamps become constant-rate
timing. Processing is not real-time: with training running, default-size demo
rates were 3–4 fps with semantics and 5–6 fps without. Real iPhone HDR color handling has
not been validated. See [media behavior and limitations](docs/INFERENCE.md).

## Web UI

```sh
python app.py
```

Open **http://localhost:7860**, upload a video or image, choose palette,
side-by-side view, smoothing, semantic enable/strength, device and auto contrast,
then click **Run** to preview and download.
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

For a new accelerator run, use a **separate output directory**. Overrides can
bypass the benchmark when both batch size and iteration count are supplied:

```sh
python scripts/train.py --device cuda --batch-size 16 --iters 50000 --crop 384 --output outputs/gpu-run
# Apple Silicon: --device mps; use a macOS PyTorch wheel, not the CPU wheel index.
```

Training uses ImageNet-pretrained ResNet18 (downloaded by torchvision), paired
geometric augmentation, per-image percentile-normalized targets, balanced
LLVIP/KAIST/FLIR sampling, L1 + SSIM + gradient loss, and AdamW; no GAN.
Training commands start a new run and overwrite training outputs: do not run
them against an existing active run. Export only trusted local checkpoints.

- [Training architecture, benchmark and validation](docs/TRAINING.md)
- [Dataset preparation, counts, splits and alignment caveats](docs/DATA_REPORT.md)
- [Inference, demos and performance](docs/INFERENCE.md)
- [Research roadmap: temporal filtering, video data rescue, one-step generation](docs/ROADMAP.md)

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

The bundled `fake_thermal/weights/fake_thermal.pt` was trained on **LLVIP, KAIST and FLIR**,
which carry research/non-commercial terms or other restrictions. **Weights are
for non-commercial/research use only until retrained on permissively licensed
data.** They are not MIT-licensed. Upstream terms still apply; this notice does
not grant rights beyond those terms, and unrestricted redistribution or
commercial use is not established. Verify upstream permissions for your use.

The bundled weights are the best checkpoint (iteration 17,500 of a 30,000-iteration
M1 Pro MPS run, validation L1 0.169593, ~8% below the previous CPU-run export
at iteration 3,500). See [weights notice](fake_thermal/weights/README.md).

## Tests

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s scripts -p 'test_*.py' -v
```

Tests run without `data/`: preparation tests use synthetic images/keys, sampler
tests use synthetic source lists, and CLI tests generate random model weights
and a one-second video with audio. Semantic tests mock segmentation and use
synthetic class probabilities, colors and edges; no segmentation checkpoint is
downloaded by tests. GitHub Actions runs both suites with CPU Torch and ffmpeg on Python 3.10.
