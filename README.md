# fake-thermal

Turn RGB videos and photos into a synthetic thermal-camera look. Two engines:
a fast, CPU-friendly ResNet18 U-Net (default), and `--engine turbo`, a
one-step SD-Turbo model fine-tuned on paired RGB/thermal footage that is far
more photographic and much slower.

Premiere Pro and DaVinci Resolve plugins are planned; this version provides
the CLI and web UI only.

**This predicts typical heat appearance, not measured temperature.** It cannot
recover real thermal information from phone RGB sensors. Colors are relative
predicted intensity, not Celsius/Fahrenheit. Predictions may be wrong: do not
use them for medical, safety, surveillance decisions, or temperature measurement.
In blind tests, judges shown a turbo frame and the real thermal frame of the
same scene picked the real one 71-87% of the time, depending on the panel of
judges (82% in the latest round; on KAIST street scenes 50-63%, where 50% is
chance); the U-Net is spotted every time. Outputs are plausible thermal
appearance, not thermometry. Measurements: [experiment log](docs/EXPERIMENTS.md).

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

Palettes: **ironbow**, **white-hot**, **black-hot**, **rainbow**, **arctic**, **lava**, **glowbow**.
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

### Photographic engine (`--engine turbo`)

```sh
python -m pip install diffusers transformers peft mediapipe
fake-thermal input.mov -o out.mp4 --engine turbo
fake-thermal input.mov -o out.mp4 --engine turbo --turbo-weights path/to/turbo.pt --side-by-side
fake-thermal input.mov -o draft.mp4 --engine turbo --turbo-preview
```

The default U-Net engine is fast but its output stays visibly smooth. The
turbo engine runs pix2pix-Turbo: one denoising step of SD-Turbo with LoRA
adapters fine-tuned on paired RGB/thermal footage. It renders much sharper,
more photographic thermal texture, but is intended for offline video, not
live use. The earlier M1 Pro benchmark took about 2.5 s per vertical 9:16
frame at a 512 px working short side. Inference now folds the LoRA adapters
into the model weights at startup to avoid running separate adapter layers on
every frame. `--stride 2` can reuse motion-warped heat on alternate frames;
it runs the model sooner if the scene cuts or the warp does not match the
current image. It preserves the frame count and FPS, but the speedup depends
on the footage. The default `--stride 1` still renders every frame.
For a quick editorial pass, use `--turbo-preview` (or **Turbo draft preview**
in the web UI). It renders at a 256 px working short side and uses motion
from both neighboring keyframes between model predictions. It rerenders
unmatched changes and never interpolates across detected cuts. Output size,
frame rate and audio are unchanged; detail is softer. Leave the option off
for the full-quality export, or add `--size 512` to keep full working detail
on the preview's keyframes.
It downloads SD-Turbo (~5 GB) from Hugging Face on first use; check
[SD-Turbo's licence](https://huggingface.co/stabilityai/sd-turbo) before any
commercial use. With the turbo engine the semantic prior warms only people and
animals (vehicles excluded, edges fitted to the photo) and sensor simulation
defaults off (it synthesizes its own sensor texture); temporal smoothing, auto
contrast and palettes work as before. The fine alternating-row and
alternating-column pattern SD-Turbo's decoder leaves, which real thermal
cameras do not make, is removed from every frame.

With either engine, people large enough to segment get body-part heat instead
of one uniform level: exposed skin hottest, clothing warm but cooler, hair,
caps, masks and glasses cooler still. Parts come from Google's MediaPipe
multiclass selfie segmenter (`pip install mediapipe`, Apache 2.0 model card;
the ~16 MB model downloads to the Torch cache on first use). Distant people keep
the uniform level, and without `mediapipe` every person does. In a blind test on
close-up frames, judges preferred the part-aware turbo rendering 46 times out of
48. Realism and flicker measurements for both engines are in
[the experiment log](docs/EXPERIMENTS.md).

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

Bundled weights: `fake_thermal.pt` (U-Net engine, adversarially fine-tuned)
and `turbo.pt` (turbo engine LoRA weights; SD-Turbo is downloaded separately
under its own licence). See [weights notice](fake_thermal/weights/README.md).

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
