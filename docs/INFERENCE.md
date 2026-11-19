# User-facing conversion

## CLI and web UI

```sh
.venv/bin/pip install -e .
.venv/bin/python scripts/export_weights.py outputs/ckpt/best.pt
.venv/bin/fake-thermal input.mov -o out.mp4 --palette ironbow --side-by-side --raw
.venv/bin/fake-thermal input.mov -o out.mp4 --size 512 --smoothing .5 --weights fake_thermal/weights/fake_thermal.pt
.venv/bin/fake-thermal input.jpg -o out.png --palette white-hot
.venv/bin/python app.py
```

`ffmpeg` and `ffprobe` must be installed separately. The installed console entry
point is `fake-thermal`. The Gradio UI is on port **7860** and offers image/video
upload, palette, side-by-side, smoothing, optional grayscale, preview and download.
UI jobs are serialized to avoid simultaneous inference competing with training.
Outputs persist under `outputs/ui/<unique-id>/`; remove old results manually
when no longer needed. No public Gradio share tunnel is created. The UI binds
all local network interfaces; use only on a trusted machine/network.

## Media behavior

- `--device auto|cpu|mps|cuda` applies to both models. Auto selects CUDA, then
  MPS, then CPU. Device choice is also available in the UI, and final timing
  JSON includes the effective device. Model weights/inputs are fp32 on every
  device; MPS entry points enable CPU fallback for unsupported Torch ops.
  CPU-based guided filtering, palettes, decode and encoding remain on CPU.
  CUDA and MPS need compatible platform PyTorch wheels and hardware; they were
  not hardware-tested on this host.
- `--auto-contrast` defaults on (also in UI). Immediately before palette
  mapping, guided full-size heat is stretched between its 2nd/99.5th percentiles.
  Bounds retain 90% of their previous value per frame, resetting with the heat
  smoother on a detected cut or shape change. Constant/near-flat maps retain
  their original intensity instead of dividing by zero. `--no-auto-contrast`
  restores fixed intensity display. Raw heat exports are **not** display
  normalized. The stretch changes only presentation, not temperature estimates.

- CPU inference defaults to **two Torch threads**, OpenCV one thread and one
  thread per ffmpeg codec/filter pipeline. Training is not stopped or altered.
- Working short side defaults to **384**. The model pads to multiples of 32 and
  unpads internally; heat maps use a fast guided filter with RGB luminance as
  guide (box-filtered linear coefficients at working resolution, reconstructed
  at upright source size). This follows visible edges, not true thermal edges.
- ffmpeg handles MOV/HEVC decode and rotation metadata. Image EXIF orientation
  is applied. Output is H.264/yuv420p MP4 at the source average frame rate;
  variable-rate timestamps become constant average-rate timing, not an exact
  VFR timestamp copy. Odd dimensions get at most one pixel of padding for H.264.
- Audio: copy the first source audio stream without re-encoding when compatible
  with MP4; otherwise fall back to AAC 192 kbps (e.g. MOV PCM). Silent inputs
  remain silent. Synthetic tests confirm AAC packet payloads are unchanged.
- EMA smoothing is at working resolution. `.65` means 65% previous heat map;
  zero disables smoothing. Mean pixel change on a reduced RGB frame detects
  hard cuts and resets history. This is a heuristic, not optical-flow tracking;
  fast motion can trail and similar-looking cuts can evade detection.
- Semantic priors default on at strength .7. `--no-semantic` disables semantic
  and emissive priors; `--semantic-strength 0` also bypasses segmentation.
  DeepLabV3 MobileNetV3 runs at 320px short side, using the pretrained
  COCO-with-VOC-labels weights from torchvision, cached on first use in Torch's
  model cache (`TORCH_HOME` can relocate it). Network failures are reported,
  not silently treated as successful semantic inference; disable semantic mode
  to work offline before weights have been cached.
- Soft person probabilities target .9; bird/cat/dog/horse/sheep/cow target .85;
  car/bus/motorbike/train target .6. Other classes add no semantic confidence.
  Saturated bright orange/yellow suggests intensity 1; very bright near-white
  suggests .8, with Gaussian feathering. The semantic and emissive targets are
  confidence-weighted together, with maximum confidence used for fusion:
  `model + strength * confidence * max(prior - model, 0)`. Fusion precedes EMA.
  This is a typical-appearance assumption, not a thermal detector. Orange
  objects and white surfaces can falsely heat up; a cold person or vehicle can
  also be incorrectly heated. The original training validation L1 does not
  measure or validate this inference-only prior.
- `--raw` writes `_raw.mp4` alongside videos, or lossless grayscale `_raw.png`
  alongside images. Raw video is grayscale H.264, so it is display-intensity
  data with codec loss, not scientific floating-point measurements.
- Processing streams frames rather than loading entire clips. Progress reports
  frame rate and ETA. Final JSON timing includes startup, decode, encode/mux.
- Images support PNG/JPEG/WebP output; grayscale PNG is recommended for editors.
  Existing result files can be overwritten, but input=output is rejected.

Default weights are `fake_thermal/weights/fake_thermal.pt`, an fp16 **state dict** exported
from trusted local training checkpoints. Inference loads those weights into an
fp32 CPU model. The local export is approximately 24 MiB, from **iteration 3500**,
validation L1 **0.184265**. Re-export after training to use improved weights.
Checkpoint export uses pickle and must only read trusted checkpoints. Normal
inference uses `weights_only=True`. Early weights are bundled under the
[non-commercial/research weights notice](../fake_thermal/weights/README.md); media are not committed.

## Latest display-contrast demos

The semantic-on videos and the six `*_semantic_frame_*.jpg` images below were
regenerated with `--device auto --auto-contrast`. Auto selected CPU on this host.
The newest available best checkpoint was re-exported and is still iteration
3500 / validation L1 0.184265. Training was left running. Bounds use a .9 EMA
of 2nd/99.5th heat percentiles, reset on cuts. Raw heat is unchanged by display
contrast. Logs: `outputs/demo/{kaist,commons}_auto_contrast.log`.

The regenerated clips ran at 3.69 fps (KAIST) and 3.93 fps (Commons), including
loading/encoding while training continued. Sample frames were inspected: the
stretch increases visible dynamic range and makes close-up people brighter,
but the KAIST building remains bright because percentile mapping preserves the
model's relative heat ordering. It cannot selectively cool incorrect hot
background predictions. Updated UI controls have callback tests but remain
visually unverified; CUDA/MPS hardware execution is also unverified here.

## Semantic-prior timing (before display contrast, iteration 3500 weights)

Both demos were regenerated with ironbow, side-by-side, guided upsampling,
default model size 384, default semantic size 320 and strength .7. The same
inputs were processed with `--no-semantic` for comparison. Two Torch threads
were used; the original training process remained running. Cached semantic
weights avoided including network download time. Timings are end-to-end on a
shared CPU, including model loading and video muxing, not isolated model speed.

| Input | Semantic on | Semantic off |
| --- | ---: | ---: |
| KAIST, 200 frames | 61.59 s / **3.25 fps** | 38.28 s / **5.22 fps** |
| Commons, 115 frames | 28.74 s / **4.00 fps** | 18.63 s / **6.17 fps** |

```sh
python scripts/export_weights.py outputs/ckpt/best.pt
fake-thermal outputs/demo/kaist_day_input.mp4 -o outputs/demo/kaist_day_thermal.mp4 --side-by-side --palette ironbow --semantic
fake-thermal outputs/demo/commons_input.mp4 -o outputs/demo/commons_thermal.mp4 --side-by-side --palette ironbow --semantic
fake-thermal outputs/demo/kaist_day_input.mp4 -o outputs/demo/kaist_day_model_only.mp4 --side-by-side --palette ironbow --no-semantic
fake-thermal outputs/demo/commons_input.mp4 -o outputs/demo/commons_model_only.mp4 --side-by-side --palette ironbow --no-semantic
```

Semantic-on sample frames: `outputs/demo/kaist_semantic_frame_{1,5,9}.jpg`
and `outputs/demo/commons_semantic_frame_{1,2,4}.jpg`. Model-only comparison
frames: `kaist_model_only_frame_5.jpg` and `commons_model_only_frame_2.jpg`
in the same directory. Detailed timings are in `{kaist,commons}_semantic_{on,off}.log`.

Inspected samples show substantially hotter, more recognizable close-up people
in the Commons clip versus model-only output. Distant KAIST pedestrians remain
faint: the small segmentation model misses small people, and guided filtering
cannot invent missing detections. Real flame footage was not available in these
demos; emissive behavior is tested on synthetic color patches. New Gradio
controls are wired to conversion but have not been visually reverified.

## Original demos run while training continued (historical iteration 2000)

Commands:

```sh
.venv/bin/python scripts/make_demo.py
.venv/bin/fake-thermal outputs/demo/kaist_day_input.mp4 -o outputs/demo/kaist_day_thermal.mp4 --side-by-side --raw
.venv/bin/fake-thermal outputs/demo/commons_input.mp4 -o outputs/demo/commons_thermal.mp4 --side-by-side
```

| Input | Frames/fps | Default-size end-to-end time | Throughput |
| --- | --- | ---: | ---: |
| KAIST day, 640×512 | 200 / 20 | 41.58 s | **4.81 fps** |
| Wikimedia, 720×720 | 115 / 24000÷1001 | 21.04 s | **5.47 fps** |

These measurements include loading weights and final encoding; KAIST also
encodes a second grayscale video. Six-thread training was running concurrently.
Loop-only rates were 5.07 and 6.12 fps respectively. This is not real-time 60fps.

KAIST uses held-out **set06_V003** with pedestrians, 200 available RGB frames
rendered at 20fps into a ten-second clip. **The mirror is sparse**: this is a
time-compressed montage of sequential available frames, not native consecutive
20fps footage. Exact archive members are in `outputs/demo/kaist_frame_members.json`.

Wikimedia source: [Kamala Harris and Joe Biden walking at the White House in
slow motion](https://commons.wikimedia.org/wiki/File:Kamala_Harris_and_Joe_Biden_walking_at_the_White_House_in_slow_motion.ogv),
White House, 2021-01-26. Commons identifies it as **public domain US government
work** (with personality-rights notice). Original: `outputs/demo/commons_original.ogv`.
The 4.8-second input was H.264-transcoded from the original OGV without audio
(the downloaded source is silent). This is a format-generalization demo, not
an assertion about anyone's actual body temperature.

Outputs under `outputs/demo/`:

- `kaist_day_thermal.mp4`, `kaist_day_thermal_raw.mp4`
- `commons_thermal.mp4`
- `kaist_frame_1.jpg`, `kaist_frame_5.jpg`, `kaist_frame_9.jpg`
- `commons_frame_1.jpg`, `commons_frame_3.jpg`
- `ui_result.png` — UI image upload, Run, side-by-side preview and downloads

Sample frames and the UI result were visually inspected. **Early predictions
are visibly blurry/coarse**, not finished-model quality. They do establish the
input → model → smoothing → palette → video pipeline. The ongoing training may
improve detail; neither these demos nor the final model measure temperature.

## Checks

`python -m unittest discover -s tests -v`: 26 tests passed, including device
auto priority/unavailable choices, CPU fp32 transfers, training overrides,
synthetic CPU training/validation/checkpointing with a custom crop, UI option
forwarding/errors, percentile EMA/cut reset/constant handling and raw invariance,
as well as semantic
soft probability mapping, never-cooling/zero-strength fusion, synthetic flame
and bright-white/negative-color patches, guided edge/shape/empty handling,
mocked lazy segmentation and semantic controls without network access, palette
endpoints/clipping, EMA and scene-cut reset, one-second video CLI conversion,
AAC preservation, exact frame count/fps, raw export, MOV rotation, single-image
conversion, invalid inputs, existing shape and training-loss tests.

A separate synthetic **3840×2160 HEVC MOV at 60fps** converted successfully:
`outputs/demo/hevc4k60_thermal.mp4`, same resolution, 60/1 fps, all three frames.
It is a short decode/encode smoke test, not genuine iPhone/HDR footage. Real
iPhone HEVC/HDR footage was not available to visually validate color handling.
No HDR-specific tone mapping or learned superresolution is implemented.

The system ffmpeg predates `-fps_mode`; the decoder uses the equivalent
`-vsync 0`. Failure cleanup has bounded process waits to avoid hanging encoders.
