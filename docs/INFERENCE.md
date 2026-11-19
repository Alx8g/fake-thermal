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

- CPU inference defaults to **two Torch threads**, OpenCV one thread and one
  thread per ffmpeg codec/filter pipeline. Training is not stopped or altered.
- Working short side defaults to **384**. The model pads to multiples of 32 and
  unpads internally; heat maps are bilinearly upsampled to upright source size.
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
- `--raw` writes `_raw.mp4` alongside videos, or lossless grayscale `_raw.png`
  alongside images. Raw video is grayscale H.264, so it is display-intensity
  data with codec loss, not scientific floating-point measurements.
- Processing streams frames rather than loading entire clips. Progress reports
  frame rate and ETA. Final JSON timing includes startup, decode, encode/mux.
- Images support PNG/JPEG/WebP output; grayscale PNG is recommended for editors.
  Existing result files can be overwritten, but input=output is rejected.

Default weights are `fake_thermal/weights/fake_thermal.pt`, an fp16 **state dict** exported
from trusted local training checkpoints. Inference loads those weights into an
fp32 CPU model. The local export is approximately 24 MiB, from **iteration 2000**,
validation L1 **0.191197**. Re-export after training to use improved weights.
Checkpoint export uses pickle and must only read trusted checkpoints. Normal
inference uses `weights_only=True`. Early weights are bundled under the
[non-commercial/research weights notice](../fake_thermal/weights/README.md); media are not committed.

## Demos run while training continued

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

`python -m unittest discover -s tests -v`: 10 tests passed, including palette
endpoints/clipping, EMA and scene-cut reset, one-second video CLI conversion,
AAC preservation, exact frame count/fps, raw export, MOV rotation, single-image
conversion, invalid inputs, existing shape and training-loss tests.

A separate synthetic **3840×2160 HEVC MOV at 60fps** converted successfully:
`outputs/demo/hevc4k60_thermal.mp4`, same resolution, 60/1 fps, all three frames.
It is a short decode/encode smoke test, not genuine iPhone/HDR footage. Real
iPhone HEVC/HDR footage was not available to visually validate color handling.
No HDR-specific tone mapping or edge-guided superresolution is implemented.

The system ffmpeg predates `-fps_mode`; the decoder uses the equivalent
`-vsync 0`. Failure cleanup has bounded process waits to avoid hanging encoders.
