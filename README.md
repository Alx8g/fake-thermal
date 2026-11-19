# fake-thermal

CPU-friendly RGB-to-relative-thermal prediction. Public repository:
https://github.com/Alx8g/fake-thermal (origin configured; not pushed).
The local checkout remains `~/repos/thermalize`.

## Convert a clip or photo

Requires `ffmpeg` and `ffprobe` on PATH, plus the exported local weights:

```sh
.venv/bin/pip install -e .
.venv/bin/python scripts/export_weights.py outputs/ckpt/best.pt
.venv/bin/fake-thermal input.mov -o out.mp4 --palette ironbow --side-by-side
.venv/bin/fake-thermal photo.jpg -o thermal.png --palette arctic --raw
.venv/bin/python app.py  # upload/preview/download UI on port 7860
```

Palettes: `ironbow`, `white-hot`, `black-hot`, `rainbow`, `arctic`. Default working
short side is 384 (`--size 512` for more detail); inference uses two CPU threads.
`--smoothing .65` retains 65% of the prior heat map and resets on detected hard
cuts; use `--smoothing 0` to disable. `--raw` adds a grayscale `_raw` file.
See [INFERENCE.md](INFERENCE.md) for behavior, demos, speed and limitations.

See [TRAINING.md](TRAINING.md) for the ResNet18 U-Net, CPU benchmark, training
commands, checkpoints and validation protocol. Predictions are normalized
white-hot relative intensity, **not measured temperatures**.

## Data preparation

**Current dataset:** see [DATA_REPORT.md](DATA_REPORT.md) for the full LLVIP +
subsampled KAIST expansion, current counts and per-source previews. The remainder
of this README is the **historical initial audit**; its LLVIP/KAIST exclusions,
download budget and output totals have been superseded. Reproduction commands
below still apply; `--download` now fetches the full LLVIP and both KAIST ZIPs too.

CPU-only groundwork for RGB-to-synthetic-thermal video translation. No model or
training pipeline yet. These are display-intensity thermal images, not calibrated
temperature measurements.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/pip install torch==2.14.1+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py --download
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v
```

The downloader fetches the three inspected archives/parquet sources, totaling
approximately 3.77 GB compressed. KAIST is deliberately not downloaded. Existing
populated `data/pairs/` is protected against overwrite; move it before rebuilding.
Preparation reads ZIP members directly and streams parquet in 16-row batches.
Additional dependency `pyarrow` is used to inspect the video parquet files.

## Inspected sources

| Hugging Face dataset | Actual contents / resolution | Usability |
| --- | --- | --- |
| `hieupth/llvip_384` | `images.zip`: 15,488 JPEGs, all 384×384, under `images/LLVIP/visible/{train,test}` (12,025/3,463). | **Zero usable pairs: the archive contains no infrared images.** The card's counts are visible-image counts, not paired files. Cannot check this mirror's cross-modal alignment. |
| `jsonhash/FLIR_aligned` | `aligned.zip`: `align/{JPEGImages,Annotations,AnnotatedImages}`. Exactly 5,142 matching `_RGB.jpg` / `_PreviewData.jpeg` pairs in `JPEGImages`, all 640×512. Other folders contain annotations/masks and rendered annotations, not extra pairs. | Used. RGB has already been geometrically transformed to thermal dimensions. Random samples show good approximate registration of road markings, vehicles and buildings; some local parallax/edge mismatch remains. Not perfectly pixel-aligned ground truth. |
| `AmineMarnissi/ThermalKaist` | Card describes 9,853 images (wording does not establish matched-pair count), `JPEGImages`, `Annotations`, `ImageSets`, Pascal VOC XML. `RGBKaist.zip` 3,023,506,314 bytes plus `thermalKaist.zip` 1,023,706,345 bytes: **4.05 GB combined**. | Not downloaded because both modalities exceed the ~3 GB budget. Card says original KAIST has aligned day/night pairs, but this modified mirror's matching, resolution and alignment were **not verified**. |
| `Maxim37/flir_adas_video_pairs` | Four parquet shards, 1,588,194,544 bytes; 3,749 paired rows from eight RGB recordings. Embedded image structs plus filenames, sizes, COCO IDs and annotations. RGB 1224×1024 (2,151 rows) or 1024×768 (1,598); thermal always 640×512. | Downloaded and inspected, **excluded**: paired in time, not pixel-registered. Contact sheets show different fields of view, framing and object positions; simply resizing is insufficient. No speculative automatic warp applied. |

### Day and night

No machine-readable day/night labels were found in these inspected payloads/cards.
LLVIP is explicitly low-light: all 12 sampled visible frames are night scenes.
FLIR aligned contains both: its 12 random samples show 10 daylight / 2 night.
FLIR video also contains day and night (4 daylight/twilight / 8 night in the
12-frame random preview). These are **sample observations, not corpus-wide ratios**.
KAIST's card claims both day/night; the mirror was not downloaded to verify it.

### Licensing — do not treat public download as an open-source data license

- LLVIP mirror README defers license/citation to the original dataset. The
  [original LLVIP README](https://github.com/bupt-ai-cz/LLVIP) permits
  **non-commercial** research, teaching, publication and personal experimentation,
  subject to its linked license terms. It is not unrestricted commercial data.
- FLIR aligned README metadata says `license: other` and points to the
  [FLIR terms](https://www.flir.com/oem/adas/adas-dataset-form/), rather than
  supplying an OSI/Creative Commons license. Full downstream/model redistribution
  permissions have not been established.
- ThermalKaist card says `license: other`; no explicit grant in its README.
- FLIR video card supplies schema/statistics but **no license grant**. Original
  FLIR terms may apply; this was not independently resolved.

Do not redistribute these data or assume commercial model rights. Code licensing
has deliberately not been chosen on the owner's behalf.

## Prepared output and split limitations

```
data/pairs/{train,val}/rgb/flir_XXXXX.jpg
data/pairs/{train,val}/thermal/flir_XXXXX.png
```

Matching **stems** (JPEG RGB / PNG 8-bit grayscale thermal), both **360×288**,
aspect ratio preserved, short side 288. 4,830 train + 260 validation = **5,090
usable output pairs**; 52 additional FLIR pairs excluded as a split buffer.
Validation is 5.11% of retained pairs.

The archive provides frame IDs but no reliable scene/recording IDs. The script
uses complete contiguous index runs (break at gaps >20) as **inferred sequence
groups**, not random individual frames. The run closest to 5% is held out:
FLIR IDs 06995–07567 (260 present frames). Any other group intersecting a
150-index embargo around validation is excluded wholesale. The measured closest
train/validation frame-index distance is 181. This prevents adjacent indexed
frames straddling the split, but **does not certify a true scene-disjoint split**:
revisits or the same physical scene in another run can remain. Ground-truth
recording metadata/manual sequence curation is needed before publishing metrics.
The sampled validation run is nighttime-only, so it is not a balanced day/night
benchmark. Some pre-aligned RGB frames have black warp borders and visible
residual mismatch; no border mask or extra registration is invented here.

`data/manifest.json` records each FLIR pair and split/exclusion; `data/summary.json`
records counts, resolutions and inferred groups. `data/prepare.log` holds the run
output. The script fails on missing FLIR partners and mismatched dimensions.

## Visual evidence and storage

`data/preview.jpg`: 12 random FLIR aligned RGB|thermal pairs, 12 LLVIP visible
images with explicitly marked missing thermal slots, and 12 excluded video pairs.
Visually inspected: FLIR aligned approximately registers; video does not.
`data/val_preview.jpg`: supplemental random validation pairs.
No KAIST preview is possible without downloading its oversized paired payload.

Local disk after preparation: **5.3 GiB project total**, including **4.0 GiB data**
(3.6 GiB originals, 462 MiB prepared pairs) and **1.3 GiB venv**; rounded `du -sh`
values. Data and venv are gitignored. There is no remote repository.
