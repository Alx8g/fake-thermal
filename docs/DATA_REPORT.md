# Expanded paired dataset

Run with `.venv/bin/python scripts/prepare_data.py --download` on a fresh
`data/pairs/` directory. Existing populated output is intentionally protected;
move it before rebuilding. Omitting `--download` uses existing raw archives.
No GPU is used. All output images preserve aspect ratio with short side 288:
RGB JPEG and 8-bit grayscale thermal PNG, equal filename stems.

## Sources and splits

| Source | Available exact pairs | Train | Validation | Prepared size |
| --- | ---: | ---: | ---: | --- |
| jsonhash/LLVIP | 15,488 | 12,025 | 3,463 | 360×288 |
| AmineMarnissi/ThermalKaist | 9,853 | 1,528 | 455 | 360×288 |
| jsonhash/FLIR_aligned | 5,142 | 4,830 | 260 | 360×288 |
| **Total retained** | | **18,383** | **4,178** | **22,561 pairs** |

FLIR still excludes 52 buffer frames. The RGB-only LLVIP-384 mirror is no longer
used; FLIR video pairs remain excluded for lack of geometric alignment.

### Full LLVIP

Downloaded `jsonhash/LLVIP/LLVIP.zip` (4,003,858,172 bytes). The archive is usable;
no fallback was necessary. `LLVIP/visible/{train,test}/XXXXXX.jpg` pairs exactly
with `LLVIP/infrared/{train,test}/XXXXXX.jpg`. Original dimensions are 1280×1024.
The two-digit filename scene prefix identifies 26 scenes. We preserve original
train/test as train/validation: 19 train scenes and seven validation scenes;
the scene-prefix sets are disjoint. Validation is consequently larger than 5%,
as explicitly allowed by use of the official split.

LLVIP is a low-light dataset; sampled RGB frames are night scenes. The report
labels it `low_light` rather than inventing exact corpus-wide day/night counts.
No explicit frame-level day/night metadata was found. Visually, fixed scene
structures and people approximately register well; cross-sensor edge/parallax
and intensity differences remain. These are not calibrated temperature images.

### KAIST

Downloaded both `RGBKaist.zip` (3,023,506,314 bytes) and `thermalKaist.zip`
(1,023,706,345 bytes). Actual image counts are **9,855 RGB / 9,853 thermal**.
Pair only by exact `(set, sequence, frame)` key:

```
RGBKaist/JPEGImages/set00_V000_visible_I01225.png
thermalKaist/JPEGImages/set00_V000_lwir_I01225.png
```

There are 9,853 exact matches across 41 sequences. The two unmatched RGB entries
are filenames `set00_V000_I01629 (copy 1)` and `set00_V000_I01631 (copy 1)`;
these are excluded, never paired by ZIP order or nearest timestamp.
Both modalities are 640×512 PNGs (thermal stored as RGB, converted to grayscale).

The mirror is already filtered/sparse, not the complete 20fps video. We retain
**every fifth available matched frame within each sequence**, resetting the
stride at each sequence boundary, for 1,983 retained pairs. This is deliberately
not `frame_number % 5`, which would fail on some sampling phases. The measured
minimum retained original-frame spacing is 10 frames (0.5 s at 20fps); most
spacing is larger. No unmatched frame can shift the correspondence.

Hold out whole **sets**: sets00–05 train and sets06–11 validation (official KAIST
train/test grouping), so no sequence crosses splits. Lighting uses the KAIST
set convention, visually checked with one example per set:

| Split | Day (sets00–02 / 06–08) | Night (sets03–05 / 09–11) |
| --- | ---: | ---: |
| Train | 958 | 570 |
| Validation | 294 | 161 |

Random matched pairs show approximately registered roads, vehicles and buildings,
with local residual misregistration/parallax. Some thermal images are very dark;
their original display intensity is preserved, without per-image normalization.
This is suitable as approximate paired training data, not perfect pixel-level
ground truth. Whole-set separation prevents shared sequences, not repeated
physical routes appearing in distinct sets.

### FLIR retained unchanged

4,830 train / 260 validation, same inferred-index-run split and 150-index embargo
as the initial preparation. Day and night both occur (10 day / 2 night in its
12-sample preview). Exact corpus-wide lighting counts are unavailable. Its
validation run is nighttime-only, and true scene-disjointness remains unproven.
Some pre-aligned RGB frames have black warp borders and residual mismatch.

### SpeakingFaces (added 23 Nov 2026)

Close-up faces from the CC BY 4.0 SpeakingFaces set (FLIR T540 464x348 thermal
with the dataset's aligned webcam frames), fetched by `scripts/fetch_speakingfaces.py`.
Official subject split: subjects 1-100 train (12,753 pairs: two frames per still
clip of each trial and angle, one per spoken command), 101-142 held out (2,038
pairs). Stems are `sf_<subject>_<trial>_<session>_<angle>_<clip>_<frame>`; session
1 is still, 2 is speaking. Thermal is inverted from the 224-colour FLIR iron
palette by index; the alignment warp's black fill is trimmed from both images,
which are resized to a 512 short side. Subject 2 has no still-session zip and
subjects 16 and 28 no speaking zip upstream. The held-out frames are kept in a
separate data root, not `pairs/val`: validation selection expects exactly the
three original sources.

## Outputs and visual evidence

- `data/preview_llvip.jpg`: 12 deterministic random full-LLVIP pairs.
- `data/preview_kaist.jpg`: 12 deterministic random retained KAIST pairs.
- `data/kaist_sets_lighting.jpg`: supplementary one-pair-per-set lighting check.
- `data/preview_flir.jpg`: 12 deterministic random aligned FLIR pairs.
- `data/preview_video_excluded.jpg`: excluded video-pair alignment evidence.
- `data/preview.jpg`: combined contact sheet.
- `data/summary.json`: counts, native resolutions, lighting and split policies.
- `data/manifest.json`: exact archive members, source IDs, split and group keys.
- `data/prepare_step2.log`: complete preparation output.

Included-source previews were visually inspected; alignment conclusions are sample-based,
not a guarantee of every pixel in every pair. Older `data/val_preview.jpg` is
the initial FLIR-only supplementary sheet, not the expanded validation dataset.

Licensing caveats from the initial audit still apply: LLVIP is non-commercial;
KAIST mirror declares `other` without an explicit grant, and FLIR links its own
terms. Expanding the local dataset grants no new redistribution/model rights.
No datasets, previews or virtual environment are committed to git.

## Checks and storage

Six unit tests passed, covering exact KAIST keys, unmatched keys, empty input,
per-sequence stride reset, deterministic FLIR splitting, modes and dimension
errors. A full output scan decoded every image: 18,383 train and 4,178 validation
pairs, all matching stems, RGB/L modes and 360×288. Manifest checks confirmed
disjoint LLVIP scene prefixes (19/7) and KAIST sequence IDs (25/16).

Rounded local disk usage after removing the previous prepared output: 12 GiB
original data, 2.1 GiB prepared pairs, 1.3 GiB venv, **15 GiB project total**.
