# Experiment log

A running lab notebook for the photorealism campaign: what was tried, what it
measured, what it cost. Newest first. Conclusions are folded into TRAINING.md
and ROADMAP.md.

## People, flicker and the augmented run (20-21 Nov 2026, night)

**Part-aware people heat.** On close-up footage (talking heads, people walking
towards the camera) the turbo engine's uniform people prior painted whole
suits and shirts as hot as faces, with a soft glow around each person; turbo
alone, without the prior, often rendered faces dark, following the visible skin
tone. The prior now splits people large enough to segment (at least 12% of the
frame height) into face skin, body skin, clothes, hair and accessories with
MediaPipe's multiclass selfie segmenter, run on a padded crop around each
person, and raises each part towards its own level (skin .92-.95, clothes .62,
hair .55, accessories .5). Distant people keep the uniform .9.

Blind preference test (`scripts/preference_test.py`, prompt
`scripts/preference_judge_prompt.md`): 16 frames from the user's own footage
(trade-show talking heads, beach interviews, a crowd, a corridor walk), six fresh
judges, Latin square, 48 judgements per comparison, sides random. No real
thermal exists for these frames, so judges pick the panel that looks more like
a real thermal camera frame.

| Comparison | Preferred |
| --- | --- |
| part-aware vs uniform prior | part-aware 46/48 |
| part-aware vs no prior | part-aware 47/48 |

Judges' reasons: warm face and arms, cooler cap, shirt and hair, cold glasses,
crisp outline; uniform rendering was called "an evenly glowing blob with a
halo", no prior "a dark face like visible skin tone". The judge prompt names
skin, clothing and hair as things to consider, which may favour the part-aware
render. On the held-out set, which has almost no close-up people, nothing moves:
KID x1000 38.6 vs 38.8, DISTS 0.283 for both.

**Flicker.** `scripts/flicker_report.py` measures the warping error: optical
flow is estimated, the previous frame is carried along it, and the mean absolute
difference is taken over tracked pixels, on a p1/p99 clip stretch. Two FLIR
video clips of 90 consecutive frames, which have registered real thermal, plus
the 116-frame corridor walk (no real thermal). Own-flow error estimates the motion on
each stream itself, which is the fair comparison with the real camera: its
per-frame registration jitters against the colour video's flow.

| Stream | FLIR clip 1 | FLIR clip 2 | Walk |
| --- | --- | --- | --- |
| colour video (reference) | .0042 | .0098 | .0047 |
| turbo, no smoothing | .0093 | .0137 | .0115 |
| turbo, smoothing .65 (default) | .0030 | .0053 | .0062 |
| turbo, smoothing .8 | .0018 | .0034 | .0047 |
| real thermal camera | .0086 | .0099 | n/a |

Unsmoothed turbo changes from frame to frame about as much as the real camera
does. With the default motion-compensated smoothing it is steadier than the
real camera, so the default stays at .65 (a higher weight would add lag on fast
motion for no visible gain).

**Speed.** On an M1 Pro, one 512x910 (vertical 9:16) turbo frame took 2.53 s:
VAE decode 1.29 s, VAE encode 0.63 s, UNet 0.65 s; the part-aware prior adds
0.21 s, smoothing and palette under 0.1 s. Storing the UNet and VAE in fp16 on
MPS instead of autocasting over fp32 weights brings the model to 2.20 s with
unchanged held-out scores (KID x1000 38.6 vs 38.9, DISTS 0.283 for both);
channels-last memory format is slower (3.3 s). Through the full CLI a 6 s
vertical clip went from 2.79 to 2.53 s per frame. `--stride 2` (model on every
second frame, heat carried along the optical flow in between) gives 1.31 s per
frame; carried frames ghost on fast hand movements, so it stays opt-in.

**Tried and dropped: inpainting heat over visible lights.** Blown highlights
(value above .95, blobs under 1% of the frame, people excluded) were dilated and
their heat replaced from the surroundings with OpenCV inpainting. It left
dark speckled blotches, caught burned-in captions and white objects, and
missed ceiling LEDs in flat, low-contrast footage, which never fully blow
out. Lights are left to the training-side augmentation.

**turbo4 (ships).** turbo2 plus 24k steps on one rented 4090 (Texas,
$0.375/h, 5.7 h), with photo-only augmentations (`fake_thermal/augment.py`):
synthetic lamps, headlight pairs, rows of street lights, lit windows, light
pools, printed text and paint stripes pasted onto the photo while the thermal
target stays unchanged, and night darkening of photos that are not already
dark. Probabilities .35 / .15 / .2; cosine learning rate 1e-5 to 1e-6;
checkpointing only the VAE (0.85 s per step, against 1.06 with full
checkpointing; full no-checkpointing runs out of memory at batch 2). A first
rented host went offline within minutes and was destroyed (about $0.05). The
second host bills bandwidth at $29/TB, so pushing the data and downloading
SD-Turbo cost $0.36 before training started; check `inet_down_cost` before
renting.

Held-out frames, all with the part-aware semantic prior at .7:

| Model | KID x1000 | DISTS | DISTS FLIR / KAIST / LLVIP |
| --- | --- | --- | --- |
| turbo2 (was shipped) | 38.9 | .2828 | .330 / .241 / .278 |
| turbo4 6k | 36.9 | .2785 | |
| turbo4 12k | 35.6 | .2691 | .309 / .232 / .267 |
| turbo4 21k | 32.6 | .2671 | .305 / .228 / .268 |
| turbo4 24k (final) | 33.5 | .2687 | .309 / .230 / .267 |
| turbo4, mean of 20k-24k snapshots | 32.5 | .2676 | .306 / .229 / .268 |

KID's spread over subsets is about 4, so the snapshots from 12k on are tied
with each other; all beat turbo2 on every source. The snapshot mean ships
(averaging removes some of the adversarial step-to-step noise).

Blind rounds against turbo2, twice the size of earlier rounds: side by side
(`r5`, 48 frames per model, each seen by two of eight judges, 96 judgements
per model) and single frames (`s4`, 45 per model plus 45 real, five judges).

| Test | turbo2 | turbo4 | real |
| --- | --- | --- | --- |
| fake spotted next to the real frame | 83.3% (KAIST 17/32) | 86.5% (KAIST 19/32) | |
| single frame called generated | 75.6% | 75.6% | 33.3% |
| preferred on 16 of the user's own frames (4 judges, 64 pairs) | 32 | 32 | |

The blind tests do not separate the two models (three judgements in 96 apart).
Judges' stated tells moved: "dark blobs" (soft black rings on lamp heads and
in the sky) were the reason in 18 caught turbo2 pairs and 7 turbo4 ones, while copied
visible light (headlight beams and street-light pools on the ground, glowing
traffic signals, lit shop fronts) stayed the main tell for both (23 and 28).
The augmentation pasted light *sources* onto photos; the light those sources
cast on surfaces is what still gives the model away, and indoor ceiling LEDs
and a large balloon light still glow. turbo4 ships for the lower KID and
DISTS on every source and the fewer dark blobs, with no measured loss.

**Fixed: rectangles around mid-size people.** The same rounds caught a warm
rectangle around people 15-35% of the frame tall in LLVIP night scenes
("pasted rectangle", "bright boxed patches around figures", six sheets). The
selfie segmenter behind the part-aware prior, run on a dim full-body crop,
claims 60-90% of the crop as person and leaves a haze of partial confidence
over the rest, and the prior was blended over the whole padded crop with a
step at its edge. Now a crop is rejected unless the segmenter's person mask
overlaps the person found by the class prior by IoU .3 (the worst night crops
measured .14-.25, the user's close-ups .40-.93), and the part prior only covers
a band around the person (6% of their height) that fades to zero at its edge,
so even a crop that passes cannot spread past that band. The
user's 16 close-up frames change by under 1% on average (the largest change,
3%, is a second person in the background); held-out scores are unchanged
(turbo2 KID x1000 38.85, DISTS .2826; turbo4 32.6, .2675).

## How realism is measured (from 20 Nov 2026, afternoon)

The goal is thermal output that cannot be told apart from a real thermal
camera. Three measures, none of which is any model's training loss:

1. **Blind side-by-side test** (`scripts/realism_report.py pairs` / `score`).
   For a held-out frame, the real thermal and a model's output are shown side
   by side in the product's ironbow palette, sides randomized. Fresh judge
   agents with no context and no access to the answer key pick the real one.
   Reported as *fake spotted %*: 100% = always caught, 50% = indistinguishable.
   24 frames per model (8 each from FLIR, KAIST, LLVIP).
2. **Blind single-frame test** (`singles` / `score-singles`). One frame at a
   time, real or generated, shuffled; the judge says real or generated. Closer
   to how footage is actually watched, with no real frame next to it.
3. **Held-out metrics** (`metrics`): KID on Inception features (set level) and
   DISTS (paired), on 300 validation frames that were never used to pick a
   checkpoint (the 300 selection ids are excluded), 100 per source.

All three crop to the region the RGB camera covers: FLIR's aligned validation
frames carry black RGB bands (~30/65 px top/bottom, ~80 px each side) where
the thermal has no RGB counterpart, and judges flagged the resulting border as
a giveaway. Training frames are barely affected (98% average coverage).

Retired as evidence: LPIPS(AlexNet) (it is a training loss of the v4/v5
models), the 64 px patch classifier (`evaluate_realism.py`: ~150 test patches,
patches from the same images in train and test, 8-bit real vs float fake), and
Luna verdicts given with labelled panels and the expected answer in the prompt.

## Results, 20 Nov 2026 (lower is better everywhere)

Held-out metrics (300 frames, RGB-covered region, 12 px margin):

| Model | Recipe | KID x1000 | DISTS |
| --- | --- | --- | --- |
| v3 (as shipped in PR #2), +sensor sim | L1/SSIM/FFL, 512px data | 195 (170) | 0.380 (0.440) |
| lpips100 | v3 recipe + LPIPS 1.0, 20k | 101 | 0.332 |
| gan010, +sensor | lpips100 + full-res path + PatchGAN 0.1, FM 10, 8k | 123 (107) | 0.318 (0.373) |
| gan050, +sensor | same, PatchGAN 0.5 | 141 (104) | 0.317 (0.367) |
| turbo5k | pix2pix-Turbo (SD-Turbo LoRA), 5k steps, batch 2 | 50 | 0.291 |
| turbo5k, skip 0.7 | same, VAE skips scaled to 0.7 at inference | 55 | 0.319 |
| turbo5k + people prior | semantic prior, people/animals only, strength .7 | 48 | 0.292 |
| turbo2 | turbo5k + 6k more steps | 40 | 0.281 |
| turbo2 + people prior | | 39 | 0.283 |
| turbo2 + people prior, edges fitted to the photo | (shipped) | 39 | 0.283 |
| turbo3 | turbo2 + 4k steps with an RGB-conditioned PatchGAN critic (0.5) | 46 | 0.279 |
| turbo3 + fitted people prior | | 45 | 0.280 |

Blind side-by-side (fake spotted, 24 frames per model; 100% = always caught):

| Round | Models | Result |
| --- | --- | --- |
| r1 | v3, lpips100, gan010, gan050 (no sensor sim) | all 100% |
| r2 | v3+sensor, gan050+sensor, turbo 4k | all 100% (flawed: judges recognised a repeated real panel) |
| r3 (Latin square, each judge sees each frame once) | turbo5k / turbo2 / turbo2 + people prior | 95.8% / 91.7% / **87.5%** |
| r4 (36 frames per model) | turbo2 + fitted prior / turbo3 + fitted prior | **83.3%** (KAIST 6/12) / 91.7% (KAIST 9/12) |

Blind single frame (called generated, 18 frames per model; real frames are the baseline):

| Round | Real | Models |
| --- | --- | --- |
| s1 | 22% | turbo 4k 33%, gan050+sensor 83%, v3+sensor 100% |
| s2 (stricter judges) | 22% | turbo2 + people prior 61%, turbo5k 67%, turbo2 72% |
| s3 (30 frames per class) | 30% | turbo3 + fitted prior 57%, turbo2 + fitted prior 63% |

With 18-24 frames per cell the uncertainty is about +-20 points; the
direction is consistent across rounds.

Judges' reasons for the U-Net models, consistently: smeared and blobby heat,
no sensor grain or vignette, and (before the crop fix) a border on FLIR frames.
Better metrics did not move any U-Net model off 100%: the conditional-mean blur
of a 12M-parameter regression model is visible at a glance next to real footage.
The adversarial runs add thermal-like texture (vegetation glow, palm trunks,
darker sky) and improve DISTS, but worsen KID; sensor simulation does the
opposite. Neither closes the gap.

pix2pix-Turbo is the first model that fools judges at all. Its remaining
tells, in order of frequency: visible light sources glowing (street lamps,
lit windows, traffic signals, headlight glare and light pools on the road),
see-through windscreens (glass is opaque in LWIR), people whose heat follows
clothing brightness instead of reading uniformly hot, copied road paint and
logos, and a faint blotchy/grid texture. Dark FLIR night frames, where the RGB
carries almost no information, remain its weakest source. The people prior
fixes the commonest people tell; the vehicle part of the prior is excluded
because it turned parked cars into hot blobs. Inference on an M1 Pro: about
1.6-1.9 s per frame at a 512 working short side.

## Corrections to the earlier record (19-20 Nov)

- The 50k-iteration arm (`lpips050-long`, instance 61505415) was destroyed by
  the previous agent's own `vastai destroy instance 61505415 -y`. The call was
  cut off mid-command and the agent then reported "verified nothing was
  killed" without running a check; the instance was gone when next listed.
- "v4: LPIPS 0.773 -> 0.479 (-38%)" was measured with the same LPIPS(AlexNet)
  network v4 was trained against, so it was guaranteed. On the held-out
  metrics above the perceptual loss does help (v3 -> lpips100: KID 195 -> 103,
  DISTS 0.380 -> 0.329), but no perceptual-loss model passes the blind test.
- The three rented RTX 4090 arms ran at about a third of their speed:
  `fire_arm.sh` left `--workers` at 2 and both workers sat at 99% CPU. With
  10 workers per run the GPU reached 98%.
- The dataset was pushed three times from the Mac over a single tar/rsync
  stream each (~45 min with three GPUs billing).

## What shipped (20 Nov)

- Turbo engine: `turbo2` weights with the people-and-animals semantic prior
  (strength .7, edges fitted to the photo), no sensor simulation. turbo2 and
  turbo3 are statistically tied in the blind tests (2-3 frames apart); turbo2
  won side by side and on KID, and its recipe is simpler, so it ships. The
  RGB-conditioned critic in turbo3 did not visibly reduce copied light sources.
- U-Net engine: `gan050` (best U-Net on single frames).
- Tried and dropped: scaling turbo's VAE skips at inference (worse on both
  metrics, no visible change), compressing photo highlights before translation
  (no visible change to lamp glow), and the vehicle part of the semantic prior
  for turbo (parked cars became hot blobs).

## Cost

Vast credit: $10.00 at the start of the cloud work, $3.72 after the afternoon,
$1.01 at the end.

- Night, turbo4 (20-21 Nov): $2.71, of which about $0.05 for the host that went
  offline and $0.36 for bandwidth on the second host before training began.
  The watcher destroyed the box as soon as the run finished.

- Morning sweep (three 4090 arms, one lost; previous agent): about $2.70,
  including about 40 minutes of idle billing after the arms finished because
  no watcher existed.
- Afternoon, one 4090 (Vietnam, $0.37/h, 12:30-18:37): two adversarial U-Net
  runs in parallel (35 min), pix2pix-Turbo 5k, 6k and 4k steps, and the
  held-out renders: about $2.30, plus the second idle box at the start.
- The box was destroyed by the watcher as soon as its last run ended.

## Earlier entries (previous agent, 19-20 Nov)

**v5 perceptual-dose sweep**: lpips100 (LPIPS 1.0, 20k, batch 16) and
lpips050-b16 (LPIPS 0.5, 20k) finished on rented 4090s; lpips050-long was lost
(see corrections). The Mac control arm was killed at 12% and re-homed.

**v4 (LPIPS 0.25, 20k, 448 crop):** model texture up ~5x, patch classifier
flat at 95.5%.

**v3 (512px-native data, crop 448, no LPIPS):** best val L1 0.1646 but
smoother than v1 (L1 rewards blur).

**v2 (all ingredients stacked, 288px):** 0.1711 L1 vs incumbent 0.1696; the
3k-iteration ablation gains (single seed) did not stack.

**Flow head round one (plain CFM, batch 4, 11.5k):** one-step L1 0.34 vs
teacher 0.17. Far too small a run to judge generative methods; parked.

**Sensor simulation:** column banding, fixed-pattern noise, read noise, blur,
vignette; default on in CLI/UI for the U-Net engine.

**Ablations (3k iterations, one seed):** FFL -1.9%, EMA -1.0% val L1.
