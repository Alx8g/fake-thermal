# Experiment log

A running lab notebook for the photorealism campaign: what was tried, what it
measured, what it cost. Newest first. Conclusions are folded into TRAINING.md
and ROADMAP.md.

## Stripes, and judges who see the left panel as grainier (21 Nov 2026, late evening)

**Found: judges see the left panel of a side-by-side sheet as grainier.**
Four judges were given 16 sheets each with the same image in both panels
(crops of the user's frames, shown 1:1). Three picked the left panel on all
16 and called it "grainy, sensor-like" and the identical right panel "waxy,
denoised". The fourth split 8-8 but described more texture on the left in
nearly every answer and picked the right for it ("A shows a fine grid").
Moving the right panel to x=448, a multiple of every likely patch size,
changed nothing. Shown the same identical pairs as two separate files, two
judges called every pair identical and split 16-16.

So the side preference is in how the judge model sees one wide image, not in
the sheets. It explains the one-sided judges above. It also corrects the JPEG
explanation below: lossless PNG did not remove the effect, and "a grid on the
left one, whichever model it showed" is exactly this percept. Balanced sides
cancel it in the totals but it drowns small texture differences: a judge reads
the left panel's texture as either real sensor noise or a generator grid and
picks a side for the whole sheet set. The turbo4-cast people round still
holds: it won 49 of 49 sheets with turbo4-cast on the right and 30 of 47 with
it on the left, the side that worked against it.

`realism_report.py pairs` and `prefer` and `preference_test.py sheets` now
write each pair as two files, NNN_A.png and NNN_B.png, judged with
`scripts/pair_judge_prompt.md`; `--side-by-side` keeps the old single image.
`singles` was never affected.

**Ships: removing turbo's period-2 stripes.** Judges kept naming a "grid",
"mesh" or "checkerboard" on faces, walls and floors of the user's frames. A
period-8 pattern from the VAE decoder was the first suspect, but per-tile
period-8 templates were no stronger than period-7 or period-9 ones. The
averaged power spectrum found the real one: alternating rows and alternating
columns, at 26 times the median power of their frequency ring on the user's
frames, while the RGB photos have nothing like it. Measured the same way at
the panel size judges see (rows, columns):

| | FLIR | KAIST | LLVIP |
|---|---|---|---|
| real thermal | 3.5, 3.3 | 11.2, 11.4 | 3.4, 5.4 |
| turbo4-cast | 5.4, 11.9 | 24.8, 81.7 | 16.3, 12.9 |
| U-Net (gan050) | 3.7, 8.1 | 4.8, 3.9 | 1.7, 2.8 |
| turbo4-cast, stripes removed | 0.0, 0.0 | 1.7, 9.2 | 2.2, 1.2 |

Real cameras do not make it, and at video size (the heat is upsampled 2-2.7x)
it becomes fine stripes and moiré. `remove_stripes` in `fake_thermal/infer.py`
multiplies the heat by the alternating sign, blurs (Gaussian, sigma 3) to get
the pattern's local amplitude, and subtracts only that; it changes the heat by
0.7-1.2 grey levels on average. `TurboPredictor` applies it by default.

- Held-out, renders post-processed: KID x1000 31.4 against 32.1, DISTS .2652
  against .2659.
- Side by side (before the finding above): 49 to 47 on the user's frames,
  97 to 95 held-out, 49 to 47 on video-scale crops, with half the judges
  one-sided.
- Separate files, 1:1 crops of the frames upsampled to 1080x1920 (6 judges,
  16 pairs each, no judge one-sided): preferred 53 to 43. Where a judge blamed
  a grid, checker or mesh on one of the two, they picked the de-striped render
  25 times out of 34 (p about .01).

## The night-only cast run (21 Nov 2026, evening)

**Tried and not shipped: cast light on night photos only.** The theory for the
daytime KAIST regression below was that beams and pools cast onto daytime
photos had taught the model to read the sky and the near road as lit.
turbo4-cast was continued for 3000 more steps on the M1 Pro with
`--cast-day 0`: cast light on night photos as before, none on day photos,
everything else the same (5.6 h, mean of the 2k-3k snapshots).

- Held-out scores, same-day rerun: KID x1000 32.9 against the shipped
  model's 32.1 (both about 3.5 either way), DISTS .2664 against .2659, KAIST
  DISTS .2371 against .2380.
- Fake vs fake on held-out frames (`prefer`, round k1, the 16 frames per
  source where the two differ most, 4 judges each): 95 to 97, KAIST 34 to
  30. Five of the 8 judges picked the same side on 22 or more of their 24
  sheets. Among the three who did not, the night-only model won 32 of 72,
  KAIST 7 of 17.
- Fake vs fake on the user's 16 frames (6 judges): 50 to 46, with four of
  the six judges one-sided.

The renders barely moved. On KAIST, the night-only model's held-out renders
differ from the shipped model's by 4.8 grey levels on average; the shipped
model differs from turbo4 by 20.3. So 3000 steps without day casting did not
undo what 3000 steps with it had done, and day casting is probably not the
cause, or not the only one. The run that made the regression also changed
batch size (1, where turbo4 trained at 2) and crop (448); testing those needs
another 5.5 h run, for one street-scene look the user's footage does not have.
turbo4-cast stays.

**Found: one-sided judges mean no visible difference.** When the two renders
are nearly the same, most judges pick one side on every sheet and then
describe a consistent difference that is not there: in k1, three judges
called the right panel "colder" on all 24 sheets and two called the left
"grainier" on 22. No judge did this in the soup round (s1). Balanced sides
make such a judge's votes cancel exactly, but `preference_test.py` drew sides
at random per sheet, so a one-sided judge still moved the score. One judge
picked the night-only model 10 times out of 16 only because it was on their
favoured side 10 times. It now balances sides within each judge's sheets of
each comparison, as `prefer` does. Both scorers print each judge's left-pick
count and flag judges who picked one side on 90% or more of their sheets. A
round where most judges are flagged measured no visible difference, whatever
the totals say. No cloud credit was used.

## Lights, night noise and the cast-light run (21 Nov 2026, day)

**Ships: cast-light continuation (`turbo4-cast`).** turbo4 continued for 3000
steps on the M1 Pro (batch 1, crop 448, 6.5 s/step, 5.5 h, no cloud cost)
with a new photo-only augmentation, `--cast-aug .5`: headlight beams,
street-light pools and wall spill multiplied into the photo while the thermal
target stays as it was (`fake_thermal/augment.py`, `add_cast`). The shipped
file is the mean of the 2k-3k snapshots, in fp16.

- Held-out scores: KID x1000 31.7, DISTS .266 (turbo4 32.6, .268).
- Blind pairs against the real frame (r6, 8 judges, 96 sheets per model):
  spotted 70.8% of the time against turbo4's 72.9%, a tie, with KAIST at
  16/32 (chance). Light-related tells were as frequent as turbo4's, and a
  night lamp probe looked almost unchanged, so at first it was not shipped.
- Fake vs fake on held-out frames (`prefer`, round c1, 48 frames x 4
  judges): 99 to 93, a tie overall, but split by source. It won LLVIP and
  FLIR 41 to 23 each, and judges named light glow, light pools or lit ground
  against turbo4 50 times and against it 24 times. It lost KAIST 17 to 47:
  on daytime street scenes it warms the sky and trees and darkens the road
  nearest the camera, as if the headlight beams it was trained to ignore
  were there.
- Single frames, no reference (s5, 4 judges, 24 frames each): called
  generated 54% of the time, turbo4 63%, real frames 46%. With 24 frames the
  uncertainty is about 10 points either way.
- Fake vs fake on the user's 16 close-up frames (6 judges each): preferred
  on 79 of 96 sheets and 14 of 16 frames. The lit SEMA pillar no longer
  reads as warm as the shirt, floors and tables stop glowing, and people
  stand out from a cooler background. These judges favoured the right-hand
  panel (see below); with it on the left it still won 30 of 47.

The user's footage is what the tool is for, and the scores did not get worse,
so it ships, with the daytime street-scene regression as a known limit. A
50/50 weight average with turbo4 (KID 32.0, DISTS .266) kept that regression
and lost the night gains: turbo4 was preferred 109 to 83 on held-out frames
(KAIST 45 to 19, the rest level).

**Found and fixed: a side artifact in the preference sheets.** In a second
round on the user's frames, judges picked the right-hand panel 89 times out of
96 and described a "grid" or "crosshatch" texture on the left one, whichever
model it showed. All three models' raw renders carry the same faint periodic
pattern from turbo's VAE decoder (its spectral power share was .0012-.0015
for each). `preference_test.py` saved its sheets as JPEG with the right panel
at an odd pixel offset, so chroma subsampling treated the two panels
differently: the pattern survived on one side and was blurred away on the
other. That round was thrown out. Every judge sheet (`pairs`, `singles`,
`prefer`, `preference_test.py`) is now saved as lossless PNG. The cast-light
round above was less affected (66 of 96 right-hand picks); it is reported
with the side split. The held-out `prefer` rounds balance sides within each
judge, so a side preference cancels there. (Corrected later the same day:
judges see the left panel as grainier even when both panels are the same
image, so JPEG was at most part of this. See the late-evening entry above.)

**Tried and not shipped: three inference-time fixes.** Each went after a tell
the r6 judges named over and over, and each changed only the pipeline, not the
model. The code is on branch `exp/inference-fixes`, not in the product.

- *Lamps.* turbo paints a cold disc with a warm ring where a bright lamp or
  traffic light is, or copies the glow as heat. A light is a compact group
  of near-saturated pixels (brightest channel .9 or more, at most .15% of the
  frame) at least 2x and .45 brighter than the brighter quarter of a ring
  three to five radii out. A sunlit patch beside a shadow is not dark all
  the way round, so it fails. Over the light's glow (and, in the second
  version, a disc of three radii) the heat is pulled back to the surrounding
  heat, keeping the grain.
- *Night noise.* turbo turns photo noise into sandy speckle. Noisy photos
  (Immerkaer noise estimate 1.6 or more in 8-bit units; sunlit sand reads
  1.3-1.4) were run through non-local means at 2.5x that strength before
  translation.
- *Small people.* LLVIP people come out flame-like or as a bright ring round
  a dark body. People 4-35% of the frame tall were re-segmented on a zoomed
  crop, which gives a crisper outline than the whole-frame 320px map.

| Held-out (same-day rerun) | KID x1000 | DISTS |
|---|---|---|
| turbo4 (the baseline these were tried on) | 32.9 | .2675 |
| lamps (glow mask) | 32.7 | .2696 |
| night denoise (dark photos only) | 28.3 | .2694 |
| small-people refine | 32.6 | .2672 |
| all three | 28.2 | .2708 |
| all three, wider lamp mask, denoise on every noisy photo | 25.8 | .2762 |

The first combination tied turbo4 in a blind pairs round (r7, 96 sheets each:
fake spotted 85.4% vs 87.5%). These judges were harsher than r6's (turbo4 at
87.5% here, 72.9% there, with FLIR and LLVIP at 32/32), so a real-vs-fake
round could not show a small gain. The second combination had the best KID of
the campaign, so it was put to a more sensitive test: fake against fake. Two
renders of the same held-out frame sat side by side with no real panel, and
judges picked the one that looks more like a real camera. The 48 frames were
the 16 per source where the two renders differed most, each judged by four
judges. turbo4 was preferred on 184 of 192 sheets and on all 48 frames. The
judges' reasons:

- the lamp fills read as smooth, inpainted blobs, worst over lit buildings
  with many lamps;
- the denoised photos come out over-smoothed, and turbo paints wavy
  "fingerprint" swirls onto the smoothed roads;
- the ordinary grain was liked as real sensor noise.

Refinement alone was tested the same way on the 48 frames it changes most
(47 LLVIP). turbo4 was preferred 108 to 84 and won 20 frames to 10. One judge
picked the left panel on all 24 of their sheets; without them it is 89 to 79.
Refined people often came out translucent or fading at the legs, where the
zoomed crop missed part of the body. Nine sheets named a box seam around a
refined person, where the fade band met the crop edge. Others did gain a
crisp, solid outline.

So KID and DISTS pointed the wrong way here: the variant with the best KID
was the one the eye rejected most clearly. Fake-vs-fake on the frames a
change touches is now `scripts/realism_report.py prefer` / `score-prefer`
(prompt `scripts/scene_judge_prompt.md`). It puts each variant on the left in
exactly half of every judge's sheets, after the side-biased judge above.
Small inference changes should go through it before shipping. The day used
the Mac and judge agents only; no cloud credit.

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
