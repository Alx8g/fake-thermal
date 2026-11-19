"""Rescue unaligned FLIR video pairs by registering real thermal onto pseudo-thermal."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import itertools
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pyarrow.parquet as pq
from tqdm import tqdm

from prepare_data import resize_pair

MOTIONS = (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE)
MOTION_NAMES = ('euclidean', 'affine')
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-5)
COLUMNS = ['rgb_image', 'thermal_image', 'rgb_image_file_name']
RESIDUAL_MAX = 0.30
GOOD_ENOUGH = 0.12


def stretch(image, valid=None):
    """p2/p99.5 display stretch to [0,1], the contrast convention the model predicts.

    With `valid`, percentiles come from those pixels only, so warp borders or
    dead sensor regions cannot skew the normalization of the rest.
    """
    data = image[valid] if valid is not None else image
    low, high = np.percentile(data, (2, 99.5))
    if high - low < 1e-6:
        return np.zeros(image.shape, np.float32)
    return np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)


def prep(image):
    """CLAHE-equalized content; ECC follows gradients, not raw intensity levels."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Registration input must be 8-bit single-channel')
    return cv2.createCLAHE(2.0, (8, 8)).apply(image)


def evidence_mask(pseudo, real, low=6, min_fraction=.02):
    """Validity mask of regions where both images carry structure for ECC, or None.

    Gradients are ANDed per pixel, then dilated into contiguous regions: the
    speckled raw mask destabilizes ECC's gaussian-smoothed gradient updates.
    """
    def gradients(image):
        return np.hypot(cv2.Sobel(image, cv2.CV_32F, 1, 0, 3), cv2.Sobel(image, cv2.CV_32F, 0, 1, 3))
    mask = ((gradients(pseudo) > low) & (gradients(real) > low)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return mask if mask.mean() >= min_fraction * 255 else None


def apply_warp(image, warp, size):
    """Sample `image` onto the (w, h) grid; `warp` maps destination pixels to source."""
    return cv2.warpAffine(image, warp, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def registration_residual(pseudo, real, warp, mask, min_coverage=.5):
    """Masked mean |pseudo - warped real| on raw stretched content; 1.0 if unusable.

    Computed on the pre-CLAHE images: CLAHE is local and shift-variant, so even a
    perfect warp would otherwise carry an artificial residual floor. Warps that
    push most of the frame off-grid can only score a small cherry-picked region,
    so minimum valid coverage is enforced first.
    """
    size = pseudo.shape[::-1]
    warped = apply_warp(real, warp, size)
    valid = (mask > 0) & (apply_warp(np.ones_like(real), warp, size) > .99)
    if valid.sum() < min_coverage * (mask > 0).sum():
        return 1.
    return float(np.abs(stretch(pseudo, valid) - stretch(warped, valid))[valid].mean())


def initial_warp(pseudo, real):
    """Phase-correlation translation seed in the same inverse-map convention as ECC."""
    hanning = cv2.createHanningWindow(pseudo.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(pseudo.astype(np.float32), real.astype(np.float32), hanning)
    return np.float32([[1., 0., dx], [0., 1., dy]])


def register_thermal(pseudo, real, motions=MOTIONS, low=6, min_fraction=.02, good_enough=GOOD_ENOUGH):
    """ECC-register `real` onto the `pseudo` grid; returns (warp, residual, reason).

    `warp` is the inverse-map affine taking pseudo-grid pixels to real pixels, usable
    with apply_warp; None when the frames carry too little shared structure to trust.
    CLAHE-prepped images drive findTransformECC from a phase-correlation seed with an
    identity fallback; the residual gates on raw content, not the CLAHE rendering.
    """
    raw_pseudo, raw_real = pseudo, real
    pseudo, real = prep(pseudo), prep(real)
    mask = evidence_mask(pseudo, real, low, min_fraction)
    if mask is None:
        return None, 1., 'textureless'
    best = (None, 1., 'ecc_failed')
    seeds = (initial_warp(pseudo, real), np.eye(2, 3, dtype=np.float32))
    for motion, name in zip(motions, MOTION_NAMES):
        for seed in seeds:
            warp = seed.copy()
            try:
                cv2.findTransformECC(pseudo, real, warp, motion, CRITERIA, mask, 5)
            except cv2.error:
                continue
            residual = registration_residual(raw_pseudo, raw_real, warp, mask)
            if residual < best[1]:
                best = (warp, residual, name)
            if residual <= good_enough:
                return best
    return best


def rescale_warp(warp, inner, outer):
    """Re-express a registration-grid warp between full-resolution grids.

    `inner` scales destination (RGB) pixels down to the registration grid; `outer`
    scales registration-grid source pixels up to the thermal's native grid.
    """
    inner, outer = np.asarray(inner, np.float32), np.asarray(outer, np.float32)
    scaled = warp.astype(np.float32).copy()
    scaled[:, :2] *= inner * outer[:, None]
    scaled[:, 2] *= outer
    return scaled


def save_pair(out, stem, rgb, thermal, short_side=288):
    """prepare_data conventions: RGB JPEG q95, L-mode thermal PNG, equal stems."""
    rgb, thermal = resize_pair(rgb, thermal, short_side)
    (out / 'rgb').mkdir(parents=True, exist_ok=True)
    (out / 'thermal').mkdir(parents=True, exist_ok=True)
    rgb.save(out / 'rgb' / f'{stem}.jpg', quality=95)
    thermal.save(out / 'thermal' / f'{stem}.png')


class Reservoir:
    """Uniform random sample over a stream without holding every item."""

    def __init__(self, capacity, seed=42):
        self.capacity, self.items, self.rng, self.seen = capacity, [], random.Random(seed), 0

    def add(self, item):
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            pick = self.rng.randrange(self.seen + 1)
            if pick < self.capacity:
                self.items[pick] = item
        self.seen += 1


def frame_key(row):
    """(recording, frame), e.g. ('video-BzZspxAweF8AnKhWK', '000745')."""
    head, _, tail = row['rgb_image_file_name'].partition('-frame-')
    return head, tail.partition('-')[0]


def thumbnail(image):
    return ImageOps.contain(image.convert('RGB'), (320, 160))


def preview_tile(stem, thumbs, label):
    tile = Image.new('RGB', (1280, 190), 'white')
    for i, image in enumerate(thumbs):
        if image is None:
            ImageDraw.Draw(tile).text((i * 320 + 90, 90), 'NOT WARPED', fill='red')
        else:
            tile.paste(image, (i * 320, 25))
    ImageDraw.Draw(tile).text((4, 4), f'{stem} {label}', fill='red' if 'REJECTED' in label else 'black')
    return tile


def save_preview(tiles, path):
    sheet = Image.new('RGB', (1280, len(tiles) * 190), 'white')
    for row, tile in enumerate(tiles):
        sheet.paste(tile, (0, row * 190))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def rescue_frame(args, row, rgb, real, pseudo):
    """Register one frame, save the pair when accepted, and return its record."""
    recording, frame = frame_key(row)
    stem = f'{recording}_{frame}'
    work = pseudo.shape[::-1]
    real_work = cv2.resize(real, work, interpolation=cv2.INTER_AREA)
    warp, residual, reason = register_thermal(pseudo, real_work)
    record = {'id': stem, 'recording': recording, 'frame': frame,
              'residual': round(residual, 4), 'motion': reason, 'accepted': False}
    warped = None
    if warp is not None:
        h, w = rgb.shape[:2]
        record['warp'] = [[float(v) for v in r] for r in warp]
        native = rescale_warp(warp, (work[0] / w, work[1] / h),
                              (real.shape[1] / work[0], real.shape[0] / work[1]))
        warped = apply_warp(real, native, (w, h))
        record['accepted'] = residual <= args.residual_max
        if record['accepted']:
            save_pair(args.output, stem, Image.fromarray(rgb), Image.fromarray(warped), args.short_side)
    thumbs = (thumbnail(Image.fromarray(rgb)), thumbnail(Image.fromarray(real)),
              thumbnail(Image.fromarray(warped)) if warped is not None else None,
              thumbnail(Image.fromarray(pseudo)))
    return record, thumbs


def harvest(result, records, counts, accepted, rejected):
    record, thumbs = result
    records.append(record)
    counts[f'{record["recording"]}_{record["accepted"]}'] += 1
    label = (f'ACCEPTED residual={record["residual"]:.3f}' if record['accepted'] else
             f'REJECTED residual={record["residual"]:.3f} ({record["motion"]})')
    (accepted if record['accepted'] else rejected).add(preview_tile(record['id'], thumbs, label))


def main():
    from fake_thermal.device import DEVICES, select_device
    from fake_thermal.infer import Predictor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('data/pairs-video'))
    parser.add_argument('--weights', type=Path, default=Path('fake_thermal/weights/fake_thermal.pt'))
    parser.add_argument('--preview', type=Path, default=Path('outputs/video_rescue_preview.jpg'))
    parser.add_argument('--limit', type=int, help='process only the first N frames')
    parser.add_argument('--device', choices=DEVICES, default='auto')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--residual-max', type=float, default=RESIDUAL_MAX,
                        help=f'accept pairs at or below this registration residual (default {RESIDUAL_MAX})')
    parser.add_argument('--short-side', type=int, default=288)
    args = parser.parse_args()
    live = (args.data / 'pairs').resolve()
    if args.output.resolve().is_relative_to(live):
        parser.error(f'Refusing to write into the live training set at {live}')
    if not 0 < args.residual_max <= 1:
        parser.error('--residual-max must be in (0, 1]')
    if args.short_side <= 0:
        parser.error('--short-side must be positive')
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--workers and --limit must be positive')
    shards = sorted((args.data / 'raw/video/data').glob('*.parquet'))
    if not shards:
        parser.error(f'No video parquet shards under {args.data}/raw/video/data')
    device = select_device(args.device)
    predictor = Predictor(args.weights, threads=args.threads, device=device.type)
    rows = (row for shard in shards
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=16, columns=COLUMNS)
            for row in batch.to_pylist())
    if args.limit is not None:
        rows = itertools.islice(rows, args.limit)
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in shards)
    records, accepted, rejected, counts, pending = [], Reservoir(12), Reservoir(4), Counter(), deque()
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in tqdm(rows, total=min(total, args.limit or total), desc='Rescue'):
            rgb = np.asarray(Image.open(BytesIO(row['rgb_image']['bytes'])).convert('RGB'))
            real = np.asarray(Image.open(BytesIO(row['thermal_image']['bytes'])).convert('L'))
            pseudo = np.rint(predictor(rgb) * 255).astype(np.uint8)
            pending.append(pool.submit(rescue_frame, args, row, rgb, real, pseudo))
            while len(pending) > args.workers * 2:
                harvest(pending.popleft().result(), records, counts, accepted, rejected)
        while pending:
            harvest(pending.popleft().result(), records, counts, accepted, rejected)
    residuals = [record['residual'] for record in records]
    ok = sum(record['accepted'] for record in records)
    stats = {'accepted': ok, 'rejected': len(records) - ok, 'total': len(records),
             'residual_max': args.residual_max, 'short_side': args.short_side,
             'residual_percentiles': {name: round(float(np.percentile(residuals, p)), 4)
                                      for name, p in (('p50', 50), ('p75', 75), ('p90', 90), ('p99', 99))}
             if residuals else {},
             'recordings': {rec: {'accepted': counts[f'{rec}_True'], 'rejected': counts[f'{rec}_False']}
                            for rec in sorted({record['recording'] for record in records})},
             'seconds': round(time.perf_counter() - start, 1)}
    (args.output / 'manifest.json').write_text(json.dumps(dict(stats, frames=records), indent=2) + '\n')
    save_preview(accepted.items + rejected.items, args.preview)
    print(json.dumps({**stats, 'output': str(args.output), 'preview': str(args.preview),
                      'device': device.type}), flush=True)


if __name__ == '__main__':
    main()
