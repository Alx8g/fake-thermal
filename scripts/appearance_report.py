"""Compare turbo checkpoints under visible-only edits on aligned validation pairs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import zlib

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_thermal.appearance import MASKED_EDITS, SYNTHETIC_EDITS, edit_appearance
from fake_thermal.data import normalize_thermal
from fake_thermal.infer import DEFAULT_TURBO_WEIGHTS, TurboPredictor
from scripts.realism_report import rgb_box


def pair_metrics(clean, edited, thermal, influence):
    """Measure raw edit drift and display error without stretching each edit."""
    size = (clean.shape[1], clean.shape[0])
    influence = cv2.resize(influence, size, interpolation=cv2.INTER_AREA)
    if not influence.any():
        return None
    thermal = cv2.resize(thermal.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    thermal = normalize_thermal(thermal)
    lo, hi = np.percentile(clean, [1, 99])
    span = max(hi - lo, 1e-6)
    clean_error = np.abs(np.clip((clean - lo) / span, 0, 1) - thermal)
    edited_error = np.abs(np.clip((edited - lo) / span, 0, 1) - thermal)
    drift = np.abs(clean - edited)
    inside = influence.sum()
    outside = influence == 0
    return {
        'edit_drift': float((drift * influence).sum() / inside),
        'context_drift': float(drift[outside].mean()) if outside.any() else None,
        'clean_mae': float(clean_error.mean()),
        'clean_edit_region_mae': float((clean_error * influence).sum() / inside),
        'edited_region_mae': float((edited_error * influence).sum() / inside),
        'clean_range': float(hi - lo),
    }


def sample_cases(root, kinds, count, seed, stems):
    by_stem = defaultdict(list)
    for kind in kinds:
        eligible = stems if kind not in MASKED_EDITS else [
            stem for stem in stems if (root / 'physics' / kind / f'{stem}.png').is_file()]
        rng = random.Random(seed)
        for stem in sorted(rng.sample(eligible, min(count, len(eligible)))):
            by_stem[stem].append(kind)
    return by_stem


def report(root, cases, pred, seed):
    rows = []
    for stem, kinds in sorted(cases.items()):
        case_seed = zlib.crc32(f'{seed}/{stem}'.encode())
        with Image.open(root / 'rgb' / f'{stem}.jpg') as im:
            rgb = np.array(im.convert('RGB'))
        with Image.open(root / 'thermal' / f'{stem}.png') as im:
            thermal = np.array(im.convert('L'))
        if thermal.shape != rgb.shape[:2]:
            raise ValueError(f'Thermal must match RGB dimensions: {stem}')
        crop = (slice(None), slice(None))
        if stem.startswith('flir_'):
            y0, y1, x0, x1 = rgb_box(rgb)
            crop = (slice(y0, y1), slice(x0, x1))
        torch.manual_seed(case_seed)
        clean = pred(np.ascontiguousarray(rgb[crop]))
        for kind in kinds:
            mask = None
            if kind in MASKED_EDITS:
                with Image.open(root / 'physics' / kind / f'{stem}.png') as im:
                    mask = np.array(im.convert('L'), np.float32) / 255
                if mask.shape != thermal.shape:
                    raise ValueError(f'{kind} mask must match RGB dimensions: {stem}')
                mask = mask[crop]
            edited, influence = edit_appearance(
                rgb[crop].astype(np.float32) / 255, kind, mask,
                rng=random.Random(case_seed), np_rng=np.random.RandomState(case_seed))
            if not influence.any():
                continue
            torch.manual_seed(case_seed)
            edit_rgb = np.round(edited * 255).astype(np.uint8)
            metrics = pair_metrics(clean, pred(edit_rgb), thermal[crop], influence)
            if metrics is not None:
                rows.append({'stem': stem, 'kind': kind, **metrics})
    return rows


def summarize(rows, kinds):
    summary = {}
    for kind in kinds:
        subset = [row for row in rows if row['kind'] == kind]
        summary[kind] = {'count': len(subset)}
        if subset:
            for metric in subset[0].keys() - {'stem', 'kind'}:
                values = [row[metric] for row in subset if row[metric] is not None]
                summary[kind][metric] = float(np.mean(values)) if values else None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=DEFAULT_TURBO_WEIGHTS)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--size', type=int, default=384)
    parser.add_argument('--count', type=int, default=12, help='maximum frames per edit kind')
    parser.add_argument('--ids', type=Path, help='JSON list or source-grouped mapping of validation stems')
    parser.add_argument('--kinds', nargs='+', choices=(*SYNTHETIC_EDITS, *MASKED_EDITS),
                        default=[*SYNTHETIC_EDITS, *MASKED_EDITS])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    root = args.data / 'pairs/val'
    stems = json.loads(args.ids.read_text()) if args.ids else [
        p.stem for p in (root / 'rgb').glob('*.jpg') if not p.name.startswith('._')]
    if isinstance(stems, dict):
        stems = [stem for group in stems.values() for stem in group]
    if not stems:
        parser.error('no validation frames found')
    cases = sample_cases(root, args.kinds, args.count, args.seed, sorted(set(stems)))
    rows = []
    if cases:
        predictor = TurboPredictor(args.weights, size=args.size, device=args.device)
        rows = report(root, cases, predictor, args.seed)
    summary = summarize(rows, args.kinds)
    result = {'weights': str(args.weights), 'size': args.size, 'seed': args.seed,
              'summary': summary, 'frames': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
