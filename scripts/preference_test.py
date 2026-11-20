"""Blind preference test between renderings of the same frames, with no real reference.

For footage that has no thermal counterpart (close-up people, the user's own
videos), fresh judges see two renders of one frame side by side and pick the
one that looks more like a real thermal camera frame. A Latin square over
judges means no judge sees a frame twice.

    python scripts/preference_test.py render FRAMES_DIR OUT            # none / uniform / parts
    python scripts/preference_test.py sheets OUT parts:uniform parts:none --judges 6
    # give each judge scripts/preference_judge_prompt.md + its sheets (judges.json), never key.json
    python scripts/preference_test.py score OUT answers.json            # {"001": "A", ...}
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw


def stretch8(heat):
    lo, hi = np.percentile(heat, [2, 99.5])
    return (np.clip((heat - lo) / max(hi - lo, 1e-5), 0, 1) * 255).round().astype(np.uint8)


def render(args):
    from fake_thermal.infer import TurboPredictor, resize_upsample, DEFAULT_TURBO_WEIGHTS
    from fake_thermal.semantic import SemanticPrior, fuse_heat
    predictor = TurboPredictor(args.weights or DEFAULT_TURBO_WEIGHTS, device=args.device)
    priors = {'uniform': SemanticPrior(device=args.device, emissive=False, vehicles=False, sharp=True),
              'parts': SemanticPrior(device=args.device, emissive=False, vehicles=False, sharp=True, parts=True)}
    frames = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    (args.out / 'rgb').mkdir(parents=True, exist_ok=True)
    for path in frames:
        rgb = np.asarray(Image.open(path).convert('RGB'))
        Image.fromarray(rgb).save(args.out / 'rgb' / f'{path.stem}.png')
        heat = predictor(rgb)
        variants = {'none': heat}
        for name, prior in priors.items():
            variants[name] = fuse_heat(heat, *prior(rgb, heat.shape), args.strength)
        for name, value in variants.items():
            (args.out / name).mkdir(exist_ok=True)
            Image.fromarray(stretch8(resize_upsample(value, rgb))).save(args.out / name / f'{path.stem}.png')
        print(path.stem, flush=True)


def sheets(args):
    from fake_thermal.infer import apply_palette
    rng = random.Random(args.seed)
    frames = sorted(p.stem for p in (args.out / 'rgb').glob('*.png'))
    comparisons = [c.split(':') for c in args.comparisons]
    folder = args.out / 'blind' / 'sheets'
    folder.mkdir(parents=True, exist_ok=True)

    def panel(name, frame):
        gray = np.asarray(Image.open(args.out / name / f'{frame}.png').convert('L'), np.float32) / 255
        image = Image.fromarray(apply_palette(gray, 'ironbow'))
        return image.resize((round(image.width * args.height / image.height), args.height), Image.LANCZOS)

    trials = [(judge, frame, comparisons[(index + judge) % len(comparisons)])
              for judge in range(args.judges) for index, frame in enumerate(frames)]
    rng.shuffle(trials)
    trials.sort(key=lambda trial: trial[0])
    key, assignment = [], {}
    for number, (judge, frame, (first, second)) in enumerate(trials, 1):
        left, right = (first, second) if rng.random() < .5 else (second, first)
        a, b = panel(left, frame), panel(right, frame)
        sheet = Image.new('RGB', (a.width + b.width + 24, args.height + 36), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((a.width // 2, 10), 'A', fill='black')
        draw.text((a.width + 24 + b.width // 2, 10), 'B', fill='black')
        sheet.paste(a, (0, 36))
        sheet.paste(b, (a.width + 24, 36))
        sheet.save(folder / f'{number:03d}.jpg', quality=90)
        assignment.setdefault(f'judge{judge + 1}', []).append(f'{number:03d}')
        key.append({'sheet': f'{number:03d}', 'frame': frame, 'A': left, 'B': right})
    (args.out / 'blind' / 'key.json').write_text(json.dumps(key, indent=1) + '\n')
    (args.out / 'blind' / 'judges.json').write_text(json.dumps(assignment, indent=1) + '\n')
    print(f'{len(key)} sheets for {args.judges} judges in {folder}')


def score(args):
    key = {k['sheet']: k for k in json.loads((args.out / 'blind' / 'key.json').read_text())}
    answers = json.loads(args.answers.read_text())
    wins, totals = Counter(), Counter()
    for sheet, choice in answers.items():
        entry = key[sheet]
        pair = tuple(sorted((entry['A'], entry['B'])))
        totals[pair] += 1
        wins[pair, entry[choice]] += 1
    for pair, total in totals.items():
        print(' vs '.join(f'{name} {wins[pair, name]}/{total}' for name in pair))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('render')
    p.add_argument('frames', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--weights', type=Path)
    p.add_argument('--strength', type=float, default=.7)
    p.add_argument('--device', default='auto')
    p = commands.add_parser('sheets')
    p.add_argument('out', type=Path)
    p.add_argument('comparisons', nargs='+', help='pairs of variants, e.g. parts:uniform')
    p.add_argument('--judges', type=int, default=6)
    p.add_argument('--height', type=int, default=640)
    p.add_argument('--seed', type=int, default=5)
    p = commands.add_parser('score')
    p.add_argument('out', type=Path)
    p.add_argument('answers', type=Path)
    args = parser.parse_args()
    {'render': render, 'sheets': sheets, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
