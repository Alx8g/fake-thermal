"""Honest realism scoring: held-out metrics plus a blind forced-choice test.

Earlier evaluation scored models with LPIPS(AlexNet) while training on the same
loss, and with a small patch classifier whose test split shared images with its
training split. This script replaces both:

  ids       fix a held-out test set: validation frames never used to pick a
            checkpoint (the 300 selection ids are excluded), stratified by source
  real      write the real thermal frames, stretched to 8 bits like the product
  generate  one model's thermal for every test frame through the product
            pipeline (model, guided edge transfer, optional sensor simulation)
  metrics   KID (Inception features, set level) and DISTS (paired); neither is a
            training loss of any model here
  pairs     blind two-alternative forced-choice sheets: real and generated
            thermal for the same frame side by side, sides randomized, ironbow
            palette as the product renders it; the key is written separately
  score     accuracy of a judge's answers per model. A judge who can always
            spot the fake scores 100%; 50% means indistinguishable.
  prefer    fake-vs-fake sheets: two models' renders of the same held-out frame,
            no real panel, judges pick the one that looks more like a real
            camera (scripts/scene_judge_prompt.md). Uses the frames where the
            two differ most, so it measures whether a change, where it shows,
            looks better; far more sensitive to small changes than pairs.
  score-prefer  how often each model was preferred, by source and by frame
"""
import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_thermal.data import normalize_thermal, validation_ids
from fake_thermal.infer import apply_palette

ROOT = Path('outputs/realism')


def stretch8(image):
    return (normalize_thermal(image) * 255).round().astype(np.uint8)


def rgb_box(rgb, threshold=20, margin=12):
    """Bounding box of the part of the frame the RGB camera actually covers.
    FLIR's aligned RGB leaves ~30/65 px black bands top/bottom and ~80 px at the
    sides; the thermal there has no RGB counterpart, so it is excluded."""
    gray = np.asarray(rgb).max(axis=2)
    rows, cols = np.where(gray.max(1) > threshold)[0], np.where(gray.max(0) > threshold)[0]
    return (rows[0] + margin, rows[-1] + 1 - margin, cols[0] + margin, cols[-1] + 1 - margin)


_BOXES = {}


def crop_to_rgb(stem, image, data=Path('data512')):
    if stem not in _BOXES:
        with Image.open(data / f'pairs/val/rgb/{stem}.jpg') as im:
            _BOXES[stem] = rgb_box(im.convert('RGB'))
    top, bottom, left, right = _BOXES[stem]
    return image[top:bottom, left:right]


def choose_ids(args):
    selection = set(validation_ids(args.data))
    paths = sorted((args.data / 'pairs/val/rgb').glob('*.jpg'))
    groups = {}
    for path in paths:
        if path.stem not in selection:
            groups.setdefault(path.stem.split('_', 1)[0], []).append(path.stem)
    rng = random.Random(args.seed)
    ids = {source: rng.sample(sorted(stems), args.per_source) for source, stems in sorted(groups.items())}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'ids.json').write_text(json.dumps(ids, indent=2) + '\n')
    print({source: len(v) for source, v in ids.items()})


def test_ids(limit=None):
    groups = json.loads((ROOT / 'ids.json').read_text())
    return [stem for stems in groups.values() for stem in stems[:limit]]


def write_real(args):
    out = ROOT / 'real'
    out.mkdir(parents=True, exist_ok=True)
    for stem in test_ids():
        with Image.open(args.data / f'pairs/val/thermal/{stem}.png') as im:
            Image.fromarray(stretch8(im.convert('L'))).save(out / f'{stem}.png')
    print(f'wrote {len(test_ids())} real frames to {out}')


def generate(args):
    from fake_thermal.infer import Predictor, TurboPredictor, resize_upsample
    from fake_thermal.semantic import SemanticPrior, fuse_heat, guided_upsample
    from fake_thermal.sensor import ThermalSensor
    if args.engine == 'turbo':
        predictor = TurboPredictor(args.weights, size=args.size, device=args.device, skip=args.skip)
        args.guided = False
    else:
        predictor = Predictor(weights=args.weights, size=args.size, device=args.device, threads=4)
    prior = (SemanticPrior(device=args.device, emissive=args.emissive, vehicles=args.vehicles,
                           sharp=args.sharp_prior, parts=args.parts_prior) if args.semantic else None)
    out = ROOT / 'fake' / args.name
    out.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(test_ids()):
        with Image.open(args.data / f'pairs/val/rgb/{stem}.jpg') as im:
            rgb = np.asarray(im.convert('RGB'))
        heat = predictor(rgb)
        if prior is not None:
            heat = fuse_heat(heat, *prior(rgb, heat.shape), args.semantic)
        heat = guided_upsample(heat, rgb) if args.guided else resize_upsample(heat, rgb)
        if args.sensor:
            heat = ThermalSensor(seed=i).update(heat.astype(np.float32))
        Image.fromarray(stretch8(heat)).save(out / f'{stem}.png')
    print(f'wrote {len(test_ids())} frames for {args.name}')


def load_gray(path, size=None):
    with Image.open(path) as im:
        im = im.convert('L')
        if size is not None and im.size != size:
            im = im.resize(size, Image.BICUBIC)
        return np.asarray(im)


def metrics(args):
    import torch
    from torchmetrics.image.kid import KernelInceptionDistance
    import piq
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    stems = test_ids()
    full = {s: load_gray(ROOT / 'real' / f'{s}.png') for s in stems}
    real = {s: crop_to_rgb(s, full[s], args.data) for s in stems}
    results = {}
    dists = piq.DISTS(reduction='none').to(device)
    for name in args.names:
        folder = ROOT / 'fake' / name
        fake = {s: crop_to_rgb(s, load_gray(folder / f'{s}.png', full[s].shape[::-1]), args.data)
                for s in stems}
        kid = KernelInceptionDistance(subset_size=min(100, len(stems)), subsets=50).to(device)
        square = lambda a: np.asarray(Image.fromarray(a).resize((299, 299), Image.BICUBIC))
        as_rgb = lambda frames: torch.from_numpy(np.stack([square(f) for f in frames])).unsqueeze(1).repeat(1, 3, 1, 1)
        for start in range(0, len(stems), 32):
            chunk = stems[start:start + 32]
            kid.update(as_rgb([real[s] for s in chunk]).to(device), real=True)
            kid.update(as_rgb([fake[s] for s in chunk]).to(device), real=False)
        kid_mean, kid_std = kid.compute()
        scores = {}
        with torch.no_grad():
            for s in stems:
                to_tensor = lambda a: torch.from_numpy(a).float().div(255)[None, None].repeat(1, 3, 1, 1).to(device)
                scores[s] = float(dists(to_tensor(fake[s]), to_tensor(real[s]))[0])
        by_source = {}
        for s, value in scores.items():
            by_source.setdefault(s.split('_', 1)[0], []).append(value)
        results[name] = {'kid_x1000': round(float(kid_mean) * 1000, 2),
                         'kid_std_x1000': round(float(kid_std) * 1000, 2),
                         'dists': round(float(np.mean(list(scores.values()))), 4),
                         'dists_per_source': {k: round(float(np.mean(v)), 4)
                                              for k, v in sorted(by_source.items())}}
        print(name, results[name], flush=True)
    path = ROOT / 'metrics.json'
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(results)
    path.write_text(json.dumps(existing, indent=2) + '\n')


def panel(gray, width):
    colored = apply_palette(gray.astype(np.float32) / 255, 'ironbow')
    h, w = colored.shape[:2]
    return Image.fromarray(colored).resize((width, round(h * width / w)), Image.LANCZOS)


def pairs(args):
    rng = random.Random(args.seed)
    stems = test_ids(args.per_source)
    out = ROOT / 'blind' / args.round
    sheets = out / 'sheets'
    sheets.mkdir(parents=True, exist_ok=True)
    # Latin square: judge j sees frame i rendered by model (i + j) mod M, so no
    # judge sees the same scene twice (a repeated real panel gives the answer
    # away) and every judge sees every model equally often.
    judges = len(args.names)
    trials = [(name, stem, (index + m) % judges) for index, stem in enumerate(stems)
              for m, name in enumerate(args.names)]
    rng.shuffle(trials)
    trials.sort(key=lambda trial: trial[2])
    assignment = {}
    key = []
    for number, (name, stem, judge) in enumerate(trials, 1):
        assignment.setdefault(f'judge{judge + 1}', []).append(f'{number:03d}')
        real = load_gray(ROOT / 'real' / f'{stem}.png')
        fake = crop_to_rgb(stem, load_gray(ROOT / 'fake' / name / f'{stem}.png', real.shape[::-1]), args.data)
        real = crop_to_rgb(stem, real, args.data)
        real_left = rng.random() < .5
        left, right = (real, fake) if real_left else (fake, real)
        a, b = panel(left, args.width), panel(right, args.width)
        sheet = Image.new('RGB', (2 * args.width + 24, a.height + 36), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((args.width // 2, 10), 'A', fill='black')
        draw.text((args.width + 24 + args.width // 2, 10), 'B', fill='black')
        sheet.paste(a, (0, 36))
        sheet.paste(b, (args.width + 24, 36))
        sheet.save(sheets / f'{number:03d}.png')  # lossless, see prefer()
        key.append({'sheet': f'{number:03d}', 'model': name, 'frame': stem,
                    'real': 'A' if real_left else 'B'})
    (out / 'key.json').write_text(json.dumps(key, indent=2) + '\n')
    (out / 'judges.json').write_text(json.dumps(assignment, indent=2) + '\n')
    print(f'wrote {len(key)} blind sheets to {sheets} for {judges} judges '
          f'(judges.json); key in {out / "key.json"}')


def singles(args):
    """No-reference test: one frame per sheet, real or generated, shuffled.
    Closer to how footage is actually watched: nobody sees the real thermal
    next to it. Different frames are used for real and generated sheets so a
    judge never sees the same scene twice."""
    rng = random.Random(args.seed)
    groups = json.loads((ROOT / 'ids.json').read_text())
    out = ROOT / 'single' / args.round
    sheets = out / 'sheets'
    sheets.mkdir(parents=True, exist_ok=True)
    trials = []
    offset = args.offset
    for source, stems in sorted(groups.items()):
        pool = stems[offset:offset + args.per_source * (len(args.names) + 1)]
        rng.shuffle(pool)
        chunks = [pool[i::len(args.names) + 1] for i in range(len(args.names) + 1)]
        trials += [('real', stem) for stem in chunks[0]]
        for name, chunk in zip(args.names, chunks[1:]):
            trials += [(name, stem) for stem in chunk]
    rng.shuffle(trials)
    key = []
    for number, (name, stem) in enumerate(trials, 1):
        folder = ROOT / 'real' if name == 'real' else ROOT / 'fake' / name
        reference = load_gray(ROOT / 'real' / f'{stem}.png')
        gray = crop_to_rgb(stem, load_gray(folder / f'{stem}.png', reference.shape[::-1]), args.data)
        panel(gray, args.width).save(sheets / f'{number:03d}.png')
        key.append({'sheet': f'{number:03d}', 'model': name, 'frame': stem})
    (out / 'key.json').write_text(json.dumps(key, indent=2) + chr(10))
    print(f'wrote {len(key)} single-frame sheets to {sheets}; key in {out / "key.json"}')


def score_singles(args):
    out = ROOT / 'single' / args.round
    key = {entry['sheet']: entry for entry in json.loads((out / 'key.json').read_text())}
    answers = {}
    for path in args.answers:
        answers.update(json.loads(Path(path).read_text()))
    totals = {}
    for sheet, answer in answers.items():
        model = key[sheet]['model']
        stats = totals.setdefault(model, [0, 0])
        stats[0] += str(answer['verdict']).strip().lower().startswith('gen')
        stats[1] += 1
    report = {model: {'judged': n, 'called_generated_pct': round(100 * flagged / n, 1)}
              for model, (flagged, n) in totals.items()}
    for model, entry in sorted(report.items(), key=lambda kv: kv[1]['called_generated_pct']):
        print(f"{model:28s} called generated {entry['called_generated_pct']:5.1f}% of {entry['judged']}")
    (out / 'score.json').write_text(json.dumps(report, indent=2) + chr(10))


def score(args):
    out = ROOT / 'blind' / args.round
    key = {entry['sheet']: entry for entry in json.loads((out / 'key.json').read_text())}
    answers = {}
    for path in args.answers:
        answers.update(json.loads(Path(path).read_text()))
    totals = {}
    for sheet, answer in answers.items():
        entry = key[sheet]
        stats = totals.setdefault(entry['model'], {'n': 0, 'spotted': 0, 'by_source': {}})
        correct = str(answer['real']).strip().upper() == entry['real']
        stats['n'] += 1
        stats['spotted'] += correct
        source = stats['by_source'].setdefault(entry['frame'].split('_', 1)[0], [0, 0])
        source[0] += correct
        source[1] += 1
    report = {}
    for model, stats in sorted(totals.items(), key=lambda kv: kv[1]['spotted'] / kv[1]['n']):
        rate = stats['spotted'] / stats['n']
        report[model] = {'judged': stats['n'], 'fake_spotted_pct': round(100 * rate, 1),
                         'by_source': {k: f'{a}/{b}' for k, (a, b) in sorted(stats['by_source'].items())}}
        print(f"{model:28s} fake spotted {100 * rate:5.1f}% of {stats['n']}  {report[model]['by_source']}")
    missing = sorted(set(key) - set(answers))
    if missing:
        print(f'{len(missing)} sheets unanswered: {missing[:10]}')
    (out / 'score.json').write_text(json.dumps(report, indent=2) + '\n')


def changed_frames(first, second, count, overall=False):
    """Held-out frames where two renders differ most: the share of pixels that
    moved by more than 8 levels, the top ``count`` overall or equally many per
    source."""
    changes = []
    for stem in test_ids():
        a = load_gray(ROOT / 'fake' / first / f'{stem}.png')
        b = load_gray(ROOT / 'fake' / second / f'{stem}.png', a.shape[::-1])
        a, b = (crop_to_rgb(stem, x).astype(np.float32) for x in (a, b))
        changes.append((float((np.abs(a - b) > 8).mean()), stem))
    changes.sort(reverse=True)
    if overall:
        return [stem for _, stem in changes[:count]]
    sources = sorted({stem.split('_', 1)[0] for _, stem in changes})
    chosen = []
    for source in sources:
        chosen += [stem for _, stem in changes if stem.split('_', 1)[0] == source][:count // len(sources)]
    return chosen


def prefer(args):
    rng = random.Random(args.seed)
    first, second = args.names
    stems = changed_frames(first, second, args.frames, args.overall)
    out = ROOT / 'prefer' / args.round
    sheets = out / 'sheets'
    sheets.mkdir(parents=True, exist_ok=True)
    # each frame is judged by --per-frame different judges; a judge sees a frame once
    batches = []
    for _ in range(args.per_frame):
        order = stems[:]
        rng.shuffle(order)
        batches += [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
    key, assignment, number = [], {}, 0
    for judge, batch in enumerate(batches, 1):
        # each model on the left in half of every judge's sheets, so a judge who
        # always picks one side cannot favour either model
        flips = [i % 2 == 0 for i in range(len(batch))]
        rng.shuffle(flips)
        for stem, flip in zip(batch, flips):
            number += 1
            pair = [second, first] if flip else [first, second]
            a, b = (panel(crop_to_rgb(stem, load_gray(ROOT / 'fake' / name / f'{stem}.png'), args.data),
                          args.width) for name in pair)
            sheet = Image.new('RGB', (2 * args.width + 24, a.height + 36), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((args.width // 2, 10), 'A', fill='black')
            draw.text((args.width + 24 + args.width // 2, 10), 'B', fill='black')
            sheet.paste(a, (0, 36))
            sheet.paste(b, (args.width + 24, 36))
            # lossless: with JPEG the panels' chroma blocks align differently and a
            # faint periodic decoder pattern can show on one side only
            sheet.save(sheets / f'{number:03d}.png')
            key.append({'sheet': f'{number:03d}', 'frame': stem, 'A': pair[0], 'B': pair[1]})
            assignment.setdefault(f'judge{judge}', []).append(f'{number:03d}')
    (out / 'key.json').write_text(json.dumps(key, indent=2) + chr(10))
    (out / 'judges.json').write_text(json.dumps(assignment, indent=2) + chr(10))
    print(f'wrote {len(key)} preference sheets ({len(stems)} frames x {args.per_frame}) for '
          f'{len(assignment)} judges to {sheets}; key in {out / "key.json"}')


def score_prefer(args):
    out = ROOT / 'prefer' / args.round
    key = {entry['sheet']: entry for entry in json.loads((out / 'key.json').read_text())}
    answers, sides = {}, {}
    for path in args.answers:
        judge = json.loads(Path(path).read_text())
        answers.update(judge)
        picks = [str(answer['realistic']).strip().upper() for answer in judge.values()]
        # a judge who could not tell the renders apart tends to pick one side every
        # time; sides are balanced per judge, so those votes cancel, but a round where
        # most judges are one-sided measured no visible difference
        lopsided = len(picks) >= 8 and max(picks.count('A'), picks.count('B')) >= .9 * len(picks)
        sides[Path(path).stem] = {'A': picks.count('A'), 'B': picks.count('B'), 'one_sided': lopsided}
        print(f"{Path(path).name}: picked A {picks.count('A')}/{len(picks)}"
              + ('  one-sided: could not tell them apart' if lopsided else ''))
    wins, sources, frames = {}, {}, {}
    for sheet, answer in answers.items():
        entry = key[sheet]
        pick = entry[str(answer['realistic']).strip().upper()]
        wins[pick] = wins.get(pick, 0) + 1
        source = sources.setdefault(entry['frame'].split('_', 1)[0], {})
        source[pick] = source.get(pick, 0) + 1
        frame = frames.setdefault(entry['frame'], {})
        frame[pick] = frame.get(pick, 0) + 1
    names = sorted({name for entry in key.values() for name in (entry['A'], entry['B'])})
    majority = {name: sum(1 for votes in frames.values() if 2 * votes.get(name, 0) > sum(votes.values()))
                for name in names}
    report = {'preferred': {name: wins.get(name, 0) for name in names}, 'judged': len(answers),
              'by_source': sources, 'frames_won': majority, 'frames': len(frames), 'sides': sides}
    for name in names:
        print(f"{name:28s} preferred {wins.get(name, 0):3d} of {len(answers)}, "
              f"won {majority[name]} of {len(frames)} frames by majority")
    (out / 'score.json').write_text(json.dumps(report, indent=2) + chr(10))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=Path('data512'))
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('ids')
    p.add_argument('--per-source', type=int, default=100)
    p.add_argument('--seed', type=int, default=7)
    commands.add_parser('real')
    p = commands.add_parser('generate')
    p.add_argument('--name', required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--device', default='auto')
    p.add_argument('--no-guided', dest='guided', action='store_false')
    p.add_argument('--engine', choices=('unet', 'turbo'), default='unet')
    p.add_argument('--skip', type=float, default=1., help='turbo skip-connection strength')
    p.add_argument('--semantic', type=float, default=0., help='semantic prior strength (0 = off)')
    p.add_argument('--no-emissive', dest='emissive', action='store_false',
                   help='drop the flame/light colour heuristic from the semantic prior')
    p.add_argument('--no-vehicles', dest='vehicles', action='store_false',
                   help='semantic prior for people and animals only')
    p.add_argument('--sharp-prior', action='store_true', help='fit the prior edges to the photo')
    p.add_argument('--parts-prior', action='store_true',
                   help='skin, clothing and hair heat for people large enough to segment (needs mediapipe)')
    p.add_argument('--sensor', action='store_true')
    p = commands.add_parser('metrics')
    p.add_argument('names', nargs='+')
    p = commands.add_parser('pairs')
    p.add_argument('names', nargs='+')
    p.add_argument('--round', required=True)
    p.add_argument('--per-source', type=int, default=8)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--seed', type=int, default=11)
    p = commands.add_parser('score')
    p.add_argument('--round', required=True)
    p.add_argument('answers', nargs='+')
    p = commands.add_parser('singles')
    p.add_argument('names', nargs='+')
    p.add_argument('--round', required=True)
    p.add_argument('--per-source', type=int, default=6)
    p.add_argument('--offset', type=int, default=40, help='skip the frames used by the paired test')
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--seed', type=int, default=13)
    p = commands.add_parser('score-singles')
    p.add_argument('--round', required=True)
    p.add_argument('answers', nargs='+')
    p = commands.add_parser('prefer')
    p.add_argument('names', nargs=2)
    p.add_argument('--round', required=True)
    p.add_argument('--frames', type=int, default=48)
    p.add_argument('--overall', action='store_true',
                   help='the most-changed frames overall instead of equally many per source')
    p.add_argument('--per-frame', type=int, default=4, help='judges per frame')
    p.add_argument('--batch', type=int, default=24, help='sheets per judge')
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--seed', type=int, default=17)
    p = commands.add_parser('score-prefer')
    p.add_argument('--round', required=True)
    p.add_argument('answers', nargs='+')
    args = parser.parse_args()
    {'ids': choose_ids, 'real': write_real, 'generate': generate, 'metrics': metrics,
     'pairs': pairs, 'score': score, 'singles': singles,
     'score-singles': score_singles, 'prefer': prefer,
     'score-prefer': score_prefer}[args.command](args)


if __name__ == '__main__':
    main()
