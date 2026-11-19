"""Prepare only aligned pairs; preserve originals and record every split decision."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import random
import zipfile

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageOps
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def download(raw):
    jobs = [('hieupth/llvip_384', 'images.zip', 'llvip'),
            ('jsonhash/FLIR_aligned', 'aligned.zip', 'flir')]
    jobs += [('Maxim37/flir_adas_video_pairs', f'data/test-{i:05d}-of-00004.parquet', 'video')
             for i in range(4)]
    for repo, filename, source in jobs:
        hf_hub_download(repo, filename, repo_type='dataset', local_dir=raw / source)


def sequence_split(ids, fraction=0.05, seed=42, gap=20, embargo=150):
    """Infer recording groups from index gaps; keep groups whole, buffer validation."""
    ids = sorted(set(ids))
    if not ids:
        raise ValueError('No aligned pairs found')
    groups = [[ids[0]]]
    for n in ids[1:]:
        if n - groups[-1][-1] > gap:
            groups.append([])
        groups[-1].append(n)
    if len(groups) < 2:
        raise ValueError('Cannot create a sequence-disjoint split from one group')
    random.Random(seed).shuffle(groups)
    target = len(ids) * fraction
    chosen = min(groups, key=lambda group: abs(len(group) - target))
    val = set(chosen)
    ranges = [(min(g) - embargo, max(g) + embargo) for g in groups if g[0] in val]
    split = {}
    for group in groups:
        disposition = ('val' if group[0] in val else 'excluded' if any(
            min(group) <= b and max(group) >= a for a, b in ranges) else 'train')
        split.update({n: disposition for n in group})
    return split, sorted([min(g), max(g), len(g), split[g[0]]] for g in groups)


def resize_pair(rgb, thermal, short_side=288):
    if rgb.size != thermal.size:
        raise ValueError(f'Unaligned dimensions: {rgb.size} vs {thermal.size}')
    w, h = rgb.size
    size = (round(w * short_side / min(w, h)), round(h * short_side / min(w, h)))
    return (rgb.convert('RGB').resize(size, Image.Resampling.LANCZOS),
            thermal.convert('L').resize(size, Image.Resampling.LANCZOS))


def panel(rgb, thermal, label):
    tile = Image.new('RGB', (640, 190), 'white')
    tile.paste(ImageOps.contain(rgb.convert('RGB'), (320, 160)), (0, 25))
    if thermal is not None:
        tile.paste(ImageOps.contain(thermal.convert('RGB'), (320, 160)), (320, 25))
    else:
        ImageDraw.Draw(tile).text((330, 80), 'THERMAL MISSING - EXCLUDED', fill='red')
    ImageDraw.Draw(tile).text((4, 4), label, fill='black')
    return tile


def prepare(data, short_side=288):
    raw, out = data / 'raw', data / 'pairs'
    if out.exists() and any(out.rglob('*.*')):
        raise ValueError('Output already populated; choose a fresh --data-dir or move pairs/ first')
    for split in ('train', 'val'):
        for mode in ('rgb', 'thermal'):
            (out / split / mode).mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    tiles, records, counts = [], [], Counter()
    with zipfile.ZipFile(raw / 'flir/aligned.zip') as z:
        names = set(z.namelist())
        rgbs = sorted(n for n in names if n.startswith('align/JPEGImages/') and n.endswith('_RGB.jpg'))
        pairs = {int(n.split('_')[1]): (n, n.replace('_RGB.jpg', '_PreviewData.jpeg')) for n in rgbs}
        if any(t not in names for _, t in pairs.values()):
            raise ValueError('FLIR has unmatched RGB frames')
        splits, groups = sequence_split(pairs)
        preview_ids = set(rng.sample(sorted(pairs), min(12, len(pairs))))
        resolutions = Counter()
        for n, (r, t) in tqdm(sorted(pairs.items()), desc='FLIR aligned'):
            with Image.open(z.open(r)) as ri, Image.open(z.open(t)) as ti:
                resolutions[str(ri.size)] += 1
                rgb, thermal = resize_pair(ri, ti, short_side)
                if n in preview_ids:
                    tiles.append(panel(rgb, thermal, f'FLIR aligned {n:05d} [{splits[n]}]'))
                split = splits[n]
                counts[split] += 1
                stem = f'flir_{n:05d}'
                if split != 'excluded':
                    rgb.save(out / split / 'rgb' / f'{stem}.jpg', quality=95)
                    thermal.save(out / split / 'thermal' / f'{stem}.png')
                records.append({'source': 'jsonhash/FLIR_aligned', 'id': stem, 'split': split,
                                'rgb_member': r, 'thermal_member': t})
    with zipfile.ZipFile(raw / 'llvip/images.zip') as z:
        names = sorted(n for n in z.namelist() if n.endswith('.jpg'))
        modalities = Counter(n.split('/')[2] for n in names)
        for n in rng.sample(names, min(12, len(names))):
            with Image.open(z.open(n)) as im:
                tiles.append(panel(im, None, f'LLVIP RGB only: {Path(n).name} - EXCLUDED'))
    video_count, video_groups, video_sizes = 0, Counter(), Counter()
    # Reservoir sampling over streamed batches avoids loading gigabytes of images into RAM.
    samples = []
    for f in sorted((raw / 'video/data').glob('*.parquet')):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=16, columns=[
                'rgb_image', 'thermal_image', 'rgb_image_file_name']):
            for row in batch.to_pylist():
                video_count += 1
                video_groups[row['rgb_image_file_name'].split('-frame-')[0]] += 1
                with Image.open(BytesIO(row['rgb_image']['bytes'])) as ri, \
                        Image.open(BytesIO(row['thermal_image']['bytes'])) as ti:
                    video_sizes[str((ri.size, ti.size))] += 1
                    j = rng.randrange(video_count)
                    if len(samples) < 12:
                        samples.append(panel(ri, ti, 'FLIR video UNALIGNED - EXCLUDED'))
                    elif j < 12:
                        samples[j] = panel(ri, ti, 'FLIR video UNALIGNED - EXCLUDED')
    tiles.extend(samples)
    sheet = Image.new('RGB', (1280, ((len(tiles) + 1) // 2) * 190), 'white')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 640, (i // 2) * 190))
    sheet.save(data / 'preview.jpg', quality=92)
    summary = {'flir': {'available_pairs': len(pairs), 'counts': dict(counts),
                       'resolutions': dict(resolutions), 'inferred_sequence_groups': groups,
                       'split_policy': 'whole index runs (gap >20), 150-index embargo around val; seed 42; approximate scenes only'},
               'llvip': {'images': len(names), 'modalities': dict(modalities), 'usable_pairs': 0},
               'video': {'pairs': video_count, 'recordings': dict(video_groups),
                         'resolutions': dict(video_sizes), 'usable_pairs': 0}}
    (data / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    (data / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--short-side', type=int, default=288)
    args = parser.parse_args()
    if args.short_side <= 0:
        parser.error('--short-side must be positive')
    if args.download:
        download(args.data_dir / 'raw')
    prepare(args.data_dir, args.short_side)
