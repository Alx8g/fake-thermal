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
    jobs += [('jsonhash/LLVIP', 'LLVIP.zip', 'llvip_full'),
             ('AmineMarnissi/ThermalKaist', 'RGBKaist.zip', 'kaist'),
             ('AmineMarnissi/ThermalKaist', 'thermalKaist.zip', 'kaist')]
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


def save_sheet(tiles, path):
    sheet = Image.new('RGB', (1280, ((len(tiles) + 1) // 2) * 190), 'white')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 640, (i // 2) * 190))
    sheet.save(path, quality=92)


def kaist_index(names, modality):
    return {Path(n).stem.replace(f'_{modality}_', '_'): n for n in names
            if '/JPEGImages/' in n and n.endswith('.png')}


def subsample_sequences(keys, stride=5):
    sequences = {}
    for key in sorted(keys):
        sequences.setdefault(key.rsplit('_', 1)[0], []).append(key)
    return sorted(key for frames in sequences.values() for key in frames[::stride])


def prepare_extra(raw, out, data, records, short_side):
    summaries, all_tiles = {}, []

    def emit(source, rz, tz, pairs, info):
        counts, sizes, lighting = Counter(), Counter(), Counter()
        tiles = []
        sample = set(random.Random(42).sample(range(len(pairs)), min(12, len(pairs))))
        for i, (key, r, t, split, group, light) in enumerate(tqdm(pairs, desc=source)):
            with Image.open(rz.open(r)) as ri, Image.open(tz.open(t)) as ti:
                sizes[str((ri.size, ti.size))] += 1
                rgb, thermal = resize_pair(ri, ti, short_side)
                stem = f'{source}_{key}'
                rgb.save(out / split / 'rgb' / f'{stem}.jpg', quality=95)
                thermal.save(out / split / 'thermal' / f'{stem}.png')
                counts[split] += 1
                lighting[f'{split}_{light}'] += 1
                records.append({'source': source, 'id': stem, 'split': split,
                                'group': group, 'lighting': light, 'rgb_member': r,
                                'thermal_member': t})
                if i in sample:
                    tiles.append(panel(rgb, thermal, f'{source} {key} [{split}, {light}]'))
        save_sheet(tiles, data / f'preview_{source}.jpg')
        summaries[source] = dict(info, counts=dict(counts), resolutions=dict(sizes),
                                 lighting=dict(lighting))
        all_tiles.extend(tiles)

    with zipfile.ZipFile(raw / 'llvip_full/LLVIP.zip') as z:
        names = set(z.namelist())
        pairs, scenes = [], {'train': set(), 'val': set()}
        for r in sorted(n for n in names if '/visible/' in n and n.endswith('.jpg')):
            t = r.replace('/visible/', '/infrared/')
            if t not in names:
                raise ValueError(f'Missing LLVIP thermal partner: {r}')
            key = Path(r).stem
            split = 'val' if '/test/' in r else 'train'
            scenes[split].add(key[:2])
            pairs.append((key, r, t, split, key[:2], 'low_light'))
        if scenes['train'] & scenes['val']:
            raise ValueError('LLVIP scene overlap between train and test')
        emit('llvip', z, z, pairs, {'available_pairs': len(pairs),
             'scenes': {k: sorted(v) for k, v in scenes.items()},
             'split_policy': 'original train/test, disjoint two-digit scene prefixes'})
    with zipfile.ZipFile(raw / 'kaist/RGBKaist.zip') as rz, \
            zipfile.ZipFile(raw / 'kaist/thermalKaist.zip') as tz:
        rgb, thermal = kaist_index(rz.namelist(), 'visible'), kaist_index(tz.namelist(), 'lwir')
        matched = rgb.keys() & thermal.keys()
        pairs = []
        for key in subsample_sequences(matched):
            set_id = int(key[3:5])
            split = 'train' if set_id < 6 else 'val'
            light = 'day' if set_id in (0, 1, 2, 6, 7, 8) else 'night'
            pairs.append((key, rgb[key], thermal[key], split, key.rsplit('_', 1)[0], light))
        emit('kaist', rz, tz, pairs, {'available_pairs': len(matched),
             'unmatched_rgb': sorted(rgb.keys() - thermal.keys()),
             'unmatched_thermal': sorted(thermal.keys() - rgb.keys()),
             'split_policy': 'sets00-05 train / sets06-11 val; every fifth available matched frame per sequence',
             'lighting_policy': 'KAIST set convention: 00-02/06-08 day, 03-05/09-11 night'})
    return summaries, all_tiles


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
    save_sheet(tiles, data / 'preview_flir.jpg')
    extra_summary, extra_tiles = prepare_extra(raw, out, data, records, short_side)
    tiles.extend(extra_tiles)
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
    save_sheet(samples, data / 'preview_video_excluded.jpg')
    tiles.extend(samples)
    save_sheet(tiles, data / 'preview.jpg')
    summary = {'flir': {'available_pairs': len(pairs), 'counts': dict(counts),
                       'resolutions': dict(resolutions), 'inferred_sequence_groups': groups,
                       'split_policy': 'whole index runs (gap >20), 150-index embargo around val; seed 42; approximate scenes only'},
               **extra_summary,
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
