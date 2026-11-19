"""CPU training and a 50-iteration throughput benchmark for fake-thermal."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from fake_thermal.data import PairedDataset, seed_worker, validation_ids
from fake_thermal.losses import thermal_loss
from fake_thermal.model import ThermalUNet


def loader(dataset, batch_size, workers, samples=None):
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=dataset.sampler(samples) if samples else None,
                      num_workers=workers, persistent_workers=workers > 0,
                      worker_init_fn=seed_worker, drop_last=samples is not None)


def optimizer_for(model):
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
    return torch.optim.AdamW([{'params': encoder, 'lr': 3e-5},
                              {'params': decoder, 'lr': 3e-4}], weight_decay=.01)


def step(model, optimizer, rgb, target):
    rgb = rgb.contiguous(memory_format=torch.channels_last)
    optimizer.zero_grad(set_to_none=True)
    pred = model(rgb)
    loss = thermal_loss(pred, target)
    if not torch.isfinite(loss):
        raise RuntimeError(f'Non-finite loss: {loss.item()}')
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.inference_mode()
def validate(model, batches, grid_path=None):
    model.eval()
    total, count, by_source, examples = 0., 0, {}, {}
    for rgb, target, names in batches:
        pred = model(rgb.contiguous(memory_format=torch.channels_last))
        errors = (pred - target).abs().mean(dim=(1, 2, 3))
        total += errors.sum().item()
        count += len(rgb)
        for i, name in enumerate(names):
            source = name.split('_', 1)[0]
            by_source.setdefault(source, []).append(errors[i].item())
            examples.setdefault(source, [])
            if len(examples[source]) < 4:
                examples[source].append((rgb[i], target[i], pred[i], name))
    if grid_path:
        rows = [row for source in sorted(examples) for row in examples[source]]
        grid = Image.new('RGB', (768, len(rows) * 280), 'white')
        draw = ImageDraw.Draw(grid)
        for row, (rgb, target, pred, name) in enumerate(rows):
            draw.text((4, row * 280 + 3), name + '   RGB | normalized GT | prediction', fill='black')
            for col, x in enumerate((rgb, target, pred)):
                a = (x.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype('uint8')
                if a.shape[-1] == 1:
                    a = a[..., 0]
                grid.paste(Image.fromarray(a).convert('RGB'), (col * 256, row * 280 + 24))
        grid.save(grid_path, quality=92)
    model.train()
    return total / count, {k: float(np.mean(v)) for k, v in by_source.items()}


def benchmark(args, train, val):
    results = []
    for batch in (2, 4):
        model = ThermalUNet().to(memory_format=torch.channels_last)
        optimizer = optimizer_for(model)
        batches = iter(loader(train, batch, args.workers, 25 * batch))
        timings = []
        model.train()
        for i in range(25):
            start = time.perf_counter()
            rgb, target, _ = next(batches)
            loss = step(model, optimizer, rgb, target)
            timings.append(time.perf_counter() - start)
            print(f'benchmark batch={batch} iter={i+1}/25 loss={loss:.5f} seconds={timings[-1]:.3f}', flush=True)
        seconds = float(np.mean(timings[5:]))
        results.append({'batch_size': batch, 'seconds_per_iter': seconds,
                        'images_per_second': batch / seconds})
        del batches, optimizer, model
    selected = max(results, key=lambda r: r['images_per_second'])
    model = ThermalUNet().to(memory_format=torch.channels_last)
    start = time.perf_counter()
    validate(model, loader(val, selected['batch_size'], args.workers))
    val_seconds = time.perf_counter() - start
    seconds = selected['seconds_per_iter']
    iterations = max(500, math.floor(.95 * args.hours * 3600 / (seconds + val_seconds / 500)))
    result = dict(selected, candidates=results, validation_seconds=val_seconds,
                  iterations=iterations, hours=args.hours, threads=args.threads,
                  workers=args.workers, memory_format='channels_last',
                  estimated_hours=iterations * (seconds + val_seconds / 500) / 3600)
    (args.output / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def train_run(args, train, val, ids):
    config = json.loads((args.output / 'benchmark.json').read_text())
    batch, iterations = config['batch_size'], config['iterations']
    model = ThermalUNet().to(memory_format=torch.channels_last)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    batches = loader(train, batch, args.workers, iterations * batch)
    validation = loader(val, batch, args.workers)
    (args.output / 'val_ids.json').write_text(json.dumps(ids, indent=2) + '\n')
    config.update(seed=args.seed, learning_rates=[3e-5, 3e-4],
                  loss='L1 + 0.5*(1-SSIM) + 0.1*gradient', target='per-image p1/p99 stretch',
                  source_counts=dict(Counter(train.sources)))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    best, start = float('inf'), time.perf_counter()
    print(f'START pid={os.getpid()} batch={batch} iterations={iterations} config={config}', flush=True)
    model.train()
    for iteration, (rgb, target, _) in enumerate(batches, 1):
        loss = step(model, optimizer, rgb, target)
        scheduler.step()
        if iteration % 10 == 0:
            elapsed = time.perf_counter() - start
            print(f'iter={iteration}/{iterations} loss={loss:.6f} elapsed_s={elapsed:.1f} '
                  f'wall_s_per_iter={elapsed/iteration:.3f} lr={scheduler.get_last_lr()[1]:.8f}', flush=True)
        if iteration % 500 == 0 or iteration == iterations:
            grid = args.output / f'val_{iteration:06d}.jpg'
            l1, source_l1 = validate(model, validation, grid)
            improved = l1 < best
            best = min(best, l1)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'iteration': iteration,
                     'val_l1': l1, 'best_val_l1': best, 'config': config,
                     'val_ids': ids, 'torch_rng': torch.get_rng_state(),
                     'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
            path = args.output / 'ckpt/last.pt'
            temporary = path.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(path)
            if improved:
                temporary = args.output / 'ckpt/best.tmp'
                torch.save(state, temporary)
                temporary.replace(args.output / 'ckpt/best.pt')
            print(f'CHECKPOINT iter={iteration} val_l1={l1:.6f} per_source={source_l1} best={best:.6f} grid={grid}', flush=True)
    print(f'FINISHED elapsed_hours={(time.perf_counter()-start)/3600:.3f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--hours', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    (args.output / 'ckpt').mkdir(parents=True, exist_ok=True)
    train = PairedDataset(args.data)
    ids = validation_ids(args.data)
    val = PairedDataset(args.data, 'val', ids)
    if args.benchmark:
        benchmark(args, train, val)
    else:
        train_run(args, train, val, ids)
