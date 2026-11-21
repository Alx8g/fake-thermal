"""Training and a 50-iteration throughput benchmark for fake-thermal."""
import argparse
from collections import Counter
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from fake_thermal.data import PairedDataset, seed_worker, validation_ids
from fake_thermal.device import DEVICES, memory_format, select_device, synchronize
from fake_thermal.discriminator import (MultiScaleDiscriminator, discriminator_loss,
                                        feature_matching_loss, generator_adversarial_loss)
from fake_thermal.losses import thermal_loss
from fake_thermal.model import ThermalUNet

PRECISIONS = ('fp32', 'fp16', 'bf16')
LAYOUTS = ('auto', 'contiguous', 'channels_last')
RESUME_DEFAULTS = {'precision': 'fp32', 'layout': 'auto', 'frequency_alpha': 0.,
                   'ema_decay': 0., 'conditioning': False, 'lpips_alpha': 0.,
                   'fullres': False, 'gan_weight': 0., 'fm_weight': 0.}


def amp_dtype(precision):
    """Autocast dtype for a --precision value, or None for full fp32."""
    if precision in (True, 'fp16'):
        return torch.float16
    if precision == 'bf16':
        return torch.bfloat16
    if isinstance(precision, torch.dtype):
        return precision
    return None


def forward(model, rgb, fp16=False):
    """Autocast forward that always returns float32 so losses stay stable.
    ``fp16`` accepts a bool, a --precision name or an autocast dtype."""
    dtype = amp_dtype(fp16)
    with torch.autocast(device_type=rgb.device.type, dtype=dtype or torch.float16,
                        enabled=dtype is not None):
        return model(rgb).float()


class ResumeSampler:
    """Replays the unconsumed tail of a seeded sampling stream."""

    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class WeightEMA:
    """Exponential moving average of weights; non-float buffers are copied verbatim."""

    def __init__(self, model, decay):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0, 1)')
        self.decay = decay
        self.shadow = {key: value.detach().clone()
                       for key, value in model.state_dict().items()}

    @torch.inference_mode()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value, alpha=1 - self.decay)
            else:
                self.shadow[key].copy_(value)


def load_scheduler(scheduler, stored, iterations):
    """Restored state carries the saved cosine horizon; extensions adopt the
    new one so learning rates keep decaying to the extended endpoint."""
    scheduler.load_state_dict(stored)
    if scheduler.T_max != iterations:
        scheduler.T_max = iterations


def resume_mismatches(stored, current):
    keys = ('batch_size', 'crop', 'seed', 'precision', 'layout', 'frequency_alpha', 'ema_decay',
            'conditioning', 'lpips_alpha', 'fullres', 'gan_weight', 'fm_weight')
    stored = {**RESUME_DEFAULTS, **stored}
    current = {**RESUME_DEFAULTS, **current}
    return [f'{key}: {stored.get(key)!r} != {current.get(key)!r}'
            for key in keys if stored.get(key) != current.get(key)]


def loader(dataset, batch_size, workers, samples=None, sampler=None, pin=False):
    selected = sampler if sampler is not None else (dataset.sampler(samples) if samples else None)
    return DataLoader(dataset, batch_size=batch_size, sampler=selected,
                      num_workers=workers, persistent_workers=workers > 0,
                      worker_init_fn=seed_worker, pin_memory=pin,
                      prefetch_factor=4 if workers > 0 else None,
                      drop_last=samples is not None or sampler is not None)


def default_workers():
    """Data augmentation is CPU-bound (~20-30 ms per 448px sample), so a fast
    GPU needs many workers; two starved a rented RTX 4090 to ~1/3 of its speed."""
    return max(2, min(16, (os.cpu_count() or 4) - 2))


def worker_count(value):
    return default_workers() if value == 'auto' else int(value)


def load_init(model, path):
    """Warm-start weights (EMA preferred) from a training checkpoint or an
    exported state dict; optimizer state is deliberately not carried over."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state and 'iteration' in state:
        state = state['ema'] if state.get('ema') is not None else state['model']
    model.load_compatible({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model


def save_snapshot(weights, path):
    """Small fp16 copy of the evaluated weights, so mid-run models survive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({k: v.detach().half().cpu() if v.is_floating_point() else v.cpu()
                for k, v in weights.items()}, temporary)
    temporary.replace(path)


def optimizer_for(model):
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
    return torch.optim.AdamW([{'params': encoder, 'lr': 3e-5},
                              {'params': decoder, 'lr': 3e-4}], weight_decay=.01)


def perceptual_metric(device):
    """LPIPS(AlexNet) as a training loss term (Johnson et al. pattern)."""
    try:
        import lpips
    except ImportError:
        raise SystemExit('--lpips-alpha needs the optional lpips package: uv pip install lpips')
    return lpips.LPIPS(net='alex').to(device).eval()


class Adversary:
    """Critic, its optimizer and the loss weights for adversarial training."""

    def __init__(self, device, gan_weight, fm_weight, lr=2e-4):
        self.critic = MultiScaleDiscriminator().to(device)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(.5, .999))
        self.gan_weight, self.fm_weight = gan_weight, fm_weight
        self.last = {}


def step(model, optimizer, rgb, target, fp16=False, layout='auto', frequency_alpha=0.,
         metric=None, lpips_alpha=0., adversary=None):
    device = next(model.parameters()).device
    rgb = rgb.to(device=device, dtype=torch.float32, non_blocking=True,
                 memory_format=memory_format(device, layout))
    target = target.to(device=device, dtype=torch.float32, non_blocking=True)
    dtype = amp_dtype(fp16)
    optimizer.zero_grad(set_to_none=True)
    pred = forward(model, rgb, fp16)
    loss = thermal_loss(pred, target, frequency_alpha)
    if metric is not None:
        triplet = lambda frame: frame.repeat(1, 3, 1, 1) * 2 - 1
        loss = loss + lpips_alpha * metric(triplet(pred), triplet(target)).mean()
    if adversary is not None:
        critic = adversary.critic
        critic.requires_grad_(False)
        with torch.autocast(device_type=device.type, dtype=dtype or torch.float16,
                            enabled=dtype is not None):
            real, fake = critic(rgb, target), critic(rgb, pred)
            adversarial = generator_adversarial_loss(fake).float()
            matching = feature_matching_loss(real, fake).float()
        loss = loss + adversary.gan_weight * adversarial + adversary.fm_weight * matching
        adversary.last = {'g_adv': adversarial.item(), 'fm': matching.item()}
    if not torch.isfinite(loss):
        raise RuntimeError(f'Non-finite loss: {loss.item()}')
    loss.backward()
    optimizer.step()
    if adversary is not None:
        critic.requires_grad_(True)
        adversary.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype or torch.float16,
                            enabled=dtype is not None):
            d_loss = discriminator_loss(critic(rgb, target), critic(rgb, pred.detach())).float()
        if not torch.isfinite(d_loss):
            raise RuntimeError(f'Non-finite critic loss: {d_loss.item()}')
        d_loss.backward()
        adversary.optimizer.step()
        adversary.last['d'] = d_loss.item()
    return loss.item()


@torch.inference_mode()
def validate(model, batches, grid_path=None, fp16=False, layout='auto'):
    model.eval()
    device = next(model.parameters()).device
    total, count, by_source, examples = 0., 0, {}, {}
    for rgb, target, names in batches:
        rgb = rgb.to(device=device, dtype=torch.float32, memory_format=memory_format(device, layout))
        target = target.to(device=device, dtype=torch.float32)
        pred = forward(model, rgb, fp16)
        errors = (pred - target).abs().mean(dim=(1, 2, 3))
        total += errors.sum().item()
        count += len(rgb)
        for i, name in enumerate(names):
            source = name.split('_', 1)[0]
            by_source.setdefault(source, []).append(errors[i].item())
            examples.setdefault(source, [])
            if len(examples[source]) < 4:
                examples[source].append((rgb[i], target[i], pred[i], name))
    if not count:
        raise ValueError('Validation requires at least one image')
    if grid_path:
        rows = [row for source in sorted(examples) for row in examples[source]]
        tile = rows[0][0].shape[-1]
        row_height = tile + 24
        grid = Image.new('RGB', (3 * tile, len(rows) * row_height), 'white')
        draw = ImageDraw.Draw(grid)
        for row, (rgb, target, pred, name) in enumerate(rows):
            draw.text((4, row * row_height + 3), name + '   RGB | normalized GT | prediction', fill='black')
            for col, x in enumerate((rgb, target, pred)):
                a = (x.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype('uint8')
                if a.shape[-1] == 1:
                    a = a[..., 0]
                grid.paste(Image.fromarray(a).convert('RGB'), (col * tile, row * row_height + 24))
        grid.save(grid_path, quality=92)
    model.train()
    return total / count, {k: float(np.mean(v)) for k, v in by_source.items()}


def benchmark(args, train, val):
    results = []
    fp16 = args.precision
    metric = perceptual_metric(args.device) if args.lpips_alpha else None
    candidates = (args.batch_size,) if args.batch_size else (2, 4)
    steps = 50 // len(candidates)
    for batch in candidates:
        model = ThermalUNet(conditioning=args.conditioning, fullres=args.fullres).to(
            device=args.device, dtype=torch.float32,
            memory_format=memory_format(args.device, args.layout))
        optimizer = optimizer_for(model)
        batches = iter(loader(train, batch, args.workers, steps * batch, pin=args.device.type == 'cuda'))
        timings = []
        model.train()
        for i in range(steps):
            synchronize(args.device)
            start = time.perf_counter()
            rgb, target, _ = next(batches)
            loss = step(model, optimizer, rgb, target, fp16, args.layout, args.frequency_alpha,
                         metric, args.lpips_alpha)
            synchronize(args.device)
            timings.append(time.perf_counter() - start)
            print(f'benchmark batch={batch} iter={i+1}/{steps} loss={loss:.5f} seconds={timings[-1]:.3f}', flush=True)
        seconds = float(np.mean(timings[5:]))
        results.append({'batch_size': batch, 'seconds_per_iter': seconds,
                        'images_per_second': batch / seconds})
        del batches, optimizer, model
    selected = max(results, key=lambda r: r['images_per_second'])
    model = ThermalUNet(conditioning=args.conditioning, fullres=args.fullres).to(
        device=args.device, dtype=torch.float32,
        memory_format=memory_format(args.device, args.layout))
    synchronize(args.device)
    start = time.perf_counter()
    validate(model, loader(val, selected['batch_size'], args.workers), fp16=fp16, layout=args.layout)
    synchronize(args.device)
    val_seconds = time.perf_counter() - start
    seconds = selected['seconds_per_iter']
    iterations = args.iters or max(500, math.floor(.95 * args.hours * 3600 / (seconds + val_seconds / 500)))
    result = dict(selected, candidates=results, validation_seconds=val_seconds,
                  iterations=iterations, hours=args.hours, threads=args.threads,
                  workers=args.workers, memory_format=str(memory_format(args.device, args.layout)),
                  device=args.device.type, crop=args.crop, precision=args.precision, layout=args.layout,
                  estimated_hours=iterations * (seconds + val_seconds / 500) / 3600)
    (args.output / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def training_config(args):
    path = args.output / 'benchmark.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    batch = args.batch_size or config.get('batch_size')
    iterations = args.iters or config.get('iterations')
    if batch is None or iterations is None:
        raise ValueError('Run --benchmark first or provide both --batch-size and --iters')
    config.update(batch_size=batch, iterations=iterations, crop=args.crop, device=args.device.type,
                  memory_format=str(memory_format(args.device, args.layout)),
                  precision=args.precision, layout=args.layout, conditioning=args.conditioning)
    return config


def train_run(args, train, val, ids):
    config = training_config(args)
    config.update(seed=args.seed, frequency_alpha=args.frequency_alpha,
                  ema_decay=args.ema_decay, lpips_alpha=args.lpips_alpha, fullres=args.fullres,
                  gan_weight=args.gan_weight, fm_weight=args.fm_weight,
                  init=str(args.init) if args.init else None)
    batch, iterations = config['batch_size'], config['iterations']
    fp16 = args.precision
    metric = perceptual_metric(args.device) if args.lpips_alpha else None
    model = ThermalUNet(conditioning=args.conditioning, fullres=args.fullres)
    if args.init and not args.resume:
        load_init(model, args.init)
        print(f'INIT weights from {args.init}', flush=True)
    model = model.to(device=args.device, dtype=torch.float32,
                     memory_format=memory_format(args.device, args.layout))
    optimizer = optimizer_for(model)
    adversary = (Adversary(args.device, args.gan_weight, args.fm_weight, args.d_lr)
                 if args.gan_weight or args.fm_weight else None)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    start, best, sampler = 1, float('inf'), None
    ema = WeightEMA(model, args.ema_decay) if args.ema_decay else None
    if args.resume:
        path = args.output / 'ckpt/last.pt'
        if not path.exists():
            raise ValueError(f'--resume requires an existing checkpoint at {path}')
        state = torch.load(path, map_location='cpu', weights_only=False)
        mismatches = resume_mismatches(state['config'], config)
        if mismatches:
            raise ValueError('Checkpoint differs from the requested run: ' + '; '.join(mismatches))
        if state['config'].get('iterations', 0) > iterations:
            raise ValueError('Checkpoint iterations exceed the configured iterations')
        if state['val_ids'] != ids:
            raise ValueError('Checkpoint validation split differs from the current data')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        load_scheduler(scheduler, state['scheduler'], iterations)
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        if ema is not None:
            if state.get('ema') is None:
                raise ValueError('Checkpoint has no EMA state for --ema-decay')
            ema.shadow = {key: value.to(device=args.device)
                          for key, value in state['ema'].items()}
        if adversary is not None:
            if state.get('critic') is None:
                raise ValueError('Checkpoint has no critic state for adversarial training')
            adversary.critic.load_state_dict(state['critic'])
            adversary.optimizer.load_state_dict(state['critic_optimizer'])
        start, best = state['iteration'] + 1, state['best_val_l1']
        stream = iter(train.sampler(iterations * batch))
        remaining = list(islice(stream, (start - 1) * batch, iterations * batch))
        sampler = ResumeSampler(remaining)
        print(f'RESUMED iteration={start} best_val_l1={best:.6f} '
              f'remaining_batches={len(remaining) // batch}', flush=True)
    pin = args.device.type == 'cuda'
    batches = (loader(train, batch, args.workers, sampler=sampler, pin=pin) if sampler is not None
               else loader(train, batch, args.workers, iterations * batch, pin=pin))
    validation = loader(val, batch, min(args.workers, 4), pin=pin)
    (args.output / 'val_ids.json').write_text(json.dumps(ids, indent=2) + '\n')
    config.update(learning_rates=[3e-5, 3e-4],
                  loss='L1 + 0.5*(1-SSIM) + 0.1*gradient', target='per-image p1/p99 stretch',
                  source_counts=dict(Counter(train.sources)))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    clock = time.perf_counter()
    print(f'START pid={os.getpid()} batch={batch} iterations={iterations} precision={args.precision} '
          f'layout={args.layout} workers={args.workers} config={config}', flush=True)
    model.train()
    for offset, (rgb, target, _) in enumerate(batches):
        iteration = start + offset
        loss = step(model, optimizer, rgb, target, fp16, args.layout, args.frequency_alpha,
                    metric, args.lpips_alpha, adversary)
        scheduler.step()
        if ema is not None:
            ema.update(model)
        if iteration % 10 == 0:
            elapsed = time.perf_counter() - clock
            critic = ''.join(f' {k}={v:.4f}' for k, v in adversary.last.items()) if adversary else ''
            print(f'iter={iteration}/{iterations} loss={loss:.6f}{critic} elapsed_s={elapsed:.1f} '
                  f'wall_s_per_iter={elapsed / (offset + 1):.3f} lr={scheduler.get_last_lr()[1]:.8f}', flush=True)
        if iteration % 500 == 0 or iteration == iterations:
            grid = args.output / f'val_{iteration:06d}.jpg'
            if ema is not None:
                raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
                model.load_state_dict(ema.shadow)
            l1, source_l1 = validate(model, validation, grid, fp16, args.layout)
            if ema is not None:
                model.load_state_dict(raw)
            improved = l1 < best
            best = min(best, l1)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'iteration': iteration,
                     'val_l1': l1, 'best_val_l1': best, 'config': config,
                     'ema': ema.shadow if ema is not None else None,
                     'critic': adversary.critic.state_dict() if adversary else None,
                     'critic_optimizer': adversary.optimizer.state_dict() if adversary else None,
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
            if args.snapshot_every and (iteration % args.snapshot_every == 0 or iteration == iterations):
                save_snapshot(ema.shadow if ema is not None else model.state_dict(),
                              args.output / f'snapshots/iter_{iteration:06d}.pt')
            print(f'CHECKPOINT iter={iteration} val_l1={l1:.6f} per_source={source_l1} best={best:.6f} grid={grid}', flush=True)
    print(f'FINISHED elapsed_hours={(time.perf_counter() - clock) / 3600:.3f}', flush=True)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--workers', type=worker_count, default='auto',
                        help='dataloader workers; auto = CPU count - 2, capped at 16')
    parser.add_argument('--hours', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=DEVICES, default='auto')
    parser.add_argument('--precision', choices=PRECISIONS, default='fp32',
                        help='fp16 autocast on CUDA/MPS; bf16 on CUDA (default fp32)')
    parser.add_argument('--layout', choices=LAYOUTS, default='auto',
                        help='tensor memory layout; auto keeps MPS contiguous, CUDA channels-last')
    parser.add_argument('--resume', action='store_true',
                        help='continue the run saved at outputs/ckpt/last.pt; iterations may be extended, not shortened')
    parser.add_argument('--frequency-alpha', type=float, default=0.,
                        help='focal frequency loss exponent; 0 disables (default), 1.0 recommended')
    parser.add_argument('--ema-decay', type=float, default=0.,
                        help='weight EMA decay used for checkpointing; 0 disables (default), 0.999 recommended')
    parser.add_argument('--lpips-alpha', type=float, default=0.,
                        help='LPIPS perceptual loss weight; 0 disables (default). Counteracts the '
                             'regression-to-mean blur that L1 rewards; needs the lpips package')
    parser.add_argument('--conditioning', action='store_true',
                        help='FiLM style conditioning from input luminance statistics (default off)')
    parser.add_argument('--fullres', action='store_true',
                        help='add the full-resolution RGB detail path into the head')
    parser.add_argument('--init', type=Path,
                        help='warm-start weights from a checkpoint or exported state dict')
    parser.add_argument('--gan-weight', type=float, default=0.,
                        help='adversarial (hinge PatchGAN) generator loss weight; 0 disables')
    parser.add_argument('--fm-weight', type=float, default=0.,
                        help='critic feature-matching loss weight (pix2pixHD); 0 disables')
    parser.add_argument('--d-lr', type=float, default=2e-4, help='critic learning rate')
    parser.add_argument('--snapshot-every', type=int, default=0,
                        help='also save fp16 evaluated weights every N iterations (multiple of 500)')
    parser.add_argument('--batch-size', type=positive_int)
    parser.add_argument('--iters', type=positive_int)
    parser.add_argument('--crop', type=positive_int, default=256)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    if args.crop < 32:
        parser.error('--crop must be at least 32')
    try:
        args.device = select_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    if args.precision == 'fp16' and args.device.type not in ('cuda', 'mps'):
        parser.error('--precision fp16 requires CUDA or MPS')
    if args.precision == 'bf16' and args.device.type != 'cuda':
        parser.error('--precision bf16 requires CUDA')
    if args.gan_weight < 0 or args.fm_weight < 0:
        parser.error('--gan-weight and --fm-weight must be nonnegative')
    if args.snapshot_every and args.snapshot_every % 500:
        parser.error('--snapshot-every must be a multiple of the 500-iteration checkpoint interval')
    if args.device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.frequency_alpha < 0:
        parser.error('--frequency-alpha must be nonnegative')
    if not 0 <= args.ema_decay < 1:
        parser.error('--ema-decay must be in [0, 1)')
    if args.lpips_alpha < 0:
        parser.error('--lpips-alpha must be nonnegative')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    (args.output / 'ckpt').mkdir(parents=True, exist_ok=True)
    train = PairedDataset(args.data, crop=args.crop)
    ids = validation_ids(args.data)
    val = PairedDataset(args.data, 'val', ids, crop=args.crop)
    if args.benchmark:
        benchmark(args, train, val)
    else:
        train_run(args, train, val, ids)
