"""Phase 5 v2 training entry point: one-step conditional flow-matching head.

Trains FlowUNet with flow_matching_loss and validates every 500 iterations by
L1 of one-step samples (sample_onestep) against targets — sampling replaces
the regression average (docs/ROADMAP.md Phase 5; MeanFlow arXiv 2505.13447,
Shortcut Models ICLR 2025, TReFT arXiv 2511.20307). Warmstarts the encoder
from the v1 regression checkpoint (teacher_warmstart); scripts/train.py (v1)
is untouched and NOT run by this file. Not run yet — this is the v2 skeleton.
"""
from collections import Counter
from itertools import islice
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fake_thermal.data import PairedDataset, validation_ids
from fake_thermal.device import DEVICES, memory_format, select_device
from fake_thermal.flow_model import FlowUNet, flow_matching_loss, sample_onestep, teacher_warmstart
from train import ResumeSampler, WeightEMA, loader, optimizer_for, positive_int, resume_mismatches

PRECISIONS = ('fp32', 'fp16')
LAYOUTS = ('auto', 'contiguous', 'channels_last')


def step(model, optimizer, rgb, target, fp16=False, layout='auto'):
    device = next(model.parameters()).device
    rgb = rgb.to(device=device, dtype=torch.float32, memory_format=memory_format(device, layout))
    target = target.to(device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        loss = flow_matching_loss(model, rgb, target).float()
    if not torch.isfinite(loss):
        raise RuntimeError(f'Non-finite loss: {loss.item()}')
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.inference_mode()
def validate(model, batches, fp16=False, layout='auto'):
    """L1 of one-step samples against targets — the v2 counterpart of the v1
    regression validate(); per-source breakdown and grids stay in v1's infer."""
    model.eval()
    device = next(model.parameters()).device
    total, count = 0., 0
    for rgb, target, _ in batches:
        rgb = rgb.to(device=device, dtype=torch.float32, memory_format=memory_format(device, layout))
        target = target.to(device=device, dtype=torch.float32)
        pred = sample_onestep(model, rgb).float()
        total += (pred - target).abs().mean(dim=(1, 2, 3)).sum().item()
        count += len(rgb)
    if not count:
        raise ValueError('Validation requires at least one image')
    model.train()
    return total / count


def train_run(args, train, val, ids):
    batch, iterations = args.batch_size, args.iters
    fp16 = args.precision == 'fp16'
    config = {'batch_size': batch, 'iterations': iterations, 'crop': args.crop,
              'device': args.device.type, 'precision': args.precision, 'layout': args.layout,
              'seed': args.seed, 'frequency_alpha': 0., 'ema_decay': args.ema_decay,
              'head': 'onestep-flow-v2'}
    model = FlowUNet(pretrained=False).to(device=args.device, dtype=torch.float32,
                                          memory_format=memory_format(args.device, args.layout))
    optimizer = optimizer_for(model)
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
        scheduler.load_state_dict(state['scheduler'])
        if scheduler.T_max != iterations:
            scheduler.T_max = iterations
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        if ema is not None:
            if state.get('ema') is None:
                raise ValueError('Checkpoint has no EMA state for --ema-decay')
            ema.shadow = {key: value.to(device=args.device)
                          for key, value in state['ema'].items()}
        start, best = state['iteration'] + 1, state['best_val_l1']
        stream = iter(train.sampler(iterations * batch, seed=args.seed))
        remaining = list(islice(stream, (start - 1) * batch, iterations * batch))
        sampler = ResumeSampler(remaining)
        print(f'RESUMED iteration={start} best_val_l1={best:.6f} '
              f'remaining_batches={len(remaining) // batch}', flush=True)
    elif args.warmstart:
        if not args.warmstart.exists():
            print(f'WARMSTART skipped: {args.warmstart} not found', flush=True)
        else:
            state = torch.load(args.warmstart, map_location='cpu', weights_only=False)
            report = teacher_warmstart(model, state['model'] if 'model' in state else state)
            print(f'WARMSTART source={args.warmstart} copied={len(report["copied"])} '
                  f'partial={report["partial"]} unmatched_model={report["unmatched_model"]} '
                  f'unmatched_teacher={report["unmatched_teacher"]}', flush=True)
    batches = (loader(train, batch, args.workers, sampler=sampler) if sampler is not None
               else loader(train, batch, args.workers, iterations * batch))
    validation = loader(val, batch, args.workers)
    (args.output / 'val_ids.json').write_text(json.dumps(ids, indent=2) + '\n')
    config.update(loss='conditional flow matching: MSE on x_1 - x_0',
                  sampling='one-step: x_0 + v(x_0, t=1 | rgb), clamped to [0, 1]',
                  warmstart=str(args.warmstart) if args.warmstart else None,
                  learning_rates=[3e-5, 3e-4],
                  source_counts=dict(Counter(train.sources)))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    clock = time.perf_counter()
    print(f'START pid={os.getpid()} batch={batch} iterations={iterations} precision={args.precision} '
          f'layout={args.layout} config={config}', flush=True)
    model.train()
    for offset, (rgb, target, _) in enumerate(batches):
        iteration = start + offset
        loss = step(model, optimizer, rgb, target, fp16, args.layout)
        scheduler.step()
        if ema is not None:
            ema.update(model)
        if iteration % 10 == 0:
            elapsed = time.perf_counter() - clock
            print(f'iter={iteration}/{iterations} loss={loss:.6f} elapsed_s={elapsed:.1f} '
                  f'wall_s_per_iter={elapsed / (offset + 1):.3f} lr={scheduler.get_last_lr()[1]:.8f}', flush=True)
        if iteration % 500 == 0 or iteration == iterations:
            if ema is not None:
                raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
                model.load_state_dict(ema.shadow)
            l1 = validate(model, validation, fp16, args.layout)
            if ema is not None:
                model.load_state_dict(raw)
            improved = l1 < best
            best = min(best, l1)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'iteration': iteration,
                     'val_l1': l1, 'best_val_l1': best, 'config': config,
                     'ema': ema.shadow if ema is not None else None,
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
            print(f'CHECKPOINT iter={iteration} onestep_val_l1={l1:.6f} best={best:.6f}', flush=True)
    print(f'FINISHED elapsed_hours={(time.perf_counter() - clock) / 3600:.3f}', flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs_flow'),
                        help='v2 run directory; kept separate from the v1 outputs/')
    parser.add_argument('--warmstart', type=Path, default=Path('outputs/ckpt/best.pt'),
                        help='v1 regression checkpoint used as encoder teacher; skipped if missing')
    parser.add_argument('--resume', action='store_true',
                        help='continue the run saved at <output>/ckpt/last.pt')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=DEVICES, default='auto')
    parser.add_argument('--precision', choices=PRECISIONS, default='fp32',
                        help='fp16 enables autocast on CUDA/MPS (default fp32)')
    parser.add_argument('--layout', choices=LAYOUTS, default='auto',
                        help='tensor memory layout; auto keeps MPS contiguous, CUDA channels-last')
    parser.add_argument('--ema-decay', type=float, default=0.,
                        help='weight EMA decay used for checkpointing; 0 disables (default), 0.999 recommended')
    parser.add_argument('--batch-size', type=positive_int, required=True)
    parser.add_argument('--iters', type=positive_int, required=True)
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
    if not 0 <= args.ema_decay < 1:
        parser.error('--ema-decay must be in [0, 1)')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    (args.output / 'ckpt').mkdir(parents=True, exist_ok=True)
    train = PairedDataset(args.data, crop=args.crop)
    ids = validation_ids(args.data)
    val = PairedDataset(args.data, 'val', ids, crop=args.crop)
    train_run(args, train, val, ids)
