"""Compare trained checkpoints against the semantic heat prior baseline."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from fake_thermal.data import PairedDataset, normalize_thermal, seed_worker
from fake_thermal.device import DEVICES, memory_format, select_device
from fake_thermal.model import ThermalUNet
from fake_thermal.semantic import SemanticPrior, fuse_heat

VARIANTS = ('model', 'fused', 'prior', 'stretched_prior')


@torch.inference_mode()
def evaluate(model, prior, batches, strength, device):
    if model is not None:
        model.eval()
    totals = {name: {} for name in VARIANTS}
    for rgb, target, names in batches:
        rgb = rgb.to(device=device, dtype=torch.float32, memory_format=memory_format(device))
        heat = model(rgb)[:, 0].cpu().numpy() if model is not None else None
        target = target.numpy()
        for i, name in enumerate(names):
            source = name.split('_', 1)[0]
            variants = {}
            if heat is not None:
                variants['model'] = heat[i]
            if prior is not None:
                rgb_frame = (rgb[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                prior_map, confidence = prior(rgb_frame, heat[i].shape if heat is not None
                                              else target[i, 0].shape)
                if heat is not None:
                    variants['fused'] = fuse_heat(heat[i], prior_map, confidence, strength)
                variants['prior'] = prior_map
                variants['stretched_prior'] = normalize_thermal(prior_map)
            for variant, prediction in variants.items():
                error = float(np.abs(prediction - target[i, 0]).mean())
                entry = totals[variant].setdefault(source, [0., 0])
                entry[0] += error
                entry[1] += 1
    return {variant: {'per_source': {source: round(sum_ / count, 6) for source, (sum_, count)
                                     in sorted(sources.items())},
                      'overall': round(sum(v[0] for v in sources.values()) /
                                       sum(v[1] for v in sources.values()), 6)}
            for variant, sources in totals.items() if sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+',
                        default=[Path('outputs/ckpt/last.pt'), Path('outputs/ckpt/best.pt')])
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs/eval.json'))
    parser.add_argument('--device', choices=DEVICES, default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--semantic-strength', type=float, default=.7)
    args = parser.parse_args()
    if not 0 <= args.semantic_strength <= 1:
        parser.error('Semantic strength must be between 0 and 1')
    device = select_device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    states = [torch.load(path, map_location='cpu', weights_only=False)
              for path in args.checkpoints]
    ids = states[0]['val_ids']
    if any(state['val_ids'] != ids for state in states[1:]):
        parser.error('Checkpoints disagree on validation images')
    crop = states[0]['config']['crop']
    dataset = PairedDataset(args.data, 'val', ids, crop=crop)
    batches = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        persistent_workers=args.workers > 0, worker_init_fn=seed_worker)

    prior = SemanticPrior(size=320, device=device.type)
    results = {'device': device.type, 'images': len(ids), 'crop': crop,
               'semantic_strength': args.semantic_strength,
               'baseline': evaluate(None, prior, batches, args.semantic_strength, device),
               'checkpoints': []}
    for path, state in zip(args.checkpoints, states):
        model = ThermalUNet(pretrained=False).eval()
        weights = state['ema'] if state.get('ema') is not None else state['model']
        model.load_state_dict(weights, strict=True)
        model.to(device=device, dtype=torch.float32, memory_format=memory_format(device))
        entry = {'path': str(path), 'iteration': state['iteration'],
                 'stored_val_l1': round(state['val_l1'], 6),
                 'stored_best_val_l1': round(state['best_val_l1'], 6),
                 **evaluate(model, prior, batches, args.semantic_strength, device)}
        results['checkpoints'].append(entry)
        print(f'{path.name}: iter={entry["iteration"]} model_L1={entry["model"]["overall"]} '
              f'fused_L1={entry["fused"]["overall"]} (stored {entry["stored_val_l1"]})', flush=True)

    for name, metrics in (('raw prior', results['baseline']['prior']),
                          ('stretched prior', results['baseline']['stretched_prior'])):
        print(f'baseline {name}: L1={metrics["overall"]} per_source={metrics["per_source"]}', flush=True)
    best = min(results['checkpoints'], key=lambda entry: entry['model']['overall'])
    print(f'BEST {best["path"]} iter={best["iteration"]} model_L1={best["model"]["overall"]}', flush=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
