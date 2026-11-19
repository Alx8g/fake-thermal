"""Real-vs-fake patch classification: the photorealism benchmark.

Trains a small CNN on image patches from real thermal footage and the
model's predictions of the same scenes, then reports held-out accuracy.
50% (chance) means a classifier cannot tell the outputs apart; the gap above
chance is the indistinguishability gap.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from fake_thermal.data import PairedDataset, normalize_thermal, validation_ids
from fake_thermal.device import select_device
from fake_thermal.infer import Predictor
from fake_thermal.sensor import ThermalSensor


def patches(image, count, size, rng):
    if image.shape[0] < size or image.shape[1] < size:
        raise ValueError(f'Image {image.shape} smaller than patch {size}')
    result = []
    for _ in range(count):
        top, left = rng.integers(0, image.shape[0] - size), rng.integers(0, image.shape[1] - size)
        result.append(image[top:top + size, left:left + size])
    return result


class PatchDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def train_discriminator(train_x, train_y, epochs, seed=42):
    torch.manual_seed(seed)
    model = PatchDiscriminator()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    x = torch.from_numpy(train_x).float().unsqueeze(1)
    y = torch.from_numpy(train_y).long()
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        for start in range(0, len(x), 128):
            index = order[start:start + 128]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x[index]), y[index])
            loss.backward()
            optimizer.step()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('fake_thermal/weights/fake_thermal.pt'))
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs/realism.json'))
    parser.add_argument('--images', type=int, default=60)
    parser.add_argument('--patches', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    ids = validation_ids(args.data, size=300)[:args.images]
    dataset = PairedDataset(args.data, 'val', ids, crop=256)
    predictor = Predictor(weights=args.checkpoint, device=args.device, threads=args.threads)
    rng = np.random.default_rng(42)

    real_x, fake_x = [], []
    for i in range(len(dataset)):
        rgb_tensor, target_tensor, name = dataset[i]
        rgb = (rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        real = normalize_thermal(target_tensor[0].numpy())
        fake = cv2.resize(predictor(rgb), real.shape[::-1])
        real_x += patches(real, args.patches, args.patch_size, rng)
        fake_x += patches(fake, args.patches, args.patch_size, rng)
        if (i + 1) % 20 == 0:
            print(f'sampled {i + 1}/{len(dataset)} images', flush=True)

    def variance(images):
        return float(np.median([cv2.Laplacian((p * 255).astype(np.float32), cv2.CV_32F).var()
                                for p in images]))

    results = {'checkpoint': str(args.checkpoint), 'patches': len(real_x),
               'real_laplacian_var': variance(real_x)}
    for label, sensor in (('model', False), ('model+sensor', True)):
        sim = ThermalSensor(seed=7) if sensor else None
        source = [sim.update(p.copy()) if sim else p for p in fake_x]
        x = np.concatenate([np.stack(real_x), np.stack(source)], 0).astype(np.float32)
        y = np.concatenate([np.ones(len(real_x)), np.zeros(len(source))]).astype(np.int64)
        order = np.random.default_rng(0).permutation(len(x))
        split = int(.8 * len(x))
        train, test = order[:split], order[split:]
        model = train_discriminator(x[train], y[train], args.epochs)
        model.eval()
        with torch.inference_mode():
            scores = model(torch.from_numpy(x[test]).float().unsqueeze(1))
        accuracy = float((scores.argmax(1) == torch.from_numpy(y[test])).float().mean())
        results[label] = {'accuracy': round(accuracy, 4),
                          'laplacian_var': variance(source)}
        print(f'{label}: classifier accuracy {accuracy:.3f} '
              f'(chance .500, real var {results["real_laplacian_var"]:.1f} vs {variance(source):.1f})',
              flush=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
