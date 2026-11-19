from collections import Counter
from io import BytesIO
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF


def normalize_thermal(image, dequantize=False):
    """Per-image p1/p99 stretch. ``dequantize`` adds uniform +-0.5 level noise
    first, so stretched 8-bit targets stop sitting on a visible lattice that a
    critic could separate from continuous predictions."""
    a = np.asarray(image, dtype=np.float32)
    if dequantize:
        a = a + np.random.uniform(-.5, .5, a.shape).astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


class PairedDataset(Dataset):
    def __init__(self, data_dir, split='train', records=None, crop=256):
        if crop < 32:
            raise ValueError('Crop size must be at least 32')
        self.crop = crop
        self.root = Path(data_dir) / 'pairs' / split
        self.training = split == 'train'
        self.paths = sorted((self.root / 'rgb').glob('*.jpg')) if records is None else [
            self.root / 'rgb' / f'{stem}.jpg' for stem in records]
        if not self.paths:
            raise ValueError(f'No paired images in {self.root}')
        self.sources = [p.stem.split('_', 1)[0] for p in self.paths]
        self.jitter = T.ColorJitter(brightness=.6, contrast=.6, saturation=.6, hue=.12)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        with Image.open(path) as im:
            rgb = im.convert('RGB')
        with Image.open(self.root / 'thermal' / f'{path.stem}.png') as im:
            target = Image.fromarray(normalize_thermal(im.convert('L'), dequantize=self.training))
        if self.training:
            top, left, h, w = T.RandomResizedCrop.get_params(rgb, scale=(.6, 1.), ratio=(.9, 1.1))
            rgb = TF.resized_crop(rgb, top, left, h, w, (self.crop, self.crop))
            target = TF.resized_crop(target, top, left, h, w, (self.crop, self.crop))
            if random.random() < .5:
                rgb, target = TF.hflip(rgb), TF.hflip(target)
            rgb = self.jitter(rgb)
            rgb = TF.adjust_gamma(rgb, random.uniform(.55, 1.8), gain=random.uniform(.7, 1.3))
            if random.random() < .2:
                rgb = TF.to_grayscale(rgb, num_output_channels=3)
            if random.random() < .25:
                rgb = rgb.filter(ImageFilter.GaussianBlur(random.uniform(.1, 1.1)))
            if random.random() < .3:
                buffer = BytesIO()
                rgb.save(buffer, format='JPEG', quality=random.randint(40, 90))
                buffer.seek(0)
                rgb = Image.open(buffer).convert('RGB')
        else:
            rgb, target = TF.center_crop(rgb, (self.crop, self.crop)), TF.center_crop(target, (self.crop, self.crop))
        rgb = TF.to_tensor(rgb)
        if self.training and random.random() < .3:
            rgb = (rgb + torch.randn_like(rgb) * random.uniform(.005, .035)).clamp(0, 1)
        return rgb, TF.to_tensor(target), path.stem

    def sampler(self, samples, seed=42):
        counts = Counter(self.sources)
        weights = [1. / counts[source] for source in self.sources]
        return WeightedRandomSampler(weights, samples, replacement=True,
                                     generator=torch.Generator().manual_seed(seed))


def validation_ids(data_dir, size=300, seed=42):
    paths = sorted((Path(data_dir) / 'pairs/val/rgb').glob('*.jpg'))
    groups = {}
    for p in paths:
        groups.setdefault(p.stem.split('_', 1)[0], []).append(p.stem)
    if len(groups) != 3 or any(len(v) < size // 3 for v in groups.values()):
        raise ValueError('Validation needs three sources with at least 100 images each')
    rng = random.Random(seed)
    return [key for source in sorted(groups) for key in rng.sample(groups[source], size // 3)]
