"""Fine-tune pix2pix-Turbo (SD-Turbo, one denoising step) for RGB-to-thermal.

Losses follow img2img-turbo's paired recipe: L2 + LPIPS(VGG) reconstruction and
an adversarial term from a CLIP-feature (vision-aided) discriminator, which
supplies the realism prior a small regression U-Net lacks. Trained LoRA, skip
and input-conv weights are saved as small snapshots; at the end the held-out
realism frames (outputs/realism/ids.json) are rendered for scoring.

    python scripts/train_turbo.py --data ~/data512 --output outputs/turbo \
        --iters 5000 --batch-size 2 --ids /root/init/ids.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from fake_thermal.data import normalize_thermal, seed_worker


class TurboPairs(Dataset):
    """512x512 random crops; thermal stretched, dequantized, 3 channels in [-1, 1]."""

    def __init__(self, data, split='train', stems=None, crop=512, training=True):
        root = Path(data) / 'pairs' / split
        self.rgb, self.thermal = root / 'rgb', root / 'thermal'
        self.stems = stems or sorted(p.stem for p in self.rgb.glob('*.jpg') if not p.name.startswith('._'))
        self.crop, self.training = crop, training
        self.jitter = T.ColorJitter(brightness=.2, contrast=.2, saturation=.15)

    def __len__(self):
        return len(self.stems)

    def sampler(self, samples, seed=42):
        sources = [s.split('_', 1)[0] for s in self.stems]
        counts = Counter(sources)
        return WeightedRandomSampler([1. / counts[s] for s in sources], samples, replacement=True,
                                     generator=torch.Generator().manual_seed(seed))

    def __getitem__(self, index):
        stem = self.stems[index]
        with Image.open(self.rgb / f'{stem}.jpg') as im:
            rgb = im.convert('RGB')
        with Image.open(self.thermal / f'{stem}.png') as im:
            thermal = Image.fromarray(normalize_thermal(im.convert('L'), dequantize=self.training))
        if min(rgb.size) < self.crop:
            rgb, thermal = TF.resize(rgb, self.crop), TF.resize(thermal, self.crop)
        w, h = rgb.size
        if self.training:
            top, left = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
        else:
            top, left = (h - self.crop) // 2, (w - self.crop) // 2
        rgb = TF.crop(rgb, top, left, self.crop, self.crop)
        thermal = TF.crop(thermal, top, left, self.crop, self.crop)
        if self.training:
            if random.random() < .5:
                rgb, thermal = TF.hflip(rgb), TF.hflip(thermal)
            rgb = self.jitter(rgb)
        target = TF.to_tensor(thermal).repeat(3, 1, 1) * 2 - 1
        return TF.to_tensor(rgb), target, stem


def stretch8(heat):
    return (normalize_thermal(heat) * 255).round().astype(np.uint8)


def amp_dtype(device):
    """bf16 autocast on CUDA; MPS lacks bf16 kernels for some ops, so fp32 there."""
    return torch.bfloat16 if device.type == 'cuda' else None


def autocast(device):
    dtype = amp_dtype(device)
    return torch.autocast(device.type, dtype=dtype or torch.float16, enabled=dtype is not None)


def save_grid(net, batch, path):
    rgb, target, _ = batch
    net.set_eval()
    with torch.no_grad(), autocast(net.device_name):
        pred = net(rgb.to(net.device_name)).float().cpu()
    tiles = []
    for i in range(len(rgb)):
        row = [rgb[i], (target[i] + 1) / 2, (pred[i] + 1) / 2]
        tiles.append(torch.cat([t.clamp(0, 1) for t in row], dim=2))
    grid = torch.cat(tiles, dim=1).permute(1, 2, 0).numpy()
    Image.fromarray((grid * 255).astype(np.uint8)).resize(
        (grid.shape[1] // 2, grid.shape[0] // 2), Image.LANCZOS).save(path, quality=90)


def build_discriminator(kind, device):
    if kind == 'vagan':
        import vision_aided_loss
        disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type='multilevel_sigmoid_s',
                                               device=device.type)
        disc = disc.to(device)
        disc.requires_grad_(True)
        disc.cv_ensemble.requires_grad_(False)
        for name, module in disc.named_modules():
            if 'attn' in name:
                module.fused_attn = False
        return disc
    raise ValueError(f'Unknown discriminator {kind}')


def generate(net, args):
    groups = json.loads(Path(args.ids).read_text())
    stems = [s for values in groups.values() for s in values]
    out = args.output / 'fake'
    out.mkdir(parents=True, exist_ok=True)
    net.set_eval()
    for stem in stems:
        with Image.open(Path(args.data) / f'pairs/val/rgb/{stem}.jpg') as im:
            rgb = np.asarray(im.convert('RGB'))
        heat = net.translate(rgb, dtype=amp_dtype(net.device_name))
        Image.fromarray(stretch8(heat)).save(out / f'{stem}.png')
    print(f'GENERATED {len(stems)} held-out frames to {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=Path('data512'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup', type=int, default=300)
    parser.add_argument('--lambda-l2', type=float, default=1.)
    parser.add_argument('--lambda-lpips', type=float, default=5.)
    parser.add_argument('--lambda-gan', type=float, default=.5)
    parser.add_argument('--disc', default='vagan')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--snapshot-every', type=int, default=1000)
    parser.add_argument('--grid-every', type=int, default=250)
    parser.add_argument('--ids', type=Path, help='held-out realism ids to render at the end')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-checkpointing', dest='checkpointing', action='store_false')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import lpips
    from fake_thermal.turbo import Pix2PixTurbo

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    (args.output / 'snapshots').mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + '\n')

    net = Pix2PixTurbo(device=device)
    net.set_train()
    if args.checkpointing:
        net.unet.enable_gradient_checkpointing()
    params = net.trainable_parameters()
    print(f'START trainable={sum(p.numel() for p in params) / 1e6:.1f}M params', flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    disc = build_discriminator(args.disc, device)
    optimizer_d = torch.optim.AdamW([p for p in disc.parameters() if p.requires_grad], lr=args.lr,
                                    weight_decay=1e-2)
    schedule = lambda step: min(1., (step + 1) / args.warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, schedule)
    perceptual = lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False)

    train = TurboPairs(args.data, crop=args.crop)
    batches = iter(DataLoader(train, batch_size=args.batch_size, sampler=train.sampler(args.iters * args.batch_size),
                              num_workers=args.workers, worker_init_fn=seed_worker, pin_memory=True,
                              drop_last=True, persistent_workers=args.workers > 0))
    val_stems = []
    for source in ('flir', 'kaist', 'llvip'):
        val_stems += sorted(p.stem for p in (Path(args.data) / 'pairs/val/rgb').glob(f'{source}_*.jpg'))[:2]
    val = TurboPairs(args.data, 'val', val_stems, crop=args.crop, training=False)
    val_batch = next(iter(DataLoader(val, batch_size=len(val))))

    clock = time.perf_counter()
    for step in range(1, args.iters + 1):
        rgb, target, _ = next(batches)
        rgb, target = rgb.to(device, non_blocking=True), target.to(device, non_blocking=True)
        net.set_train()
        with autocast(device):
            pred = net(rgb)
            l2 = F.mse_loss(pred.float(), target)
            lp = perceptual(pred.float(), target).mean()
            disc.requires_grad_(False)
            g_adv = disc(pred, for_G=True).mean()
        loss = args.lambda_l2 * l2 + args.lambda_lpips * lp + args.lambda_gan * g_adv.float()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}: {loss.item()}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        scheduler.step()

        disc.requires_grad_(True)
        disc.cv_ensemble.requires_grad_(False)
        with autocast(device):
            d_real = disc(target, for_real=True).mean()
            d_fake = disc(pred.detach(), for_real=False).mean()
        d_loss = args.lambda_gan * (d_real + d_fake).float()
        optimizer_d.zero_grad(set_to_none=True)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.)
        optimizer_d.step()
        scheduler_d.step()

        if step % 20 == 0:
            elapsed = time.perf_counter() - clock
            print(f'iter={step}/{args.iters} loss={loss.item():.4f} l2={l2.item():.4f} lpips={lp.item():.4f} '
                  f'g_adv={g_adv.item():.4f} d={d_loss.item():.4f} elapsed_s={elapsed:.0f} '
                  f'wall_s_per_iter={elapsed / step:.3f}', flush=True)
        if step % args.grid_every == 0 or step == args.iters:
            save_grid(net, val_batch, args.output / f'val_{step:06d}.jpg')
        if step % args.snapshot_every == 0 or step == args.iters:
            net.save_weights(args.output / f'snapshots/iter_{step:06d}.pt')
            print(f'CHECKPOINT iter={step}', flush=True)
    if args.ids:
        generate(net, args)
    print(f'FINISHED elapsed_hours={(time.perf_counter() - clock) / 3600:.3f}', flush=True)


if __name__ == '__main__':
    main()
