"""Export a trusted local training checkpoint to a small fp16 state dict."""
import argparse
from pathlib import Path

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('-o', '--output', type=Path, default=Path('weights/fake_thermal.pt'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = {k: v.half() if v.is_floating_point() else v for k, v in checkpoint['model'].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(args.output)
    print(f'Exported iteration {checkpoint["iteration"]}, val L1 {checkpoint["val_l1"]:.6f} to {args.output}')
