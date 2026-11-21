"""Launch a fake-thermal training run on a rented GPU (Vast.ai / RunPod / any
CUDA host). Provider-agnostic: the same command works on any Linux box with
CUDA torch installed and the repo + data present.

Setup on the rented host (once):
    git clone https://github.com/Alx8g/fake-thermal && cd fake-thermal
    python -m venv .venv && source .venv/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install -e . lpips huggingface_hub pyarrow
    # copy the prepared dataset from your machine (~8 GB):
    rsync -azP <local>/data512/ host:data512/

Run all experiment arms in parallel as separate instances (or sequentially on
one box). Checkpoints resume across preemptions: atomic writes + --resume.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ARMS = {
    'v5-lpips025': [],                                    # v4 recipe (control)
    'v5-lpips050': ['--lpips-alpha', '0.5'],
    'v5-lpips100': ['--lpips-alpha', '1.0'],
    'v5-long': ['--iters', '50000'],
    'v5-nocond': [],                                      # overrides below
    'v5-noema': [],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--data', type=Path, default=Path('data512'))
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    output = args.output or Path('outputs') / args.arm
    command = [sys.executable, 'scripts/train.py',
               '--device', 'cuda', '--data', str(args.data),
               '--batch-size', str(args.batch_size), '--iters', str(args.iters),
               '--crop', '448', '--frequency-alpha', '1', '--ema-decay', '0.999',
               '--conditioning', '--output', str(output),
               *ARMS[args.arm]]
    if args.arm == 'v5-nocond':
        command.remove('--conditioning')
    if args.arm == 'v5-noema':
        command.remove('--ema-decay'); command.remove('0.999')
    if args.resume:
        command.append('--resume')
    print(' '.join(command), flush=True)
    subprocess.run(command, check=True)
