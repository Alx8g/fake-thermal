"""Rent, feed, run and reap Vast.ai GPU boxes for fake-thermal training.

Run on the machine that holds the Vast API key and the prepared dataset:

    python scripts/vast_fleet.py up --offer 60749257 --label gan
    python scripts/vast_fleet.py push --label gan            # repo + data512, one tar stream
    python scripts/vast_fleet.py launch --label gan --arm gan010 -- --gan-weight .1 ...
    python scripts/vast_fleet.py watch --label gan --floor 0.5

`watch` pulls each arm's snapshots, grids and log every few minutes, pulls the
final checkpoint when an arm prints FINISHED (or dies), and destroys the box
once every arm on it is done, unless outputs/vast/<label>.hold exists. It also
destroys every box when account credit falls to --floor dollars, so a budget
cannot be overspent while nobody is watching.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'outputs/vast'
VAST = str(ROOT / '.venv/bin/vastai')
IMAGE = 'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime'
REPO = 'https://github.com/Alx8g/fake-thermal.git'
KEY = Path.home() / '.ssh/id_ed25519'


def vast(*args):
    out = subprocess.run([VAST, *args, '--raw'], capture_output=True, text=True, check=True).stdout
    return json.loads(out) if out.strip() else None


def credit():
    return float(vast('show', 'user')['credit'])


def state_path(label):
    return STATE / f'{label}.json'


def load(label):
    return json.loads(state_path(label).read_text())


def save(label, data):
    STATE.mkdir(parents=True, exist_ok=True)
    state_path(label).write_text(json.dumps(data, indent=2) + '\n')


def ssh_args(box):
    return ['ssh', '-i', str(KEY), '-p', str(box['port']), '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=30', f"root@{box['host']}"]


def remote(box, command, check=True, **kwargs):
    return subprocess.run([*ssh_args(box), command], text=True, check=check, **kwargs)


def ssh_url(instance):
    url = subprocess.run([VAST, 'ssh-url', str(instance)], capture_output=True,
                         text=True).stdout.strip().splitlines()[-1]
    host_port = url.split('@', 1)[1]
    host, port = host_port.rsplit(':', 1)
    return host, int(port)


def up(args):
    if credit() < args.floor + 0.5:
        sys.exit(f'Credit {credit():.2f} too close to the floor {args.floor:.2f}; not renting')
    result = vast('create', 'instance', str(args.offer), '--image', IMAGE, '--disk', str(args.disk),
                  '--ssh', '--label', f'fake-thermal-{args.label}')
    instance = result['new_contract']
    print(f'rented instance {instance}', flush=True)
    box = {'instance': instance, 'arms': {}, 'rented': time.time()}
    save(args.label, box)
    for _ in range(60):
        info = vast('show', 'instance', str(instance))
        if info and info.get('actual_status') == 'running':
            break
        time.sleep(10)
    else:
        sys.exit('instance never reached running; destroy it by hand')
    box['host'], box['port'] = ssh_url(instance)
    for _ in range(30):
        if remote(box, 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && nproc',
                  check=False, capture_output=True).returncode == 0:
            break
        time.sleep(10)
    save(args.label, box)
    remote(box, 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; '
                'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"')


def push(args):
    box = load(args.label)
    branch = args.branch
    remote(box, f'set -e; test -d fake-thermal || git clone -q {REPO}; cd fake-thermal; '
                f'git fetch -q origin && git checkout -q -B {branch} origin/{branch}; '
                'pip install -q --no-deps -e . && pip install -q lpips opencv-python-headless tqdm '
                '2>&1 | grep -v "already satisfied" | tail -n 3; mkdir -p ~/data512')
    started = time.time()
    tar = subprocess.Popen(['tar', '-C', str(ROOT / args.data), '-cf', '-', 'pairs', 'manifest.json'],
                           stdout=subprocess.PIPE)
    subprocess.run([*ssh_args(box), 'tar -C ~/data512 -xf -'], stdin=tar.stdout, check=True)
    tar.wait()
    print(f'data pushed in {time.time() - started:.0f}s', flush=True)
    remote(box, 'ls ~/data512/pairs/train/rgb | wc -l; du -sh ~/data512')


def launch(args):
    box = load(args.label)
    extra = ' '.join(shlex.quote(a) for a in args.train_args)
    script = args.script or 'scripts/train.py --device cuda --data ~/data512'
    run = (f'cd ~/fake-thermal && git fetch -q origin && git checkout -q -B {args.branch} '
           f'origin/{args.branch} && python {script} --output outputs/{args.arm} {extra} '
           f'> ~/train_{args.arm}.log 2>&1')
    if args.after:
        # chain behind another arm on the same box: start once it has ended
        run = (f'while ! grep -q -E "FINISHED|Traceback" ~/train_{args.after}.log 2>/dev/null; '
               f'do sleep 30; done; {run}')
    remote(box, f'nohup sh -c {shlex.quote(run)} > /dev/null 2>&1 &')
    box['arms'][args.arm] = {'launched': time.time(), 'done': False, 'after': args.after}
    save(args.label, box)
    print(f'launched {args.arm} on {args.label}', flush=True)


def pull(box, arm, final, destination):
    target = destination / arm
    target.mkdir(parents=True, exist_ok=True)
    ssh = ' '.join(shlex.quote(a) for a in ssh_args(box)[:-1])
    include = ['--include=snapshots/***', '--include=val_*.jpg', '--include=config.json',
               '--include=val_ids.json']
    if final:
        include += ['--include=ckpt/', '--include=ckpt/last.pt', '--include=ckpt/best.pt']
    subprocess.run(['rsync', '-a', '-e', ssh, *include, '--exclude=*',
                    f"{ssh_args(box)[-1]}:fake-thermal/outputs/{arm}/", f'{target}/'], check=False)
    subprocess.run(['rsync', '-a', '-e', ssh, f"{ssh_args(box)[-1]}:train_{arm}.log",
                    f'{target}/train.log'], check=False)


def destroy(label, box, reason):
    subprocess.run([VAST, 'destroy', 'instance', str(box['instance']), '-y'], check=False,
                   capture_output=True)
    for _ in range(12):
        alive = [i['id'] for i in vast('show', 'instances')]
        if box['instance'] not in alive:
            break
        time.sleep(5)
    else:
        print(f'WARNING: instance {box["instance"]} still listed after destroy', flush=True)
        return
    box['destroyed'] = time.time()
    box['destroy_reason'] = reason
    save(label, box)
    print(f'DESTROYED {label} ({box["instance"]}): {reason}; credit now {credit():.2f}', flush=True)


def watch(args):
    destination = ROOT / args.into
    while True:
        box = load(args.label)
        if box.get('destroyed'):
            print('box already destroyed', flush=True)
            return
        money = credit()
        for arm, info in box['arms'].items():
            if info['done']:
                continue
            log = remote(box, f'tail -n 2 ~/train_{arm}.log 2>/dev/null; '
                              f'grep -c -E "FINISHED|Traceback" ~/train_{arm}.log 2>/dev/null',
                         check=False, capture_output=True).stdout.strip().splitlines()
            ended = bool(log) and log[-1].isdigit() and int(log[-1]) > 0
            print(f'{time.strftime("%H:%M")} {arm}: {log[-2][:160] if len(log) > 1 else "(no log yet)"}',
                  flush=True)
            pull(box, arm, final=ended, destination=destination)
            if ended:
                info['done'] = True
                save(args.label, box)
                print(f'ARM ENDED {arm}', flush=True)
        box = load(args.label)
        everything_done = box['arms'] and all(i['done'] for i in box['arms'].values())
        if money <= args.floor:
            for arm in box['arms']:
                pull(box, arm, final=True, destination=destination)
            destroy(args.label, box, f'credit {money:.2f} reached floor {args.floor:.2f}')
            return
        if everything_done and not (STATE / f'{args.label}.hold').exists():
            destroy(args.label, box, 'all arms finished')
            return
        if args.max_hours and time.time() - box['rented'] > args.max_hours * 3600:
            for arm in box['arms']:
                pull(box, arm, final=True, destination=destination)
            destroy(args.label, box, f'max {args.max_hours}h reached')
            return
        time.sleep(args.every)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('up')
    p.add_argument('--offer', type=int, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--disk', type=int, default=60)
    p.add_argument('--floor', type=float, default=.5)
    p = commands.add_parser('push')
    p.add_argument('--label', required=True)
    p.add_argument('--data', default='data512')
    p.add_argument('--branch', default='feat/realism-campaign')
    p = commands.add_parser('launch')
    p.add_argument('--label', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--after', help='start only when this arm on the same box has ended')
    p.add_argument('--script', help='training command instead of scripts/train.py (args before --output)')
    p.add_argument('--branch', default='feat/realism-campaign')
    p.add_argument('train_args', nargs=argparse.REMAINDER)
    p = commands.add_parser('watch')
    p.add_argument('--label', required=True)
    p.add_argument('--floor', type=float, default=.5, help='destroy everything at this credit')
    p.add_argument('--max-hours', type=float, default=0.)
    p.add_argument('--every', type=int, default=180)
    p.add_argument('--into', default='outputs/v6-cloud')
    args = parser.parse_args()
    if getattr(args, 'train_args', None) and args.train_args[0] == '--':
        args.train_args = args.train_args[1:]
    {'up': up, 'push': push, 'launch': launch, 'watch': watch}[args.command](args)


if __name__ == '__main__':
    main()
