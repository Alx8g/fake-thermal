"""Temporal flicker of thermal outputs on consecutive video frames.

Warping error (Lai et al., "Learning Blind Video Temporal Consistency", 2018):
optical flow is estimated on the colour video, the previous output frame is
carried along it to the current one, and the mean absolute difference is taken
over pixels the flow tracks reliably (forward-backward consistent, not
occluded). Lower is steadier. Every stream is stretched with one p1/p99 range
over the whole clip, so values are comparable between streams and with the
real thermal camera when its registered frames are available.

    python scripts/flicker_report.py --rgb data512/pairs/train/rgb \
        --thermal data512/pairs/train/thermal --prefix video_RMxN6a4CcCeLGu4tA_ \
        --start 200 --count 90 --out outputs/flicker/flir_a
    python scripts/flicker_report.py --video clip.mp4 --count 120 --out outputs/flicker/walk
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image


def gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def flow_pair(previous, current, dis):
    """Backward flow (current -> previous) and a mask of reliably tracked pixels."""
    back = dis.calc(current, previous, None)
    fore = dis.calc(previous, current, None)
    h, w = current.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x, y = gx + back[..., 0], gy + back[..., 1]
    fore_at = cv2.remap(fore, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e3)
    trip = np.hypot(back[..., 0] + fore_at[..., 0], back[..., 1] + fore_at[..., 1])
    scale = .01 * ((back ** 2).sum(-1) + (fore_at ** 2).sum(-1)) + .5
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    return x, y, (trip < scale) & inside


def warp_error(frames, flows):
    errors = []
    for (x, y, valid), previous, current in zip(flows, frames[:-1], frames[1:]):
        warped = cv2.remap(previous, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        errors.append(float(np.abs(current - warped)[valid].mean()))
    return errors


def clip_stretch(frames):
    stack = np.stack(frames)
    lo, hi = np.percentile(stack, [1, 99])
    return [np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32) for f in frames]


def load_frames(args):
    if args.video:
        h = args.size
        command = ['ffmpeg', '-v', 'error', '-i', str(args.video), '-vf', f'scale=-2:{h}', '-frames:v',
                   str(args.count + args.start), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                'stream=width,height', '-of', 'json', str(args.video)],
                               capture_output=True, text=True, check=True)
        stream = json.loads(probe.stdout)['streams'][0]
        w = round(stream['width'] * h / stream['height'] / 2) * 2
        raw = subprocess.run(command, capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)[args.start:]
        return list(frames), None
    rgb, thermal = [], []
    for index in range(args.start, args.start + args.count):
        path = args.rgb / f'{args.prefix}{index:06d}.jpg'
        if not path.is_file():
            break  # keep the run of strictly consecutive frames
        rgb.append(np.asarray(Image.open(path).convert('RGB')))
        if args.thermal:
            thermal.append(np.asarray(Image.open(args.thermal / f'{path.stem}.png').convert('L'),
                                      np.float32))
    return rgb, thermal or None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rgb', type=Path)
    parser.add_argument('--thermal', type=Path)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=90)
    parser.add_argument('--weights', type=Path, default=None, help='turbo weights (default: bundled)')
    parser.add_argument('--smoothing', type=float, nargs='+', default=[.5, .65, .8])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from fake_thermal.infer import DEFAULT_TURBO_WEIGHTS, TemporalSmoother, TurboPredictor
    from fake_thermal.semantic import SemanticPrior, fuse_heat
    rgb, thermal = load_frames(args)
    if len(rgb) < 3:
        sys.exit('need at least three consecutive frames')
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / 'turbo_raw.npy'
    if cache.is_file() and len(np.load(cache)) == len(rgb):
        raw = list(np.load(cache))
    else:
        predictor = TurboPredictor(args.weights or DEFAULT_TURBO_WEIGHTS, size=args.size, device=args.device)
        prior = SemanticPrior(device=args.device, emissive=False, vehicles=False, sharp=True, parts=True)
        raw = []
        for frame in rgb:
            heat = predictor(frame)
            raw.append(fuse_heat(heat, *prior(frame, heat.shape), .7).astype(np.float32))
        np.save(cache, np.stack(raw))
    shape = raw[0].shape
    rgb_small = [cv2.resize(f, shape[::-1], interpolation=cv2.INTER_AREA) for f in rgb]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flows = [flow_pair(gray(a), gray(b), dis) for a, b in zip(rgb_small[:-1], rgb_small[1:])]
    streams = {'rgb luminance (flow reference)': clip_stretch([gray(f).astype(np.float32) for f in rgb_small]),
               'turbo, no smoothing': clip_stretch(raw)}
    for weight in args.smoothing:
        smoother = TemporalSmoother(weight)
        streams[f'turbo, smoothing {weight:g}'] = clip_stretch(
            [smoother.update(h, f) for h, f in zip(raw, rgb_small)])
    if thermal:
        streams['real thermal camera'] = clip_stretch(
            [cv2.resize(t, shape[::-1], interpolation=cv2.INTER_AREA) for t in thermal])
    report = {'frames': len(rgb), 'tracked_fraction': float(np.mean([v.mean() for _, _, v in flows]))}
    for name, frames in streams.items():
        errors = warp_error(frames, flows)
        report[name] = {'mean': float(np.mean(errors)), 'p90': float(np.percentile(errors, 90))}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
