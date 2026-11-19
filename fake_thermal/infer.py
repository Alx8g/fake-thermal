"""Image/video inference. Requires ffmpeg and ffprobe on PATH."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm

from fake_thermal.model import ThermalUNet
from fake_thermal.device import DEVICES, memory_format, select_device
from fake_thermal.semantic import SemanticPrior, fuse_heat, guided_upsample
from fake_thermal.sensor import ThermalSensor

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / 'weights/fake_thermal.pt'
UNREADABLE_MEDIA = "Couldn't read this file as a video or image — try an MP4/MOV/JPG/PNG."
PALETTES = {
    'white-hot': [(0, 0, 0), (255, 255, 255)],
    'black-hot': [(255, 255, 255), (0, 0, 0)],
    'ironbow': [(0, 0, 0), (35, 0, 80), (135, 12, 90), (230, 60, 20), (255, 190, 30), (255, 255, 235)],
    'rainbow': [(0, 0, 90), (0, 50, 255), (0, 240, 220), (100, 255, 0), (255, 220, 0), (255, 20, 0)],
    'arctic': [(0, 0, 25), (20, 30, 120), (0, 150, 220), (110, 235, 255), (255, 255, 255)],
    'lava': [(0, 0, 0), (40, 0, 60), (150, 10, 10), (230, 90, 10), (255, 170, 40), (255, 240, 170)],
    'glowbow': [(10, 0, 60), (70, 20, 140), (160, 40, 130), (220, 90, 60), (255, 170, 40), (255, 230, 140)],
}


def apply_palette(heat, palette):
    anchors = np.asarray(PALETTES[palette], dtype=np.float32)
    x = np.clip(heat, 0, 1) * (len(anchors) - 1)
    lo = np.minimum(x.astype(np.int32), len(anchors) - 2)
    frac = (x - lo)[..., None]
    return np.rint(anchors[lo] * (1 - frac) + anchors[lo + 1] * frac).astype(np.uint8)


class TemporalSmoother:
    def __init__(self, smoothing=.65, cut_threshold=.22, flow=True, tolerance=1.5):
        if not 0 <= smoothing < 1:
            raise ValueError('Smoothing must be in [0, 1)')
        self.smoothing, self.threshold = smoothing, cut_threshold
        self.flow, self.tolerance = flow, tolerance
        self.previous, self.scene, self.scene_gray = None, None, None
        self.reset_history = True

    def update(self, heat, rgb):
        scene = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        cut = self.scene is not None and np.abs(scene - self.scene).mean() > self.threshold
        self.reset_history = self.previous is None or cut or self.previous.shape != heat.shape
        current = None
        if self.flow:
            small = cv2.resize(rgb, flow_grid(heat.shape), interpolation=cv2.INTER_AREA)
            current = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        if self.reset_history:
            result = heat.copy()
        elif not self.flow:
            result = self.smoothing * self.previous + (1 - self.smoothing) * heat
        else:
            result = self.propagate(heat, current, self.scene_gray)
        self.previous, self.scene = result, scene
        self.scene_gray = current
        return result

    def propagate(self, heat, current, previous):
        """Blend the previous heat map carried along the estimated motion field."""
        h, w = heat.shape
        back = optical_flow(current, previous)
        magnitude = np.hypot(back[..., 0], back[..., 1])
        if float(magnitude.max()) < .5:
            mask = np.ones_like(magnitude, np.float32)
        else:
            fore = optical_flow(previous, current)
            mask = consistent_mask(back, fore, self.tolerance)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        scale = np.float32([w / back.shape[1], h / back.shape[0]])
        back_full = cv2.resize(back, (w, h), interpolation=cv2.INTER_LINEAR) * scale
        x = gx + back_full[..., 0]
        y = gy + back_full[..., 1]
        inside = ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)).astype(np.float32)
        warped = cv2.remap(self.previous, x, y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        alpha = self.smoothing * (cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) * inside)
        return alpha * warped + (1 - alpha) * heat


def flow_grid(shape):
    """Quarter-resolution (width, height) used for motion estimation."""
    h, w = shape
    return max(w // 4, 32), max(h // 4, 32)


def optical_flow(source, target):
    """Farneback displacement field taking pixels of `source` to `target`."""
    return cv2.calcOpticalFlowFarneback(source, target, None, .5, 3, 21, 3, 5, 1.2, 0)


def consistent_mask(back, fore, tolerance=1.5):
    """Forward-backward consistency: reliable where the round trip is short."""
    h, w = back.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = np.clip((gx + back[..., 0]).round().astype(int), 0, w - 1)
    y = np.clip((gy + back[..., 1]).round().astype(int), 0, h - 1)
    trip = np.hypot(back[..., 0] + fore[y, x, 0], back[..., 1] + fore[y, x, 1])
    return (trip <= tolerance).astype(np.float32)


class DisplayNormalizer:
    def __init__(self, smoothing=.9):
        if not 0 <= smoothing < 1:
            raise ValueError('Contrast smoothing must be in [0, 1)')
        self.smoothing = smoothing
        self.bounds = None

    def update(self, heat, reset=False):
        if not heat.size or not np.isfinite(heat).all():
            raise ValueError('Display heat must be nonempty and finite')
        bounds = np.percentile(heat, [2, 99.5])
        if self.bounds is None or reset:
            self.bounds = bounds
        else:
            self.bounds = self.smoothing * self.bounds + (1 - self.smoothing) * bounds
        low, high = self.bounds
        if high - low < 1e-5:
            return np.clip(heat, 0, 1).copy()
        return np.clip((heat - low) / (high - low), 0, 1).astype(np.float32)


def probe(path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                                 '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        raise ValueError(UNREADABLE_MEDIA) from None
    info = json.loads(result.stdout)
    stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    if stream is None:
        raise ValueError(UNREADABLE_MEDIA)
    w, h = stream.get('width', 0) or 0, stream.get('height', 0) or 0
    if w <= 0 or h <= 0:
        raise ValueError(UNREADABLE_MEDIA)
    rotation = float(stream.get('tags', {}).get('rotate', 0))
    for side in stream.get('side_data_list', []):
        rotation = float(side.get('rotation', rotation))
    if round(rotation) % 180 == 90:
        w, h = h, w
    fps = stream.get('avg_frame_rate', '0/0')
    if fps in ('0/0', '0'):
        fps = stream.get('r_frame_rate', '30/1')
    if fps in ('0/0', '0'):
        fps = '30/1'
    rate = float(Fraction(fps))
    duration = float(info.get('format', {}).get('duration', 0))
    count = int(stream.get('nb_frames', 0) or 0) or round(duration * rate)
    return w, h, fps, count


def fps_mode_args():
    """Frame passthrough flag: ffmpeg 7 removed -vsync, 5.1+ takes -fps_mode."""
    if fps_mode_args.passthrough is None:
        probe = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                                'nullsrc=s=8x8:d=0.04', '-fps_mode', 'passthrough',
                                '-f', 'null', '-'], capture_output=True)
        fps_mode_args.passthrough = probe.returncode == 0
    return (['-fps_mode', 'passthrough'] if fps_mode_args.passthrough
            else ['-vsync', '0'])


fps_mode_args.passthrough = None


class Predictor:
    def __init__(self, weights=DEFAULT_WEIGHTS, size=384, threads=2, device='auto'):
        if size < 32:
            raise ValueError('Working short side must be at least 32')
        torch.set_num_threads(threads)
        cv2.setNumThreads(1)
        self.size = size
        self.device = select_device(device)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        self.model = ThermalUNet(pretrained=False,
                                 conditioning=any(key.startswith('style.') for key in state)).eval()
        self.model.load_state_dict(state, strict=True)
        self.model.to(device=self.device, dtype=torch.float32, memory_format=memory_format(self.device))

    @torch.inference_mode()
    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        size = (round(w * self.size / min(w, h)), round(h * self.size / min(w, h)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        x = x.to(self.device, memory_format=memory_format(self.device))
        return self.model(x)[0, 0].cpu().numpy()


def encoder(path, w, h, fps):
    # yuv420p requires even dimensions; pad at most one pixel without stretching.
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-threads', '1', '-filter_threads', '1', '-f', 'rawvideo',
                             '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', fps, '-i', 'pipe:0',
                             '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
                             '-threads', '1', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                             str(path)], stdin=subprocess.PIPE)


def mux_audio(video, original, output):
    command = ['ffmpeg', '-v', 'error', '-y', '-threads', '1', '-i', str(video), '-i', str(original),
               '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'copy', '-c:a']
    result = subprocess.run(command + ['copy', '-movflags', '+faststart', str(output)], capture_output=True)
    if result.returncode:
        subprocess.run(command + ['aac', '-b:a', '192k', '-threads', '1', '-movflags', '+faststart',
                                  str(output)], check=True)


def convert(input_path, output_path, palette='ironbow', size=384, side_by_side=False,
            weights=DEFAULT_WEIGHTS, smoothing=.65, raw=False, threads=2,
            semantic=True, semantic_strength=.7, device='auto', auto_contrast=True,
            flow=True, sensor=True):
    start = time.perf_counter()
    source, output = Path(input_path), Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError('Input and output must be different files')
    if not source.is_file():
        raise FileNotFoundError(source)
    if palette not in PALETTES:
        raise ValueError(f'Unknown palette: {palette}')
    if not np.isfinite(semantic_strength) or not 0 <= semantic_strength <= 1:
        raise ValueError('Semantic strength must be between 0 and 1')
    smoother = TemporalSmoother(smoothing, flow=flow)
    selected_device = select_device(device)
    sensor_sim = ThermalSensor() if sensor else None
    predictor = Predictor(weights, size, threads, selected_device.type)
    semantic_predictor = SemanticPrior(device=selected_device.type) if semantic and semantic_strength > 0 else None
    normalizer = DisplayNormalizer() if auto_contrast else None

    def predict_heat(rgb):
        heat = predictor(rgb)
        if semantic_predictor is not None:
            prior, confidence = semantic_predictor(rgb, heat.shape)
            heat = fuse_heat(heat, prior, confidence, semantic_strength)
        return heat

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_name(output.stem + '_raw' + output.suffix) if raw else None
    try:
        with Image.open(source) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    except (OSError, ValueError):
        rgb = None
    if rgb is not None:
        if output.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
            raise ValueError('For an image, use a .png/.jpg/.webp output')
        raw_path = output.with_name(output.stem + '_raw.png') if raw else None
        if raw_path is not None and source.resolve() == raw_path.resolve():
            raise ValueError('Input and raw output must be different files')
        heat = guided_upsample(predict_heat(rgb), rgb)
        if sensor_sim is not None:
            heat = sensor_sim.update(heat)
        colored = apply_palette(normalizer.update(heat) if normalizer else heat, palette)
        Image.fromarray(np.concatenate([rgb, colored], axis=1) if side_by_side else colored).save(output)
        if raw:
            Image.fromarray(np.rint(heat * 255).astype(np.uint8)).save(raw_path)
        frames = 1
    else:
        w, h, fps, total = probe(source)
        if output.suffix.lower() != '.mp4':
            raise ValueError('Video output must be .mp4')
        if raw_path is not None and source.resolve() == raw_path.resolve():
            raise ValueError('Input and raw output must be different files')
        frames = 0
        with tempfile.TemporaryDirectory(prefix='fake-thermal-', dir=output.parent) as temporary:
            temp = Path(temporary)
            with (temp / 'decode.log').open('wb') as errors:
                decoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '1', '-filter_threads', '1', '-i', str(source),
                                             '-map', '0:v:0', '-an', *fps_mode_args(),
                                             '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                                            stdout=subprocess.PIPE, stderr=errors)
                enc = encoder(temp / 'color.mp4', w * (2 if side_by_side else 1), h, fps)
                raw_enc = encoder(temp / 'raw.mp4', w, h, fps) if raw else None
                try:
                    with tqdm(total=total or None, unit='frame', desc='Thermalizing') as bar:
                        while True:
                            buffer = decoder.stdout.read(w * h * 3)
                            if not buffer:
                                break
                            if len(buffer) != w * h * 3:
                                raise ValueError(UNREADABLE_MEDIA)
                            rgb = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 3)
                            heat = smoother.update(predict_heat(rgb), rgb)
                            heat = guided_upsample(heat, rgb)
                            if sensor_sim is not None:
                                heat = sensor_sim.update(heat)
                            display = normalizer.update(heat, reset=smoother.reset_history) if normalizer else heat
                            colored = apply_palette(display, palette)
                            enc.stdin.write((np.concatenate([rgb, colored], axis=1) if side_by_side else colored).tobytes())
                            if raw_enc:
                                gray = np.rint(heat * 255).astype(np.uint8)
                                raw_enc.stdin.write(np.repeat(gray[..., None], 3, axis=2).tobytes())
                            frames += 1
                            bar.update()
                    if decoder.wait() != 0:
                        raise ValueError(UNREADABLE_MEDIA)
                    if frames == 0:
                        raise ValueError(UNREADABLE_MEDIA)
                    enc.stdin.close()
                    if enc.wait() != 0:
                        raise RuntimeError('ffmpeg encoding failed')
                    if raw_enc:
                        raw_enc.stdin.close()
                        if raw_enc.wait() != 0:
                            raise RuntimeError('Raw video encoding failed')
                    mux_audio(temp / 'color.mp4', source, output)
                    if raw:
                        mux_audio(temp / 'raw.mp4', source, raw_path)
                finally:
                    for process in (decoder, enc, raw_enc):
                        if process is not None:
                            if process.poll() is None:
                                process.terminate()
                            try:
                                process.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
    elapsed = time.perf_counter() - start
    stats = {'frames': frames, 'seconds': elapsed, 'fps': frames / elapsed,
             'output': str(output), 'raw': str(raw_path) if raw_path else None,
             'device': selected_device.type}
    print(json.dumps(stats), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description='Predict a thermal look, not real temperatures.')
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--palette', choices=PALETTES, default='ironbow')
    parser.add_argument('--size', type=int, default=384, help='working short side (default 384)')
    parser.add_argument('--side-by-side', action='store_true')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--smoothing', type=float, default=.65, help='previous-frame weight, [0,1); 0 disables')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--semantic', action=argparse.BooleanOptionalAction, default=True,
                        help='semantic/emissive heat prior (default on; downloads weights on first use)')
    parser.add_argument('--semantic-strength', type=float, default=.7, help='prior strength, 0 to 1 (default .7)')
    parser.add_argument('--device', choices=DEVICES, default='auto', help='auto selects CUDA, then MPS, then CPU')
    parser.add_argument('--auto-contrast', action=argparse.BooleanOptionalAction, default=True,
                        help='temporally stable percentile display contrast (default on; raw is unaffected)')
    parser.add_argument('--flow-smoothing', action=argparse.BooleanOptionalAction, default=True,
                        help='carry the previous heat map along estimated motion with occlusion masking (default on)')
    parser.add_argument('--sensor-sim', action=argparse.BooleanOptionalAction, default=True,
                        help='add measured microbolometer noise, banding and vignette (default on)')
    args = parser.parse_args()
    try:
        convert(args.input, args.output, args.palette, args.size, args.side_by_side,
                args.weights, args.smoothing, args.raw, args.threads,
                args.semantic, args.semantic_strength, args.device, args.auto_contrast,
                args.flow_smoothing, args.sensor_sim)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'fake-thermal: {error}\n')


if __name__ == '__main__':
    main()
