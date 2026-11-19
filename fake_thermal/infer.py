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

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / 'weights/fake_thermal.pt'
PALETTES = {
    'white-hot': [(0, 0, 0), (255, 255, 255)],
    'black-hot': [(255, 255, 255), (0, 0, 0)],
    'ironbow': [(0, 0, 0), (35, 0, 80), (135, 12, 90), (230, 60, 20), (255, 190, 30), (255, 255, 235)],
    'rainbow': [(0, 0, 90), (0, 50, 255), (0, 240, 220), (100, 255, 0), (255, 220, 0), (255, 20, 0)],
    'arctic': [(0, 0, 25), (20, 30, 120), (0, 150, 220), (110, 235, 255), (255, 255, 255)],
}


def apply_palette(heat, palette):
    anchors = np.asarray(PALETTES[palette], dtype=np.float32)
    x = np.clip(heat, 0, 1) * (len(anchors) - 1)
    lo = np.minimum(x.astype(np.int32), len(anchors) - 2)
    frac = (x - lo)[..., None]
    return np.rint(anchors[lo] * (1 - frac) + anchors[lo + 1] * frac).astype(np.uint8)


class TemporalSmoother:
    def __init__(self, smoothing=.65, cut_threshold=.22):
        if not 0 <= smoothing < 1:
            raise ValueError('Smoothing must be in [0, 1)')
        self.smoothing, self.threshold = smoothing, cut_threshold
        self.previous, self.scene = None, None

    def update(self, heat, rgb):
        scene = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        cut = self.scene is not None and np.abs(scene - self.scene).mean() > self.threshold
        if self.previous is None or cut or self.previous.shape != heat.shape:
            result = heat.copy()
        else:
            result = self.smoothing * self.previous + (1 - self.smoothing) * heat
        self.previous, self.scene = result, scene
        return result


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                             '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    if stream is None:
        raise ValueError('Input has no video stream')
    w, h = stream['width'], stream['height']
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


class Predictor:
    def __init__(self, weights=DEFAULT_WEIGHTS, size=384, threads=2):
        if size < 32:
            raise ValueError('Working short side must be at least 32')
        torch.set_num_threads(threads)
        cv2.setNumThreads(1)
        self.size = size
        self.model = ThermalUNet(pretrained=False).eval().to(memory_format=torch.channels_last)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)

    @torch.inference_mode()
    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        size = (round(w * self.size / min(w, h)), round(h * self.size / min(w, h)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        return self.model(x.contiguous(memory_format=torch.channels_last))[0, 0].numpy()


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
            weights=DEFAULT_WEIGHTS, smoothing=.65, raw=False, threads=2):
    start = time.perf_counter()
    source, output = Path(input_path), Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError('Input and output must be different files')
    if not source.is_file():
        raise FileNotFoundError(source)
    if palette not in PALETTES:
        raise ValueError(f'Unknown palette: {palette}')
    smoother = TemporalSmoother(smoothing)
    predictor = Predictor(weights, size, threads)
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
        heat = cv2.resize(predictor(rgb), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        colored = apply_palette(heat, palette)
        Image.fromarray(np.concatenate([rgb, colored], axis=1) if side_by_side else colored).save(output)
        if raw:
            Image.fromarray(np.rint(heat * 255).astype(np.uint8)).save(raw_path)
        frames = 1
    else:
        if output.suffix.lower() != '.mp4':
            raise ValueError('Video output must be .mp4')
        if raw_path is not None and source.resolve() == raw_path.resolve():
            raise ValueError('Input and raw output must be different files')
        w, h, fps, total = probe(source)
        frames = 0
        with tempfile.TemporaryDirectory(prefix='fake-thermal-', dir=output.parent) as temporary:
            temp = Path(temporary)
            with (temp / 'decode.log').open('wb') as errors:
                decoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '1', '-filter_threads', '1', '-i', str(source),
                                             '-map', '0:v:0', '-an', '-vsync', '0',
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
                                raise RuntimeError('Truncated decoded frame')
                            rgb = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 3)
                            heat = smoother.update(predictor(rgb), rgb)
                            heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
                            colored = apply_palette(heat, palette)
                            enc.stdin.write((np.concatenate([rgb, colored], axis=1) if side_by_side else colored).tobytes())
                            if raw_enc:
                                gray = np.rint(heat * 255).astype(np.uint8)
                                raw_enc.stdin.write(np.repeat(gray[..., None], 3, axis=2).tobytes())
                            frames += 1
                            bar.update()
                    if decoder.wait() != 0:
                        raise RuntimeError('ffmpeg decode failed: ' + (temp / 'decode.log').read_text()[-2000:])
                    enc.stdin.close()
                    if enc.wait() != 0:
                        raise RuntimeError('ffmpeg encoding failed')
                    if raw_enc:
                        raw_enc.stdin.close()
                        if raw_enc.wait() != 0:
                            raise RuntimeError('Raw video encoding failed')
                    if frames == 0:
                        raise ValueError('No frames decoded')
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
             'output': str(output), 'raw': str(raw_path) if raw_path else None}
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
    args = parser.parse_args()
    try:
        convert(args.input, args.output, args.palette, args.size, args.side_by_side,
                args.weights, args.smoothing, args.raw, args.threads)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'fake-thermal: {error}\n')


if __name__ == '__main__':
    main()
