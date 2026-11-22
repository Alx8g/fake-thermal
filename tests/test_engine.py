from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from fake_thermal.infer import convert, remove_stripes, resize_upsample


class EngineTests(unittest.TestCase):
    def test_resize_upsample_matches_frame_and_stays_in_range(self):
        heat = np.random.default_rng(0).random((16, 24)).astype(np.float32)
        rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        out = resize_upsample(heat, rgb)
        self.assertEqual(out.shape, (32, 48))
        self.assertGreaterEqual(out.min(), 0)
        self.assertLessEqual(out.max(), 1)
        same = resize_upsample(heat, np.zeros((16, 24, 3), dtype=np.uint8))
        self.assertTrue(np.array_equal(same, heat))

    def test_remove_stripes_drops_period_two_and_keeps_other_texture(self):
        y, x = np.mgrid[0:64, 0:96].astype(np.float32)
        smooth = .3 + .4 * x / 95
        texture = .05 * np.sin(2 * np.pi * (x + y) / 4)
        stripes = .03 * np.where(x % 2, -1, 1) + .02 * np.where(y % 2, -1, 1)
        out = remove_stripes(smooth + texture + stripes)
        self.assertEqual(out.dtype, np.float32)
        inner = (slice(8, -8), slice(8, -8))
        self.assertLess(np.abs(out - smooth - texture)[inner].max(), .004)
        self.assertLess(np.abs(remove_stripes(smooth + texture) - smooth - texture)[inner].max(), .004)

    def test_turbo_engine_runs_without_guided_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'in.png'
            Image.fromarray(np.full((40, 60, 3), 128, dtype=np.uint8)).save(source)
            heat = np.linspace(0, 1, 20 * 30, dtype=np.float32).reshape(20, 30)
            with patch('fake_thermal.infer.TurboPredictor') as turbo, \
                    patch('fake_thermal.infer.guided_upsample') as guided:
                turbo.return_value = lambda rgb: heat
                stats = convert(source, Path(directory) / 'out.png', engine='turbo',
                                semantic=False, sensor=False, device='cpu')
            turbo.assert_called_once()
            guided.assert_not_called()
            self.assertEqual(stats['frames'], 1)
            with Image.open(Path(directory) / 'out.png') as image:
                self.assertEqual(image.size, (60, 40))

    def test_unknown_engine_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'in.png'
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(source)
            with self.assertRaises(ValueError):
                convert(source, Path(directory) / 'out.png', engine='nope')

    def test_stride_refreshes_on_local_change_cut_and_max_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'in.mp4'
            dark = np.zeros((128, 128, 3), np.uint8)
            subject = dark.copy()
            subject[48:80, 48:80] = 255
            bright = np.full_like(dark, 255)
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '128x128',
                            '-r', '24', '-i', 'pipe:0', '-c:v', 'libx264', '-threads', '1', '-pix_fmt',
                            'yuv420p', str(source)],
                           input=np.stack([dark, subject, subject, bright, bright, bright, bright]).tobytes(),
                           check=True)
            predictions = []

            def predict(rgb):
                predictions.append((rgb[0, 0, 0] > 128, rgb[64, 64, 0] > 128))
                return rgb[..., 0].astype(np.float32) / 255

            with patch('fake_thermal.infer.TurboPredictor', return_value=predict):
                stats = convert(source, Path(directory) / 'out.mp4', engine='turbo', semantic=False,
                                sensor=False, device='cpu', stride=3)
            self.assertEqual(predictions, [(False, False), (False, True), (True, True), (True, True)])
            self.assertEqual(stats['model_frames'], 4)
            self.assertEqual(stats['frames'], 7)


if __name__ == '__main__':
    unittest.main()
