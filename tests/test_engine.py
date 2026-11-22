from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from fake_thermal.infer import convert, preview_frames, probe, remove_stripes, resize_upsample


class EngineTests(unittest.TestCase):
    def test_preview_interpolates_in_order_and_predicts_the_tail(self):
        frames = [np.zeros((128, 128, 3), np.uint8) for _ in range(6)]
        indexes = {id(frame): index for index, frame in enumerate(frames)}
        calls = []

        def predict(rgb):
            index = indexes[id(rgb)]
            calls.append(index)
            return np.full(rgb.shape[:2], index / 10, np.float32)

        result = list(preview_frames(iter(frames), predict, stride=4, cut_threshold=.22))
        self.assertEqual(calls, [0, 4, 5])
        self.assertEqual([indexes[id(rgb)] for rgb, _ in result], list(range(6)))
        for index, (_, heat) in enumerate(result):
            np.testing.assert_allclose(heat, index / 10, atol=1e-6)

    def test_preview_refreshes_local_changes_and_does_not_blend_across_cuts(self):
        black = np.zeros((128, 128, 3), np.uint8)
        subject = black.copy()
        subject[48:80, 48:80] = 160
        bright = np.full_like(black, 255)
        frames = [black, subject, black, bright, bright, bright]
        calls = []

        def predict(rgb):
            calls.append(rgb)
            return rgb[..., 0].astype(np.float32) / 255

        result = list(preview_frames(iter(frames), predict, stride=4, cut_threshold=.22))
        self.assertEqual([id(rgb) for rgb in calls], list(map(id, [black, black, subject, bright, bright])))
        for rgb, heat in result:
            np.testing.assert_allclose(heat, rgb[..., 0] / 255, atol=1e-6)

    def test_preview_uses_the_matching_anchor_for_a_new_subject(self):
        black = np.zeros((128, 128, 3), np.uint8)
        subject = black.copy()
        subject[40:88, 40:88] = 160
        calls = []

        def predict(rgb):
            calls.append(rgb)
            return rgb[..., 0].astype(np.float32) / 255

        result = list(preview_frames(iter([black, subject, subject]), predict, stride=4, cut_threshold=.22))
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(result[1][1][64, 64], 160 / 255, places=6)

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
                            '-r', '24', '-i', 'pipe:0', '-f', 'lavfi', '-i', 'sine=frequency=440',
                            '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                            '-shortest', str(source)],
                           input=np.stack([dark, subject, subject, bright, bright, bright, bright]).tobytes(),
                           check=True)
            predictions = []

            def predict(rgb):
                predictions.append((rgb[0, 0, 0] > 128, rgb[64, 64, 0] > 128))
                return rgb[..., 0].astype(np.float32) / 255

            with patch('fake_thermal.infer.TurboPredictor', return_value=predict) as turbo:
                stats = convert(source, Path(directory) / 'out.mp4', engine='turbo', semantic=False,
                                sensor=False, device='cpu', stride=3)
                self.assertEqual(predictions, [(False, False), (False, True), (True, True), (True, True)])
                self.assertEqual(stats['model_frames'], 4)
                self.assertEqual(stats['frames'], 7)

                output = Path(directory) / 'preview.mp4'
                stats = convert(source, output, engine='turbo', semantic=False, sensor=False,
                                device='cpu', turbo_preview=True)
                self.assertEqual(turbo.call_args.args[1], 256)
                self.assertEqual(stats['model_frames'], 4)
                self.assertEqual(stats['frames'], 7)
                self.assertEqual(probe(source), probe(output))

                def audio_packets(path):
                    return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map',
                                                    '0:a:0', '-c:a', 'copy', '-f', 'hash', '-hash', 'sha256', '-'])

                self.assertEqual(audio_packets(source), audio_packets(output))

    def test_preview_requires_turbo_and_bounds_lookahead(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'in.png'
            Image.new('RGB', (128, 128)).save(source)
            with self.assertRaisesRegex(ValueError, 'requires the turbo engine'):
                convert(source, Path(directory) / 'out.png', turbo_preview=True)
            with self.assertRaisesRegex(ValueError, 'stride up to'):
                convert(source, Path(directory) / 'out.png', engine='turbo', turbo_preview=True, stride=99)


if __name__ == '__main__':
    unittest.main()
