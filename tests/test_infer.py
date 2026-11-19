import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from fake_thermal.infer import apply_palette, PALETTES, TemporalSmoother, probe, Predictor
from fake_thermal.model import ThermalUNet


class PaletteTests(unittest.TestCase):
    def test_bundled_default_weights(self):
        heat = Predictor(size=64, threads=2)(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(heat.shape, (64, 64))
        self.assertTrue(np.isfinite(heat).all())
        self.assertTrue(((heat >= 0) & (heat <= 1)).all())

    def test_endpoints_and_clamping(self):
        for name, anchors in PALETTES.items():
            result = apply_palette(np.array([[-1., 0., 1., 2.]]), name)
            self.assertEqual(result.shape, (1, 4, 3))
            self.assertEqual(result.dtype, np.uint8)
            np.testing.assert_array_equal(result[0, :2], [anchors[0], anchors[0]])
            np.testing.assert_array_equal(result[0, 2:], [anchors[-1], anchors[-1]])

    def test_smoothing_and_cut_reset(self):
        smooth = TemporalSmoother(.75)
        rgb = np.zeros((16, 16, 3), np.uint8)
        np.testing.assert_array_equal(smooth.update(np.zeros((4, 4)), rgb), 0)
        np.testing.assert_array_equal(smooth.update(np.ones((4, 4)), rgb), .25)
        np.testing.assert_array_equal(smooth.update(np.ones((4, 4)), rgb + 255), 1)
        disabled = TemporalSmoother(0)
        disabled.update(np.zeros((4, 4)), rgb)
        np.testing.assert_array_equal(disabled.update(np.ones((4, 4)), rgb), 1)
        for value in (-.1, 1.):
            with self.assertRaises(ValueError):
                TemporalSmoother(value)


def streams(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                               '-of', 'json', str(path)]))['streams']


class CLIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.weights = cls.root / 'weights.pt'
        torch.save(ThermalUNet(pretrained=False).state_dict(), cls.weights)
        cls.source = cls.root / 'input.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '1', '-f', 'lavfi', '-i',
                        'testsrc2=size=96x64:rate=6:duration=1', '-f', 'lavfi', '-i',
                        'sine=frequency=440:duration=1', '-c:v', 'libx264', '-threads', '1',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(cls.source)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def cli(self, source, output, *options):
        return subprocess.run([str(Path(sys.executable).with_name('fake-thermal')), str(source), '-o', str(output), '--size', '64',
                               '--weights', str(self.weights), '--no-semantic', *options], capture_output=True, text=True)

    def test_video_audio_frames_fps_and_raw(self):
        output = self.root / 'converted.mp4'
        result = self.cli(self.source, output, '--side-by-side', '--raw')
        self.assertEqual(result.returncode, 0, result.stderr)
        info = streams(output)
        video = next(s for s in info if s['codec_type'] == 'video')
        self.assertEqual((video['width'], video['height']), (192, 64))
        self.assertEqual((video['nb_frames'], video['avg_frame_rate']), ('6', '6/1'))
        self.assertEqual(next(s for s in info if s['codec_type'] == 'audio')['codec_name'], 'aac')
        def audio_hash(path):
            return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map',
                                             '0:a:0', '-c:a', 'copy', '-f', 'hash', '-hash', 'sha256', '-'])
        self.assertEqual(audio_hash(self.source), audio_hash(output))
        self.assertEqual(probe(self.root / 'converted_raw.mp4')[:3], (96, 64, '6/1'))

    def test_rotation(self):
        rotated = self.root / 'rotated.mov'
        help_text = subprocess.check_output(['ffmpeg', '-h', 'full'], stderr=subprocess.DEVNULL,
                                            text=True)
        if '-display_rotation' in help_text:
            args = ['-display_rotation:v:0', '90', '-i', str(self.source), '-c', 'copy']
        else:
            args = ['-i', str(self.source), '-c', 'copy', '-metadata:s:v:0', 'rotate=90']
        subprocess.run(['ffmpeg', '-v', 'error', '-y', *args, str(rotated)], check=True)
        self.assertEqual(probe(rotated)[:2], (64, 96), 'Fixture must contain rotation metadata')
        output = self.root / 'upright.mp4'
        result = self.cli(rotated, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(probe(output)[:2], (64, 96))

    def test_image_and_error(self):
        source = self.root / 'image.png'
        Image.new('RGB', (75, 53), 'orange').save(source)
        output = self.root / 'result.png'
        result = self.cli(source, output, '--palette', 'arctic', '--raw')
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(output) as im:
            self.assertEqual(im.size, (75, 53))
        with Image.open(self.root / 'result_raw.png') as im:
            self.assertEqual(im.mode, 'L')
        self.assertNotEqual(self.cli(source, source).returncode, 0)
        self.assertNotEqual(self.cli(self.root / 'missing.mov', output).returncode, 0)

    def test_raw_image_cannot_overwrite_input(self):
        source = self.root / 'holiday_raw.png'
        Image.new('RGB', (75, 53), 'orange').save(source)
        original = source.read_bytes()
        for suffix in ('.png', '.jpg'):
            with self.subTest(suffix=suffix):
                output = self.root / ('holiday' + suffix)
                result = self.cli(source, output, '--raw')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Input and raw output must be different files', result.stderr)
                self.assertEqual(source.read_bytes(), original)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
