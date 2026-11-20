from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from fake_thermal.infer import convert, resize_upsample


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


if __name__ == '__main__':
    unittest.main()
