from collections import Counter
import unittest

import numpy as np
from PIL import Image
import torch

from fake_thermal.data import normalize_thermal, PairedDataset
from fake_thermal.losses import thermal_loss, ssim
from fake_thermal.model import ThermalUNet


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_model_odd_and_small_sizes(self):
        model = ThermalUNet(pretrained=False).eval()
        with torch.no_grad():
            for h, w in ((257, 319), (33, 65), (1, 7), (256, 256)):
                y = model(torch.rand(1, 3, h, w))
                self.assertEqual(y.shape, (1, 1, h, w))
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(((y >= 0) & (y <= 1)).all())

    def test_loss_identity_and_backward(self):
        target = torch.rand(2, 1, 32, 32)
        self.assertAlmostEqual(thermal_loss(target, target).item(), 0., places=5)
        self.assertAlmostEqual(ssim(target, target).item(), 1., places=5)
        pred = torch.full_like(target, .5, requires_grad=True)
        loss = thermal_loss(pred, target)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_constant_loss_is_finite(self):
        a = torch.zeros(1, 1, 16, 16)
        self.assertTrue(torch.isfinite(thermal_loss(a, a)))
        self.assertGreater(thermal_loss(a, torch.ones_like(a)).item(), 1.)

    def test_percentile_stretch_and_constant(self):
        self.assertEqual(normalize_thermal(Image.new('L', (16, 16), 42)).sum(), 0)
        a = normalize_thermal(Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)))
        self.assertEqual((a.min(), a.max()), (0, 1))
        self.assertAlmostEqual(float(a[8, 0]), .502, places=2)

    def test_balanced_sampler(self):
        dataset = object.__new__(PairedDataset)
        dataset.sources = ['llvip'] * 100 + ['kaist'] * 10 + ['flir'] * 30
        counts = Counter(dataset.sources[i] for i in dataset.sampler(6000))
        self.assertTrue(all(1800 < n < 2200 for n in counts.values()), counts)


if __name__ == '__main__':
    unittest.main()
