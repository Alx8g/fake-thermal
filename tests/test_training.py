from collections import Counter
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from fake_thermal.data import normalize_thermal, PairedDataset
from fake_thermal.losses import thermal_loss, ssim
from fake_thermal.model import ThermalUNet

WEIGHTS = Path(__file__).resolve().parents[1] / 'fake_thermal/weights/fake_thermal.pt'
STYLE_KEYS = {'style.head.0.weight', 'style.head.0.bias', 'style.head.2.weight', 'style.head.2.bias'}


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

    def test_focal_frequency_loss_identity_finite_and_gradients(self):
        from fake_thermal.losses import focal_frequency_loss
        target = torch.rand(2, 1, 32, 32)
        self.assertAlmostEqual(focal_frequency_loss(target, target).item(), 0., places=6)
        self.assertAlmostEqual(thermal_loss(target, target, frequency_alpha=1.).item(), 0., places=5)
        pred = torch.full_like(target, .5, requires_grad=True)
        value = focal_frequency_loss(pred, target, alpha=1.)
        self.assertTrue(torch.isfinite(value) and value.item() > 0)
        value.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_focal_frequency_loss_is_bounded_across_crop_sizes(self):
        from fake_thermal.losses import focal_frequency_loss
        values = {}
        for size in (32, 128, 256):
            pred = torch.full((1, 1, size, size), .5)
            target = torch.full((1, 1, size, size), .6)
            value = focal_frequency_loss(pred, target)
            values[size] = value.item()
            self.assertTrue(torch.isfinite(value))
            self.assertLess(value.item(), 1.)
        self.assertLess(abs(values[256] - values[32]) / values[32], .5)
        harder = focal_frequency_loss(torch.full((1, 1, 64, 64), .5),
                                      torch.full((1, 1, 64, 64), .6), alpha=2.)
        self.assertTrue(torch.isfinite(harder) and harder.item() < 1.)

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

    def test_unconditioned_weights_strict_load(self):
        model = ThermalUNet(pretrained=False)
        state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)

    def test_conditioning_adds_only_style_keys(self):
        plain = ThermalUNet(pretrained=False)
        cond = ThermalUNet(pretrained=False, conditioning=True)
        plain_keys, cond_keys = set(plain.state_dict()), set(cond.state_dict())
        self.assertEqual(cond_keys - plain_keys, STYLE_KEYS)
        self.assertFalse(plain_keys - cond_keys)
        self.assertEqual(tuple(cond.state_dict()['style.head.2.weight'].shape), (2048, 32))
        self.assertEqual(sum(p.numel() for p in cond.style.parameters()), 67744)

    def test_conditioned_forward_shape_and_backward(self):
        model = ThermalUNet(pretrained=False, conditioning=True)
        rgb = torch.rand(2, 3, 64, 96)
        pred = model(rgb)
        self.assertEqual(pred.shape, (2, 1, 64, 96))
        self.assertTrue(torch.isfinite(pred).all())
        loss = thermal_loss(pred, torch.rand(2, 1, 64, 96))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.style.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_film_init_is_identity(self):
        plain = ThermalUNet(pretrained=False).eval()
        cond = ThermalUNet(pretrained=False, conditioning=True).eval()
        result = cond.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(set(result.missing_keys), STYLE_KEYS)
        rgb = torch.rand(2, 3, 96, 128)
        with torch.no_grad():
            for h, w in ((257, 319), (33, 65), (256, 256)):
                y = cond(torch.rand(1, 3, h, w))
                self.assertEqual(y.shape, (1, 1, h, w))
            self.assertTrue(torch.allclose(plain(rgb), cond(rgb), atol=1e-6))
            cond.style.head[-1].bias.add_(1.)
            self.assertGreater((plain(rgb) - cond(rgb)).abs().max().item(), 1e-4)

    def test_luminance_stats_match_numpy(self):
        cond = ThermalUNet(pretrained=False, conditioning=True)
        rgb = torch.rand(3, 3, 17, 29)
        gray = rgb.mean(dim=1).reshape(3, -1).numpy()
        expected = np.stack([gray.mean(1), gray.std(1, ddof=1),
                             np.percentile(gray, 5, axis=1),
                             np.percentile(gray, 95, axis=1)], axis=1)
        actual = cond.style.stats(rgb).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
