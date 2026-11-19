from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageFilter

from rescue_video_pairs import (apply_warp, register_thermal, rescale_warp,
                                save_pair, stretch)


def texture(shape=(128, 160), seed=7):
    """Smooth deterministic pattern with gradients across the whole frame."""
    rng = np.random.default_rng(seed)
    noise = (rng.random(shape) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(noise, 'L').filter(ImageFilter.GaussianBlur(3)), np.uint8)


def interior_difference(a, b, border=10):
    h, w = a.shape
    valid = np.zeros(a.shape, bool)
    valid[border:h - border, border:w - border] = True
    return float(np.abs(stretch(a, valid) - stretch(b, valid))[valid].mean())


class RegistrationTests(unittest.TestCase):
    def test_identity_alignment_near_zero_residual(self):
        pseudo = texture()
        warp, residual, reason = register_thermal(pseudo, pseudo.copy())
        self.assertEqual(reason, 'euclidean')
        self.assertLess(residual, .01)
        np.testing.assert_allclose(warp[:, :2], np.eye(2), atol=.01)
        np.testing.assert_allclose(warp[:, 2], (0., 0.), atol=.5)

    def test_apply_warp_identity_is_identity(self):
        pseudo = texture()
        warped = apply_warp(pseudo, np.float32(np.eye(2, 3)), pseudo.shape[::-1])
        np.testing.assert_allclose(warped, pseudo, atol=1)

    def test_known_shift_recovered(self):
        base = texture((128, 160))
        padded = np.pad(base, 24, mode='reflect')
        sy, sx = 7, -5
        real = padded[24 - sy:24 - sy + 128, 24 - sx:24 - sx + 160]
        warp, residual, reason = register_thermal(base, real)
        self.assertEqual(reason, 'euclidean')
        self.assertLess(residual, .05)
        np.testing.assert_allclose(warp[:, :2], np.eye(2), atol=.01)
        np.testing.assert_allclose(warp[:, 2], (sx, sy), atol=1.)
        self.assertLess(interior_difference(base, apply_warp(real, warp, base.shape[::-1])), .02)

    def test_known_rotation_recovered(self):
        base = texture((128, 160))
        padded = np.pad(base, 32, mode='reflect')
        rotated = np.asarray(Image.fromarray(padded, 'L').rotate(3, resample=Image.BILINEAR))
        real = rotated[32:160, 32:192]
        warp, residual, _ = register_thermal(base, real)
        self.assertLess(residual, .08)
        angle = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
        self.assertLess(abs(angle), 4.)
        self.assertAlmostEqual(abs(angle), 3., delta=.5)
        self.assertAlmostEqual(float(np.linalg.det(warp[:, :2])), 1., delta=.03)
        self.assertLess(interior_difference(base, apply_warp(real, warp, base.shape[::-1])), .05)

    def test_blank_image_rejected_not_garbage(self):
        pseudo = texture()
        blank = np.full(pseudo.shape, 128, np.uint8)
        for a, b in ((pseudo, blank), (blank, pseudo), (blank, blank)):
            warp, residual, reason = register_thermal(a, b)
            self.assertIsNone(warp)
            self.assertGreaterEqual(residual, 1.)
            self.assertEqual(reason, 'textureless')

    def test_rescale_warp_scales_transform_between_grids(self):
        warp = np.float32([[1., 0., 4.], [0., 1., -2.]])
        scaled = rescale_warp(warp, (.5, .25), (3., 4.))
        np.testing.assert_allclose(scaled, np.float32([[1.5, 0., 12.], [0., 1., -8.]]), atol=1e-6)
        shear = np.float32([[2., 1., 4.], [0., 1., -2.]])
        sheared = rescale_warp(shear, (.5, .25), (3., 4.))
        np.testing.assert_allclose(sheared[:, :2], np.float32([[3., .75], [0., 1.]]), atol=1e-6)
        np.testing.assert_allclose(sheared[:, 2], (12., -8.), atol=1e-6)

    def test_save_pair_modes_and_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rgb = Image.new('RGB', (640, 512), (10, 20, 30))
            thermal = Image.new('L', (640, 512), 5)
            save_pair(out, 'video-X_000001', rgb, thermal)
            with Image.open(out / 'rgb' / 'video-X_000001.jpg') as saved_rgb, \
                    Image.open(out / 'thermal' / 'video-X_000001.png') as saved_thermal:
                self.assertEqual((saved_rgb.size, saved_rgb.mode), ((360, 288), 'RGB'))
                self.assertEqual((saved_thermal.size, saved_thermal.mode), ((360, 288), 'L'))
        with self.assertRaises(ValueError):
            save_pair(Path('/nonexistent-safe'), 'x', Image.new('RGB', (100, 100)),
                      Image.new('L', (200, 100)))


if __name__ == '__main__':
    unittest.main()
