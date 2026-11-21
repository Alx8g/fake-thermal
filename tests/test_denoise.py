import unittest

import numpy as np

from fake_thermal.denoise import denoise_photo, noise_sigma


class DenoiseTests(unittest.TestCase):
    def test_noise_estimate_tracks_added_noise(self):
        rng = np.random.default_rng(0)
        flat = np.full((96, 128, 3), 60, np.float32)
        estimates = []
        for sigma in (3, 6, 12):
            noisy = np.clip(flat + rng.normal(0, sigma, flat.shape), 0, 255).astype(np.uint8)
            estimates.append(noise_sigma(noisy))
            # measured on grey, which averages independent channel noise down to about 2/3
            self.assertGreater(estimates[-1], .5 * sigma)
            self.assertLess(estimates[-1], sigma)
        self.assertAlmostEqual(estimates[2] / estimates[1], 2, delta=.3)
        self.assertLess(noise_sigma(flat.astype(np.uint8)), .1)

    def test_letterbox_border_does_not_dilute_the_estimate(self):
        rng = np.random.default_rng(1)
        photo = np.zeros((128, 128, 3), np.uint8)
        photo[20:108, 20:108] = np.clip(80 + rng.normal(0, 8, (88, 88, 3)), 0, 255)
        self.assertGreater(noise_sigma(photo), 4)  # about 2/3 of 8 on grey; borders included would give ~2.5

    def test_clean_photo_is_untouched_and_noisy_one_is_smoothed(self):
        rng = np.random.default_rng(2)
        y = np.linspace(20, 120, 96, dtype=np.float32)[:, None, None]
        clean = np.broadcast_to(y, (96, 128, 3)).astype(np.uint8)
        self.assertIs(denoise_photo(clean), clean)
        noisy = np.clip(clean + rng.normal(0, 8, clean.shape), 0, 255).astype(np.uint8)
        rng2 = np.random.default_rng(3)
        sand = np.clip(clean + rng2.normal(0, 1.5, clean.shape), 0, 255).astype(np.uint8)
        self.assertIs(denoise_photo(sand), sand)  # fine daylight texture is not noise
        out = denoise_photo(noisy)
        self.assertEqual(out.shape, noisy.shape)
        self.assertLess(np.abs(out.astype(np.float32) - clean).mean(),
                        .6 * np.abs(noisy.astype(np.float32) - clean).mean())


if __name__ == '__main__':
    unittest.main()
