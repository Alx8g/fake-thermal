import random
import unittest

import numpy as np

from fake_thermal.augment import add_cast, add_lights, add_print, darken, photo_only


class AugmentTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)

    def test_each_augmentation_keeps_shape_and_range(self):
        rgb = np.random.default_rng(0).random((64, 80, 3)).astype(np.float32)
        for function in (add_cast, add_lights, add_print, darken):
            with self.subTest(function=function.__name__):
                out = function(rgb)
                self.assertEqual(out.shape, rgb.shape)
                self.assertGreaterEqual(float(out.min()), 0.)
                self.assertLessEqual(float(out.max()), 1.)

    def test_no_probability_means_no_change(self):
        rgb = np.full((48, 48, 3), .5, np.float32)
        np.testing.assert_array_equal(photo_only(rgb), rgb)

    def test_cast_day_zero_casts_light_on_night_photos_only(self):
        day = np.full((64, 64, 3), .6, np.float32)
        night = np.full((64, 64, 3), .1, np.float32)
        for _ in range(5):
            np.testing.assert_array_equal(photo_only(day, p_cast=1., cast_day=0.), day)
        self.assertTrue(any(np.abs(photo_only(night, p_cast=1., cast_day=0.) - night).max() > .01
                            for _ in range(5)))
        self.assertTrue(any(np.abs(photo_only(day, p_cast=1.) - day).max() > .01 for _ in range(5)))


if __name__ == '__main__':
    unittest.main()
