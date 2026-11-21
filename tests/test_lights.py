import unittest

import numpy as np

from fake_thermal.lights import find_lights, neutralize_lights


def night_with_lamp(size=256, centre=(80, 120), radius=3):
    """A dark photo with one small saturated lamp and a soft glow around it."""
    yy, xx = np.mgrid[0:size, 0:size]
    distance = np.hypot(yy - centre[0], xx - centre[1])
    value = .12 + .5 * np.exp(-(distance / (2.5 * radius)) ** 2)
    value[distance <= radius] = 1.
    return (np.repeat(value[..., None], 3, 2) * 255).astype(np.uint8), distance


class LightTests(unittest.TestCase):
    def test_finds_a_lamp_on_a_dark_scene(self):
        rgb, distance = night_with_lamp()
        mask = find_lights(rgb, (256, 256))
        self.assertIsNotNone(mask)
        self.assertTrue(mask[distance <= 3].all())
        self.assertFalse(mask[distance > 20].any())

    def test_ignores_bright_things_in_daylight_and_large_areas(self):
        rgb = np.full((128, 128, 3), 170, np.uint8)
        rgb[50:56, 50:56] = 250  # a white patch on a sunlit scene is not a light source
        self.assertIsNone(find_lights(rgb, (128, 128)))
        dark = np.full((128, 128, 3), 20, np.uint8)
        dark[20:100, 20:100] = 255  # a lit screen filling the frame is too big for a lamp
        self.assertIsNone(find_lights(dark, (128, 128)))

    def test_removes_cold_disc_and_keeps_the_rest(self):
        rgb, distance = night_with_lamp()
        rng = np.random.default_rng(0)
        heat = (.5 + rng.normal(0, .01, (256, 256))).astype(np.float32)
        heat[distance <= 6] = .05  # the cold disc the translator paints at lamps
        fixed = neutralize_lights(heat, rgb)
        self.assertGreater(fixed[distance <= 6].mean(), .45)
        np.testing.assert_array_equal(fixed[distance > 30], heat[distance > 30])
        hot = heat.copy()
        hot[distance <= 6] = .98  # or the glow it copies as heat
        self.assertLess(neutralize_lights(hot, rgb)[distance <= 6].mean(), .55)

    def test_excluded_pixels_are_left_alone(self):
        rgb, distance = night_with_lamp()
        heat = np.full((256, 256), .5, np.float32)
        heat[distance <= 6] = .05
        fixed = neutralize_lights(heat, rgb, exclude=np.ones((256, 256), np.float32))
        np.testing.assert_array_equal(fixed, heat)

    def test_mask_follows_heat_resolution(self):
        rgb, _ = night_with_lamp(size=256, centre=(80, 120), radius=4)
        mask = find_lights(rgb, (64, 64))
        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(mask[20, 30])


if __name__ == '__main__':
    unittest.main()
