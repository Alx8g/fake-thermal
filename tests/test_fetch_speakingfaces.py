import io
import unittest

import numpy as np
from PIL import Image

from scripts.fetch_speakingfaces import PALETTE, thermal_to_grey, valid_box


def png(array):
    buf = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buf, 'PNG')
    return buf.getvalue()


class PaletteTests(unittest.TestCase):
    def test_palette_is_a_smooth_cold_to_hot_curve(self):
        self.assertEqual(PALETTE.shape, (224, 3))
        self.assertEqual(len({tuple(c) for c in PALETTE.tolist()}), 224)
        steps = np.linalg.norm(np.diff(PALETTE, axis=0), axis=1)
        self.assertLessEqual(steps.max(), 10.5)
        self.assertLess(PALETTE[0].sum(), 60)  # near black
        self.assertGreater(PALETTE[-1].sum(), 740)  # near white

    def test_palette_image_inverts_to_its_index(self):
        index = np.arange(224).reshape(14, 16)
        grey, miss = thermal_to_grey(png(PALETTE[index]))
        self.assertEqual(miss, 0)
        np.testing.assert_array_equal(grey, np.round(index * 255 / 223).astype(np.uint8))

    def test_off_palette_colour_takes_the_nearest_entry(self):
        image = PALETTE[[[100, 100]]].copy()
        image[0, 1] += [1, -1, 0]
        grey, miss = thermal_to_grey(png(image))
        self.assertEqual(miss, 1)
        self.assertEqual(grey[0, 0], round(100 * 255 / 223))
        self.assertLessEqual(abs(int(grey[0, 1]) - int(grey[0, 0])), 2)


class TrimTests(unittest.TestCase):
    def test_warp_fill_is_trimmed(self):
        rgb = np.full((120, 160, 3), 90, np.uint8)
        rgb[:7] = 0
        rgb[:, 150:] = 0
        rgb[60, 40] = 0  # a lone black pixel inside the frame stays
        self.assertEqual(valid_box(rgb), (7, 120, 0, 150))


if __name__ == '__main__':
    unittest.main()
