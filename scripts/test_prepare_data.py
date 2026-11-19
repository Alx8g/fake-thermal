from collections import Counter
import unittest

from PIL import Image

from prepare_data import resize_pair, sequence_split


class PreparationTests(unittest.TestCase):
    def test_resize_and_modes(self):
        a, b = resize_pair(Image.new('RGB', (640, 512)), Image.new('RGB', (640, 512)))
        self.assertEqual(a.size, (360, 288))
        self.assertEqual(a.size, b.size)
        self.assertEqual((a.mode, b.mode), ('RGB', 'L'))

    def test_reject_different_dimensions(self):
        with self.assertRaises(ValueError):
            resize_pair(Image.new('RGB', (100, 100)), Image.new('L', (200, 100)))

    def test_split_empty_and_single_sequence(self):
        for ids in ([], range(100)):
            with self.assertRaises(ValueError):
                sequence_split(ids)

    def test_split_is_deterministic_and_groups_stay_whole(self):
        ids = [n for start in range(0, 20000, 1000) for n in range(start, start + 100)]
        split, groups = sequence_split(ids)
        self.assertEqual((split, groups), sequence_split(ids))
        self.assertEqual(Counter(split.values())['val'], 100)
        for a, b, count, _ in groups:
            self.assertEqual(len({split[n] for n in range(a, b + 1)}), 1)
        val = [n for n in ids if split[n] == 'val']
        train = [n for n in ids if split[n] == 'train']
        self.assertTrue(all(abs(a - b) > 150 for a in val for b in train))


if __name__ == '__main__':
    unittest.main()
