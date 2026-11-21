from argparse import Namespace
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preference_test


class SheetTests(unittest.TestCase):
    def make_round(self, root, frames, variants):
        for folder in ('rgb', *variants):
            (root / folder).mkdir(parents=True)
            for index in range(frames):
                gray = np.full((24, 16), 40 * index % 255, np.uint8)
                Image.fromarray(gray).save(root / folder / f'f{index:02d}.png')

    def test_each_variant_on_the_left_in_half_of_every_judges_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_round(root, 6, ('parts', 'uniform', 'none'))
            preference_test.sheets(Namespace(out=root, comparisons=['parts:uniform', 'parts:none'],
                                             judges=4, height=24, seed=3))
            key = json.loads((root / 'blind' / 'key.json').read_text())
            judges = json.loads((root / 'blind' / 'judges.json').read_text())
            sheets = {entry['sheet']: entry for entry in key}
            for numbers in judges.values():
                left = Counter((tuple(sorted((sheets[n]['A'], sheets[n]['B']))), sheets[n]['A'])
                               for n in numbers)
                totals = Counter(tuple(sorted((sheets[n]['A'], sheets[n]['B']))) for n in numbers)
                for pair, total in totals.items():
                    self.assertLessEqual(abs(left[pair, pair[0]] - left[pair, pair[1]]), total % 2)
                frames = [sheets[n]['frame'] for n in numbers]
                self.assertEqual(len(frames), len(set(frames)))

    def test_choice_of_accepts_both_answer_shapes(self):
        self.assertEqual(preference_test.choice_of('b'), 'B')
        self.assertEqual(preference_test.choice_of({'realistic': ' a ', 'confidence': 2}), 'A')

    def test_one_sided_judges_are_flagged(self):
        self.assertTrue(preference_test.one_sided(['A'] * 15 + ['B']))
        self.assertFalse(preference_test.one_sided(['A'] * 12 + ['B'] * 4))
        self.assertFalse(preference_test.one_sided(['A'] * 5))


if __name__ == '__main__':
    unittest.main()
