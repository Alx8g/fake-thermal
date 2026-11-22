from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from fake_thermal.appearance import edit_appearance
from fake_thermal.losses import appearance_loss
from scripts.appearance_report import pair_metrics, sample_cases, summarize
from scripts.train_turbo import TurboPairs


class AppearanceTests(unittest.TestCase):
    def test_edits_need_registered_material_masks(self):
        rgb = np.full((64, 64, 3), .6, np.float32)
        for kind in ('clothing', 'glass'):
            for mask in (None, np.ones((2, 3), np.float32)):
                with self.assertRaisesRegex(ValueError, 'aligned'):
                    edit_appearance(rgb, kind, mask)
            edited, influence = edit_appearance(rgb, kind, np.zeros((64, 64), np.float32))
            np.testing.assert_array_equal(edited, rgb)
            np.testing.assert_array_equal(influence, 0)

    def test_masked_edits_preserve_geometry_and_leave_background_alone(self):
        mask = np.zeros((64, 64), np.float32)
        mask[16:48, 16:48] = 1
        rgb = np.full((64, 64, 3), .7, np.float32)
        rgb[16:48, 16:48:2] = .3
        for kind in ('clothing', 'glass'):
            edited, influence = edit_appearance(rgb, kind, mask)
            np.testing.assert_array_equal(edited[mask == 0], rgb[mask == 0])
            self.assertGreater(float(influence[20:44, 20:44].mean()), .25)
            np.testing.assert_array_equal(influence[:12], 0)
            self.assertTrue(np.isfinite(edited).all())
            self.assertTrue(((0 <= edited) & (edited <= 1)).all())
        self.assertLess(float(edited[25, 25, 0] - edited[25, 24, 0]), .25)

    def test_seeded_print_and_glare_are_repeatable(self):
        rgb = np.full((192, 192, 3), .3, np.float32)
        for kind in ('print', 'glare'):
            random.seed(19)
            np.random.seed(19)
            a, m = edit_appearance(rgb, kind)
            random.seed(19)
            np.random.seed(19)
            b, n = edit_appearance(rgb, kind)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(m, n)
            self.assertGreater(float(m.sum()), 0)
            self.assertTrue(((0 <= m) & (m <= 1)).all())
        np.testing.assert_array_equal(rgb, np.full_like(rgb, .3))

    def test_loss_is_local_and_does_not_update_clean_anchor(self):
        edit = torch.full((2, 3, 2, 2), .5, requires_grad=True)
        clean = torch.zeros_like(edit, requires_grad=True)
        target = torch.zeros_like(edit)
        mask = torch.zeros((2, 1, 2, 2))
        mask[0, 0, 1, 1] = .1
        self.assertEqual(appearance_loss(edit, clean, target, mask * 0).item(), 0)
        loss = appearance_loss(edit, clean, target, mask)
        self.assertAlmostEqual(loss.item(), .375)
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertTrue(torch.isfinite(edit.grad).all())
        self.assertEqual(torch.count_nonzero(edit.grad).item(), 3)

    def test_dataset_shares_resize_crop_flip_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'pairs/train'
            for sub in ('rgb', 'thermal', 'physics/clothing'):
                (root / sub).mkdir(parents=True)
            Image.new('RGB', (24, 12), (160, 160, 160)).save(root / 'rgb/frame.jpg')
            y, x = np.mgrid[:12, :24]
            image = np.uint8(x < 8) * 255
            Image.fromarray(image).save(root / 'thermal/frame.png')
            Image.fromarray(image).save(root / 'physics/clothing/frame.png')
            baseline = TurboPairs(directory, crop=16)
            augmented = TurboPairs(directory, crop=16, appearance=True)
            rng = random.Random(7)
            augmented.appearance_rng = rng, np.random.RandomState(7)
            with patch('scripts.train_turbo.random.randint', side_effect=[0, 0, 0, 0]), \
                 patch.object(rng, 'choice', return_value='clothing'), \
                 patch.object(rng, 'random', return_value=.1), \
                 patch('scripts.train_turbo.random.random', return_value=.1):
                np.random.seed(7)
                _, target0, _ = baseline[0]
                np.random.seed(7)
                base, target, stem, edited, influence = augmented[0]
            torch.testing.assert_close(target0, target)
            self.assertEqual(stem, 'frame')
            self.assertEqual(edited.shape, base.shape)
            self.assertEqual(influence.shape, (1, 16, 16))
            self.assertTrue(torch.equal(edited[:, :, :3], base[:, :, :3]))
            self.assertTrue((edited[:, :, 9:] < base[:, :, 9:]).all())
            self.assertLess(target[:, :, :3].max(), target[:, :, 9:].min())

    def test_appearance_does_not_change_the_clean_training_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'pairs/train'
            (root / 'rgb').mkdir(parents=True)
            (root / 'thermal').mkdir()
            image = np.random.RandomState(1).randint(0, 256, (128, 160), dtype=np.uint8)
            Image.fromarray(image).convert('RGB').save(root / 'rgb/frame.jpg')
            Image.fromarray(image).save(root / 'thermal/frame.png')

            def batches(appearance):
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                pairs = TurboPairs(directory, crop=96, augment={'p_print': .5}, appearance=appearance)
                return [pairs[0][:2] for _ in range(4)]

            for control, candidate in zip(batches(False), batches(True)):
                torch.testing.assert_close(candidate, control, atol=1e-7, rtol=0)

    def test_metrics_keep_same_display_mapping_and_report_missing_kinds(self):
        clean = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
        mask = np.zeros_like(clean)
        mask[1, 1] = 1
        edited = clean + .1
        result = pair_metrics(clean, edited, clean, mask)
        self.assertAlmostEqual(result['edit_drift'], .1, places=6)
        self.assertAlmostEqual(result['context_drift'], .1, places=6)
        self.assertAlmostEqual(result['clean_mae'], 0)
        self.assertGreater(result['edited_region_mae'], .1)
        self.assertIsNone(pair_metrics(clean, edited, clean, mask * 0))
        summary = summarize([{'stem': 'a', 'kind': 'print', **result}], ('print', 'glass'))
        self.assertEqual(summary['glass'], {'count': 0})
        self.assertEqual(summary['print']['count'], 1)

    def test_case_sampling_keeps_material_kinds_to_annotated_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / 'physics/glass'
            masks.mkdir(parents=True)
            (masks / 'b.png').touch()
            self.assertEqual(sample_cases(root, ('glass',), 4, 2, ['a', 'b', 'c']), {'b': ['glass']})


if __name__ == '__main__':
    unittest.main()
