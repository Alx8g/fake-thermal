from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from fake_thermal.infer import convert
from fake_thermal.semantic import (
    CLASS_HEAT, PART_HEAT, PartSegmenter, SemanticPrior, emissive_prior, fuse_heat, guided_upsample,
    part_prior, probability_prior,
)


class SemanticTests(unittest.TestCase):
    def test_soft_probabilities_and_background(self):
        categories = ['background', *CLASS_HEAT]
        probabilities = np.zeros((len(categories), 2, 3), np.float32)
        prior, confidence = probability_prior(probabilities, categories)
        np.testing.assert_array_equal(confidence, 0)
        np.testing.assert_array_equal(prior, 0)
        for name, target in CLASS_HEAT.items():
            probabilities.fill(0)
            probabilities[categories.index(name)] = .4
            prior, confidence = probability_prior(probabilities, categories)
            np.testing.assert_allclose(prior, target)
            np.testing.assert_allclose(confidence, .4)
        probabilities.fill(0)
        probabilities[categories.index('person')] = .4
        probabilities[categories.index('car')] = .2
        prior, confidence = probability_prior(probabilities, categories)
        np.testing.assert_allclose(prior, .8)
        np.testing.assert_allclose(confidence, .6)

    def test_fusion_never_cools_and_zero_strength_is_identity(self):
        model = np.array([[.1, .95, .6]], np.float32)
        prior = np.array([[.9, .8, 1]], np.float32)
        confidence = np.array([[.5, 1, 0]], np.float32)
        np.testing.assert_array_equal(fuse_heat(model, prior, confidence, 0), model)
        fused = fuse_heat(model, prior, confidence, .7)
        np.testing.assert_allclose(fused, [[.38, .95, .6]])
        self.assertTrue((fused >= model).all())
        for strength in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                fuse_heat(model, prior, confidence, strength)

    def test_emissive_colors_and_feathering(self):
        for color, expected in [((255, 130, 0), 1), ((255, 220, 0), 1), ((255, 255, 255), .8)]:
            rgb = np.zeros((24, 24, 3), np.uint8)
            rgb[8:16, 8:16] = color
            prior, confidence = emissive_prior(rgb)
            self.assertAlmostEqual(float(prior[12, 12]), expected, places=5)
            self.assertGreater(confidence[12, 12], .6)
            self.assertGreater(confidence[7, 12], 0)
            self.assertLess(confidence[7, 12], confidence[12, 12])
            self.assertEqual(confidence[0, 0], 0)
        for color in ((0, 0, 0), (80, 40, 0), (0, 0, 255), (0, 255, 0), (150, 150, 150)):
            _, confidence = emissive_prior(np.full((16, 16, 3), color, np.uint8))
            np.testing.assert_array_equal(confidence, 0)

    def test_guided_shapes_constants_and_edge(self):
        for h, w in ((53, 75), (1, 7), (64, 64)):
            result = guided_upsample(np.full((8, 9), .4, np.float32), np.zeros((h, w, 3), np.uint8))
            self.assertEqual(result.shape, (h, w))
            np.testing.assert_allclose(result, .4, atol=1e-6)
            self.assertTrue(np.isfinite(result).all())
        rgb = np.zeros((64, 64, 3), np.uint8)
        rgb[:, 32:] = 255
        heat = np.zeros((16, 16), np.float32)
        heat[:, 8:] = 1
        result = guided_upsample(heat, rgb)
        self.assertGreater(float(result[32, 32] - result[32, 31]), .8)
        self.assertTrue(((result >= 0) & (result <= 1)).all())
        with self.assertRaises(ValueError):
            guided_upsample(np.zeros((0, 0)), rgb)

    def test_semantic_model_uses_softmax_and_lazy_loading(self):
        torch.set_num_threads(2)
        with patch('fake_thermal.semantic.deeplabv3_mobilenet_v3_large') as factory:
            semantic = SemanticPrior(size=32, device='cpu')
            factory.assert_not_called()
            logits = torch.zeros((1, 21, 32, 48))
            logits[:, 15] = 3
            factory.return_value.eval.return_value.to.return_value.return_value = {'out': logits}
            prior, confidence = semantic(np.zeros((16, 24, 3), np.uint8), (8, 12))
            self.assertEqual(prior.shape, (8, 12))
            self.assertTrue(((confidence > 0) & (confidence < 1)).all())
            self.assertGreater(float(prior.mean()), .8)
            semantic(np.zeros((16, 24, 3), np.uint8), (8, 12))
            factory.assert_called_once_with(weights=semantic.weights)
            factory.return_value.eval.return_value.to.assert_called_once_with(
                device=torch.device('cpu'), dtype=torch.float32)

    def test_part_prior_levels(self):
        masks = np.zeros((3, 4, 6), np.float32)
        masks[0, :, 3] = 1  # face skin
        masks[1, :, 4] = 1  # clothes
        masks[2, :, 0] = 1  # background
        target, person = part_prior(masks)
        np.testing.assert_allclose(target[0], PART_HEAT[3])
        np.testing.assert_allclose(target[1], PART_HEAT[4])
        np.testing.assert_allclose(target[2], CLASS_HEAT['person'])
        np.testing.assert_allclose(person, [[1] * 4, [1] * 4, [0] * 4])
        self.assertGreater(PART_HEAT[3], PART_HEAT[4])
        self.assertGreater(PART_HEAT[4], PART_HEAT[1])

    def test_part_segmenter_resolves_large_people_only(self):
        person = np.zeros((40, 40), np.float32)
        person[4:36, 4:16] = 1   # tall: resolved by parts
        person[30:34, 30:33] = 1  # distant: left to the class prior
        segmenter = PartSegmenter(min_height=.2)
        segmenter.segmenter = object()
        crops = []

        def face(crop):
            crops.append(crop.shape)
            masks = np.zeros((*crop.shape[:2], 6), np.float32)
            masks[..., 3] = 1
            return masks

        with patch.object(segmenter, 'segment', side_effect=face):
            target, seen, cover = segmenter(np.zeros((80, 80, 3), np.uint8), person, (40, 40))
        self.assertEqual(len(crops), 1)
        self.assertAlmostEqual(float(target[20, 10]), PART_HEAT[3], places=5)
        self.assertEqual(float(cover[20, 10]), 1.)
        self.assertEqual(float(cover[32, 31]), 0.)
        with patch.object(segmenter, 'segment', side_effect=lambda crop: np.eye(6, dtype=np.float32)[
                np.zeros(crop.shape[:2], int)]):
            self.assertIsNone(segmenter(np.zeros((80, 80, 3), np.uint8), person, (40, 40)))

    def test_part_segmenter_ignores_crop_background(self):
        person = np.zeros((40, 40), np.float32)
        person[4:36, 10:22] = 1
        segmenter = PartSegmenter(min_height=.2, pad=.5)
        segmenter.segmenter = object()
        rgb = np.zeros((80, 80, 3), np.uint8)
        # a part model that calls the whole crop a face, as the selfie model does on
        # small night-time figures
        whole = lambda crop: np.eye(6, dtype=np.float32)[np.full(crop.shape[:2], 3)]
        with patch.object(segmenter, 'segment', side_effect=whole):
            _, _, cover = segmenter(rgb, person, (40, 40))
            self.assertEqual(float(cover[20, 16]), 1.)
            self.assertEqual(float(cover[20, 5]), 0.)  # inside the crop, beyond the band
            self.assertEqual(float(cover[0, 16]), 0.)
            segmenter.min_overlap = .5
            self.assertIsNone(segmenter(rgb, person, (40, 40)))

    def test_convert_controls_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'in.png', Path(directory) / 'out.png'
            Image.new('RGB', (16, 24)).save(source)
            with patch('fake_thermal.infer.Predictor') as model, patch('fake_thermal.infer.SemanticPrior') as semantic:
                model.return_value.return_value = np.full((8, 8), .2, np.float32)
                semantic.return_value.return_value = (np.full((8, 8), .9, np.float32), np.ones((8, 8), np.float32))
                convert(source, output, semantic=True, semantic_strength=.7, palette='white-hot', sensor=False)
                self.assertGreater(np.asarray(Image.open(output)).mean(), 150)
                semantic.return_value.assert_called_once()
                semantic.reset_mock()
                for options in ({'semantic': False}, {'semantic_strength': 0}):
                    convert(source, output, **options)
                    semantic.assert_not_called()
                with self.assertRaises(ValueError):
                    convert(source, output, semantic_strength=2)


if __name__ == '__main__':
    unittest.main()
