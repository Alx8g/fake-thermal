from itertools import islice
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train


class ResumeTests(unittest.TestCase):
    def test_sampler_tail_matches_full_stream(self):
        """Skipping draws from a reseeded sampler reproduces the tail of the stream."""
        weights = [1.] * 6
        full = list(WeightedRandomSampler(weights, 12, replacement=True,
                                          generator=torch.Generator().manual_seed(42)))
        stream = iter(WeightedRandomSampler(weights, 12, replacement=True,
                                            generator=torch.Generator().manual_seed(42)))
        self.assertEqual(list(islice(stream, 8, 12)), full[8:])

    def test_resume_sampler_replays_indices(self):
        sampler = train.ResumeSampler([3, 1, 2])
        self.assertEqual(list(iter(sampler)), [3, 1, 2])
        self.assertEqual(len(sampler), 3)

    def test_resume_mismatches_fills_defaults_and_skips_iterations(self):
        stored = {'batch_size': 16, 'iterations': 30000, 'crop': 256, 'seed': 42}
        current = {'batch_size': 16, 'iterations': 40000, 'crop': 256, 'seed': 42,
                   'precision': 'fp32', 'layout': 'auto'}
        self.assertEqual(train.resume_mismatches(stored, current), [])
        current['layout'] = 'channels_last'
        self.assertEqual(train.resume_mismatches(stored, current),
                         ["layout: 'auto' != 'channels_last'"])

    def test_forward_returns_float32_without_autocast(self):
        model = train.ThermalUNet(pretrained=False).eval()
        rgb = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            pred = train.forward(model, rgb)
        self.assertEqual(pred.dtype, torch.float32)
        self.assertEqual(pred.shape[-2:], (64, 64))


    def test_weight_ema_converges_toward_parameters(self):
        linear = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            linear.weight.fill_(0.)
        ema = train.WeightEMA(linear, decay=.5)
        with torch.no_grad():
            linear.weight.fill_(1.)
        ema.update(linear)
        self.assertTrue(torch.allclose(ema.shadow['weight'], torch.full((4, 4), .5)))
        ema.update(linear)
        self.assertTrue(torch.allclose(ema.shadow['weight'], torch.full((4, 4), .75)))
        with self.assertRaises(ValueError):
            train.WeightEMA(linear, decay=1.)

    def test_resume_mismatches_reject_frequency_and_ema_changes(self):
        stored = {'batch_size': 16, 'crop': 256, 'seed': 42, 'frequency_alpha': 1., 'ema_decay': .999}
        current = dict(stored, ema_decay=0.)
        self.assertEqual(train.resume_mismatches(stored, current), ['ema_decay: 0.999 != 0.0'])


if __name__ == '__main__':
    unittest.main()
