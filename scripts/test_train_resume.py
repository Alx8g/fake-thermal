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

    def test_extended_resume_adopts_new_cosine_horizon(self):
        model = train.ThermalUNet(pretrained=False)
        optimizer = train.optimizer_for(model)
        stored = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        stored.step()
        extended = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        train.load_scheduler(extended, stored.state_dict(), 40)
        self.assertEqual(extended.T_max, 40)
        unchanged = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        train.load_scheduler(unchanged, stored.state_dict(), 20)
        self.assertEqual(unchanged.T_max, 20)

    def test_resume_mismatches_reject_frequency_and_ema_changes(self):
        stored = {'batch_size': 16, 'crop': 256, 'seed': 42, 'frequency_alpha': 1., 'ema_decay': .999}
        current = dict(stored, ema_decay=0.)
        self.assertEqual(train.resume_mismatches(stored, current), ['ema_decay: 0.999 != 0.0'])


class RealismTrainingTests(unittest.TestCase):
    def test_amp_dtype_accepts_bools_names_and_dtypes(self):
        self.assertIsNone(train.amp_dtype(False))
        self.assertIsNone(train.amp_dtype('fp32'))
        self.assertIs(train.amp_dtype(True), torch.float16)
        self.assertIs(train.amp_dtype('bf16'), torch.bfloat16)
        self.assertIs(train.amp_dtype(torch.bfloat16), torch.bfloat16)

    def test_adversarial_step_trains_generator_and_critic(self):
        torch.manual_seed(0)
        model = train.ThermalUNet(pretrained=False, fullres=True)
        optimizer = train.optimizer_for(model)
        adversary = train.Adversary(torch.device('cpu'), gan_weight=.1, fm_weight=1.)
        critic_before = [p.detach().clone() for p in adversary.critic.parameters()]
        head_before = model.head[0].weight.detach().clone()
        rgb, target = torch.rand(2, 3, 64, 64), torch.rand(2, 1, 64, 64)
        loss = train.step(model, optimizer, rgb, target, adversary=adversary)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(set(adversary.last), {'g_adv', 'fm', 'd'})
        self.assertFalse(torch.equal(head_before, model.head[0].weight))
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(critic_before, adversary.critic.parameters())))
        self.assertTrue(all(p.requires_grad for p in adversary.critic.parameters()))

    def test_resume_mismatches_cover_adversarial_settings(self):
        stored = {'batch_size': 16, 'crop': 256, 'seed': 42}
        self.assertEqual(train.resume_mismatches(stored, dict(stored, gan_weight=.1)),
                         ['gan_weight: 0.0 != 0.1'])

    def test_init_prefers_ema_and_grows_into_fullres(self):
        import tempfile
        torch.manual_seed(0)
        source, ema = train.ThermalUNet(pretrained=False), train.ThermalUNet(pretrained=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            torch.save({'model': source.state_dict(), 'ema': ema.state_dict(), 'iteration': 5}, path)
            grown = train.load_init(train.ThermalUNet(pretrained=False, fullres=True), path).eval()
            x = torch.rand(1, 3, 64, 64)
            with torch.no_grad():
                self.assertTrue(torch.allclose(grown(x), ema.eval()(x), atol=1e-6))
            snapshot = Path(directory) / 'snapshots/iter_000500.pt'
            train.save_snapshot(grown.state_dict(), snapshot)
            saved = torch.load(snapshot, weights_only=True)
            self.assertIs(saved['head.0.weight'].dtype, torch.float16)

    def test_auto_workers_scale_with_cpus(self):
        self.assertGreaterEqual(train.worker_count('auto'), 2)
        self.assertEqual(train.worker_count('0'), 0)
        self.assertEqual(train.build_parser().parse_args([]).workers, train.default_workers())


if __name__ == '__main__':
    unittest.main()
