import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch

from fake_thermal.data import PairedDataset
from fake_thermal.device import memory_format, select_device
from fake_thermal.infer import DisplayNormalizer, Predictor, TemporalSmoother, UNREADABLE_MEDIA, apply_palette, convert
from scripts.train import build_parser, training_config, train_run, validate


class DeviceTests(unittest.TestCase):
    def test_auto_priority_and_explicit_devices(self):
        for cuda, mps, expected in ((True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')):
            with patch('torch.cuda.is_available', return_value=cuda), patch('torch.backends.mps.is_available', return_value=mps):
                self.assertEqual(select_device().type, expected)
                self.assertEqual(select_device('cpu').type, 'cpu')
                for name, available in (('cuda', cuda), ('mps', mps)):
                    if available:
                        self.assertEqual(select_device(name).type, name)
                    else:
                        with self.assertRaisesRegex(ValueError, 'unavailable'):
                            select_device(name)
        with self.assertRaises(ValueError):
            select_device('invalid')
        self.assertEqual(memory_format(torch.device('mps')), torch.contiguous_format)

    def test_predictor_cpu_float32(self):
        predictor = Predictor(size=32, threads=2, device='cpu')
        self.assertEqual(next(predictor.model.parameters()).device.type, 'cpu')
        self.assertEqual(next(predictor.model.parameters()).dtype, torch.float32)
        self.assertEqual(predictor(np.zeros((35, 43, 3), np.uint8)).shape, (32, 39))

    def test_ui_controls_forwarded_and_errors(self):
        import app

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.png'
            Image.new('RGB', (32, 32)).save(source)
            with patch('app.convert') as converter:
                converter.return_value = dict(frames=1, seconds=1., fps=1., raw=None, device='cpu')
                result = app.run_media(str(source), "ironbow", False, .65, False, False, .7, "cpu", False, True, True)
                self.assertEqual(converter.call_args.kwargs['device'], 'cpu')
                self.assertFalse(converter.call_args.kwargs['auto_contrast'])
                self.assertIn('cpu', result[-1])
                app.run_media(str(source), "ironbow", False, .65, False, False, .7, "cpu", True, True, False,
                              engine='turbo', turbo_preview=True)
                self.assertTrue(converter.call_args.kwargs['turbo_preview'])
                self.assertIsNone(converter.call_args.kwargs['size'])
                converter.side_effect = ValueError('invalid device')
                with self.assertRaises(app.gr.Error):
                    app.run_media(str(source), "ironbow", False, .65, False, False, .7, "bad", True, True, True)
            with self.assertRaises(app.gr.Error):
                app.run_media(None, "ironbow", False, .65, False, False, .7, "cpu", True, True, True)

    def test_training_overrides_and_missing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--output', directory, '--device', 'cpu',
                                              '--batch-size', '8', '--iters', '10', '--crop', '384'])
            args.device = select_device(args.device)
            config = training_config(args)
            self.assertEqual((config['batch_size'], config['iterations'], config['crop'], config['device']),
                             (8, 10, 384, 'cpu'))
            (Path(directory) / 'benchmark.json').write_text(json.dumps({'batch_size': 4, 'iterations': 500}))
            self.assertEqual(training_config(args)['batch_size'], 8)
            args.batch_size = args.iters = None
            self.assertEqual(training_config(args)['iterations'], 500)
            (Path(directory) / 'benchmark.json').unlink()
            with self.assertRaisesRegex(ValueError, '--benchmark'):
                training_config(args)
            for flag in ('--batch-size', '--iters', '--crop'):
                with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        build_parser().parse_args([flag, '0'])

    def test_unreadable_upload_has_friendly_ui_error(self):
        import app

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'x.mp4'
            source.write_text('not media')
            with patch('app.ROOT', Path(directory)), patch('fake_thermal.infer.Predictor'):
                with self.assertRaises(app.gr.Error) as caught:
                    app.run_media(str(source), "ironbow", False, .65, False, False, .7, "cpu", True, True, True)
            self.assertEqual(caught.exception.message, UNREADABLE_MEDIA)
            self.assertNotIn(str(source), str(caught.exception))

    def test_unreadable_decode_or_empty_video_has_friendly_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'broken.mp4'
            source.write_text('not media')
            for code in (0, 1):
                with self.subTest(decoder_exit_code=code):
                    decoder, encoder = MagicMock(), MagicMock()
                    decoder.stdout.read.return_value = b''
                    decoder.wait.return_value = code
                    decoder.poll.return_value = encoder.poll.return_value = 0
                    with patch('fake_thermal.infer.Predictor'), patch('fake_thermal.infer.probe', return_value=(64, 64, '6/1', 0)), patch('fake_thermal.infer.fps_mode_args', return_value=['-fps_mode', 'passthrough']), patch('fake_thermal.infer.subprocess.Popen', side_effect=[decoder, encoder]):
                        with self.assertRaises(ValueError) as caught:
                            convert(source, Path(directory) / 'out.mp4', semantic=False, device='cpu')
                    self.assertEqual(str(caught.exception), UNREADABLE_MEDIA)

    def test_cpu_training_crop_validation_and_checkpoint(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val'):
                for modality in ('rgb', 'thermal'):
                    folder = root / 'pairs' / split / modality
                    folder.mkdir(parents=True)
                    mode = 'RGB' if modality == 'rgb' else 'L'
                    suffix = '.jpg' if modality == 'rgb' else '.png'
                    Image.new(mode, (64, 64), 80).save(folder / ('llvip_test' + suffix))
            train, val = PairedDataset(root, crop=48), PairedDataset(root, 'val', crop=48)
            self.assertEqual(train[0][0].shape, (3, 48, 48))
            self.assertEqual(val[0][1].shape, (1, 48, 48))
            with self.assertRaises(ValueError):
                PairedDataset(root, crop=0)
            output = root / 'run'
            (output / 'ckpt').mkdir(parents=True)
            args = build_parser().parse_args(['--output', str(output), '--batch-size', '2',
                                              '--iters', '1', '--crop', '48', '--workers', '0'])
            args.device = select_device('cpu')

            class TinyModel(torch.nn.Module):
                def __init__(self, **_):
                    super().__init__()
                    self.encoder = torch.nn.Conv2d(3, 4, 3, padding=1)
                    self.head = torch.nn.Conv2d(4, 1, 1)

                def forward(self, rgb):
                    return self.head(self.encoder(rgb).relu()).sigmoid()

            with patch('scripts.train.ThermalUNet', side_effect=TinyModel):
                train_run(args, train, val, ['llvip_test'])
            checkpoint = torch.load(output / 'ckpt/best.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['iteration'], 1)
            self.assertEqual(checkpoint['config']['crop'], 48)
            with Image.open(output / 'val_000001.jpg') as grid:
                self.assertEqual(grid.size, (144, 72))
            with self.assertRaises(ValueError):
                validate(TinyModel(), [])


class ContrastTests(unittest.TestCase):
    def test_stretch_smoothing_and_reset(self):
        heat = np.linspace(.5, .8, 1000, dtype=np.float32).reshape(20, 50)
        normalizer = DisplayNormalizer(smoothing=.9)
        result = normalizer.update(heat)
        self.assertEqual((float(result.min()), float(result.max())), (0, 1))
        bounds = normalizer.bounds.copy()
        normalizer.update(heat + .1)
        np.testing.assert_allclose(normalizer.bounds, bounds + .01, atol=1e-6)
        normalizer.update(heat + .1, reset=True)
        np.testing.assert_allclose(normalizer.bounds, bounds + .1, atol=1e-6)

    def test_constant_empty_and_cut(self):
        heat = np.full((8, 8), .7, np.float32)
        np.testing.assert_array_equal(DisplayNormalizer().update(heat), heat)
        for heat in (np.empty((0, 0)), np.full((8, 8), np.nan)):
            with self.assertRaises(ValueError):
                DisplayNormalizer().update(heat)
        with self.assertRaises(ValueError):
            DisplayNormalizer(smoothing=1)
        smoother = TemporalSmoother()
        normalizer = DisplayNormalizer()
        first = np.linspace(.2, .4, 64, dtype=np.float32).reshape(8, 8)
        second = first + .4
        for heat, rgb in ((first, np.zeros((8, 8, 3), np.uint8)),
                          (second, np.full((8, 8, 3), 255, np.uint8))):
            smoothed = smoother.update(heat, rgb)
            self.assertTrue(smoother.reset_history)
            normalizer.update(smoothed, reset=smoother.reset_history)
        np.testing.assert_allclose(normalizer.bounds, np.percentile(second, [2, 99.5]))

    def test_display_only_and_disable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.png'
            Image.new('RGB', (64, 64)).save(source)
            heat = np.tile(np.linspace(.5, .8, 64, dtype=np.float32), (64, 1))
            with patch('fake_thermal.infer.Predictor') as predictor, patch('fake_thermal.infer.guided_upsample', return_value=heat):
                predictor.return_value.return_value = heat
                for enabled in (True, False):
                    convert(source, root / f'out_{enabled}.png', palette='white-hot', raw=True,
                            semantic=False, device='cpu', auto_contrast=enabled, sensor=False)
                with Image.open(root / 'out_True.png') as im:
                    displayed = np.asarray(im)
                    self.assertEqual((displayed.min(), displayed.max()), (0, 255))
                with Image.open(root / 'out_False.png') as im:
                    np.testing.assert_array_equal(np.asarray(im), apply_palette(heat, 'white-hot'))
                self.assertEqual((root / 'out_True_raw.png').read_bytes(), (root / 'out_False_raw.png').read_bytes())


if __name__ == '__main__':
    unittest.main()
