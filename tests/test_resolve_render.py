import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, call, patch


RESOLVE = Path(__file__).resolve().parents[1] / 'resolve'


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, RESOLVE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render = load_module('resolve_render', 'Fake Thermal Render.py')
installer = load_module('resolve_install', 'install.py')


class TestResolveRender(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.source = self.directory / 'Source Clip.mov'
        self.source.touch()
        self.resolve = Mock()
        self.pool = self.resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
        self.clip = Mock()
        self.clip.GetClipProperty.return_value = str(self.source)
        self.pool.GetSelectedClips.return_value = [self.clip]
        self.config = {
            'python': sys.executable, 'mode': 'turbo', 'palette': 'arctic',
            'media_tool_dirs': [str(self.directory / 'Media Tools')],
        }

    @staticmethod
    def success(command, **kwargs):
        Path(command[command.index('-o') + 1]).write_bytes(b'output')
        return subprocess.CompletedProcess(command, 0, stdout='done')

    def test_turbo_draft_render_imports_and_preserves_existing_files(self):
        output_dir = self.directory / 'Rendered Videos'
        output_dir.mkdir()
        existing = output_dir / 'Source Clip_thermal.mp4'
        existing.write_bytes(b'keep')
        self.config.update(mode='turbo-draft', output_dir=str(output_dir))
        self.pool.GetSelectedClips.return_value = {'first': self.clip, 'duplicate': self.clip}
        run = Mock(side_effect=self.success)
        output = str(output_dir / 'Source Clip_thermal_2.mp4')
        self.assertEqual(render.render_selected(self.resolve, self.config, run, Mock()), [output])
        self.assertEqual(run.call_args.args[0],
                         [sys.executable, '-I', '-m', 'fake_thermal.infer', str(self.source.resolve()),
                          '-o', output, '--palette', 'arctic', '--engine', 'turbo', '--turbo-preview'])
        self.assertEqual(run.call_args.kwargs['env']['PATH'].split(os.pathsep)[0],
                         self.config['media_tool_dirs'][0])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.clip.GetClipProperty.call_args_list, [call('File Path')] * 2)
        self.pool.ImportMedia.assert_called_once_with([output])
        self.assertEqual(existing.read_bytes(), b'keep')

    def test_batch_skips_missing_and_reports_render_and_import_failures(self):
        missing, second = Mock(), Mock()
        missing.GetClipProperty.return_value = ''
        missing.GetName.return_value = 'Offline Clip'
        other_source = self.directory / 'Another.mov'
        other_source.touch()
        second.GetClipProperty.return_value = str(other_source)
        self.pool.GetSelectedClips.return_value = [missing, self.clip, second]
        self.pool.ImportMedia.return_value = []
        run = Mock(side_effect=lambda command, **kw: (
            self.success(command, **kw) if str(other_source.resolve()) in command
            else subprocess.CompletedProcess(command, 1, stdout='ffmpeg error')))
        report = Mock()
        output = str(other_source.resolve().with_name('Another_thermal.mp4'))
        self.assertEqual(render.render_selected(self.resolve, self.config, run, report), [output])
        self.assertTrue(all(call.args[0][-2:] == ['--engine', 'turbo'] for call in run.call_args_list))
        self.assertFalse(self.source.with_name('Source Clip_thermal.mp4').exists())
        self.assertTrue(Path(output).is_file())
        self.pool.ImportMedia.assert_called_once_with([output])
        messages = '\n'.join(entry.args[0] for entry in report.call_args_list)
        self.assertIn('Skipping Offline Clip', messages)
        self.assertIn('ffmpeg error', messages)
        self.assertIn('The render is still on disk', messages)

    def test_requires_a_media_pool_selection(self):
        self.pool.GetSelectedClips.return_value = []
        run = Mock()
        with self.assertRaisesRegex(RuntimeError, 'Select one or more source videos'):
            render.render_selected(self.resolve, self.config, run, Mock())
        run.assert_not_called()
        self.pool.ImportMedia.assert_not_called()


class TestResolveInstaller(unittest.TestCase):
    def test_installs_effect_and_render_configuration_to_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            luts, scripts, output = (home / name for name in ('LUTs', 'Scripts', 'Output'))
            with patch.object(Path, 'home', return_value=home), \
                    patch.object(shutil, 'which', side_effect=lambda tool: str(home / 'bin' / tool)), \
                    redirect_stdout(io.StringIO()):
                installer.main(['--lut-dir', str(luts), '--scripts-dir', str(scripts),
                                '--output-dir', str(output), '--mode', 'unet', '--palette', 'white-hot'])
            self.assertEqual((luts / 'Fake Thermal' / installer.DCTL_NAME).read_bytes(),
                             (RESOLVE / installer.DCTL_NAME).read_bytes())
            self.assertEqual((scripts / installer.SCRIPT_NAME).read_bytes(),
                             (RESOLVE / installer.SCRIPT_NAME).read_bytes())
            config = json.loads((home / render.CONFIG_NAME).read_text(encoding='utf-8'))
            self.assertEqual(config, {'python': sys.executable, 'mode': 'unet', 'palette': 'white-hot',
                                      'output_dir': str(output.resolve()), 'media_tool_dirs': [str(home / 'bin')]})

    def test_effect_only_does_not_require_ffmpeg_or_write_a_config(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with patch.object(Path, 'home', return_value=home), \
                    patch.object(shutil, 'which', return_value=None) as which, redirect_stdout(io.StringIO()):
                installer.main(['--dctl-only', '--lut-dir', str(home)])
            self.assertTrue((home / 'Fake Thermal' / installer.DCTL_NAME).is_file())
            self.assertFalse((home / render.CONFIG_NAME).exists())
            which.assert_not_called()
