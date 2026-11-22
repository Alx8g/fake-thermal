import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location('install_user', Path(__file__).resolve().parents[1] / 'scripts/install_user.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class MediaLinksTests(unittest.TestCase):
    def test_links_stable_path_tools_for_gui_and_repeated_installs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bindir = root / 'bin'
            bindir.mkdir()
            tools = {}
            for name in ('ffmpeg', 'ffprobe'):
                target = root / (name + '-real')
                target.touch()
                launcher = root / name
                launcher.symlink_to(target)
                tools[name] = str(launcher)
            for platform in ('darwin', 'linux'):
                with patch.object(installer.sys, 'platform', platform), patch.object(installer.shutil, 'which', side_effect=tools.get):
                    self.assertTrue(installer.link_path_media_tools(bindir))
                    self.assertTrue(installer.link_path_media_tools(bindir))
                for name in tools:
                    self.assertTrue((bindir / name).is_symlink())
                    self.assertEqual((bindir / name).readlink(), root / name)
            replacement = root / 'ffmpeg-upgraded'
            replacement.touch()
            (root / 'ffmpeg').unlink()
            (root / 'ffmpeg').symlink_to(replacement)
            (root / 'ffmpeg-real').unlink()
            self.assertTrue((bindir / 'ffmpeg').is_file())
            self.assertEqual((bindir / 'ffmpeg').resolve(), replacement)

    def test_venv_tools_on_path_are_not_replaced_with_self_links(self):
        with tempfile.TemporaryDirectory() as folder:
            bindir = Path(folder)
            for name in ('ffmpeg', 'ffprobe'):
                (bindir / name).touch()
            with patch.object(installer.sys, 'platform', 'darwin'), patch.object(installer.shutil, 'which', side_effect=lambda name: str(bindir / name)):
                self.assertTrue(installer.link_path_media_tools(bindir))
            for name in ('ffmpeg', 'ffprobe'):
                self.assertTrue((bindir / name).is_file())
                self.assertFalse((bindir / name).is_symlink())

    def test_missing_tool_leaves_directory_untouched_for_download(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(installer.shutil, 'which', side_effect=['/tools/ffmpeg', None]):
            self.assertFalse(installer.link_path_media_tools(Path(folder)))
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_windows_does_not_link_path_tools(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(installer.sys, 'platform', 'win32'), patch.object(installer.shutil, 'which', return_value='C:/tools/ffmpeg.exe'):
            self.assertTrue(installer.link_path_media_tools(Path(folder)))
            self.assertEqual(list(Path(folder).iterdir()), [])


class ElevationTests(unittest.TestCase):
    def test_passwordless_sudo_works_with_piped_stdin(self):
        with patch.object(installer.sys, 'platform', 'darwin'), patch.object(installer.subprocess, 'run', return_value=Mock(returncode=0)) as run:
            self.assertTrue(installer.elevate_dctl(['python', 'install.py', '--dctl-only']))
            self.assertEqual(run.call_args.args[0], ['sudo', '-n', 'python', 'install.py', '--dctl-only'])

    def test_no_terminal_prints_exact_command(self):
        with patch.object(installer.sys, 'platform', 'darwin'), patch.object(installer.subprocess, 'run', return_value=Mock(returncode=1)), patch('builtins.open', side_effect=OSError), patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertFalse(installer.elevate_dctl(['/path with space/python', 'install.py', '--uninstall-dctl']))
            self.assertIn("sudo '/path with space/python' install.py --uninstall-dctl", output.getvalue())

    def test_terminal_sudo_uses_controlling_terminal(self):
        tty = Mock()
        tty.__enter__ = Mock(return_value=tty)
        tty.__exit__ = Mock(return_value=False)
        with patch.object(installer.sys, 'platform', 'darwin'), patch.object(installer.subprocess, 'run', side_effect=[Mock(returncode=1), Mock(returncode=0)]) as run, patch('builtins.open', return_value=tty):
            self.assertTrue(installer.elevate_dctl(['python', 'install.py']))
            self.assertIs(run.call_args.kwargs['stdin'], tty)
            self.assertIs(run.call_args.kwargs['stdout'], tty)

    def test_windows_noninteractive_prints_manual_command(self):
        with patch.object(installer.sys, 'platform', 'win32'), patch.object(installer.sys.stdin, 'isatty', return_value=False), patch.object(installer.subprocess, 'run') as run, patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertFalse(installer.elevate_dctl(['python.exe', 'install.py']))
            run.assert_not_called()
            self.assertIn("& 'python.exe' 'install.py'", output.getvalue())

    def test_windows_interactive_elevation_waits(self):
        with patch.object(installer.sys, 'platform', 'win32'), patch.object(installer.sys.stdin, 'isatty', return_value=True), patch.object(installer.subprocess, 'run', return_value=Mock(returncode=0)) as run:
            self.assertTrue(installer.elevate_dctl(['python.exe', 'install.py']))
            self.assertIn('-Verb RunAs -Wait', run.call_args.args[0][-1])
