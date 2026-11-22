import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location('install_user', Path(__file__).resolve().parents[1] / 'scripts/install_user.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


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
