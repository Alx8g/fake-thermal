from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fake_thermal.media import media_tool


class MediaToolTests(unittest.TestCase):
    def test_path_takes_precedence(self):
        with patch('fake_thermal.media.shutil.which', return_value='/tools/ffmpeg'):
            self.assertEqual(media_tool('ffmpeg'), '/tools/ffmpeg')

    def test_adjacent_and_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('fake_thermal.media.shutil.which', return_value=None), patch('fake_thermal.media.sys.executable', str(Path(folder) / 'python')), patch('fake_thermal.media.sys.platform', 'linux'):
                self.assertEqual(media_tool('ffprobe'), 'ffprobe')
                (Path(folder) / 'ffprobe').touch()
                self.assertEqual(media_tool('ffprobe'), str(Path(folder) / 'ffprobe'))

    def test_windows_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'ffmpeg.exe').touch()
            with patch('fake_thermal.media.shutil.which', return_value=None), patch('fake_thermal.media.sys.executable', str(Path(folder) / 'python.exe')), patch('fake_thermal.media.sys.platform', 'win32'):
                self.assertEqual(media_tool('ffmpeg'), str(Path(folder) / 'ffmpeg.exe'))
