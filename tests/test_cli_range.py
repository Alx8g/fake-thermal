import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fake_thermal.infer import convert, range_args


class RangeTests(unittest.TestCase):
    def test_no_audio_skips_mux_for_color_and_raw_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'in.mp4', Path(folder) / 'out.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'testsrc2=size=64x64:rate=4:duration=3', '-f', 'lavfi',
                            '-i', 'sine=frequency=440:duration=3', '-c:v', 'libx264',
                            '-threads', '1', '-c:a', 'aac', str(source)], check=True)
            with patch('fake_thermal.infer.mux_audio') as mux, contextlib.redirect_stdout(io.StringIO()):
                convert(source, output, size=64, semantic=False, start=1,
                        duration=1, no_audio=True, raw=True)
            mux.assert_not_called()
            for video in (output, output.with_name('out_raw.mp4')):
                info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                    '-show_streams', '-of', 'json', str(video)]))
                self.assertEqual([stream['codec_type'] for stream in info['streams']], ['video'])
                self.assertAlmostEqual(float(info['streams'][0]['duration']), 1, delta=.05)

    def test_range_arguments(self):
        self.assertEqual(range_args(0, None), [])
        self.assertEqual(range_args(1, 2), ['-ss', '1', '-t', '2'])

    def test_invalid_ranges(self):
        for options in ({'start': -1}, {'start': float('nan')},
                        {'duration': 0}, {'duration': float('inf')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                convert('missing', 'out.mp4', **options)

    def test_trimmed_video_audio_and_json_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'in.mp4', Path(folder) / 'out.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'testsrc2=size=64x64:rate=4:duration=3', '-f', 'lavfi',
                            '-i', 'sine=frequency=440:duration=3', '-c:v', 'libx264',
                            '-threads', '1', '-c:a', 'aac', str(source)], check=True)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                stats = convert(source, output, size=64, semantic=False, start=1,
                                duration=1, progress_json=True)
            lines = [json.loads(line) for line in stderr.getvalue().splitlines()]
            self.assertEqual(stats['frames'], 4)
            self.assertEqual(lines[:-1], [{'frame': n, 'total': 4} for n in range(1, 5)])
            self.assertEqual(lines[-1], {'done': True, 'output': str(output)})
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                '-show_streams', '-of', 'json', str(output)]))
            for stream in info['streams']:
                self.assertAlmostEqual(float(stream['start_time']), 0, delta=.05)
                self.assertAlmostEqual(float(stream['duration']), 1, delta=.1)
            self.assertIn('audio', [stream['codec_type'] for stream in info['streams']])
            with self.assertRaisesRegex(ValueError, "Couldn't read"):
                convert(source, output, size=64, semantic=False, start=10, progress_json=True)

    def test_still_progress_and_range_rejection(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'in.png', Path(folder) / 'out.png'
            Image.new('RGB', (64, 64)).save(source)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                convert(source, output, size=64, semantic=False, progress_json=True)
            self.assertEqual(json.loads(stderr.getvalue().splitlines()[0]), {'frame': 1, 'total': 1})
            with self.assertRaisesRegex(ValueError, 'only supported for video'):
                convert(source, output, size=64, semantic=False, duration=1)
