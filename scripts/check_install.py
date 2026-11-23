"""Assertions used by clean-runner installation CI."""
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'resolve'))
from install import default_paths, DCTL_NAME, SCRIPT_NAME
from fake_thermal.media import media_tool

home = Path.home()
root = Path(sys.executable).parents[2]
lut, scripts = default_paths(sys.platform, home, os.environ)
assert (lut / 'Fake Thermal' / DCTL_NAME).is_file()
assert (scripts / SCRIPT_NAME).is_file()
assert (home / '.fake-thermal-resolve.json').is_file()
support = Path(os.environ['APPDATA']) if sys.platform == 'win32' else home / 'Library/Application Support'
assert (support / 'Adobe/CEP/extensions/com.alx8g.fakethermal.panel/CSXS/manifest.xml').is_file()
result = subprocess.check_output([media_tool('ffprobe'), '-v', 'error', '-show_format', '-of', 'json', str(root / 'ci.mp4')])
assert float(json.loads(result)['format']['duration']) > 0
print('Verified MP4, DCTL, Resolve render script/config and Premiere panel.')
