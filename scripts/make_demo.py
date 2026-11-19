"""Render 200 available held-out KAIST daytime RGB frames at 20 fps."""
import json
from pathlib import Path
import subprocess
import zipfile

from PIL import Image


if __name__ == '__main__':
    output = Path('outputs/demo')
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile('data/raw/kaist/RGBKaist.zip') as archive:
        names = sorted(n for n in archive.namelist() if '/JPEGImages/set06_V003_visible_' in n)[:200]
        if len(names) != 200:
            raise ValueError('Need 200 held-out set06_V003 frames')
        process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-threads', '1', '-filter_threads', '1',
                                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x512', '-r', '20',
                                    '-i', 'pipe:0', '-c:v', 'libx264', '-threads', '1', '-crf', '18',
                                    '-pix_fmt', 'yuv420p', str(output / 'kaist_day_input.mp4')], stdin=subprocess.PIPE)
        for name in names:
            with Image.open(archive.open(name)) as im:
                process.stdin.write(im.convert('RGB').tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError('Demo encoding failed')
        (output / 'kaist_frame_members.json').write_text(json.dumps(names, indent=2) + '\n')
        print('200 sparse available set06_V003 frames at 20 fps; time-compressed, not native contiguous video')
