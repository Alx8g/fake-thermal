"""Install the Resolve DCTL and optional offline rendering script."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent
DCTL_NAME = 'Fake Thermal.dctl'
SCRIPT_NAME = 'Fake Thermal Render.py'
CONFIG_NAME = '.fake-thermal-resolve.json'
PALETTES = ('ironbow', 'white-hot', 'black-hot', 'rainbow', 'arctic', 'lava', 'glowbow')


def default_paths(platform, home, env):
    vendor = Path('Blackmagic Design') / 'DaVinci Resolve'
    if platform == 'darwin':
        support = Path('Library') / 'Application Support' / vendor
        return Path('/') / support / 'LUT', home / support / 'Fusion' / 'Scripts' / 'Utility'
    if platform == 'win32':
        program_data = Path(env.get('PROGRAMDATA', r'C:\ProgramData'))
        roaming = Path(env.get('APPDATA', home / 'AppData' / 'Roaming'))
        return (program_data / vendor / 'Support' / 'LUT',
                roaming / vendor / 'Support' / 'Fusion' / 'Scripts' / 'Utility')
    if platform.startswith('linux'):
        return (Path('/opt/resolve/LUT'),
                home / '.local' / 'share' / 'DaVinciResolve' / 'Fusion' / 'Scripts' / 'Utility')
    raise ValueError('Unsupported OS: {}'.format(platform))


def install_file(name, directory):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    shutil.copyfile(HERE / name, destination)
    print('Installed {}'.format(destination))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lut-dir', type=Path, help='Resolve LUT root (Color Management > Open LUT Folder)')
    parser.add_argument('--scripts-dir', type=Path, help='Resolve Scripts/Utility directory')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dctl-only', action='store_true', help='install only the GPU preview effect')
    group.add_argument('--render-only', action='store_true', help='install only the Media Pool render script')
    parser.add_argument('--mode', choices=('turbo', 'turbo-draft', 'unet'), default='turbo')
    parser.add_argument('--palette', choices=PALETTES, default='ironbow')
    parser.add_argument('--output-dir', type=Path, help='render folder, defaults to beside each source')
    args = parser.parse_args(argv)
    home = Path.home()
    try:
        lut_dir, scripts_dir = default_paths(sys.platform, home, os.environ)
        config = None
        if not args.dctl_only:
            tools = [shutil.which(tool) or next((str(p) for p in
                     [Path(sys.executable).parent / (tool + ('.exe' if sys.platform == 'win32' else ''))]
                     if p.is_file()), None) for tool in ('ffmpeg', 'ffprobe')]
            if not all(tools):
                parser.error('Install ffmpeg and ffprobe and put them on PATH before installing the render script.')
            config = {
                'python': sys.executable,
                'mode': args.mode,
                'palette': args.palette,
                'media_tool_dirs': list(dict.fromkeys(str(Path(tool).absolute().parent) for tool in tools)),
                'output_dir': str(args.output_dir.expanduser().resolve()) if args.output_dir else None,
            }
        if not args.render_only:
            install_file(DCTL_NAME, (args.lut_dir or lut_dir).expanduser() / 'Fake Thermal')
        if config is not None:
            install_file(SCRIPT_NAME, (args.scripts_dir or scripts_dir).expanduser())
            config_path = home / CONFIG_NAME
            config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
            print('Wrote {}'.format(config_path))
    except PermissionError as error:
        parser.exit(3, '{}\nAdministrator permission is required for this directory.\n'.format(error))
    except (OSError, ValueError) as error:
        parser.exit(1, '{}\nUse --lut-dir or --scripts-dir to select a writable Resolve directory.\n'.format(error))


if __name__ == '__main__':
    main()
