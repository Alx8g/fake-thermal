"""Shared per-user installer, run only with the bootstrapped managed Python."""
import argparse
import ctypes
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def download(url, path):
    print('Downloading ' + url, flush=True)
    if sys.platform == 'darwin':
        run('curl', '-fLsS', '--retry', '3', url, '-o', path)
    else:
        urllib.request.urlretrieve(url, path)


def notify_path_change():
    result = ctypes.c_size_t()
    ctypes.windll.user32.SendMessageTimeoutW(65535, 26, 0, 'Environment', 2, 5000, ctypes.byref(result))


def link_path_media_tools(bindir):
    tools = {name: shutil.which(name) for name in ('ffmpeg', 'ffprobe')}
    if not all(tools.values()):
        return False
    if sys.platform != 'win32':
        for name, found in tools.items():
            target = Path(found).resolve()
            destination = bindir / name
            if destination.resolve() != target:
                destination.unlink(missing_ok=True)
                destination.symlink_to(target)
    return True


def elevate_dctl(command):
    if sys.platform == 'darwin':
        if subprocess.run(['sudo', '-n'] + command, check=False).returncode == 0:
            return True
        try:
            tty = open('/dev/tty', 'rb+', buffering=0)
        except OSError:
            tty = None
        if tty is not None:
            with tty:
                tty.write(b'Resolve stores its DCTL preview in /Library. Administrator permission is needed for this file only.\n')
                if subprocess.run(['sudo'] + command, stdin=tty, stdout=tty, stderr=tty, check=False).returncode == 0:
                    return True
        manual = 'sudo ' + shlex.join(command)
    elif sys.platform == 'win32':
        arguments = subprocess.list2cmdline(command[1:])
        expression = "Start-Process -FilePath '{}' -ArgumentList '{}' -Verb RunAs -Wait -PassThru | ForEach-Object {{ exit $_.ExitCode }}".format(command[0].replace("'", "''"), arguments.replace("'", "''"))
        if sys.stdin.isatty():
            if subprocess.run(['powershell.exe', '-NoProfile', '-Command', expression], check=False).returncode == 0:
                return True
        manual = '& ' + ' '.join("'" + arg.replace("'", "''") + "'" for arg in command)
    else:
        manual = 'sudo ' + shlex.join(command)
    print('DCTL action requires an elevated shell. Run: ' + manual)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'uv', 'source'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('turbo', 'skip-plugins', 'uninstall'):
        parser.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    root, source = args.root.absolute(), args.source.absolute()
    windows = sys.platform == 'win32'
    bindir = root / 'venv' / ('Scripts' if windows else 'bin')
    python = bindir / ('python.exe' if windows else 'python')
    cli = bindir / ('fake-thermal.exe' if windows else 'fake-thermal')
    spec = importlib.util.spec_from_file_location('resolve_install', source / 'resolve/install.py')
    resolve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolve)
    lut, scripts = resolve.default_paths(sys.platform, Path.home(), os.environ)
    if args.uninstall:
        if windows:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_ALL_ACCESS) as key:
                value = winreg.QueryValueEx(key, 'Path')[0]
                winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, ';'.join(p for p in value.split(';') if p.lower() != str(bindir).lower()))
            notify_path_change()
        else:
            link = Path.home() / '.local/bin/fake-thermal'
            if link.is_symlink() and link.resolve() == cli.resolve():
                link.unlink()
        for path in (lut / 'Fake Thermal', scripts / resolve.SCRIPT_NAME, Path.home() / resolve.CONFIG_NAME):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except PermissionError:
                if path == lut / 'Fake Thermal':
                    if windows:
                        command = ['powershell.exe', '-NoProfile', '-Command', "Remove-Item -LiteralPath '{}' -Recurse -Force".format(str(path).replace("'", "''"))]
                    else:
                        command = ['/bin/rm', '-rf', str(path)]
                    elevate_dctl(command)
                else:
                    print('Remove with administrator privileges: ' + str(path))
        premiere = (Path(os.environ['APPDATA']) if windows else Path.home() / 'Library/Application Support') / 'Adobe/CEP/extensions/com.alx8g.fakethermal.panel'
        shutil.rmtree(premiere, ignore_errors=True)
        if not windows:
            shutil.rmtree(root)
        print('Uninstalled fake-thermal (shared model caches and CEP debug settings retained).')
        return
    stored = root / 'installer'
    (stored / 'scripts').mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'scripts/install_user.py', stored / 'scripts/install_user.py')
    shutil.copytree(source / 'resolve', stored / 'resolve', dirs_exist_ok=True)
    pip = [args.uv, 'pip', 'install', '--python', python, '--upgrade']
    run(*pip, 'torch', 'torchvision', *([] if sys.platform == 'darwin' else ['--index-url', 'https://download.pytorch.org/whl/cpu']))
    run(*pip, source)
    if args.turbo:
        run(*pip, 'diffusers', 'transformers', 'peft', 'mediapipe', 'huggingface_hub')
    if not link_path_media_tools(bindir):
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            temporary = Path(temporary)
            if windows:
                url = 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip'
            elif sys.platform == 'darwin':
                arch = 'arm64' if platform.machine() == 'arm64' else 'amd64'
                build = '1795207104_9.1' if arch == 'arm64' else '1795212433_9.1'
                url = 'https://ffmpeg.martin-riedl.de/download/macos/' + arch + '/' + build + '/ffmpeg.zip'
            else:
                url = 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-' + ('arm64' if platform.machine() == 'aarch64' else 'amd64') + '-static.tar.xz'
            archive = temporary / 'download'
            download(url, archive)
            if sys.platform == 'darwin':
                with zipfile.ZipFile(archive) as z:
                    z.extractall(temporary)
                download(url.replace('/ffmpeg.zip', '/ffprobe.zip'), archive)
            if windows or sys.platform == 'darwin':
                with zipfile.ZipFile(archive) as z:
                    z.extractall(temporary)
            else:
                with tarfile.open(archive) as tar:
                    tar.extractall(temporary, filter='data')
            for name in ('ffmpeg', 'ffprobe'):
                filename = name + ('.exe' if windows else '')
                tool = next(temporary.rglob(filename))
                if (bindir / filename).is_symlink():
                    (bindir / filename).unlink()
                shutil.copy2(tool, bindir / filename)
                (bindir / filename).chmod(0o755)
    os.environ['PATH'] = str(bindir) + os.pathsep + os.environ.get('PATH', '')
    if windows:
        import winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
            try:
                value = winreg.QueryValueEx(key, 'Path')[0]
            except FileNotFoundError:
                value = ''
            if str(bindir).lower() not in [p.lower() for p in value.split(';')]:
                winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, value.rstrip(';') + ';' + str(bindir))
        notify_path_change()
    else:
        link = Path.home() / '.local/bin/fake-thermal'
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        link.symlink_to(cli)
        if str(link.parent) not in os.environ['PATH'].split(os.pathsep):
            print('Add to your shell profile: export PATH="$HOME/.local/bin:$PATH"')
    if not args.skip_plugins:
        run(python, source / 'resolve/install.py', '--render-only', '--mode', 'turbo' if args.turbo else 'unet')
        command = [str(python), str(source / 'resolve/install.py'), '--dctl-only']
        try:
            run(*command)
        except subprocess.CalledProcessError as error:
            if error.returncode != 3:
                raise
            elevate_dctl(command)
        if windows:
            run('powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', source / 'plugins/premiere/install-windows.ps1')
        elif sys.platform == 'darwin':
            run('sh', source / 'plugins/premiere/install-mac.sh')
    run('ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=128x128:rate=4:duration=1', '-c:v', 'libx264', root / 'verification-input.mp4')
    run(cli, root / 'verification-input.mp4', '-o', root / 'verification.mp4')
    run('ffprobe', '-v', 'error', '-show_entries', 'format=duration', root / 'verification.mp4')
    print('Installed CLI + weights: {}\nPython, media tools and verification render: {}\nRestart editors and open a fresh terminal.'.format(cli, root))


if __name__ == '__main__':
    main()
