"""Render selected Media Pool source videos with the fake-thermal CLI."""
import json
import os
from pathlib import Path
import subprocess


CONFIG_NAME = '.fake-thermal-resolve.json'
MODE_ARGS = {
    'unet': ['--engine', 'unet'],
    'turbo-draft': ['--engine', 'turbo', '--turbo-preview'],
    'turbo': ['--engine', 'turbo'],
}


def reserve_output(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        suffix = '' if version == 1 else '_{}'.format(version)
        path = directory / ('{}_thermal{}.mp4'.format(source.stem, suffix))
        try:
            path.touch(exist_ok=False)
            return path
        except FileExistsError:
            version += 1


def render_selected(resolve, config, run=subprocess.run, report=print):
    project = resolve.GetProjectManager().GetCurrentProject()
    if not project:
        raise RuntimeError('Open a Resolve project first.')
    pool = project.GetMediaPool()
    select = getattr(pool, 'GetSelectedClips', None)
    if not select:
        raise RuntimeError('This Resolve version does not expose Media Pool clip selection.')
    selection = select()
    if not selection:
        raise RuntimeError('Select one or more source videos in the Media Pool, not the timeline.')
    if isinstance(selection, dict):
        selection = selection.values()

    args = MODE_ARGS.get(config['mode'])
    if args is None:
        raise ValueError('Invalid render mode: {}'.format(config['mode']))
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join(config.get('media_tool_dirs', []) + [os.defpath, env.get('PATH', '')])
    directory = Path(config['output_dir']).expanduser() if config.get('output_dir') else None
    rendered, seen = [], set()
    for clip in selection:
        path = clip.GetClipProperty('File Path')
        if not path or not Path(path).is_file():
            report('Skipping {}: no accessible source file.'.format(clip.GetName()))
            continue
        source = Path(path).resolve()
        if source in seen:
            continue
        seen.add(source)
        output = None
        try:
            output = reserve_output(source, directory or source.parent)
            report('Rendering {} ({})...'.format(source.name, config['mode']))
            result = run([config['python'], '-I', '-m', 'fake_thermal.infer', str(source),
                          '-o', str(output), '--palette', config['palette']] + args,
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding='utf-8', errors='replace')
            if result.returncode or output.stat().st_size == 0:
                details = '\n'.join((result.stdout or '').splitlines()[-8:])
                raise RuntimeError(details or 'The CLI did not produce a video.')
        except (OSError, RuntimeError) as error:
            if output is not None and output.exists():
                output.unlink()
            report('Could not render {}: {}'.format(source, error))
        else:
            rendered.append(str(output))

    for path in rendered:
        if pool.ImportMedia([path]):
            report('Imported {}'.format(path))
        else:
            report('Resolve could not import {}. The render is still on disk.'.format(path))
    return rendered


def get_resolve(namespace):
    if namespace.get('resolve'):
        return namespace['resolve']
    bmd = namespace.get('bmd')
    if bmd:
        app = bmd.scriptapp('Resolve')
        if app:
            return app
    import DaVinciResolveScript
    return DaVinciResolveScript.scriptapp('Resolve')


def main():
    try:
        app = get_resolve(globals())
        if not app:
            raise RuntimeError('Run this script from Resolve’s Workspace > Scripts menu.')
        with (Path.home() / CONFIG_NAME).open(encoding='utf-8') as config:
            render_selected(app, json.load(config))
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print('Fake Thermal: {}'.format(error))


if __name__ == '__main__':
    main()
