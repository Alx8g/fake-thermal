'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const cp = require('child_process');
function loadLogic() {
    try {
        let root = window.__adobe_cep__.getSystemPath('extension');
        if (/^file:/i.test(root)) {
            const url = new URL(root);
            root = decodeURIComponent(url.pathname);
            if (url.hostname && url.hostname !== 'localhost') root = '//' + url.hostname + root;
        } else {
            root = decodeURIComponent(root);
        }
        if (process.platform === 'win32') {
            root = root.replace(/^\/(?=[a-z]:)/i, '').replace(/\//g, '\\');
        }
        return require(path.join(root, 'js', 'logic.js'));
    } catch (_) {
        return require('./js/logic.js');
    }
}
const logic = loadLogic();
const el = id => document.getElementById(id);
let child = null, cancelled = false, busy = false;
function host(code) {
    return new Promise((resolve, reject) => {
        if (!window.__adobe_cep__) return reject(new Error('Open this panel inside Premiere Pro.'));
        window.__adobe_cep__.evalScript(code, value => {
            try { const parsed = JSON.parse(value); if (parsed.error) reject(new Error(parsed.error)); else resolve(parsed.result); }
            catch (_) { reject(new Error('Premiere host communication failed: ' + value)); }
        });
    });
}
function findCommand() {
    const exe = process.platform === 'win32' ? 'fake-thermal.exe' : 'fake-thermal';
    const home = os.homedir();
    const dirs = (process.env.PATH || '').split(path.delimiter).concat([
        path.join(home, '.local', 'bin'), path.join(home, 'repos', 'fake-thermal', '.venv', 'bin'),
        path.join(home, 'fake-thermal', '.venv', 'bin'), path.join(home, 'fake-thermal', '.venv', 'Scripts'),
        path.join(home, 'venvs', 'fake-thermal', process.platform === 'win32' ? 'Scripts' : 'bin'), '/opt/homebrew/bin', '/usr/local/bin']);
    return dirs.map(dir => path.join(dir, exe)).find(file => fs.existsSync(file)) || 'fake-thermal';
}
logic.PALETTES.forEach(name => { const option = document.createElement('option'); option.value = option.textContent = name; el('palette').appendChild(option); });
el('palette').value = 'ironbow';
el('command').value = localStorage.getItem('fakeThermalCommand') || findCommand();
el('folder').value = localStorage.getItem('fakeThermalFolder') || '';
function render(settings, clip, output) {
    return new Promise((resolve, reject) => {
        let diagnostic = '';
        const parser = logic.progressParser(data => {
            if (data.total > 0) el('progress').value = Math.min(100, 100 * data.frame / data.total);
        }, message => { diagnostic = (diagnostic + '\n' + message).slice(-4000); el('status').textContent = diagnostic; });
        child = cp.spawn(settings.command, logic.buildArgs(settings, clip, output), {shell: false, windowsHide: true, detached: process.platform !== 'win32'});
        child.stdout.resume();
        child.stderr.setEncoding('utf8'); child.stderr.on('data', chunk => parser.push(chunk));
        child.on('error', error => { child = null; reject(error); });
        child.on('close', code => { parser.end(); child = null;
            if (cancelled) reject(new Error('Cancelled. Partial files may remain in the output folder.'));
            else if (code !== 0) reject(new Error(diagnostic || 'CLI failed with exit code ' + code));
            else resolve();
        });
    });
}
el('apply').onclick = async () => {
    if (busy) return;
    busy = true; cancelled = false; el('apply').disabled = true; el('cancel').disabled = false;
    try {
        const selection = await host('FakeThermal.selected()');
        const settings = {command: el('command').value.trim(), engine: el('engine').value, palette: el('palette').value,
            preview: el('preview').checked, usedRange: el('usedRange').checked};
        if (!settings.command) throw new Error('Enter the fake-thermal executable path.');
        let folder = el('folder').value.trim();
        if (!folder) {
            if (!selection.projectPath) throw new Error('Save the project or choose an output folder first.');
            folder = path.join(path.dirname(selection.projectPath), 'Fake Thermal Renders');
        } else if (!path.isAbsolute(folder)) {
            if (!selection.projectPath) throw new Error('Use an absolute output folder or save the project first.');
            folder = path.resolve(path.dirname(selection.projectPath), folder);
        }
        fs.mkdirSync(folder, {recursive: true});
        localStorage.setItem('fakeThermalCommand', settings.command);
        localStorage.setItem('fakeThermalFolder', el('folder').value.trim());
        for (let i = 0; i < selection.clips.length; i++) {
            if (cancelled) throw new Error('Cancelled.');
            const clip = selection.clips[i];
            let output;
            while (true) {
                output = logic.outputName(folder, clip, file => fs.existsSync(file) || fs.existsSync(file + '.lock'));
                try { fs.closeSync(fs.openSync(output + '.lock', 'wx')); break; }
                catch (error) { if (error.code !== 'EEXIST') throw error; }
            }
            try {
                el('progress').value = 0; el('status').textContent = 'Rendering ' + (i + 1) + '/' + selection.clips.length + '\n' + clip.mediaPath;
                await render(settings, clip, output);
                if (cancelled) throw new Error('Cancelled.');
                await host('FakeThermal.place(' + JSON.stringify({clip, output, usedRange: settings.usedRange}) + ')');
            } finally { fs.unlinkSync(output + '.lock'); }
        }
        el('progress').value = 100; el('status').textContent = 'Done. Renders placed above the selected clips.';
    } catch (error) { el('status').textContent = error.message; }
    finally { busy = false; el('apply').disabled = false; el('cancel').disabled = true; }
};
el('cancel').onclick = () => {
    cancelled = true;
    if (child) {
        if (process.platform === 'win32') cp.spawn('taskkill', ['/pid', String(child.pid), '/T', '/F']);
        else { try { process.kill(-child.pid, 'SIGKILL'); } catch (_) { child.kill('SIGKILL'); } }
    }
    el('status').textContent = 'Cancelling…';
};
