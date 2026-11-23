'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {buildArgs, outputName, progressParser, PALETTES} = require('../js/logic');
const clip = {mediaPath: '/media/a b.mov', sourceIn: 3, sourceOut: 5};
const settings = {engine: 'unet', palette: 'ironbow', usedRange: true};
test('panel discovers per-user installs without shell PATH', () => {
    const script = fs.readFileSync(path.join(__dirname, '../js/panel.js'), 'utf8');
    const discovery = script.slice(script.indexOf('function findCommand()'), script.indexOf('logic.PALETTES.forEach'));
    for (const [platform, home, expected] of [
        ['darwin', '/Users/editor', '/Users/editor/Library/Application Support/fake-thermal/venv/bin/fake-thermal'],
        ['linux', '/home/editor', '/home/editor/.fake-thermal/venv/bin/fake-thermal'],
        ['win32', 'C:\\Users\\editor', 'C:\\Users\\editor\\AppData\\Local\\fake-thermal\\venv\\Scripts\\fake-thermal.exe']
    ]) {
        const context = {process: {platform, env: {}}, path: platform === 'win32' ? path.win32 : path.posix,
            os: {homedir: () => home}, fs: {existsSync: file => file === expected}};
        assert.equal(vm.runInNewContext(discovery + '\nfindCommand()', context), expected);
        context.fs.existsSync = () => false;
        assert.equal(vm.runInNewContext(discovery + '\nfindCommand()', context), 'fake-thermal');
    }
});
test('arguments preserve paths, trim used range, disable side-by-side', () => {
    const args = buildArgs(settings, clip, '/out/a.mp4');
    assert.deepEqual(args.slice(-4), ['--start', '3', '--duration', '2']);
    assert.equal(args[0], clip.mediaPath); assert.ok(!args.includes('--side-by-side'));
    assert.ok(args.includes('--progress-json')); assert.equal(PALETTES.length, 7);
    assert.ok(args.includes('--no-audio'));
});
test('whole sources and stills omit video range; turbo preview is gated', () => {
    assert.ok(!buildArgs({...settings, usedRange: false}, clip, 'out').includes('--start'));
    assert.ok(!buildArgs(settings, {...clip, still: true}, 'out').includes('--start'));
    assert.ok(buildArgs({...settings, engine: 'turbo', preview: true}, clip, 'out').includes('--turbo-preview'));
    assert.ok(!buildArgs({...settings, preview: true}, clip, 'out').includes('--turbo-preview'));
});
test('invalid media and palette give explicit errors', () => {
    assert.throws(() => buildArgs(settings, {}, 'out'), /source media/);
    assert.throws(() => buildArgs({...settings, palette: ''}, clip, 'out'), /palette/);
});
test('naming handles occupied and empty output folders, still suffix', () => {
    assert.equal(outputName('/out', clip, () => false), '/out/a_b_thermal_1.mp4');
    assert.equal(outputName('/out', {...clip, still: true}, name => name.endsWith('_1.png')), '/out/a_b_thermal_2.png');
});
test('progress handles chunks, CRLF, empty input, diagnostics, final unterminated line', () => {
    const events = [], messages = []; const parser = progressParser(x => events.push(x), x => messages.push(x));
    parser.push(''); parser.push('\n{"fra'); parser.push('me":1,"total":2}\r\nwarning\n{bad}\n');
    parser.push('{"done":true,"output":"a"}'); parser.end();
    assert.deepEqual(events, [{frame: 1, total: 2}, {done: true, output: 'a'}]);
    assert.deepEqual(messages, ['warning', '{bad}']);
});
test('panel resolves logic from CEP extension file URLs and paths without depending on cwd', () => {
    const bootstrap = fs.readFileSync(path.join(__dirname, '../js/panel.js'), 'utf8').split('const el =')[0];
    const cases = [
        ['darwin', 'file:///Users/editor/Library/Application%20Support/Fake%20Thermal', '/Users/editor/Library/Application Support/Fake Thermal/js/logic.js'],
        ['darwin', '/Users/editor/Fake%20Thermal', '/Users/editor/Fake Thermal/js/logic.js'],
        ['win32', 'file:///C:/Users/editor/Fake%20Thermal', 'C:\\Users\\editor\\Fake Thermal\\js\\logic.js'],
        ['win32', 'C:\\Users\\editor\\Fake%20Thermal', 'C:\\Users\\editor\\Fake Thermal\\js\\logic.js'],
        ['win32', 'file://server/share/Fake%20Thermal', '\\\\server\\share\\Fake Thermal\\js\\logic.js'],
        ['darwin', null, './js/logic.js']
    ];
    for (const [platform, root, expected] of cases) {
        let loaded;
        vm.runInNewContext(bootstrap, {
            URL, process: {platform}, window: root === null ? {} : {__adobe_cep__: {
                getSystemPath(key) { assert.equal(key, 'extension'); return root; }
            }},
            require(name) {
                if (name === 'path') return platform === 'win32' ? path.win32 : path.posix;
                if (['fs', 'os', 'child_process'].includes(name)) return {};
                loaded = name; return {};
            }
        });
        assert.equal(loaded, expected);
    }
});
