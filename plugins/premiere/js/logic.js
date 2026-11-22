'use strict';
const path = require('path');
const PALETTES = ['white-hot', 'black-hot', 'ironbow', 'rainbow', 'arctic', 'lava', 'glowbow'];
function buildArgs(settings, clip, output) {
    if (!clip.mediaPath) throw new Error('Clip has no source media');
    if (!PALETTES.includes(settings.palette)) throw new Error('Unknown palette');
    const args = [clip.mediaPath, '-o', output, '--engine', settings.engine,
        '--palette', settings.palette, '--progress-json', '--no-audio'];
    if (settings.preview && settings.engine === 'turbo') args.push('--turbo-preview');
    if (settings.usedRange && !clip.still) args.push('--start', String(clip.sourceIn),
        '--duration', String(clip.sourceOut - clip.sourceIn));
    return args;
}
function outputName(folder, clip, exists) {
    const stem = path.basename(clip.mediaPath, path.extname(clip.mediaPath)).replace(/[^\w.-]/g, '_');
    for (let n = 1; ; n++) {
        const candidate = path.join(folder, stem + '_thermal_' + n + (clip.still ? '.png' : '.mp4'));
        if (!exists(candidate)) return candidate;
    }
}
function progressParser(onProgress, onMessage) {
    let buffer = '';
    function line(value) {
        if (!value.trim()) return;
        try {
            const data = JSON.parse(value);
            if (data.done === true || (Number.isFinite(data.frame) && Number.isFinite(data.total))) {
                onProgress(data); return;
            }
        } catch (_) { /* Non-JSON diagnostics remain visible. */ }
        onMessage(value);
    }
    return {push(chunk) { buffer += chunk; const lines = buffer.split(/\r?\n/); buffer = lines.pop(); lines.forEach(line); },
        end() { line(buffer); buffer = ''; }};
}
module.exports = {PALETTES, buildArgs, outputName, progressParser};
