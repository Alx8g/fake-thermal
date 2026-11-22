var FakeThermal = (function () {
    function json(value) {
        var parts = [], key, i;
        if (value === null) return 'null';
        if (typeof value === 'string') return '"' + value.replace(/[\\"\u0000-\u001f]/g, function (c) {
            return '\\u' + ('0000' + c.charCodeAt(0).toString(16)).slice(-4);
        }) + '"';
        if (typeof value !== 'object') return String(value);
        if (value instanceof Array) {
            for (i = 0; i < value.length; i++) parts.push(json(value[i]));
            return '[' + parts.join(',') + ']';
        }
        for (key in value) if (value.hasOwnProperty(key)) parts.push(json(key) + ':' + json(value[key]));
        return '{' + parts.join(',') + '}';
    }
    function reply(fn) {
        try { return json({result: fn()}); }
        catch (error) { return json({error: error.message || String(error)}); }
    }
    function sequence(id) {
        var seq = app.project.activeSequence;
        if (!seq) throw new Error('Open a sequence first.');
        if (id && String(seq.sequenceID) !== String(id)) throw new Error('The active sequence changed. Return to the original sequence.');
        return seq;
    }
    function selected() {
        return reply(function () {
            var seq = sequence(), clips = [], i, j, item, media, speed;
            for (i = 0; i < seq.videoTracks.numTracks; i++) {
                for (j = 0; j < seq.videoTracks[i].clips.numItems; j++) {
                    item = seq.videoTracks[i].clips[j];
                    if (!item.isSelected()) continue;
                    if (!item.projectItem || item.projectItem.isSequence()) throw new Error('Nested sequences are unsupported. Select source media clips.');
                    media = item.projectItem.getMediaPath();
                    if (!media) throw new Error('Selected clip has no source media path.');
                    speed = item.getSpeed();
                    if (Math.abs(speed - 1) > 0.00001 || item.isSpeedReversed()) throw new Error('Speed-adjusted clips are unsupported. Render and replace them first.');
                    clips.push({mediaPath: media, sourceIn: item.inPoint.seconds, sourceOut: item.outPoint.seconds,
                        timelineStart: item.start.seconds, timelineEnd: item.end.seconds, trackIndex: i,
                        sequenceId: String(seq.sequenceID), still: /\.(png|jpe?g|webp|tiff?|bmp|psd|tga|gif)$/i.test(media)});
                }
            }
            if (!clips.length) throw new Error('Select one or more video clips on the timeline.');
            return {clips: clips, projectPath: app.project.path};
        });
    }
    function place(data) {
        return reply(function () {
            var seq = sequence(data.clip.sequenceId), bin, i, j, track, occupied, item, imported;
            for (i = 0; i < app.project.rootItem.children.numItems; i++) {
                item = app.project.rootItem.children[i];
                if (item.name === 'Fake Thermal' && item.type === ProjectItemType.BIN) bin = item;
            }
            if (!bin) bin = app.project.rootItem.createBin('Fake Thermal');
            if (!app.project.importFiles([data.output], true, bin, false)) throw new Error('Could not import render. The rendered file is still on disk.');
            for (i = 0; i < bin.children.numItems; i++) {
                if (bin.children[i].getMediaPath() === data.output) imported = bin.children[i];
            }
            if (!imported) throw new Error('Imported render was not found.');
            for (i = data.clip.trackIndex + 1; ; i++) {
                if (i >= seq.videoTracks.numTracks) {
                    app.enableQE();
                    qe.project.getActiveSequence().addTracks(1);
                    if (i >= seq.videoTracks.numTracks) throw new Error('Could not create a video track. Add an empty track above the source, then retry.');
                }
                track = seq.videoTracks[i]; occupied = typeof track.isLocked === 'function' && track.isLocked();
                for (j = 0; j < track.clips.numItems; j++) {
                    item = track.clips[j];
                    if (item.start.seconds < data.clip.timelineEnd - 0.00001 && item.end.seconds > data.clip.timelineStart + 0.00001) occupied = true;
                }
                if (!occupied) break;
            }
            var sourceStart = data.usedRange || data.clip.still ? 0 : data.clip.sourceIn;
            imported.setInPoint(sourceStart, 4);
            imported.setOutPoint(sourceStart + data.clip.timelineEnd - data.clip.timelineStart, 4);
            if (track.overwriteClip(imported, String(data.clip.timelineStart)) === false) throw new Error('Could not place render on the timeline.');
            return true;
        });
    }
    return {selected: selected, place: place};
}());
