# Fake Thermal for DaVinci Resolve

The **Fake Thermal** DCTL is an instant, editable look on a Color page node.
**Fake Thermal Render** is a separate Workspace script that runs the actual
fake-thermal model and imports new media into the project. Neither measures temperature.

## Add the on-clip look

This workflow uses the **DCTL Resolve FX** in **DaVinci Resolve Studio**.
It needs no PyTorch, model weights, or FFmpeg.

1. In Resolve, go to **Project Settings > Color Management > Lookup Tables >
   Open LUT Folder**. Copy [Fake Thermal.dctl](Fake%20Thermal.dctl) into that
   folder, or run the installer with the displayed folder path:

   `python resolve/install.py --dctl-only --lut-dir "/path/to/Resolve/LUT"`

2. In Project Settings click **Update Lists**, or restart Resolve if the effect
   doesn't appear. Open the **Color** page, select a clip, and drag **DCTL**
   from **Effects** onto a node. In the effect settings select **Fake Thermal**
   in the DCTL list.
3. Choose **Palette** (all seven CLI palettes are available), adjust **Contrast**
   and **Level** to map brightness to the colors, and set **Amount** to blend
   with the input. Zero Amount leaves the input unchanged.

The effect maps Rec.709-weighted RGB brightness to color. Put it where the node
receives normalised display-referred RGB (such as Rec.709), not unconverted
camera log or scene-linear HDR. It shares the model renderer's palette
interpolation, **not** its heat prediction, temporal processing, or auto contrast.

## Render the model look

1. Follow the [CLI setup](../README.md#install-cpu-python-310), including FFmpeg
   and FFprobe. The default *turbo* mode also needs the
   [Turbo dependencies and first-run model download](../README.md#photographic-engine---engine-turbo).
   Use `--mode unet` to avoid the SD-Turbo dependencies/download, or
   `--mode turbo-draft` for the CLI's faster draft preset.
2. In the Python environment where the CLI is installed, run:

   `python resolve/install.py --render-only --mode turbo --palette ironbow`

   The installer records this interpreter and the directories containing FFmpeg
   and FFprobe, so Resolve doesn't have to inherit your terminal's `PATH`.
   Resolve itself also needs its own supported 64-bit Python runtime for
   Workspace scripts. The script uses in-app scripting; there is no need to
   enable external or network scripting.
3. Restart Resolve. Open a project, select one or more **source videos in the
   Media Pool** (not in the timeline), then run **Workspace > Scripts >
   Fake Thermal Render**. The script uses the Media Pool selection API; if it
   reports that selection is unavailable, update Resolve Studio.
4. When rendering completes, a new `_thermal.mp4` is imported for each source.
   Add it to your timeline in place of the original when you're ready. The
   original media, edit, and grade are left untouched. This renders the **full
   source file**, not a timeline trim, speed change, compound clip, or grade.
   The script waits for the CLI; check **Workspace > Console** for output and
   errors. If Resolve cannot import the encoded H.264 MP4 on your platform, the
   finished file remains on disk.

### Installation options

`python resolve/install.py --help` lists options for macOS, Windows, and Linux.
Omit `--render-only` and `--dctl-only` to install both components. Use
`--lut-dir` for a custom LUT folder (some system defaults require write access)
and `--scripts-dir` for a custom Resolve `Scripts/Utility` directory.

| Platform | Default LUT folder | Default script folder |
| --- | --- | --- |
| macOS | `/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT` | `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility` |
| Windows | `%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support\LUT` | `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Utility` |
| Linux | `/opt/resolve/LUT` | `~/.local/share/DaVinciResolve/Fusion/Scripts/Utility` |

By default each render is saved beside its source, with a numeric suffix if
needed to avoid overwriting an existing file. Use `--output-dir` to write
elsewhere (for example, if your originals are on read-only media). Rerun the
installer with `--render-only` to change the settings in
`~/.fake-thermal-resolve.json`. To uninstall, remove the installed DCTL,
Workspace script and JSON settings file.
