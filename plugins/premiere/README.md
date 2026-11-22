# Fake Thermal for Premiere Pro

A Node-enabled CEP panel for Premiere Pro 2022+ (CSXS 11). This is an
**offline renderer**, not a real-time or native per-frame effect. It runs your
local `fake-thermal` executable and imports the finished render. Synthetic heat
is not measured temperature.

## Install

Install Python 3.10+, ffmpeg/ffprobe on PATH, and the CLI from the repository:

```sh
python -m venv .venv
# macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

Turbo requires additional `diffusers`, `transformers`, `peft`, and
`huggingface_hub` packages, the SD-Turbo base model and trained Turbo weights;
follow the [repository's Turbo instructions](../../README.md). Downloads can
be large; run the CLI once outside Premiere to prepare models and dependencies.

Close Premiere, then run `sh plugins/premiere/install-mac.sh` on macOS, or
`& .\plugins\premiere\install-windows.ps1` in Windows PowerShell. If local
script execution is restricted, review the script and use a process-scoped
execution policy allowed by your organization. No administrator rights needed.
These copy into the user Adobe CEP extensions folder and enable unsigned
extensions via PlayerDebugMode for CSXS 9–12. This allows other unsigned CEP
extensions too: install only trusted extensions.

Restart Premiere and open **Window > Extensions (Legacy) > Fake Thermal**.
Uninstall with the same script plus `--uninstall` (macOS) or `-Uninstall`
(Windows). Debug settings are intentionally retained because other extensions
may use them; remove PlayerDebugMode manually if no longer needed.

## Use

1. Save the project and select one or more video clips on the active timeline.
2. Choose U-Net (fast) or Turbo (photographic), and a palette. Turbo draft preview
   trades fidelity for speed. Side-by-side is always off. The panel always passes
   `--no-audio`: renders are video-only, so placing them cannot overwrite existing
   dialogue or other audio tracks. Original timeline audio is left untouched.
3. Check the CLI path: auto-discovery searches PATH and common venv locations.
   If needed, enter the absolute `.venv/bin/fake-thermal` or
   `.venv\Scripts\fake-thermal.exe` path, **without quotes or extra arguments**.
   Premiere may not inherit your shell PATH; ffmpeg/ffprobe must also be visible
   to Premiere. The executable path is saved locally after Apply.
4. Leave the output folder empty for `Fake Thermal Renders` next to the saved
   project, or enter a folder. Apply processes selections sequentially. Progress
   comes from the CLI; model loading may take time before the first update.
5. Each unique render is imported into the **Fake Thermal** bin and placed on
   the first unlocked video track above its source with no overlap for that
   time range. A new track is created if needed. The source is not replaced.

Used-range rendering is on by default and seeks the source in/out. Turn it off
to render the whole source; insertion still uses the original clip's in point
and timeline duration. Still images render to PNG and hold for the selected
duration. Renders never intentionally reuse an existing output name. Cancel
kills the CLI process tree; partial output files may remain and are not imported.
Already imported clips in a multi-clip batch are retained.

Nested sequences, offline media, speed changes/reverse and audio-only selections
are unsupported. Source effects, transitions, scaling, proxies, color management
and other timeline processing are not baked into the render. Do not edit or
switch sequences during a render. Track creation uses Premiere's undocumented
QE API; if unavailable, add an empty video track above manually. Errors leave
completed renders on disk so they can be imported manually. This extension
requires a Premiere version with CEP support; it has no UXP fallback.

## Development

`node --test plugins/premiere/test` tests the CEP-independent argument, naming,
and progress logic. Host integration and installers require actual macOS or
Windows Premiere testing; Node syntax checks do not validate Premiere APIs.
