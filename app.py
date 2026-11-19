"""Local upload/preview UI. Run with .venv/bin/python app.py."""
import os
from pathlib import Path
import uuid

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import gradio as gr
from PIL import Image

from fake_thermal.infer import PALETTES, convert
from fake_thermal.device import DEVICES

ROOT = Path(__file__).resolve().parent


def run_media(upload, palette, side_by_side, smoothing, raw, semantic, semantic_strength,
              device, auto_contrast, flow,
              progress=gr.Progress(track_tqdm=True)):
    if not upload:
        raise gr.Error('Upload an image or video first.')
    try:
        with Image.open(upload) as image:
            image.verify()
        is_image = True
    except (OSError, ValueError):
        is_image = False
    directory = ROOT / 'outputs/ui' / uuid.uuid4().hex
    output = directory / ('thermal.png' if is_image else 'thermal.mp4')
    try:
        result = convert(upload, output, palette=palette, side_by_side=side_by_side,
                         smoothing=smoothing, raw=raw, threads=2, semantic=semantic,
                         semantic_strength=semantic_strength, device=device,
                         auto_contrast=auto_contrast, flow=flow)
    except Exception as error:
        raise gr.Error(str(error)) from error
    downloads = [str(output)] + ([result['raw']] if result['raw'] else [])
    return (None if is_image else str(output), str(output) if is_image else None,
            downloads, f'{result["frames"]} frames in {result["seconds"]:.1f}s '
            f'({result["fps"]:.2f} fps, {result["device"]})')


with gr.Blocks(title='fake-thermal') as app:
    gr.Markdown('# fake-thermal\nTurn RGB clips and photos into a thermal-camera look. '
                '**AI-generated relative heat, not measured temperatures.**')
    upload = gr.File(label='Drop a video or image', file_types=['video', 'image'], type='filepath')
    with gr.Row():
        palette = gr.Dropdown(list(PALETTES), value='ironbow', label='Palette')
        side = gr.Checkbox(label='RGB | thermal side by side')
        raw = gr.Checkbox(label='Also export grayscale heat map')
    device = gr.Dropdown(list(DEVICES), value='auto', label='Device (auto: CUDA > MPS > CPU)')
    auto_contrast = gr.Checkbox(value=True, label='Auto contrast (stable percentile display stretch)')
    smoothing = gr.Slider(0, .95, value=.65, step=.05,
                          label='Temporal smoothing (previous-frame weight; 0 = off)')
    flow = gr.Checkbox(value=True, label='Motion-compensated smoothing (carry heat along motion)')
    semantic = gr.Checkbox(value=True, label='Semantic heat prior (downloads model on first use)')
    semantic_strength = gr.Slider(0, 1, value=.7, step=.05, label='Semantic heat strength')
    gr.Markdown('Semantic mode assumes people/animals are hot and uses color cues for flames/lights. '
                'It can be wrong; it does not detect or measure actual heat.')
    button = gr.Button('Run', variant='primary')
    status = gr.Textbox(label='Timing', interactive=False)
    video = gr.Video(label='Video preview')
    image = gr.Image(label='Image preview', type='filepath')
    files = gr.File(label='Download results', file_count='multiple')
    button.click(run_media, [upload, palette, side, smoothing, raw, semantic, semantic_strength,
                             device, auto_contrast, flow],
                 [video, image, files, status], concurrency_limit=1)


if __name__ == '__main__':
    app.queue(default_concurrency_limit=1).launch(server_name='0.0.0.0', server_port=7860, share=False)
