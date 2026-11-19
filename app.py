"""Local upload/preview UI. Run with .venv/bin/python app.py."""
from pathlib import Path
import uuid

import gradio as gr
from PIL import Image

from fake_thermal.infer import PALETTES, convert

ROOT = Path(__file__).resolve().parent


def run_media(upload, palette, side_by_side, smoothing, raw, progress=gr.Progress(track_tqdm=True)):
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
                         smoothing=smoothing, raw=raw, threads=2)
    except Exception as error:
        raise gr.Error(str(error)) from error
    downloads = [str(output)] + ([result['raw']] if result['raw'] else [])
    return (None if is_image else str(output), str(output) if is_image else None,
            downloads, f'{result["frames"]} frames in {result["seconds"]:.1f}s ({result["fps"]:.2f} fps)')


with gr.Blocks(title='fake-thermal') as app:
    gr.Markdown('# fake-thermal\nTurn RGB clips and photos into a thermal-camera look. '
                '**AI-generated relative heat, not measured temperatures.**')
    upload = gr.File(label='Drop a video or image', file_types=['video', 'image'], type='filepath')
    with gr.Row():
        palette = gr.Dropdown(list(PALETTES), value='ironbow', label='Palette')
        side = gr.Checkbox(label='RGB | thermal side by side')
        raw = gr.Checkbox(label='Also export grayscale heat map')
    smoothing = gr.Slider(0, .95, value=.65, step=.05,
                          label='Temporal smoothing (previous-frame weight; 0 = off)')
    button = gr.Button('Run', variant='primary')
    status = gr.Textbox(label='Timing', interactive=False)
    video = gr.Video(label='Video preview')
    image = gr.Image(label='Image preview', type='filepath')
    files = gr.File(label='Download results', file_count='multiple')
    button.click(run_media, [upload, palette, side, smoothing, raw],
                 [video, image, files, status], concurrency_limit=1)


if __name__ == '__main__':
    app.queue(default_concurrency_limit=1).launch(server_name='0.0.0.0', server_port=7860, share=False)
