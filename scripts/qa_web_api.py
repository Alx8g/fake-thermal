"""Drive the Gradio web UI's queue API exactly as the Run button does."""
import json
import sys
import time
import urllib.request
import uuid

BASE = 'http://127.0.0.1:7860/gradio_api'


def upload(path):
    boundary = uuid.uuid4().hex
    with open(path, 'rb') as f:
        payload = f.read()
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
            f'filename="{path.split("/")[-1]}"\r\nContent-Type: '
            f'application/octet-stream\r\n\r\n').encode() + payload + f'\r\n--{boundary}--\r\n'.encode()
    request = urllib.request.Request(f'{BASE}/upload', data=body, headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())[0]


def run(fn_data, timeout=900):
    session = uuid.uuid4().hex
    join = {'data': fn_data, 'fn_index': 0, 'session_hash': session,
            'trigger_id': 12, 'event_data': None}
    request = urllib.request.Request(f'{BASE}/queue/join', data=json.dumps(join).encode(),
                                     headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(request, timeout=30).read()
    stream = urllib.request.urlopen(f'{BASE}/queue/data?session_hash={session}', timeout=timeout)
    start = time.time()
    for line in stream:
        text = line.decode().strip()
        if not text.startswith('data:'):
            continue
        message = json.loads(text[5:])
        if message.get('msg') == 'process_completed':
            output = message['output']
            if output.get('error'):
                raise RuntimeError(output['error'])
            return output['data'], time.time() - start
        if time.time() - start > timeout:
            raise TimeoutError('queue wait exceeded')
    raise RuntimeError('stream ended without completion')


def check(label, upload_path, palette='ironbow', side=False, semantic=True,
          strength=.7, device='auto', contrast=True, smoothing=.65):
    handle = upload(upload_path)
    data = [{'path': handle, 'meta': {'_type': 'gradio.FileData'}}, palette, side,
            smoothing, False, semantic, strength, device, contrast]
    result, seconds = run(data)
    files = result[2] or []
    names = [f.get('orig_name', '?') for f in files]
    assert files, f'{label}: no output files'
    assert result[3], f'{label}: no timing status'
    print(f'PASS {label}: {result[3]} files={names} queue_s={seconds:.1f}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'video'):
        check('video default (semantic+contrast)', 'outputs/demo/kaist_day_input.mp4')
        check('video no-semantic arctic', 'outputs/demo/kaist_day_input.mp4',
              palette='arctic', semantic=False)
        check('video side-by-side rainbow no-contrast', 'outputs/demo/kaist_day_input.mp4',
              palette='rainbow', side=True, contrast=False)
    if which in ('all', 'image'):
        check('image white-hot', 'data/pairs/val/rgb/flir_06995.jpg', palette='white-hot')
    if which in ('all', 'badfile'):
        try:
            check('corrupt upload', 'outputs/eval.json')
            print('FAIL corrupt upload: unexpectedly succeeded')
        except RuntimeError as error:
            print(f'PASS corrupt upload rejected: {str(error)[:90]}')
