#!/bin/sh
# Overnight pipeline: when the v2 run exits, integrate the 512px rescued
# video pairs, then launch the v3 photorealism run on the GPU.
set -e
echo "waiting for the v2 run (outputs/v2) to release the GPU..."
while pgrep -f 'train.py --device mps.*--output outputs/v2' > /dev/null; do
  sleep 60
done
echo "v2 finished at $(date)"

.venv/bin/python - <<'EOF'
import json
import shutil
from pathlib import Path

manifest_path = Path('data512/pairs-video/manifest.json')
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    suspect = 'video-msNEBxJE5PPDqenBM'
    copied = 0
    for entry in manifest['frames']:
        stem = entry['id']
        rgb = Path('data512/pairs-video/rgb') / f'{stem}.jpg'
        thermal = Path('data512/pairs-video/thermal') / f'{stem}.png'
        if (not rgb.exists() or entry['recording'] == suspect
                or not entry.get('accepted')):
            continue
        new = stem.replace('-', '_', 1)
        shutil.copy(str(rgb), str(Path('data512/pairs/train/rgb') / f'{new}.jpg'))
        shutil.copy(str(thermal), str(Path('data512/pairs/train/thermal') / f'{new}.png'))
        copied += 1
    print(f'integrated {copied} 512px video pairs into data512/pairs/train')
else:
    print('no 512px rescue manifest found; training on the three image sources')
EOF

echo "launching v3 (crop 448, 512px data) at $(date)"
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/train.py \
  --device mps --batch-size 12 --iters 20000 --crop 448 --data data512 \
  --frequency-alpha 1 --ema-decay 0.999 --conditioning \
  --output outputs/v3 > outputs_v3.log 2>&1
echo "v3 finished at $(date)"
grep CHECKPOINT outputs_v3.log | tail -n 2
