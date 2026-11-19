#!/bin/sh
# Morning battery: when the v4 run exits, export, then run the realism
# benchmark and the L1/LPIPS evaluation so results are ready on wake-up.
set -e
echo "waiting for the v4 run (outputs/v4)..."
while pgrep -f 'train.py --device mps.*--output outputs/v4' > /dev/null; do
  sleep 60
done
echo "v4 finished at $(date)"
grep CHECKPOINT outputs_v4.log | tail -n 3 > outputs/v4_result.txt
cat outputs/v4_result.txt

.venv/bin/python scripts/export_weights.py outputs/v4/ckpt/best.pt
.venv/bin/python scripts/evaluate.py outputs/v4/ckpt/best.pt --data data512 --device mps \
  --lpips --output outputs/eval_v4.json 2>&1 | tail -n 4
.venv/bin/python scripts/evaluate_realism.py --data data512 --images 48 --patches 8 \
  --output outputs/realism_v4.json 2>&1 | tail -n 3

.venv/bin/python - <<'EOF'
from pathlib import Path

import cv2
import numpy as np

from fake_thermal.data import normalize_thermal
from fake_thermal.infer import Predictor, apply_palette
from fake_thermal.semantic import guided_upsample
from fake_thermal.sensor import ThermalSensor

val = Path('data512/pairs/val')
predictor = Predictor(device='cpu', threads=4)
sensor = ThermalSensor()
stems = (sorted((val / 'rgb').glob('flir_*.jpg'))[:2]
         + sorted((val / 'rgb').glob('kaist_*.jpg'))[:2]
         + sorted((val / 'rgb').glob('llvip_*.jpg'))[:2])

def gray(heat):
    return cv2.cvtColor((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

rows = []
for stem in stems:
    rgb = cv2.cvtColor(cv2.imread(str(val / 'rgb' / f'{stem.stem}.jpg')), cv2.COLOR_BGR2RGB)
    real = normalize_thermal(cv2.imread(str(val / 'thermal' / f'{stem.stem}.png'),
                                        cv2.IMREAD_GRAYSCALE))
    fake = sensor.update(guided_upsample(predictor(rgb), rgb))
    row = np.concatenate([gray(real), gray(fake),
                          cv2.cvtColor(apply_palette(fake, 'ironbow'), cv2.COLOR_RGB2BGR)],
                          axis=1)
    rows.append(cv2.resize(row, (int(row.shape[1] * 288 / row.shape[0]), 288)))
cv2.imwrite('outputs/morning_v4.png', np.concatenate(rows, axis=0))
print('morning sheet: outputs/morning_v4.png')
EOF
echo "MORNING BATTERY DONE $(date)"
