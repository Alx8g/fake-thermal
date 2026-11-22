#!/bin/sh
# 3k-iteration ablation arms for frequency loss and weight EMA (fp16 for speed;
# the precision confound is identical across arms).
set -e
mkdir -p outputs/ablation
run() {
  name=$1; shift
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/train.py \
    --device mps --batch-size 16 --iters 3000 --precision fp16 \
    --output "outputs/ablation/$name" "$@" > "outputs/ablation/$name.log" 2>&1
  grep 'CHECKPOINT iter=3000' "outputs/ablation/$name.log"
}
run base
run ffl --frequency-alpha 1
run ema --ema-decay 0.999
run both --frequency-alpha 1 --ema-decay 0.999
echo ABLATION_DONE
