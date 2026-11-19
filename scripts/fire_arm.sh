#!/bin/sh
# Launch one v5 sweep arm on a rented Vast host. Run from the Mac:
#   sh scripts/fire_arm.sh <ssh-port> <ssh-host> <arm-name> <extra-args...>
set -e
PORT=$1; HOST=$2; ARM=$3; shift 3
ssh -i ~/.ssh/id_ed25519 -p $PORT root@$HOST "
  cd fake-thermal && git fetch -q origin && git checkout -q origin/v2-phases &&
  pip -q install -e . lpips 2>/dev/null;
  nohup python scripts/train.py --device cuda --data ~/data512 \
    --batch-size 16 --iters 20000 --crop 448 \
    --frequency-alpha 1 --ema-decay 0.999 --conditioning \
    --output outputs/$ARM $* > train_$ARM.log 2>&1 &
  echo LAUNCHED_$ARM"
