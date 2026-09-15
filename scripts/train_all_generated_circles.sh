#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"

for res in 64 128 256; do
  for n in 1 2 3 4; do
    python train_fm_unet.py \
      --data "data/circles-${res}-n${n}.pt" \
      --out-dir "outputs/fm_unet/circles-${res}/n${n}" \
      --epochs 100 \
      --batch 512 \
      --lr 0.0002 \
      --wd 0.0001 \
      --ema-decay 0.999 \
      --euler-steps 200 \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}"
  done
done
