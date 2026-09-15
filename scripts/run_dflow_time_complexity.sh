#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-outputs/dflow_time_complexity_tv_1e-4_lr_1p0_750_steps}"
GPU_ID="${GPU_ID:-0}"
RESOLUTIONS_TEXT="${RESOLUTIONS_TEXT:-64 96 128 160}"
read -r -a RESOLUTIONS <<< "${RESOLUTIONS_TEXT}"

python3 scripts/benchmark_dflow_blur_scalability.py \
  --resolutions "${RESOLUTIONS[@]}" \
  --n-circles 1 \
  --num-samples 5 \
  --device "cuda:${GPU_ID}" \
  --epoch 50 \
  --blur-sigma 5.0 \
  --euler-steps 5 \
  --lbfgs-lr 1.0 \
  --lbfgs-max-iter 20 \
  --lbfgs-history-size 25 \
  --line-search-fn none \
  --outer-steps 750 \
  --stop-loss 1e-12 \
  --tv-weight 1e-4 \
  --tv-eps 1e-6 \
  --record-iteration-timings \
  --timing-warmup-steps 10 \
  --out-dir "${OUT_DIR}" \
  --save-figures
