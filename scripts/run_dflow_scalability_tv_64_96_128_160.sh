#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:1}"
OUT_DIR="${OUT_DIR:-outputs/dflow_scalability_tv_64_96_128_160}"
TV_WEIGHT="${TV_WEIGHT:-1e-5}"

python3 scripts/benchmark_dflow_blur_scalability.py \
  --resolutions 64 96 128 160 \
  --n-circles 1 \
  --num-samples 30 \
  --device "${DEVICE}" \
  --epoch 50 \
  --blur-sigma 5.0 \
  --euler-steps 5 \
  --lbfgs-lr 1.0 \
  --lbfgs-max-iter 20 \
  --lbfgs-history-size 25 \
  --outer-steps 300 \
  --stop-loss 1e-12 \
  --tv-weight "${TV_WEIGHT}" \
  --tv-eps 1e-6 \
  --out-dir "${OUT_DIR}" \
  --save-figures
