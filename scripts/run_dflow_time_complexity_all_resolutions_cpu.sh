#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-outputs/dflow_time_complexity_all_resolutions_cpu_sample_0}"
CPU_THREADS="${CPU_THREADS:-32}"

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"

python3 scripts/benchmark_dflow_blur_scalability.py \
  --resolutions 64 96 128 160 \
  --n-circles 1 \
  --sample-indices 0 \
  --device cpu \
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
