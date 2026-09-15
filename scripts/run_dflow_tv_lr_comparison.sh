#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_BASE="${OUT_BASE:-outputs/dflow_tv_lr_comparison_750_steps}"
OUTER_STEPS="${OUTER_STEPS:-750}"
MAX_CONCURRENT_JOBS_PER_GPU="${MAX_CONCURRENT_JOBS_PER_GPU:-8}"

mkdir -p "${OUT_BASE}"

OUT_ROOT="${OUT_BASE}/lr_0.3" \
GPU_IDS=0 \
LINE_SEARCH_FN=none \
LBFGS_LR=0.3 \
OUTER_STEPS="${OUTER_STEPS}" \
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS_PER_GPU}" \
./scripts/run_dflow_tv_coarse_sweep.sh \
  >"${OUT_BASE}/lr_0.3_launcher.log" 2>&1 &
lr_03_pid=$!

OUT_ROOT="${OUT_BASE}/lr_0.1" \
GPU_IDS=1 \
LINE_SEARCH_FN=none \
LBFGS_LR=0.1 \
OUTER_STEPS="${OUTER_STEPS}" \
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS_PER_GPU}" \
./scripts/run_dflow_tv_coarse_sweep.sh \
  >"${OUT_BASE}/lr_0.1_launcher.log" 2>&1 &
lr_01_pid=$!

status=0
if ! wait "${lr_03_pid}"; then
  echo "LR 0.3 sweep failed; see ${OUT_BASE}/lr_0.3_launcher.log" >&2
  status=1
fi
if ! wait "${lr_01_pid}"; then
  echo "LR 0.1 sweep failed; see ${OUT_BASE}/lr_0.1_launcher.log" >&2
  status=1
fi

exit "${status}"
