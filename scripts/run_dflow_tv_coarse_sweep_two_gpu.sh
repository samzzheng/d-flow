#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-outputs/dflow_tv_coarse_sweep_no_line_search}"
LBFGS_LR="${LBFGS_LR:-1.0}"
OUTER_STEPS="${OUTER_STEPS:-750}"
mkdir -p "${OUT_ROOT}"

OUT_ROOT="${OUT_ROOT}" \
GPU_ID=1 \
LBFGS_LR="${LBFGS_LR}" \
OUTER_STEPS="${OUTER_STEPS}" \
SHARD_IDS_TEXT=0 \
AGGREGATE_ON_COMPLETE=0 \
./scripts/run_dflow_tv_coarse_sweep_16.sh \
  >"${OUT_ROOT}/gpu1_shard0_launcher.log" 2>&1 &
gpu1_pid=$!

OUT_ROOT="${OUT_ROOT}" \
GPU_ID=0 \
LBFGS_LR="${LBFGS_LR}" \
OUTER_STEPS="${OUTER_STEPS}" \
SHARD_IDS_TEXT=1 \
AGGREGATE_ON_COMPLETE=0 \
./scripts/run_dflow_tv_coarse_sweep_16.sh \
  >"${OUT_ROOT}/gpu0_shard1_launcher.log" 2>&1 &
gpu0_pid=$!

status=0
if ! wait "${gpu1_pid}"; then
  echo "GPU 1 shard group failed" >&2
  status=1
fi
if ! wait "${gpu0_pid}"; then
  echo "GPU 0 shard group failed" >&2
  status=1
fi

if [[ ${status} -eq 0 ]]; then
  python3 scripts/aggregate_dflow_tv_sweep_shards.py \
    --root "${OUT_ROOT}" \
    --expected-samples-per-weight 20
fi

exit "${status}"
