#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-outputs/dflow_tv_coarse_sweep_no_line_search}"
GPU_ID="${GPU_ID:-1}"
LBFGS_LR="${LBFGS_LR:-1.0}"
OUTER_STEPS="${OUTER_STEPS:-750}"
TV_WEIGHTS_TEXT="${TV_WEIGHTS_TEXT:-0 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3}"
SHARD_IDS_TEXT="${SHARD_IDS_TEXT:-0 1}"
AGGREGATE_ON_COMPLETE="${AGGREGATE_ON_COMPLETE:-1}"
read -r -a TV_WEIGHTS <<< "${TV_WEIGHTS_TEXT}"
read -r -a SHARD_IDS <<< "${SHARD_IDS_TEXT}"
SAMPLE_SHARDS=("0 1 2" "3 4")

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/shards"
pids=()
labels=()

for weight in "${TV_WEIGHTS[@]}"; do
  for shard_index in "${SHARD_IDS[@]}"; do
    read -r -a sample_indices <<< "${SAMPLE_SHARDS[$shard_index]}"
    label="tv_${weight}_shard_${shard_index}"
    out_dir="${OUT_ROOT}/shards/${label}"
    log_path="${OUT_ROOT}/logs/${label}.log"
    echo "Starting ${label} on cuda:${GPU_ID}"
    python3 scripts/benchmark_dflow_blur_scalability.py \
      --resolutions 64 96 128 160 \
      --n-circles 1 \
      --sample-indices "${sample_indices[@]}" \
      --device "cuda:${GPU_ID}" \
      --epoch 50 \
      --blur-sigma 5.0 \
      --euler-steps 5 \
      --lbfgs-lr "${LBFGS_LR}" \
      --lbfgs-max-iter 20 \
      --lbfgs-history-size 25 \
      --line-search-fn none \
      --outer-steps "${OUTER_STEPS}" \
      --stop-loss 1e-12 \
      --tv-weight "${weight}" \
      --tv-eps 1e-6 \
      --out-dir "${out_dir}" \
      --save-figures \
      >"${log_path}" 2>&1 &
    pids+=("$!")
    labels+=("${label}")
  done
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${labels[$index]}"
  else
    echo "Failed ${labels[$index]} (see ${OUT_ROOT}/logs/${labels[$index]}.log)" >&2
    status=1
  fi
done

if [[ ${status} -eq 0 && ${AGGREGATE_ON_COMPLETE} -eq 1 ]]; then
  python3 scripts/aggregate_dflow_tv_sweep_shards.py \
    --root "${OUT_ROOT}" \
    --expected-samples-per-weight 20
fi

exit "${status}"
