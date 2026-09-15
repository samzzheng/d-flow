#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-outputs/dflow_tv_coarse_sweep}"
GPU_IDS_TEXT="${GPU_IDS:-0 1}"
LINE_SEARCH_FN="${LINE_SEARCH_FN:-strong_wolfe}"
LBFGS_LR="${LBFGS_LR:-1.0}"
OUTER_STEPS="${OUTER_STEPS:-750}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-8}"
TV_WEIGHTS_TEXT="${TV_WEIGHTS_TEXT:-0 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3}"
read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_TEXT}"
read -r -a TV_WEIGHTS <<< "${TV_WEIGHTS_TEXT}"

if [[ ${#GPU_IDS_ARRAY[@]} -eq 0 ]]; then
  echo "GPU_IDS must contain at least one CUDA device index" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs"

pids=()
labels=()
status=0

wait_for_active_jobs() {
  for active_index in "${!pids[@]}"; do
    if wait "${pids[$active_index]}"; then
      echo "Completed ${labels[$active_index]}"
    else
      echo "Failed ${labels[$active_index]} (see ${OUT_ROOT}/logs/${labels[$active_index]}.log)" >&2
      status=1
    fi
  done
  pids=()
  labels=()
}

for index in "${!TV_WEIGHTS[@]}"; do
  weight="${TV_WEIGHTS[$index]}"
  gpu="${GPU_IDS_ARRAY[$((index % ${#GPU_IDS_ARRAY[@]}))]}"
  label="tv_${weight}"
  out_dir="${OUT_ROOT}/${label}"
  log_path="${OUT_ROOT}/logs/${label}.log"

  echo "Starting ${label} on cuda:${gpu}; lr=${LBFGS_LR}; line_search=${LINE_SEARCH_FN}; log=${log_path}"
  python3 scripts/benchmark_dflow_blur_scalability.py \
    --resolutions 64 96 128 160 \
    --n-circles 1 \
    --num-samples 5 \
    --device "cuda:${gpu}" \
    --epoch 50 \
    --blur-sigma 5.0 \
    --euler-steps 5 \
    --lbfgs-lr "${LBFGS_LR}" \
    --lbfgs-max-iter 20 \
    --lbfgs-history-size 25 \
    --line-search-fn "${LINE_SEARCH_FN}" \
    --outer-steps "${OUTER_STEPS}" \
    --stop-loss 1e-12 \
    --tv-weight "${weight}" \
    --tv-eps 1e-6 \
    --out-dir "${out_dir}" \
    --save-figures \
    >"${log_path}" 2>&1 &
  pids+=("$!")
  labels+=("${label}")

  if [[ ${#pids[@]} -ge ${MAX_CONCURRENT_JOBS} ]]; then
    wait_for_active_jobs
  fi
done

wait_for_active_jobs

exit "${status}"
