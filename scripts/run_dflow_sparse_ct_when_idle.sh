#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
LOG="${LOG:-outputs/dflow_sparse_ct_shepp_logan/benchmark.log}"
POLL_SECONDS="${POLL_SECONDS:-30}"
REQUIRED_FREE_MIB="${REQUIRED_FREE_MIB:-45000}"
MAX_UTIL="${MAX_UTIL:-10}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-3}"
mkdir -p "$(dirname "${LOG}")"

echo "[$(date --iso-8601=seconds)] waiting for an uncontended GPU" >> "${LOG}"
previous=""
idle_checks=0
while true; do
  chosen=""
  while IFS=, read -r index free_mib utilization; do
    index="${index// /}"
    free_mib="${free_mib// /}"
    utilization="${utilization// /}"
    if (( free_mib >= REQUIRED_FREE_MIB && utilization <= MAX_UTIL )); then
      chosen="${index}"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  if [[ -n "${chosen}" ]]; then
    if [[ "${chosen}" == "${previous}" ]]; then
      idle_checks=$((idle_checks + 1))
    else
      previous="${chosen}"
      idle_checks=1
    fi
  else
    previous=""
    idle_checks=0
  fi
  if [[ -n "${chosen}" && "${idle_checks}" -ge "${REQUIRED_IDLE_CHECKS}" ]]; then
    echo "[$(date --iso-8601=seconds)] launching on cuda:${chosen}" >> "${LOG}"
    exec "${PYTHON}" scripts/benchmark_dflow_sparse_ct.py \
      --device "cuda:${chosen}" \
      --resolutions 64 128 \
      --samples 5 \
      --angles 15 \
      --flow-steps 5 \
      --outer-steps 100 \
      --lbfgs-lr 0.3 \
      --lbfgs-max-iter 10 \
      --history-size 100 \
      --tv-weight 0 \
      --stop-mse 1e-12 \
      --out-dir outputs/dflow_sparse_ct_shepp_logan \
      >> "${LOG}" 2>&1
  fi
  sleep "${POLL_SECONDS}"
done
