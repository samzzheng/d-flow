#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/dflow_sparse_ct_circles_gpu_100step_noise_sweep}"
SESSION_PREFIX="${SESSION_PREFIX:-dflow-ct-noise}"
LBFGS_LR="${LBFGS_LR:-0.1}"
ROLLBACK_POLICY="${ROLLBACK_POLICY:-factor}"

wait_for_idle_gpu() {
  local device="$1"
  local index="${device#cuda:}"
  while true; do
    local values
    values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits -i "${index}")"
    local memory_used="${values%%,*}"
    local utilization="${values##*,}"
    memory_used="${memory_used// /}"
    utilization="${utilization// /}"
    if [[ "${memory_used}" -le 1024 && "${utilization}" -le 10 ]]; then
      echo "$(date --iso-8601=seconds) ${device} is idle; starting queue"
      return
    fi
    echo "$(date --iso-8601=seconds) waiting for ${device}: ${memory_used} MiB, ${utilization}%"
    sleep 60
  done
}

run_noise() {
  local mode="$1"
  local device="$2"
  local tag="$3"
  local noise="$4"
  local samples="1"
  local resolutions=(128)
  if [[ "${mode}" == "full" ]]; then
    samples="5"
    resolutions=(128 160)
  fi
  local out_dir="${OUT_ROOT}/noise_${tag}"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" -u "${ROOT}/scripts/benchmark_dflow_sparse_ct_circles_cpu.py" \
    --resolutions "${resolutions[@]}" \
    --samples "${samples}" \
    --epoch 50 \
    --angles 15 \
    --noise-level "${noise}" \
    --euler-flow-steps 100 \
    --rk4-flow-steps 100 \
    --outer-steps 100 \
    --lbfgs-lr "${LBFGS_LR}" \
    --lbfgs-max-iter 5 \
    --history-size 100 \
    --rollback-factor 1.5 \
    --rollback-policy "${ROLLBACK_POLICY}" \
    --cpu-threads 8 \
    --device "${device}" \
    --activation-checkpointing \
    --out-dir "${out_dir}" \
    2>&1 | tee -a "${out_dir}/${mode}.log"
}

worker() {
  local mode="$1"
  local device="$2"
  local slot="$3"
  shift 3
  local session="${SESSION_PREFIX}-${mode}"
  SIGNAL_CHANNEL="${session}-${slot}"
  export SIGNAL_CHANNEL
  trap 'tmux wait-for -S "${SIGNAL_CHANNEL}"' EXIT
  if [[ "${SKIP_GPU_IDLE_WAIT:-0}" != "1" ]]; then
    wait_for_idle_gpu "${device}"
  else
    echo "$(date --iso-8601=seconds) idle check overridden for ${device}"
  fi
  while [[ "$#" -gt 0 ]]; do
    local tag="${1%%:*}"
    local noise="${1#*:}"
    run_noise "${mode}" "${device}" "${tag}" "${noise}"
    shift
  done
}

launch() {
  local mode="$1"
  if [[ "${mode}" != "pilot" && "${mode}" != "full" ]]; then
    echo "Usage: $0 pilot|full" >&2
    return 2
  fi
  local session="${SESSION_PREFIX}-${mode}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    return 1
  fi
  for resolution in 128 160; do
    test -f "${ROOT}/outputs/fm_unet/circles-${resolution}/n1/checkpoints/epoch_0050.pt"
    test -f "${ROOT}/data/circles-${resolution}-n1.pt"
  done
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader

  local expected_resolutions="128"
  local expected_samples="1"
  if [[ "${mode}" == "full" ]]; then
    expected_resolutions="128 160"
    expected_samples="5"
  fi
  tmux new-session -d -s "${session}" -n summary -c "${ROOT}" \
    "tmux wait-for '${session}-gpu0'; tmux wait-for '${session}-gpu1'; \
     '${PYTHON_BIN}' scripts/summarize_dflow_sparse_ct_noise_sweep.py \
       --root '${OUT_ROOT}' --expected-resolutions ${expected_resolutions} \
       --expected-samples '${expected_samples}' --require-complete; exec zsh"
  tmux set-window-option -t "${session}:summary" remain-on-exit on
  tmux new-window -d -t "${session}" -n gpu0 -c "${ROOT}" \
    "OUT_ROOT='${OUT_ROOT}' SESSION_PREFIX='${SESSION_PREFIX}' \
     LBFGS_LR='${LBFGS_LR}' ROLLBACK_POLICY='${ROLLBACK_POLICY}' \
     SKIP_GPU_IDLE_WAIT='${SKIP_GPU_IDLE_WAIT:-0}' \
     '$0' --worker '${mode}' cuda:0 gpu0 000:0.0 005:0.05"
  tmux new-window -d -t "${session}" -n gpu1 -c "${ROOT}" \
    "OUT_ROOT='${OUT_ROOT}' SESSION_PREFIX='${SESSION_PREFIX}' \
     LBFGS_LR='${LBFGS_LR}' ROLLBACK_POLICY='${ROLLBACK_POLICY}' \
     SKIP_GPU_IDLE_WAIT='${SKIP_GPU_IDLE_WAIT:-0}' \
     '$0' --worker '${mode}' cuda:1 gpu1 001:0.01 010:0.10"
  tmux set-window-option -t "${session}:gpu0" remain-on-exit on
  tmux set-window-option -t "${session}:gpu1" remain-on-exit on
  tmux list-windows -t "${session}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker "$@"
else
  launch "${1:-}"
fi
