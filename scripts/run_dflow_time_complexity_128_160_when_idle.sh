#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-outputs/dflow_time_complexity_128_160_rerun}"
POLL_SECONDS="${POLL_SECONDS:-30}"
IDLE_POLLS_REQUIRED="${IDLE_POLLS_REQUIRED:-4}"
MAX_IDLE_MEMORY_MIB="${MAX_IDLE_MEMORY_MIB:-256}"
MAX_IDLE_UTILIZATION="${MAX_IDLE_UTILIZATION:-10}"

idle_polls_gpu0=0
idle_polls_gpu1=0

mkdir -p "${OUT_DIR}"
echo "Waiting for an idle GPU before starting the 128x128 and 160x160 timing rerun."

while true; do
  for gpu_id in 0 1; do
    IFS=, read -r memory_used utilization < <(
      nvidia-smi \
        --id="${gpu_id}" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"

    counter_name="idle_polls_gpu${gpu_id}"
    if (( memory_used <= MAX_IDLE_MEMORY_MIB && utilization <= MAX_IDLE_UTILIZATION )); then
      printf -v "${counter_name}" '%d' "$(( ${!counter_name} + 1 ))"
    else
      printf -v "${counter_name}" '%d' 0
    fi

    echo "gpu=${gpu_id} memory_mib=${memory_used} utilization=${utilization}% idle_polls=${!counter_name}/${IDLE_POLLS_REQUIRED}"
    if (( ${!counter_name} >= IDLE_POLLS_REQUIRED )); then
      echo "Starting timing rerun on cuda:${gpu_id}."
      exec env \
        OUT_DIR="${OUT_DIR}" \
        GPU_ID="${gpu_id}" \
        RESOLUTIONS_TEXT="128 160" \
        ./scripts/run_dflow_time_complexity.sh
    fi
  done
  sleep "${POLL_SECONDS}"
done
