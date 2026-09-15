#!/usr/bin/env bash
set -euo pipefail

DEVICES_CSV="${DEVICES:-cuda:0,cuda:1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-outputs/fm_unet/logs}"
EPOCHS="${EPOCHS:-50}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MICRO_64="${MICRO_64:-64}"
MICRO_96="${MICRO_96:-256}"
MICRO_128="${MICRO_128:-16}"
MICRO_160="${MICRO_160:-64}"
MICRO_200="${MICRO_200:-40}"
MICRO_250="${MICRO_250:-16}"
RESOLUTIONS_CSV="${RESOLUTIONS:-128,160,64,96}"
N_CIRCLES_CSV="${N_CIRCLES:-1}"

IFS=',' read -r -a DEVICES_ARR <<< "${DEVICES_CSV}"
IFS=',' read -r -a RESOLUTIONS_ARR <<< "${RESOLUTIONS_CSV}"
IFS=',' read -r -a N_CIRCLES_ARR <<< "${N_CIRCLES_CSV}"
mkdir -p "${LOG_DIR}"

echo "GPU preflight (no unrelated processes will be stopped):"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

run_one() {
  local res="$1"
  local n="$2"
  local device="$3"
  local out_dir="outputs/fm_unet/circles-${res}/n${n}"
  local log="${LOG_DIR}/circles-${res}-n${n}.log"
  local micro_batch="${MICRO_64}"

  case "${res}" in
    64) micro_batch="${MICRO_64}" ;;
    96) micro_batch="${MICRO_96}" ;;
    128) micro_batch="${MICRO_128}" ;;
    160) micro_batch="${MICRO_160}" ;;
    200) micro_batch="${MICRO_200}" ;;
    250) micro_batch="${MICRO_250}" ;;
    *) echo "Unsupported resolution: ${res}" >&2; return 2 ;;
  esac

  echo "[$(date --iso-8601=seconds)] start circles-${res}-n${n} on ${device} epochs=${EPOCHS} micro_batch=${micro_batch}" | tee "${log}"
  "${PYTHON_BIN}" train_fm_unet.py \
    --data "data/circles-${res}-n${n}.pt" \
    --out-dir "${out_dir}" \
    --epochs "${EPOCHS}" \
    --batch 512 \
    --lr 0.0002 \
    --wd 0.0001 \
    --ema-decay 0.999 \
    --sample-steps 100 \
    --num-workers "${NUM_WORKERS}" \
    --micro-batch "${micro_batch}" \
    --no-compile \
    --device "${device}" \
    2>&1 | tee -a "${log}"
  echo "[$(date --iso-8601=seconds)] done circles-${res}-n${n} on ${device}" | tee -a "${log}"
}

is_complete() {
  local out_dir="$1"
  local final_ckpt
  printf -v final_ckpt "%s/checkpoints/epoch_%04d.pt" "${out_dir}" "${EPOCHS}"

  [[ -f "${final_ckpt}" ]] && "${PYTHON_BIN}" - \
    "${out_dir}/config.json" "${out_dir}/run_summary.json" "${EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
expected = int(sys.argv[3])
if not config_path.exists() or not summary_path.exists():
    raise SystemExit(1)
with config_path.open() as f:
    config = json.load(f)
with summary_path.open() as f:
    summary = json.load(f)
complete = (
    config.get("epochs") == expected
    and config.get("checkpoint_sampling", {}).get("steps") == 100
    and summary.get("status") == "complete"
    and summary.get("epochs_completed") == expected
    and summary.get("checkpoint_count") == 10
    and summary.get("evaluation_grid_count") == 10
)
raise SystemExit(0 if complete else 1)
PY
}

mark_skip() {
  local res="$1"
  local n="$2"
  local log="${LOG_DIR}/circles-${res}-n${n}.log"

  echo "[$(date --iso-8601=seconds)] skip circles-${res}-n${n}; found completed ${EPOCHS}-epoch run" | tee "${log}"
}

active=0
max_active="${#DEVICES_ARR[@]}"
device_idx=0
pending=()

for res in "${RESOLUTIONS_ARR[@]}"; do
  for n in "${N_CIRCLES_ARR[@]}"; do
    out_dir="outputs/fm_unet/circles-${res}/n${n}"
    if is_complete "${out_dir}"; then
      mark_skip "${res}" "${n}"
      continue
    fi
    pending+=("${res}:${n}")
  done
done

for item in "${pending[@]}"; do
    IFS=: read -r res n <<< "${item}"
    device="${DEVICES_ARR[$device_idx]}"
    run_one "${res}" "${n}" "${device}" &
    device_idx=$(((device_idx + 1) % max_active))
    active=$((active + 1))

    if [[ "${active}" -ge "${max_active}" ]]; then
      wait
      active=0
      device_idx=0
    fi
done

wait
