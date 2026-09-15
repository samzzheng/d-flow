#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="${LOG_DIR:-outputs/fm_unet/logs}"
EPOCHS="${EPOCHS:-50}"
mkdir -p "${LOG_DIR}"

for resolution in 64 128 256; do
  dataset="data/shepp-logan-${resolution}.pt"
  if [[ ! -f "${dataset}" ]]; then
    "${PYTHON}" -m dataset.shepp_logan \
      --resolutions "${resolution}" \
      --device cuda:0
  fi
done

run_one() {
  local resolution="$1"
  local device="$2"
  local micro_batch="$3"
  local output="outputs/fm_unet/shepp-logan-${resolution}"
  local log="${LOG_DIR}/shepp-logan-${resolution}.log"

  echo "[$(date --iso-8601=seconds)] start ${resolution}x${resolution} on ${device}" | tee "${log}"
  "${PYTHON}" train_fm_unet.py \
    --data "data/shepp-logan-${resolution}.pt" \
    --out-dir "${output}" \
    --epochs "${EPOCHS}" \
    --batch 512 \
    --lr 0.0002 \
    --wd 0.0001 \
    --ema-decay 0.999 \
    --euler-steps 200 \
    --num-workers 4 \
    --micro-batch "${micro_batch}" \
    --no-compile \
    --device "${device}" \
    2>&1 | tee -a "${log}"
  echo "[$(date --iso-8601=seconds)] done ${resolution}x${resolution} on ${device}" | tee -a "${log}"
}

# One sequential worker per GPU: no GPU ever receives two training processes.
(
  run_one 64 cuda:0 64
  run_one 256 cuda:0 16
) &
worker_gpu0=$!

run_one 128 cuda:1 16 &
worker_gpu1=$!

wait "${worker_gpu0}"
wait "${worker_gpu1}"
