#!/usr/bin/env bash
# Local worker for the 200-epoch b8 FM UNet sweep (see modal_train_fm.py).
# Workers share one job list, claiming jobs atomically via mkdir, and resume
# each run from its latest EMA checkpoint below 200 epochs.
#
# Usage: run_e200_local_queue.sh <gpu-index>   (start several per GPU if memory allows)
set -u
cd "$(dirname "$0")/.."
GPU="$1"
EPOCHS=200
CLAIMS=outputs/fm_unet/.e200_claims
mkdir -p "$CLAIMS"

# Longest remaining first, so the slowest models never start last.
JOBS="250:1 250:2 250:3 250:4 200:1 200:2 200:3 200:4 160:3 160:4 160:1 160:2
      128:4 128:1 128:2 128:3 64:4 96:1 96:2 96:3 96:4"

config_for() {
  case "$1" in
    64)  echo "1,2,2,2 64" ;;
    96)  echo "1,2,2,2,2 64" ;;
    128) echo "1,2,2,2,2 32" ;;
    160) echo "1,2,2,2,2 32" ;;
    200) echo "1,2,2,2,2,2 16" ;;
    250) echo "1,2,2,2,2,2 16" ;;
  esac
}

for JOB in $JOBS; do
  mkdir "$CLAIMS/${JOB/:/-}" 2>/dev/null || continue
  RES="${JOB%%:*}"; N="${JOB##*:}"
  OUT="outputs/fm_unet/circles-${RES}-noattn-ch16-b8-e200/n${N}"
  DATA="data/circles-${RES}-n${N}.pt"
  [ -f "$OUT/run_summary.json" ] && { echo "=== $JOB already complete"; continue; }
  until [ -f "$DATA" ]; do sleep 60; done
  read -r CH_MUL MICRO <<<"$(config_for "$RES")"
  ATT=$(sed 's/[0-9]\+/0/g' <<<"$CH_MUL")
  RESUME=()
  LAST=$(ls "$OUT"/checkpoints/epoch_*.pt 2>/dev/null | sed 's/.*epoch_0*\([0-9]*\)\.pt/\1/' \
         | awk -v e="$EPOCHS" '$1 < e' | sort -n | tail -1)
  if [ -n "$LAST" ] && [ -f "$OUT/loss_history.json" ]; then
    RESUME=(--resume-checkpoint "$OUT/checkpoints/epoch_$(printf %04d "$LAST").pt" --start-epoch "$LAST")
  fi
  mkdir -p "$OUT"
  echo "=== [gpu $GPU] $JOB start ${LAST:-0} $(date +%H:%M:%S) ==="
  TQDM_MININTERVAL=60 .venv/bin/python train_fm_unet.py \
    --data "$DATA" --out-dir "$OUT" \
    --epochs "$EPOCHS" --batch 512 --micro-batch "$MICRO" --device "cuda:${GPU}" \
    --ch 16 --ch-mul "$CH_MUL" --att-channels "$ATT" --no-mid-attn --groups 4 \
    "${RESUME[@]}" >> "$OUT/train.log" 2>&1
  echo "=== [gpu $GPU] $JOB finished rc=$? $(date +%H:%M:%S) ==="
done
echo "=== [gpu $GPU] worker done $(date +%H:%M:%S) ==="
