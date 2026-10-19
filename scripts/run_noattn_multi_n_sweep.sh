#!/usr/bin/env bash
# Attention-free ch16 flow-matching sweep over (circle-count, resolution) pairs
# on a single GPU, run serially. Mirrors run_noattn_resolution_sweep.sh but
# takes explicit res:n pairs so multiple circle counts can be queued.
#
# Usage: run_noattn_multi_n_sweep.sh <gpu-index> <res:n> [<res:n> ...]
set -u
cd "$(dirname "$0")/.."
GPU="$1"; shift
EPOCHS=100

config_for() {           # res -> "ch_mul micro_batch"
  case "$1" in
    64)  echo "1,2,2,2        64" ;;
    96)  echo "1,2,2,2,2      64" ;;
    128) echo "1,2,2,2,2      32" ;;
    160) echo "1,2,2,2,2      32" ;;
    200) echo "1,2,2,2,2,2    16" ;;
    250) echo "1,2,2,2,2,2    16" ;;
    *)   echo "UNKNOWN 0" ;;
  esac
}

for PAIR in "$@"; do
  RES="${PAIR%%:*}"; N="${PAIR##*:}"
  read -r CH_MUL MICRO <<<"$(config_for "$RES")"
  if [ "$CH_MUL" = "UNKNOWN" ]; then echo "no config for $RES" >&2; continue; fi
  LEVELS=$(awk -F, '{print NF}' <<<"$CH_MUL")
  ATT=$(python3 -c "print(','.join(['0']*$LEVELS))")
  OUT="outputs/fm_unet/circles-${RES}-noattn-ch16-b8/n${N}"
  echo "=== [gpu $GPU] res $RES n $N  ch_mul $CH_MUL  micro $MICRO  -> $OUT  $(date +%H:%M:%S) ==="
  .venv/bin/python train_fm_unet.py \
    --data "data/circles-${RES}-n${N}.pt" --out-dir "$OUT" \
    --epochs "$EPOCHS" --batch 512 --micro-batch "$MICRO" --device "cuda:${GPU}" \
    --ch 16 --ch-mul "$CH_MUL" --att-channels "$ATT" --no-mid-attn --groups 4
  echo "=== [gpu $GPU] res $RES n $N finished rc=$? $(date +%H:%M:%S) ==="
done
echo "=== [gpu $GPU] QUEUE COMPLETE $(date +%H:%M:%S) ==="
