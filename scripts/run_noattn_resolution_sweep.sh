#!/usr/bin/env bash
# Attention-free ch16 flow-matching sweep, depth scaled to hold a ~8x8 bottleneck
# at every resolution (so effective receptive field is resolution-invariant, per
# the cross-resolution comparison requirement).  One queue per GPU, run serially.
#
# Usage: run_noattn_resolution_sweep.sh <gpu-index> <res> [<res> ...]
set -u
cd "$(dirname "$0")/.."
GPU="$1"; shift
EPOCHS=100

config_for() {           # res -> "ch_mul micro_batch"
  case "$1" in
    # micro-batch chosen by probe: throughput falls with micro-batch size at
    # high resolution (cache) and rises at low resolution (launch overhead).
    # Memory is never the binding constraint here (peak 16.6 GiB at 250/64).
    64)  echo "1,2,2,2        64" ;;
    96)  echo "1,2,2,2,2      64" ;;
    128) echo "1,2,2,2,2      32" ;;
    160) echo "1,2,2,2,2      32" ;;
    200) echo "1,2,2,2,2,2    16" ;;
    250) echo "1,2,2,2,2,2    16" ;;
    *)   echo "UNKNOWN 0" ;;
  esac
}

for RES in "$@"; do
  read -r CH_MUL MICRO <<<"$(config_for "$RES")"
  if [ "$CH_MUL" = "UNKNOWN" ]; then echo "no config for $RES" >&2; continue; fi
  LEVELS=$(awk -F, '{print NF}' <<<"$CH_MUL")
  ATT=$(python3 -c "print(','.join(['0']*$LEVELS))")
  OUT="outputs/fm_unet/circles-${RES}-noattn-ch16-b8/n1"
  echo "=== [gpu $GPU] res $RES  ch_mul $CH_MUL  micro $MICRO  -> $OUT  $(date +%H:%M:%S) ==="
  .venv/bin/python train_fm_unet.py \
    --data "data/circles-${RES}-n1.pt" --out-dir "$OUT" \
    --epochs "$EPOCHS" --batch 512 --micro-batch "$MICRO" --device "cuda:${GPU}" \
    --ch 16 --ch-mul "$CH_MUL" --att-channels "$ATT" --no-mid-attn --groups 4
  echo "=== [gpu $GPU] res $RES finished rc=$? $(date +%H:%M:%S) ==="
done
echo "=== [gpu $GPU] QUEUE COMPLETE $(date +%H:%M:%S) ==="
