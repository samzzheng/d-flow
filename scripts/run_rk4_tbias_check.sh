#!/usr/bin/env bash
# Euler-100 vs RK4-100 at 64px, epoch-200 t-biased (t_alpha=3) fine-tuned model.
# usage: scripts/run_rk4_tbias_check.sh <device> <sample> [<sample> ...]
set -u
cd "$(dirname "$0")/.."
DEV=$1; shift
for S in "$@"; do
  for G in euler100 rk4100; do
    OUT=outputs/rk4_mechanism_2026_09_27/tbias_check/res64_s${S}_${G}_tbias200
    if [ -f "$OUT/results.json" ] && grep -q '"stop_reason": "' "$OUT/results.json" && ! grep -q '"stop_reason": null' "$OUT/results.json"; then
      echo "skip $OUT (complete)"; continue
    fi
    RESUME=""; [ -f "$OUT/resume.pt" ] && RESUME="--resume"
    echo "=== $(date -u +%FT%TZ) sample $S $G tbias200 on $DEV $RESUME"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/investigate_rk4_mechanism.py solve \
      --resolution 64 --sample "$S" --generator "$G" --model-epoch 200 \
      --model-template "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1_tbias" \
      --device "$DEV" --threads 4 \
      --max-steps 160 --no-tf32 --out "$OUT" $RESUME > "$OUT.log" 2>&1
    echo "=== $(date -u +%FT%TZ) exit $? sample $S $G tbias200"
  done
done
