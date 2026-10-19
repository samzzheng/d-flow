#!/usr/bin/env bash
# Flow-step sweep, stage B: DLR-driven capped Armijo L-BFGS reconstructions at
# 250px for several Euler step counts. usage: run_flow_steps_sweep.sh <device> <sample>...
set -u
cd "$(dirname "$0")/.."
DEV=$1; shift
for S in "$@"; do
  for N in ${NLIST:-10 20 30 50 100}; do
    OUT=outputs/flow_steps_sweep_2026_09_28/solves/res250_s${S}_euler${N}_dlr
    if [ -f "$OUT/results.json" ] && ! grep -q '"stop_reason": null' "$OUT/results.json"; then echo "skip $OUT"; continue; fi
    RESUME=""; [ -f "$OUT/resume.pt" ] && RESUME="--resume"
    echo "=== $(date -u +%FT%TZ) sample $S N=$N on $DEV $RESUME"
    OMP_NUM_THREADS=4 .venv/bin/python scripts/investigate_rk4_mechanism.py solve --resolution 250 \
      --sample "$S" --generator "euler$N" --gradient dlr --device "$DEV" --threads 4 \
      --max-steps 200 --no-tf32 --out "$OUT" $RESUME > "$OUT.log" 2>&1
    echo "=== $(date -u +%FT%TZ) exit $? sample $S N=$N"
  done
done
