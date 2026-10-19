#!/usr/bin/env bash
# Stage B reference: exact-gradient (checkpointed D-Flow) solves at the same N.
set -u
cd "$(dirname "$0")/.."
DEV=$1; shift
for S in "$@"; do
  for N in ${NLIST:-10 20 30 50 100}; do
    OUT=outputs/flow_steps_sweep_2026_09_28/solves/res250_s${S}_euler${N}_exact
    if [ -f "$OUT/results.json" ] && ! grep -q '"stop_reason": null' "$OUT/results.json"; then echo "skip $OUT"; continue; fi
    RESUME=""; [ -f "$OUT/resume.pt" ] && RESUME="--resume"
    echo "=== $(date -u +%FT%TZ) sample $S N=$N exact on $DEV $RESUME"
    OMP_NUM_THREADS=4 .venv/bin/python scripts/investigate_rk4_mechanism.py solve --resolution 250 \
      --sample "$S" --generator "euler$N" --gradient exact --device "$DEV" --threads 4 \
      --max-steps 200 --no-tf32 --out "$OUT" $RESUME > "$OUT.log" 2>&1
    echo "=== $(date -u +%FT%TZ) exit $? sample $S N=$N exact"
  done
done
