#!/usr/bin/env bash
# Family-vs-accuracy trajectories at 250px: for each sample run the four
# generators sequentially (cheap ones first) on one GPU, saving every accepted
# state. usage: scripts/run_rk4_mechanism_solves.sh <device> <sample> [<sample> ...]
set -u
cd "$(dirname "$0")/.."
DEV=$1; shift
for S in "$@"; do
  for G in euler100 euler400 rk4100 rk425; do
    OUT=outputs/rk4_mechanism_2026_09_27/solves/res250_s${S}_${G}
    if [ -f "$OUT/results.json" ] && grep -q '"stop_reason": "' "$OUT/results.json" && ! grep -q '"stop_reason": null' "$OUT/results.json"; then
      echo "skip $OUT (complete)"; continue
    fi
    RESUME=""; [ -f "$OUT/resume.pt" ] && RESUME="--resume"
    echo "=== $(date -u +%FT%TZ) sample $S $G on $DEV $RESUME"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/investigate_rk4_mechanism.py solve \
      --resolution 250 --sample "$S" --generator "$G" --device "$DEV" --threads 4 \
      --max-steps ${MAXSTEPS:-160} --no-tf32 --out "$OUT" $RESUME > "$OUT.log" 2>&1
    echo "=== $(date -u +%FT%TZ) exit $? sample $S $G"
  done
done
