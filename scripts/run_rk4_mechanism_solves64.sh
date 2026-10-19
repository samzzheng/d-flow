#!/usr/bin/env bash
# 64px attention-free version of the family-vs-accuracy trajectories for
# statistical power. usage: scripts/run_rk4_mechanism_solves64.sh <device> <sample> [...]
set -u
cd "$(dirname "$0")/.."
DEV=$1; shift
for S in "$@"; do
  for G in euler100 rk425 rk4100 euler400; do
    OUT=outputs/rk4_mechanism_2026_09_27/solves64/res64_s${S}_${G}
    if [ -f "$OUT/results.json" ] && ! grep -q '"stop_reason": null' "$OUT/results.json"; then
      echo "skip $OUT (complete)"; continue
    fi
    RESUME=""; [ -f "$OUT/resume.pt" ] && RESUME="--resume"
    echo "=== $(date -u +%FT%TZ) sample $S $G on $DEV $RESUME"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/investigate_rk4_mechanism.py solve \
      --resolution 64 --sample "$S" --generator "$G" --device "$DEV" --threads 2 \
      --max-steps 160 --out "$OUT" $RESUME > "$OUT.log" 2>&1
    echo "=== $(date -u +%FT%TZ) exit $? sample $S $G"
  done
done
