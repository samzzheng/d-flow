#!/usr/bin/env bash
# one (device, sample, generator) 64px job; resumes if a checkpoint exists.
set -u
cd "$(dirname "$0")/.."
DEV=$1; S=$2; G=$3
OUT=outputs/rk4_mechanism_2026_09_27/solves64/res64_s${S}_${G}
if [ -f "$OUT/results.json" ] && ! grep -q '"stop_reason": null' "$OUT/results.json"; then echo "skip $OUT"; exit 0; fi
RESUME=""; [ -f "$OUT/resume.pt" ] && RESUME="--resume"
echo "=== $(date -u +%FT%TZ) start s$S $G $DEV $RESUME"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/investigate_rk4_mechanism.py solve \
  --resolution 64 --sample "$S" --generator "$G" --device "$DEV" --threads 2 --max-steps 160 --out "$OUT" $RESUME > "$OUT.log" 2>&1
echo "=== $(date -u +%FT%TZ) exit $? s$S $G"
