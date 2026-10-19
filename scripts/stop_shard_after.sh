#!/usr/bin/env bash
# Stop one Euler-vs-RK4 shard once its figure for sample N is written, so a
# GPU 0 shard can take the rest of its range without repeating work.
#   usage: stop_shard_after.sh <shard-log> <last-sample> <python-pid> <bash-pid>
LOG=$1; LAST=$(printf "%03d" $2); PY=$3; SH=$4
until grep -q "recon_sample_${LAST}.png" "$LOG" 2>/dev/null; do
  kill -0 $PY 2>/dev/null || exit 0; sleep 30; done
kill $SH $PY 2>/dev/null; echo "stopped $LOG after sample $2"
