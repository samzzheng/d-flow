#!/usr/bin/env bash
# Poll the d-flow-outputs Modal Volume and pull each sample's results/figures
# to outputs/modal_runs/<label>/ as soon as that sample finishes (has a
# non-empty samples.csv), instead of waiting for the whole sweep to complete.
# Safe to run alongside an in-progress `modal run modal_app.py::*_sweep` --
# it only reads from the Volume and never touches the running sweep.
#
#   usage: scripts/pull_modal_results.sh <remote-prefix> <local-label> <sample-list>
#   e.g.:  scripts/pull_modal_results.sh euler_rk4 euler_rk4_250_noise001_armijo \
#            5,6,7,8,10,11,12,13,14,22,23,24,26,27,28,29
set -u
cd "$(dirname "$0")/.."
REMOTE_PREFIX=$1   # e.g. "euler_rk4" (matches out_dir_template in modal_app.py)
LABEL=$2           # local folder under outputs/modal_runs/
IFS=',' read -ra SAMPLES <<< "$3"
DEST="outputs/modal_runs/$LABEL"
mkdir -p "$DEST"
declare -A DONE
echo "watching ${#SAMPLES[@]} samples: ${SAMPLES[*]}"
while true; do
  remaining=0
  for s in "${SAMPLES[@]}"; do
    [ -n "${DONE[$s]:-}" ] && continue
    remaining=$((remaining + 1))
    if modal volume ls d-flow-outputs "$REMOTE_PREFIX/sample_$s" 2>/dev/null | grep -q "samples.csv"; then
      echo "sample $s: finished, pulling..."
      if modal volume get --force d-flow-outputs "$REMOTE_PREFIX/sample_$s" "$DEST" 2>&1 | tail -3; then
        DONE[$s]=1
        remaining=$((remaining - 1))
        echo "sample $s: pulled to $DEST/sample_$s"
      fi
    fi
  done
  [ "$remaining" -eq 0 ] && { echo "all samples pulled."; break; }
  sleep 30
done
