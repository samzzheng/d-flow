#!/usr/bin/env bash
# Remaining optimizer-diagnosis runs split across GPUs (2026-09-25).
#   usage: run_optimizer_queue.sh <cuda:N> <out-root> "<samples>" "<arms>" [wait-pids...]
# arms: wolfe adam l2 shell armijo. Each GPU writes to its own out-root: two
# processes rewriting one samples.csv would drop each other's rows.
set -u
cd /home/samzheng/d-flow
DEV=$1; ROOT=$2; SAMPLES=$3; ARMS=$4; shift 4
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
B="python3 scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py --device $DEV --samples 1"
declare -A FLAGS=(
  [wolfe]="--lbfgs-line-search strong_wolfe --lbfgs-max-iter 5 --lbfgs-max-eval 100 --outer-steps 20"
  [adam]="--optimizer adam --adam-lr 0.05 --outer-steps 110"
  [l2]="--source-prior l2 --source-prior-lambda 1.25 --outer-steps 20"
  [shell]="--source-prior shell --source-prior-lambda 1.25 --outer-steps 20"
  [armijo]="--lbfgs-line-search armijo --outer-steps 100"
)
for s in $SAMPLES; do
  for a in $ARMS; do
    $B --sample-offset $s --out-dir $ROOT/$a ${FLAGS[$a]}
  done
done
echo "QUEUE $DEV DONE"
