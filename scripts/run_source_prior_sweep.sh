#!/usr/bin/env bash
# Source-prior strength sweep, 250px, 1% noise, matched-protocol seeds (2026-09-25).
# L-BFGS + Armijo backtracking (forward-only trials), step cap 0.01 sqrt(d),
# up to 300 iterations (stops early once two line searches fail in a row),
# L2 prior lam/2 ||z||^2 with lam in {0, 0.001, 0.01, 0.1, 1}; D-Flow and DLR.
#   usage: run_source_prior_sweep.sh <cuda:N> "<samples>"
set -u
cd /home/samzheng/d-flow
DEV=$1; SAMPLES=$2
for s in $SAMPLES; do
  for lam in 0 0.001 0.01 0.1 1; do
    if [ "$lam" = 0 ]; then PRIOR="--source-prior none"; else PRIOR="--source-prior l2 --source-prior-lambda $lam"; fi
    python3 scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py --device $DEV --samples 1 \
      --sample-offset $s --out-dir outputs/source_prior_sweep/lam$lam/sample$s \
      --lbfgs-line-search armijo --armijo-step-cap 0.01 --outer-steps 300 $PRIOR
  done
done
echo "SWEEP $DEV [$SAMPLES] DONE"
