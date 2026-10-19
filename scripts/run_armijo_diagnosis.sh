#!/usr/bin/env bash
# Armijo-backtracking L-BFGS arm, same samples/seeds as run_optimizer_diagnosis.sh.
# 100 iterations = 100 gradient evaluations (+ cheap forward-only trials).
set -u
cd /home/samzheng/d-flow
for s in 3 6 0 10; do
  python3 scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py --device cuda:1 --samples 1 \
    --sample-offset $s --out-dir outputs/optimizer_diagnosis/armijo \
    --lbfgs-line-search armijo --outer-steps 100
done
echo "ARMIJO DONE"
