#!/usr/bin/env bash
# Euler vs RK4 noise test rerun (2026-09-26): 250px, 1% noise, 30 samples,
# same model/seeds as outputs/euler_rk4_250_noise001_cpu, both 100 flow steps,
# L-BFGS + Armijo (forward-only trials with the run's own integrator),
# step cap 0.01 sqrt(d), no source prior, up to 300 iterations (stops once two
# line searches fail in a row).   usage: run_euler_rk4_armijo.sh <cuda:N> <offset> <count> <shard>
set -u
cd /home/samzheng/d-flow
python3 scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py --device $1 \
  --sample-offset $2 --samples $3 --methods forward_euler rk4 \
  --euler-flow-steps 100 --rk4-flow-steps 100 \
  --lbfgs-line-search armijo --armijo-step-cap 0.01 --outer-steps 300 \
  --out-dir outputs/euler_rk4_250_noise001_armijo/$4
echo "SHARD $4 DONE"
