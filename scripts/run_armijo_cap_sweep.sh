#!/usr/bin/env bash
# Step-cap sweep for Armijo L-BFGS at 250px, 1% noise, matched-protocol seeds.
# Cap = fraction of sqrt(d) = 250. No-cap baseline: outputs/optimizer_diagnosis/armijo.
set -u
cd /home/samzheng/d-flow
for s in 3 6; do
  for cap in 0.1 0.03 0.3 0.01; do
    python3 scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py --device cuda:1 --samples 1 \
      --sample-offset $s --out-dir outputs/optimizer_diagnosis/armijo_cap$cap \
      --lbfgs-line-search armijo --armijo-step-cap $cap --outer-steps 100
  done
done
echo "CAP SWEEP DONE"
