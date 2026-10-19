#!/usr/bin/env bash
# Run with CUDA_VISIBLE_DEVICES=<gpu> so --device cuda:0 maps to that GPU only.
# Per-iteration speed: D-Flow vs DLR-D-Flow (KSL) vs DLR-D-Flow (parallel BUG),
# 250px, 1% noise, matched seeds, Armijo L-BFGS cap 0.01 sqrt(d), 4 trace probes,
# 40 iterations each on samples 0 and 1.  Waits for the integrator test to end.
set -u
cd /home/samzheng/d-flow
while ps -eo cmd | grep -q "[t]est_parallel_bug_integrator.py"; do sleep 30; done
B="python3 scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py --device cuda:0 --samples 2 \
   --lbfgs-line-search armijo --armijo-step-cap 0.01 --outer-steps 40 --trace-probes 4"
$B --methods forward_euler dlr --dlr-integrator ksl \
   --out-dir outputs/dlr_integrator_speed/ksl
$B --methods dlr --dlr-integrator parallel \
   --out-dir outputs/dlr_integrator_speed/parallel
$B --methods dlr --dlr-integrator parallel --dlr-overlap \
   --out-dir outputs/dlr_integrator_speed/parallel_overlap
$B --methods dlr --dlr-integrator bug \
   --out-dir outputs/dlr_integrator_speed/bug
echo "INTEGRATOR SPEED DONE"
