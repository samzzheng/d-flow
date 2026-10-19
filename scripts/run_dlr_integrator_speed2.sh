#!/usr/bin/env bash
# Follow-up to run_dlr_integrator_speed.sh (which ran the parallel step WITH the
# benchmark's default 4 enrichment probes): the paper's p=0 variant, KSL with p=0,
# and a D-Flow cap check on sample 1.  Waits for the first queue to finish.
set -u
cd /home/samzheng/d-flow
while ps -eo cmd | grep -q "[r]un_dlr_integrator_speed.sh"; do sleep 30; done
B="python3 scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py --device cuda:0 \
   --lbfgs-line-search armijo --outer-steps 40 --trace-probes 4"
$B --samples 2 --armijo-step-cap 0.01 --methods dlr --dlr-integrator parallel --enrichment-probes 0 \
   --out-dir outputs/dlr_integrator_speed/parallel_p0
$B --samples 2 --armijo-step-cap 0.01 --methods dlr --dlr-integrator ksl --enrichment-probes 0 \
   --out-dir outputs/dlr_integrator_speed/ksl_p0
$B --samples 1 --sample-offset 1 --armijo-step-cap 0.03 --methods forward_euler \
   --out-dir outputs/dlr_integrator_speed/dflow_cap0.03_s1
echo "INTEGRATOR SPEED 2 DONE"
