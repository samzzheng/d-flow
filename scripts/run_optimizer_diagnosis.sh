#!/usr/bin/env bash
# Fix tests for the DLR divergence at 250px, 1% noise (matched-protocol seeds).
# Arms, each run with both D-Flow (forward_euler) and DLR, ~100-110 gradient evals:
#   wolfe : L-BFGS + properly configured strong-Wolfe line search
#   adam  : Adam lr 0.05
#   l2    : L-BFGS no line search + lam/2 ||z||^2, lam=1.25 (~ old lam=1e-3 rescaled)
#   shell : L-BFGS no line search + lam/2 (||z||-sqrt(d))^2, lam=1.25
set -u
cd /home/samzheng/d-flow
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done; fi
B="python3 scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py --device cuda:1 --samples 1"
for s in 3 6 0 10; do
  $B --sample-offset $s --out-dir outputs/optimizer_diagnosis/wolfe \
     --lbfgs-line-search strong_wolfe --lbfgs-max-iter 5 --lbfgs-max-eval 100 --outer-steps 20
  $B --sample-offset $s --out-dir outputs/optimizer_diagnosis/adam \
     --optimizer adam --adam-lr 0.05 --outer-steps 110
  $B --sample-offset $s --out-dir outputs/optimizer_diagnosis/l2 \
     --source-prior l2 --source-prior-lambda 1.25 --outer-steps 20
  $B --sample-offset $s --out-dir outputs/optimizer_diagnosis/shell \
     --source-prior shell --source-prior-lambda 1.25 --outer-steps 20
done
echo "ALL ARMS DONE"
