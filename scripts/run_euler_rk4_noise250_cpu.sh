#!/usr/bin/env bash
# One CPU worker per sample: Euler then RK4 at 250px with 1% noise, then figures.
#
#   run_euler_rk4_noise250_cpu.sh worker <sample>   # single sample + its figures
#   run_euler_rk4_noise250_cpu.sh launch [n]        # spawn n workers (default 30)
#
# Single-threaded workers: intra-op threading saturates at 16 threads for this
# 0.74M-parameter model at batch 1, while independent processes scale nearly
# linearly. 30x1 measured fastest (47 h) against 16x2 (55 h) and 8x4 (78 h);
# oversubscribing SMT to 64 threads is ~4x worse.
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
OUT="${OUT:-${ROOT}/outputs/euler_rk4_250_noise001_cpu}"
LR="${LR:-1.0}"

worker() {
  local sample="$1"
  local dir="${OUT}/sample_$(printf '%03d' "$sample")"
  mkdir -p "$dir"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  echo "$(date --iso-8601=seconds) sample ${sample} starting (lr ${LR})"
  "$PY" -u "${ROOT}/scripts/benchmark_dflow_sparse_ct_circles_cpu.py" \
    --resolutions 250 --samples 1 --sample-offset "$sample" \
    --run-dir-template 'outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1' --epoch 100 \
    --angles 15 --noise-level 0.01 \
    --euler-flow-steps 100 --rk4-flow-steps 100 --outer-steps 100 \
    --lbfgs-lr "$LR" --lbfgs-max-iter 5 --history-size 100 \
    --rollback-factor 1.5 --rollback-policy nonfinite \
    --cpu-threads 1 --device cpu --activation-checkpointing \
    --out-dir "$dir"
  local rc=$?
  echo "$(date --iso-8601=seconds) sample ${sample} finished rc=${rc}; building figures"
  # Pair figure for this sample, and refresh the running progress figure.
  # flock serialises the shared progress figure across the 30 workers.
  "$PY" "${ROOT}/scripts/make_euler_rk4_noise_figures.py" \
      --root "$OUT" --sample "$sample" >> "${OUT}/figures.log" 2>&1
  echo "$(date --iso-8601=seconds) sample ${sample} figures done"
}

launch() {
  local n="${1:-30}"
  mkdir -p "${OUT}/logs" "${OUT}/figures"
  echo "launching ${n} single-threaded workers -> ${OUT}"
  for sample in $(seq 0 $((n - 1))); do
    nohup "$0" worker "$sample" \
      > "${OUT}/logs/sample_$(printf '%03d' "$sample").log" 2>&1 &
    sleep 0.4
  done
  echo "launched; logs in ${OUT}/logs/"
}

case "${1:-}" in
  worker) shift; worker "$@" ;;
  launch) shift; launch "$@" ;;
  *) echo "usage: $0 {worker <sample>|launch [n]}" >&2; exit 2 ;;
esac
