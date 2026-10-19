#!/usr/bin/env bash
# Resume the 250px 1%-noise Euler/RK4 sweep on GPU after the CPU run was cut short.
# Completed work is kept: only the 6 missing Euler samples and all 30 RK4 samples
# are (re)run.  Results land in the same root as the CPU phase, in per-job
# directories, so the figure builder picks up both phases together.
#
#   resume_euler_rk4_noise250_gpu.sh worker <gpu> <job> [<job> ...]   job = SAMPLE:METHOD
#   resume_euler_rk4_noise250_gpu.sh launch
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; PY="${ROOT}/.venv/bin/python"
OUT="${OUT:-${ROOT}/outputs/euler_rk4_250_noise001_cpu}"
LR="${LR:-1.0}"
# 3 per GPU pairs the 12 remaining jobs as one RK4 + one Euler per worker.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"

run_job() {
  local gpu="$1" sample="$2" method="$3"
  local dir="${OUT}/sample_$(printf '%03d' "$sample")_${method}"
  mkdir -p "$dir"
  echo "$(date --iso-8601=seconds) start s${sample} ${method} on cuda:${gpu}"
  "$PY" -u "${ROOT}/scripts/benchmark_dflow_sparse_ct_circles_cpu.py" \
    --resolutions 250 --samples 1 --sample-offset "$sample" --methods "$method" \
    --run-dir-template 'outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1' --epoch 100 \
    --angles 15 --noise-level 0.01 \
    --euler-flow-steps 100 --rk4-flow-steps 100 --outer-steps 100 \
    --lbfgs-lr "$LR" --lbfgs-max-iter 5 --history-size 100 \
    --rollback-factor 1.5 --rollback-policy nonfinite \
    --cpu-threads 2 --device "cuda:${gpu}" --activation-checkpointing \
    --out-dir "$dir"
  # Capture before any other command runs: a $(date) substitution resets $?.
  local rc=$?
  echo "$(date --iso-8601=seconds) done s${sample} ${method} rc=${rc}"
  if [ "$rc" -ne 0 ]; then
    echo "$(date --iso-8601=seconds) FAILED s${sample} ${method}; skipping figures" >&2
    return "$rc"
  fi
  "$PY" "${ROOT}/scripts/make_euler_rk4_noise_figures.py" \
      --root "$OUT" --sample "$sample" >> "${OUT}/figures.log" 2>&1
  echo "$(date --iso-8601=seconds) figures s${sample}"
}

worker() {
  local gpu="$1"; shift
  for job in "$@"; do run_job "$gpu" "${job%%:*}" "${job##*:}"; done
  echo "$(date --iso-8601=seconds) worker on cuda:${gpu} QUEUE COMPLETE"
}

launch() {
  mkdir -p "${OUT}/logs_gpu" "${OUT}/figures"
  # Job list: every RK4 sample, plus the Euler samples the CPU run never finished.
  local jobs=()
  for s in 24 25 26 27 28 29; do jobs+=("${s}:rk4"); done
  for s in 20 21 23 24 25 27; do jobs+=("${s}:forward_euler"); done
  local total_workers=$((WORKERS_PER_GPU * 2))
  echo "${#jobs[@]} jobs over ${total_workers} workers (${WORKERS_PER_GPU} per GPU)"
  for w in $(seq 0 $((total_workers - 1))); do
    local mine=() i
    for i in $(seq 0 $(( ${#jobs[@]} - 1 ))); do
      [ $(( i % total_workers )) -eq "$w" ] && mine+=("${jobs[$i]}")
    done
    # Alternate cards rather than splitting the list in half: the job list is
    # grouped by method, so contiguous blocks would put every RK4 job (~4x the
    # cost of a Euler job) on one card and leave the other idle.
    local gpu=$(( w % 2 ))
    echo "  worker ${w} -> cuda:${gpu}: ${mine[*]}"
    nohup "$0" worker "$gpu" "${mine[@]}" \
      > "${OUT}/logs_gpu/worker_${w}.log" 2>&1 &
    sleep 1
  done
  echo "launched"
}

case "${1:-}" in
  worker) shift; worker "$@" ;;
  launch) shift; launch "$@" ;;
  *) echo "usage: $0 {worker <gpu> <job>...|launch}" >&2; exit 2 ;;
esac
