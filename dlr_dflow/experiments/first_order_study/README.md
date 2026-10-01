# First-order study at 64×64

See [the completed pilot results and charts](results_summary.md).

The CPU pilot's short trajectories do not establish convergence. The follow-up
GPU study retunes for 200 updates and runs reconstructions and ranks for 500–2,000
updates, with objective-plateau checks after 500. See the
[live GPU accuracy report](results/gpu_convergence_20260930/report.md).

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m dlr_dflow.experiments.first_order_study.gpu_study
# Resume the same protocol and optimizer state after interruption:
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m dlr_dflow.experiments.first_order_study.gpu_study --resume
# Continue with two independent workers on each GPU:
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m dlr_dflow.experiments.first_order_study.gpu_study --resume --gpus 0 1 --workers-per-gpu 2
```

This follow-up has no four-hour cutoff. It shares GPU 0, limits its allocator to
15% of GPU memory, disables TF32, and saves CPU checkpoints every ten updates.
The image/model/noise/source inputs are copied from the pilot. DLR's device-local
random sketches differ from the CPU draws. Its factors are still recomputed at
every gradient with the same analytical alpha settings. An extra native Euler
forward makes the objective and terminal gradient identical across estimators
up to ordinary evaluation roundoff; fast sensitivity endpoint roundoff therefore
does not affect late Armijo comparisons. No timing comparisons are reported.

Two consecutive 100-update windows with less than 0.1% improvement in the best
objective trigger a plateau label after the minimum budget. Recent objective
oscillations above 5% receive a separate label. Stalls and the 2,000-update limit
are not called convergence. LBFGS gets one retry after clearing failed history;
the first-attempt multiplier is never repeated. Final exact tangent-gradient
norms are recorded. Source snapshots every 25 updates plus terminal states
support initial/midpoint/final diagnostics. Each run's optimizer state, full
metric history, and source snapshots are in its own `states/*.pt` checkpoint.

The parallel mode uses four persistent worker processes with one CPU thread each.
It resumes the original checkpoints in ten-update chunks, then runs held-out
reconstructions and rank sweeps concurrently after tuning. Workers own distinct
`shards/<run>/` outputs; only the coordinator updates `results.json`, the combined
report, and canonical `states/` files. A coordinator lock prevents two parallel
drivers from owning the same study. GPU assignments are recorded, and the same
source, optimizer state, and device-local initialization seed follow a run when
its next chunk moves between GPUs. The six diagnostic points also run in parallel.

This isolated CPU experiment compares exact D-Flow and fresh-factor DLR with GD+Armijo, Adam, and an LBFGS+Armijo control. It also runs a rank sweep and evaluates exact, independently truncated, DLR, and DLR-subspace gradients at shared sources.

The fixed protocol is circles-64 without attention, epoch 100, 100 Euler steps, 15-angle sparse CT, and Gaussian measurement noise normalized to 1% relative L2. DLR always uses analytical `alpha_t = 1 - t`, `trace_probes=0`, zero enrichment, initialization oversampling 8, and exactly zero terminal alpha. Factors are recomputed for every gradient. No production modules are modified.

From the repository root:

```sh
.venv/bin/python -m unittest dlr_dflow.experiments.first_order_study.test_study -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout --signal=TERM 14400s .venv/bin/python -m dlr_dflow.experiments.first_order_study.study --threads 2 --hours 4
```

The runner uses cumulative phase deadlines of 10, 40, 145, 190, and 235 minutes, reserving five minutes for reporting. Unused phase time therefore carries forward. The external timeout enforces the four-hour process limit. The primary runner uses CPU only. A smaller `--hours` proportionally reduces deadlines; it does not reduce the numerical protocol and may leave tuning incomplete. Use a new `--out` directory for each run.

Each optimizer receives the first-attempt multiplier `min(1, 1 / ||ambient_gradient||_1)` once in its lifetime. Every proposal is capped to Euclidean norm `0.01 sqrt(d)` after constructing the optimizer direction. Sphere runs project directions onto the tangent plane and evaluate Armijo on the retracted trial; Adam projects its preconditioned direction too. The LBFGS control stores actual displacements and differences of tangent gradients but is not a fully transported Riemannian LBFGS method. Its history length is 100.

Tuning uses validation image 0 and compares learning rates and first-order families at a common completed update count within each estimator/constraint. Held-out comparisons use test images 0–2 with paired initial Gaussian draws and noise. The initial draw is normalized only in sphere runs. Rank sweeps use test image 0 with optimizer family and rate selected from DLR validation, then frozen across ranks and the exact control. Failed searches are recorded and stop that trajectory; common-prefix and final errors are both reported.

`gradient_seconds` includes the Euler flow and gradient computation. `seconds_excluding_trials` subtracts all candidate forward solves, including Adam's single candidate, from optimizer wall time. `seconds` includes those candidate solves. Metrics and output I/O are excluded. Every history contains objective/gradient evaluation counts; all methods use the same native Euler objective without clipping.

The spectral reference is independent randomized block iteration on the exact discrete Jacobian, with two-column JVP/VJP batches. It checks both singular-triplet residuals, refinement stability of the truncated gradient, and the cutoff gap. Numerical certification is not a proof of finding the globally largest singular values. Each point gets a soft six-minute spectral allowance and the phase has a strict signal deadline. Partial or ambiguous references are labeled. Selected Strong Wolfe diagnostics check true derivatives of the search curve; they do not constitute an optimizer sweep.

Outputs are saved after each completed update under `results/`: `results.json`, `states.pt`, `inputs.pt`, `summary.json`, a report, and PNG figures. States include sources along every trajectory and optimizer state; interrupted updates do not replace the last committed source. Inputs, checkpoint/data hashes, actual source-file hashes, seeds, failures, and phase completion are recorded. Python/Torch artifacts should only be loaded from trusted runs.

Re-render without evaluating a model:

```sh
.venv/bin/python -m dlr_dflow.experiments.first_order_study.report PATH_TO_RESULTS
```

When GPU 0 is explicitly available, optional supplemental reference refinement can batch 17 directions without changing reconstruction runs or their timing. It disables TF32, checks CPU/GPU endpoint and gradient parity, and verifies final singular residuals against CPU products. The original pilot deadline still applies. This worker uses two CPU threads, so it can run alongside the two-thread primary runner within the four-thread cap:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m dlr_dflow.experiments.first_order_study.refine PATH_TO_RESULTS --device cuda:0 --both-constraints --output-name spectral_refinement_gpu.json
```

The refiner writes its own JSON and tensor factors without modifying primary results. Re-rendering selects only certified, CPU-verified supplemental references and saves `combined_results.json`, retaining the initial references for audit. Without GPU availability, `refine` defaults to CPU and unconstrained sources.

Results from three images are a pilot. Extra noise realizations at fixed sources are conditional gradient checks, not extra reconstruction samples. Ground-truth best image checkpoints are retrospective and are paired with their own residuals.
