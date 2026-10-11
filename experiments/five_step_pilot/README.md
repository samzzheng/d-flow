# Five-step Euler pilot

One held-out single-circle image (test index 0) at each of 64, 96, 128,
160, 200 and 250 pixels. Noiseless CT10 and Cartesian MRI4 give 12 paired
cases and 36 D-Flow/P-Flow/DLR D-Flow reconstructions.

Use the epoch-200 non-attention models, five Euler steps and up to 1,000
optimizer attempts. Preserve the production LBFGS/geometric Armijo settings,
first-attempt initial scaling, displacement cap `0.01 * sqrt(d)` and stop after
two consecutive failed searches. Report accepted updates separately from
attempts. No plateau stop is enabled before the full budget.

DLR keeps rank8, analytical `alpha=1-t`, zero terminal alpha, no trace probes
or enrichment, initialization oversampling8, and previous-V initialization.
Reconstruction code is reused unchanged from `experiments.ct_mri_sweep.run`.

`run init --out DIR` freezes manifests and source hashes. `run run --out DIR`
first runs two-update CT/MRI smoke jobs at 250px, then admits the production
cases sequentially on physical GPU1. Named `--shared-controller PID MIB`
reservations account for other studies' possible memory growth. Unknown
GPU1 occupants pause the worker. GPU0 is never used.

Invoke with `.venv/bin/python -m experiments.five_step_pilot.run ...`.
The output root contains `queue_status.json`, `summary.json` and `report.md`.
Refresh partial results with `report --out DIR`. Each production case saves
paired inputs, resumable optimizer/source states and a five-row comparison
figure at `production/r{resolution}_n1_{acquisition}/sample_000/comparison.png`
(also PDF). Smoke results remain separate from production.

Five Euler steps change the generator; increasing the optimization budget
does not remove discretization error. This pilot does not include new
100-step controls, and one sample per model cannot establish a general ranking.
