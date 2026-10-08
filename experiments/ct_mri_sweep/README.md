# Paired CT/MRI sweep

## Active reduced sweep (October 8 restart)

The active output directory is `outputs/ct_mri_sweep_geometric_v_20261008`.
All three methods now use geometric Armijo backtracking (halve rejected rates),
with history 100 and the existing curvature-pair rule. DLR replaces eight of
the width-16 initialization sketch columns with its previous terminal right
basis, retaining eight random columns. Its first gradient is entirely random;
every gradient still evolves fresh factors from flow time zero. The basis is
saved with optimizer checkpoints and restored on resume.

This sweep restarts from the original paired sources and noiseless observations.
The October 7 results and the interrupted CPU optimizer pilot are preserved in
their original directories. Their checkpoints must not be resumed under this
changed protocol. The new queue uses **GPU 1 only**; GPU 0 remains excluded.

`config_100_steps.json` selects resolutions 64/128/250, circle counts 1/2/3,
CT with 10 angles, Cartesian MRI at nominal 4x, and 30 paired samples:
18 acquisitions, 540 comparisons, and 1,620 method runs. Each method gets
**at most 100 optimizer attempts**, with the same LBFGS/Armijo settings below.
Two failed searches still stop a run as `stalled`; reaching 100 is `step_limit`,
not a claim of convergence. Measurements are now **noiseless**, with observed
`y = A(x_GT)` exactly. Both earlier noisy queues are paused and preserved;
their 1% measurement noise was carried over from the previous study and was
not the intended setting for this sweep. Their reconstruction results cannot
be converted to noiseless results by replotting.

The queue divides acquisitions into disjoint five-sample workers and packs
them onto GPU 1 using measured memory reservations for each resolution.
Every worker runs all three methods, including the uncheckpointed exact gradient;
its reservation covers the largest tape, persistent DLR graphs, and optimizer
state. A device margin and per-process allocator limit bound memory use.
GPU memory use and scheduler decisions are saved in `queue_history.jsonl`.
Sharing changes execution only: seeds, objective, gradient estimators, and
optimization settings are unchanged. Allocation capacity does not guarantee
linear throughput improvement with worker count.

Single-worker memory checks and the earlier **1%-noise** 100-update smoke study live in
`outputs/ct_mri_smoke_20261007/`. The smoke study uses test image 0 with the
three-circle model at each resolution, under both operators. It is separate
from the 30-sample study. Inspect its late objective/image-error histories
before interpreting 100 steps as sufficient for convergence.

That noisy smoke study **did not establish convergence within 100 updates**:
rank-8 DLR stalled in five of six cases and reached the limit with 41.91% image
error in the sixth. Exact D-Flow and P-Flow also had case-dependent failures.
Full histories, paired figures, and the late-progress assessment are in
`outputs/ct_mri_smoke_20261007/convergence100/report.md` and `convergence.pdf`.
The requested sweep therefore remains a fixed-budget comparison with explicit
failure counts, not a comparison of converged solutions.

Capacity checks passed with 17 simultaneous 64px workers and six 128px workers,
each exercising all three estimators. Peak aggregate VRAM was 40.8 and 44.7 GiB,
respectively. The 250px worker is limited to one per device; the scheduler fills
remaining capacity with smaller resolutions. The October 8 restart increases
reservations to 2816/8448/30720 MiB for 64/128/250 after measuring the new
worker's peaks, with a further 1024 MiB device margin and 768 MiB of each
reservation set aside for allocations outside PyTorch's allocator.

```sh
.venv/bin/python -m experiments.ct_mri_sweep.queue init \
  --out outputs/ct_mri_sweep_geometric_v_20261008 \
  --config experiments/ct_mri_sweep/config_100_steps.json
.venv/bin/python -m experiments.ct_mri_sweep.queue run \
  --out outputs/ct_mri_sweep_geometric_v_20261008 --gpus 1 --poll-seconds 10
```

## Original larger grid

Six resolutions (64, 96, 128, 160, 200, 250), four circle counts, four CT angle
counts (10, 20, 23, 180), and three MRI accelerations (4, 8, 24) give 168
acquisition configurations. Test images 0–29 give 5,040 paired cases and
15,120 individual D-Flow, P-Flow, and DLR D-Flow reconstructions.

The queue waits for **all 24 epoch-200 training summaries and checkpoints**.
It then runs one acquisition worker per available local GPU, preferring GPU 1.
GPU 0 is eligible only after its current external process leaves. An external
CUDA process appearing on an occupied device causes our worker to checkpoint
and pause; the queue resumes it when that device is free. No cloud GPU jobs are
created. The scheduler must remain running; logs and its PID are in the output
directory. A failed worker is recorded and excluded from automatic retries.

## Fixed protocol

- Non-attention ch16/b8 epoch-200 checkpoints from the existing training sweep.
- 100 Euler steps; float32 model/source, float64 objective reduction.
- Objective `0.5 * ||A G(z) - y||²`. No image clipping, extra prior, sphere
  projection, truth-based stopping, or truth-based checkpoint selection.
- Identical LBFGS/Armijo implementation for all methods: learning rate 1,
  history 100, `c1=1e-4`, at most 30 trials, displacement cap `0.01 sqrt(d)`.
  Rejected trial rates are halved geometrically.
  L1 initial scaling is used only on the first attempted update, including
  across restarts. Trials must strictly lower the canonical native Euler
  objective. P-Flow/DLR use their approximate gradients for the Armijo slope.
- P-Flow uses the recovered identity sensitivity proxy with this shared
  optimizer. Its original fixed-step spherical update remains available in
  `p_flow`; the sweep intentionally applies the same unconstrained optimization
  conditions to all three estimators.
- Fastest recorded DLR execution profile: rank 8, analytical alpha `1-t`,
  explicit `trace_probes=0`, terminal alpha exactly zero, no enrichment,
  initialization oversampling 8, stride 1, fresh factors at every gradient.
  Previous terminal right vectors seed eight sketch columns after the first
  gradient, with the other eight columns retaining their fixed random seed.
  Shared hand-written products, fused normalization/primal kernels, specialized
  algebra, full-step CUDA graphs, 6/4 derivative chunks, no DLR checkpointing.
- Accepted native Euler endpoints are cached for P-Flow/DLR terminal gradients;
  fused DLR endpoint roundoff cannot replace the line-search objective.
- Up to 2,000 attempted updates. Stop after two consecutive failed searches,
  or after at least 500 updates if objective improvement is below 0.1% in each
  of two consecutive 100-update windows. A stall or step limit is not labeled
  convergence. No noise-floor termination is used in this convergence study.
- Source/noise/sketch/mask seeds are paired across methods. Test image indices
  and source seeds are also shared across acquisitions for each model.
- Default noise is zero in both configuration files; no random measurement
  noise is drawn. Earlier immutable manifests retain their original 1% noise.
  Explicit nonzero settings remain supported: a Gaussian direction is rescaled
  to the requested relative norm, with complex MRI noise on acquired coefficients.

CT uses equally spaced angles over `[0,180)` degrees, with the existing
rotate-and-average linear operator. The 180-angle case is evaluated in chunks
to bound temporary memory. This is a simulated linear attenuation experiment.

MRI is a zero-phase, single-coil phantom experiment using a centered orthonormal
FFT and variable-density Cartesian lines. Exactly `round(N / acceleration)`
lines are acquired, including DC, with remaining lines sampled toward the
center. There is no fixed-width calibration region that would exceed the 24x
budget. Requested and actual acceleration, mask, and mask seed are saved.
For example, 64px/24x uses three lines, giving actual acceleration 21.33x.
The alternative `mri_pattern=points` uses `round(N² / acceleration)` coefficients.
Noise and error norms are complex-valued before display transforms.

## Queue and resume

From the repository root:

```sh
.venv/bin/python -m experiments.ct_mri_sweep.queue init --out outputs/ct_mri_sweep_20261007
.venv/bin/python -m experiments.ct_mri_sweep.queue run --out outputs/ct_mri_sweep_20261007 --gpus 1 0
.venv/bin/python -m experiments.ct_mri_sweep.queue status --out outputs/ct_mri_sweep_20261007
```

`init` freezes the configuration, source hashes, and job grid. A new protocol
requires a new output directory. A queue lock prevents duplicate controllers.
Send SIGTERM to the queue PID for a graceful pause; rerunning the same `run`
command resumes unfinished workers. Each method saves its source, complete
optimizer state and lifetime counters, history, measurements, checkpoint/test
hashes, and seeds every 25 updates and at termination. Finished methods are
skipped. Code/protocol changes are rejected on resume.

Each checkpoint gets a GPU preflight comparing graphed and eager DLR gradients
(1% tolerance) and fast/native endpoints (0.5% tolerance), with exact zero
terminal alpha. Failures stop that acquisition before reconstructions. These
checks validate execution agreement, not DLR approximation accuracy.

## Figures and outputs

Each sample directory contains paired inputs, one `.pt`/`.json` per method, and
`comparison.png`/`comparison.pdf`. Figures are refreshed as each method finishes
and contain all three final runs once complete: **5,040 complete comparisons**.
Columns are GT, D-Flow, P-Flow, DLR D-Flow. Rows are images, clean-reference
measurements, absolute image errors, residuals against observed measurements,
and **relative image error / measurement residual versus optimization step**.
All panels have the same square dimensions and aligned columns; colorbars have
their own column. Absolute image-error and residual maps use `cmocean.cm.amp`.
Each panel has a relative L2 percentage and each row shares its color limits.
MRI measurement displays use log magnitude; errors use the original complex
arrays. Method statuses and update counts are included. Histories retain the
per-update errors and residuals. Acquisition summaries record final image error
mean/std/median, update counts, elapsed time, and terminal status counts.

Use `python -m experiments.ct_mri_sweep.replot <output-directory-or-sample>`
to refresh saved PNG/PDF panels on the CPU. This uses the saved observations,
reconstructions, and histories unchanged. Thus an old noisy GT residual remains
1%, while the new noiseless GT residual is exactly zero. Curves stop at each
method's actual final attempt, including early stalls; they share axis limits.

## Runtime estimate

`runtime_estimate.json` uses historical RTX 6000 Ada timings for the same model
architecture and fastest DLR profile, scaled from 150 to 100 Euler steps. It
includes 1–2 native Armijo forward trials per update for every method and assumes
P-Flow reuses the accepted endpoint. These are scenarios, not measured completion
times for the new checkpoints or CT180/MRI operators.

| Mean updates per method | One dedicated GPU | Two dedicated GPUs |
| --- | --- | --- |
| 100 | 25–38 days | 13–19 days |
| 250 | 63–96 days | 32–48 days |
| 500 | 127–192 days | 63–96 days |
| 1,000 | 254–385 days | 127–192 days |

Training waits, external GPU occupancy, extra late Armijo backtracking, figure
and checkpoint IO, and acquisition overhead are additional. The 2,000-update
ceiling is an iteration bound, not a wall-clock bound. Previous 64px LBFGS runs
stalled at 150–985 updates for D-Flow and 81–204 for DLR; there is no comparable
convergence evidence yet for P-Flow across this new grid. Budget on a month scale
and revise the estimate from completed runs.

CPU validation:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m unittest \
  p_flow.test_p_flow p_flow.test_armijo experiments.ct_mri_sweep.test_sweep -v
```
