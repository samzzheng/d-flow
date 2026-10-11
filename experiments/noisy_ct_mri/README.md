# Paired 1%-noise CT/MRI studies

The October11 study is isolated under `outputs/noisy_ct_mri_20261011`.
The existing noiseless results and optimizer states remain separate.

- Rank study:64×64, circle counts1–4, CT10/Cartesian MRI4, samples0–9,
  fixed DLR ranks2/4/8/12/16:400 reconstructions. Each rank starts fresh;
  the source, mask, noise and DLR seeds are identical across ranks.
- Larger comparison:64/128/250, circle counts1–3, CT10/MRI4, samples0–29,
  D-Flow/P-Flow/rank8 DLR:1,620 reconstructions. The overlapping rank8 cases
  are repeated executions under the same protocol, not independent samples.
- Both use100 Euler steps, full FP32 with TF32 disabled, analytical alpha1−t
  ending exactly at zero, no trace probes/enrichment, oversampling8, and
  previous terminal V. Full-step graphs and derivative chunking match the
  current production profile.
- LBFGS history100, lr1, geometric halving, max30 trials, c1=1e−4,
  displacement cap0.01√d, first-attempt-only scaling; at most100 optimizer
  attempts. Two failed searches terminate as stalled. No noise-floor stopping,
  clipping, projection, additional prior or best-image checkpoint selection.
- Noise has `||noise||/||A(x_GT)|| = .01`. MRI noise is circular complex on
  acquired coefficients only. GT measurement is clean. Its observed residual
  is approximately1% (the plotting denominator is `||y||`, not `||A(x_GT)||`).

`rank_worker.py` restricts the existing worker's method list to DLR. It reuses
its objective, optimizer, checkpointing, backend checks and figure code.
Each rank gets a standard GT/D/P/DLR figure with D/P explicitly unavailable;
all three columns are populated in the larger comparison. The fifth row has
error/residual histories, and residual heatmaps retain CT0–1/MRI0–25 limits.
The report additionally compares five-rank completed sets; retrospective
best-image plots always use the residual from that same checkpoint.

The shared scheduler resumes unfinished noiseless GPU1 shards and admits the
new studies with combined memory reservations. GPU0 is blocked while the
existing spectral/hybrid and adaptive controllers live, including their idle
intervals; it becomes eligible afterward. Unknown GPU processes pause our
workers on that device. The legacy queue lock is held to prevent a competing
controller. Existing GPU0 jobs are neither stopped nor modified.

Before production admission, all12 two-update CT/MRI smoke jobs must complete:
ten DLR rank cases and two three-method cases (16 reconstructions total).
Every worker checks graph/eager gradients, warm-V gradients, native endpoints,
zero terminal alpha, displacement cap and strict accepted objective decrease.
The smoke validates implementation and memory fit, not100-step convergence.
Reservations are4,096MiB per64px rank worker and2,816/8,448/30,720MiB per
three-method worker at64/128/250; each includes768MiB outside the allocator.
There is a further1,536MiB device margin and at most12 workers per GPU.

```sh
.venv/bin/python -m unittest experiments.noisy_ct_mri.test_noise
.venv/bin/python -m experiments.noisy_ct_mri.setup --out outputs/noisy_ct_mri_20261011
.venv/bin/python -m experiments.noisy_ct_mri.setup --out outputs/noisy_ct_mri_smoke_20261011 --smoke
# Gracefully stop the old GPU1 controller before acquiring its queue lock.
.venv/bin/python -m experiments.noisy_ct_mri.queue \
  --out outputs/noisy_ct_mri_20261011 \
  --preflight outputs/noisy_ct_mri_smoke_20261011 \
  --continue-manifest outputs/ct_mri_sweep_fp32_20261009/manifest.json \
  --gpu0-guard-pids 2060790 2083158
.venv/bin/python -m experiments.noisy_ct_mri.report --out outputs/noisy_ct_mri_20261011
```

The new `queue_status.json` is authoritative for resumed noiseless work;
the old controller's status remains paused. Source hashes are frozen in every
manifest. Checkpoints retain optimizer state, warm V, source, measurements,
seeds, checkpoint/data hashes, failure status and histories. Final reports must
compare matched completed samples and distinguish failures from convergence.
