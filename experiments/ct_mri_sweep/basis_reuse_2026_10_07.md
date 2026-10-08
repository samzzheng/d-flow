# Previous-terminal-V initialization pilot — 2026-10-07

Paired CPU runs from original Gaussian sources; selected diagnostic cases, not a random population sample.
Both use 100 Euler steps, rank 8, analytical alpha=1-t, zero enrichment, the same LBFGS/Armijo and cap.
Random16 uses 16 random initialization columns. Previous-V8 replaces eight with previous terminal right vectors.
Factors always restart from t=0. All remaining random columns and seeds are paired.

| Case | Final image %: random / reuse | Final residual %: random / reuse | Attempts: random / reuse | Status: random / reuse |
| --- | --- | --- | --- | --- |
| r64_n1_ct10/sample_003 | 74.030 / 1.145 | 49.971 / 0.311 | 30 / 100 | stalled / step_limit |
| r64_n1_ct10/sample_008 | 30.868 / 27.090 | 13.957 / 12.048 | 51 / 46 | stalled / stalled |
| r64_n1_ct10/sample_022 | 1.375 / 1.277 | 0.404 / 0.300 | 83 / 90 | stalled / stalled |
| r64_n1_ct10/sample_028 | 1.185 / 1.173 | 0.295 / 0.294 | 80 / 81 | stalled / stalled |
| r64_n1_mri4/sample_000 | 24.909 / 19.668 | 20.686 / 16.489 | 48 / 86 | stalled / stalled |
| r64_n1_mri4/sample_003 | 1.134 / 1.145 | 0.691 / 0.696 | 97 / 67 | stalled / stalled |
| r64_n1_mri4/sample_018 | 107.132 / 107.157 | 106.671 / 106.697 | 18 / 33 | stalled / stalled |
| r64_n1_mri4/sample_026 | 93.276 / 6.085 | 90.528 / 3.857 | 38 / 100 | stalled / step_limit |

Completed paired cases: 8/8.
Final image error <=5%: random 3, reuse 4.
Mean final image error (%): random 41.739, reuse 20.592.
Median final image error (%): random 27.888, reuse 3.681.
Reuse lower final image error: 6/8.
CPU results must not be treated as exact replays of the historical GPU trajectories.

## Same-source gradient comparison after the first accepted update

Both runs have identical source, native endpoint, and terminal gradient here; all differences come from initialization.

| Case | Relative gradient error %: random / reuse | Cosine: random / reuse |
| --- | --- | --- |
| r64_n1_ct10/sample_003 | 4.230 / 4.274 | 0.999269 / 0.999175 |
| r64_n1_ct10/sample_008 | 6.762 / 7.741 | 0.998016 / 0.997317 |
| r64_n1_ct10/sample_022 | 9.158 / 9.114 | 0.996188 / 0.996151 |
| r64_n1_ct10/sample_028 | 5.673 / 5.514 | 0.998473 / 0.998534 |
| r64_n1_mri4/sample_000 | 8.553 / 8.268 | 0.997564 / 0.997601 |
| r64_n1_mri4/sample_003 | 4.115 / 4.146 | 0.999305 / 0.999220 |
| r64_n1_mri4/sample_018 | 17.628 / 17.314 | 0.986934 / 0.987554 |
| r64_n1_mri4/sample_026 | 2.620 / 2.536 | 0.999686 / 0.999691 |

Terminal-source diagnostics are saved in gradient_diagnostics.json. The two variants have different terminal sources.

## Interpretation and limits

Previous-terminal-V seeding helped in this selected CPU pilot: final image
error decreased in six of eight cases, and <=5% successes increased from
three to four. The strongest changes were CT sample 3 (74.03% to 1.14%) and
MRI sample 26 (93.28% to 6.09%). MRI sample 18 remained a severe failure.
The two worse outcomes differed by only 0.011 and 0.025 percentage points.
The mean improvement is strongly influenced by the two largest recoveries.
These eight diagnostic cases are not a population-level estimate.

Reusing the basis also reached <=5% image error earlier in some pairs: CT
sample 22 at attempt 40 versus 65, and MRI sample 3 at 28 versus 34. These
are retrospective image-error thresholds; the optimizer did not use ground
truth to stop. CT sample 28 crossed at 38 versus 39, with nearly identical
final errors. Two reuse runs reached the 100-attempt limit; the other 14
runs stopped after two failed searches. A stall does not certify convergence.
MRI sample 26 was still improving at the budget limit.

At the shared source after the first update, mean relative gradient errors
were 7.342% for the random sketch and 7.363% for reuse. Five individual
cases improved, three worsened. This does not support a uniform immediate
improvement in gradient accuracy. The observed reconstruction gains arise
after the optimization paths diverge; this pilot does not identify whether
basis persistence, altered LBFGS curvature history, or simply a favorable
trajectory perturbation is responsible. Repeated source/sketch seeds and a
random-seed control are needed to separate these explanations.

The initialization still applies the new initial residual operator and
computes fresh factors. Only the sketch directions are reused. A linear-flow
preflight verified identical JVP/VJP calls and column widths for the two
variants, no mutation of the old basis, and the correct Euler endpoint and
analytical alpha path. With rank 8 and width 16 at 100 steps, each gradient
has 16 + 99*8 = 808 local JVP columns and the same number of VJP columns in
both variants. There is extra basis storage/normalization, but no added
model-product columns. Total solve work can differ because the accepted
updates and line-search counts change; no GPU wall-time speed claim is made.

All 16 starting gradients and the first accepted source in every pair agree
between variants. Raw-array image/residual metrics, source caps, monotone
objectives, and fresh-gradient counts passed validation. All 16 terminal
native-CPU objective replays matched saved values exactly. The pilot resumed
its first four workers from saved states while increasing concurrency from
four to sixteen low-priority, one-thread CPU workers. Both GPUs and the
ongoing noiseless sweep were left unchanged; their source hashes still match.

The CPU results are not interchangeable with previous CUDA trajectories.
In particular, some historical GPU successes became failures in the CPU
baseline. Keep this initialization experimental until paired GPU trials
and independent seeds establish how much benefit transfers.

## Artifacts and reproduction

- [Convergence plot](../../outputs/basis_reuse_pilot_20261007/convergence.png)
- [Per-source gradient diagnostics](../../outputs/basis_reuse_pilot_20261007/gradient_diagnostics.json)
- [Validation and evaluation counts](../../outputs/basis_reuse_pilot_20261007/validation.json)
- [Study configuration](../../outputs/basis_reuse_pilot_20261007/study.json)
- All 16 source/optimizer/basis states, histories, and five-row reconstruction
  figures are under `outputs/basis_reuse_pilot_20261007/r64_n1_*/sample_*/*/`.

From the repository root, using a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 10 .venv/bin/python -m experiments.ct_mri_sweep.test_basis_reuse \
  --output outputs/basis_reuse_repeat --workers 16 --threads 1 --max-steps 100
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m experiments.ct_mri_sweep.analyze_basis_reuse \
  --output outputs/basis_reuse_repeat
```
