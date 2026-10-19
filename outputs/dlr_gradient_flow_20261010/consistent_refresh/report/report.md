# DLR Gradient Flow pilot

The selected samples form a small paired pilot, not a population estimate. Best image errors are retrospective selections from recorded histories; their own residuals are paired, but figures always use final saved iterates.

Corrected endpoint KSL is a secant-plus-correction variant. Literal Section 11 uses directional second derivatives. Jacobian-secant KSL instead uses the difference between the old and new true Jacobians. These are distinct algorithms.

Paired input, source/acquisition protocol, and checkpoint hashes passed for 2 cases.

Final metrics use original arrays from a single saved checkpoint. Runs may stop after two failed searches before their attempt limit; accepted updates are reported separately.

Baseline rows use the saved sweep; rerun rows use this study. Per-method `protocol.json` files record refresh frequency, refresh correction, optimizer, and derivative batch widths.

Convergence plots and companion controls prefer a new exact/proxy/fresh rerun when present; baseline panels preserve historical results. Reconstruction wall times are omitted: accuracy workers share a GPU and can pause for isolated benchmarks.

| Case | Method | Origin | Status | Attempts / accepted | Image L2 % | Residual % |
|---|---|---|---|---:|---:|---:|
| r64_n1_ct10/sample_000 | D-Flow | baseline | stalled | 32 / 30 | 53.797 | 34.466 |
| r64_n1_ct10/sample_000 | P-Flow | baseline | stalled | 25 / 21 | 55.548 | 32.854 |
| r64_n1_ct10/sample_000 | DLR D-Flow | baseline | stalled | 47 / 42 | 50.992 | 30.132 |
| r64_n1_ct10/sample_000 | DLR Gradient Flow (corrected) | rerun | stalled | 85 / 82 | 49.096 | 27.244 |
| r64_n1_mri4/sample_000 | D-Flow | baseline | step_limit | 100 / 100 | 1.553 | 1.003 |
| r64_n1_mri4/sample_000 | P-Flow | baseline | stalled | 14 / 11 | 65.980 | 62.778 |
| r64_n1_mri4/sample_000 | DLR D-Flow | baseline | step_limit | 100 / 95 | 19.553 | 15.579 |
| r64_n1_mri4/sample_000 | DLR Gradient Flow (corrected) | rerun | step_limit | 100 / 100 | 17.925 | 13.116 |

## Fixed-source gradient benchmarks

Synchronized cached-objective gradient wall time. Excludes preceding canonical objective, optimizer, line search, and warmup; exact gradient includes its necessary differentiable Euler forward/backward.

These are local gradient-accuracy measurements, not reconstruction success rates. Gradient errors are measured against the native 100-step Euler gradient at the same source.

| Case | Method | Source displacement norm | Median seconds | Gradient error % | Repeats | File |
|---|---|---:|---:|---:|---:|---|

Figures use fixed measurement-residual color scales: CT 0–1, MRI 0–25, `amp`. Companion figures explicitly replace only the fourth column with the named experimental method. CSV/JSON include retrospective best-image metrics and residuals at those same iterates.
