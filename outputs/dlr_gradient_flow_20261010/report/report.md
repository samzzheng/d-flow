# DLR Gradient Flow pilot

The selected samples form a small paired pilot, not a population estimate. Best image errors are retrospective selections from recorded histories; their own residuals are paired, but figures always use final saved iterates.

Corrected endpoint KSL is a secant-plus-correction variant. Literal Section 11 uses directional second derivatives. Jacobian-secant KSL instead uses the difference between the old and new true Jacobians. These are distinct algorithms.

Paired input, source/acquisition protocol, and checkpoint hashes passed for 4 cases.

Final metrics use original arrays from a single saved checkpoint. Runs may stop after two failed searches before their attempt limit; accepted updates are reported separately.

Baseline rows use the saved sweep; rerun rows use this study. Per-method `protocol.json` files record refresh frequency, refresh correction, optimizer, and derivative batch widths.

Convergence plots and companion controls prefer a new exact/proxy/fresh rerun when present; baseline panels preserve historical results. Reconstruction wall times are omitted: accuracy workers share a GPU and can pause for isolated benchmarks.

| Case | Method | Origin | Status | Attempts / accepted | Image L2 % | Residual % |
|---|---|---|---|---:|---:|---:|
| r64_n1_ct10/sample_000 | D-Flow | baseline | stalled | 32 / 30 | 53.797 | 34.466 |
| r64_n1_ct10/sample_000 | P-Flow | baseline | stalled | 25 / 21 | 55.548 | 32.854 |
| r64_n1_ct10/sample_000 | DLR D-Flow | baseline | stalled | 47 / 42 | 50.992 | 30.132 |
| r64_n1_ct10/sample_000 | DLR D-Flow | rerun | step_limit | 100 / 96 | 53.035 | 32.083 |
| r64_n1_ct10/sample_000 | DLR Gradient Flow (corrected) | rerun | step_limit | 100 / 100 | 112.818 | 86.258 |
| r64_n1_ct10/sample_000 | DLR Gradient Flow (Jacobian secant) | rerun | step_limit | 100 / 100 | 49.762 | 27.692 |
| r64_n1_ct10/sample_001 | D-Flow | baseline | stalled | 81 / 79 | 1.167 | 0.307 |
| r64_n1_ct10/sample_001 | P-Flow | baseline | step_limit | 100 / 86 | 7.583 | 2.989 |
| r64_n1_ct10/sample_001 | DLR D-Flow | baseline | stalled | 87 / 84 | 1.208 | 0.474 |
| r64_n1_ct10/sample_001 | DLR D-Flow | rerun | stalled | 50 / 46 | 1.144 | 0.339 |
| r64_n1_ct10/sample_001 | DLR Gradient Flow (corrected) | rerun | stalled | 60 / 58 | 1.152 | 0.296 |
| r64_n1_ct10/sample_001 | DLR Gradient Flow (Jacobian secant) | rerun | step_limit | 100 / 99 | 3.747 | 1.539 |
| r64_n1_mri4/sample_000 | D-Flow | baseline | step_limit | 100 / 100 | 1.553 | 1.003 |
| r64_n1_mri4/sample_000 | P-Flow | baseline | stalled | 14 / 11 | 65.980 | 62.778 |
| r64_n1_mri4/sample_000 | DLR D-Flow | baseline | step_limit | 100 / 95 | 19.553 | 15.579 |
| r64_n1_mri4/sample_000 | D-Flow | rerun | step_limit | 100 / 100 | 1.578 | 1.028 |
| r64_n1_mri4/sample_000 | DLR D-Flow | rerun | step_limit | 100 / 99 | 15.224 | 12.155 |
| r64_n1_mri4/sample_000 | DLR Gradient Flow (corrected) | rerun | stalled | 65 / 62 | 22.650 | 16.710 |
| r64_n1_mri4/sample_000 | DLR Gradient Flow (Jacobian secant) | rerun | stalled | 75 / 72 | 2.424 | 1.993 |
| r64_n1_mri4/sample_001 | D-Flow | baseline | step_limit | 100 / 100 | 0.845 | 0.393 |
| r64_n1_mri4/sample_001 | P-Flow | baseline | step_limit | 100 / 96 | 1.197 | 0.773 |
| r64_n1_mri4/sample_001 | DLR D-Flow | baseline | stalled | 41 / 38 | 1.270 | 0.904 |
| r64_n1_mri4/sample_001 | DLR D-Flow | rerun | stalled | 40 / 38 | 1.273 | 0.907 |
| r64_n1_mri4/sample_001 | DLR Gradient Flow (corrected) | rerun | step_limit | 100 / 100 | 1.108 | 0.709 |
| r64_n1_mri4/sample_001 | DLR Gradient Flow (Jacobian secant) | rerun | step_limit | 100 / 98 | 0.990 | 0.595 |

## Fixed-source gradient benchmarks

Synchronized cached-objective gradient wall time. Excludes preceding canonical objective, optimizer, line search, and warmup; exact gradient includes its necessary differentiable Euler forward/backward.

These are local gradient-accuracy measurements, not reconstruction success rates. Gradient errors are measured against the native 100-step Euler gradient at the same source.

| Case | Method | Source displacement norm | Median seconds | Gradient error % | Repeats | File |
|---|---|---:|---:|---:|---:|---|
| r128_n1_ct10/sample_000 | D-Flow | 1.28 | 1.2648 | 0.0000 | 2 | benchmark128_cap/benchmark.json |
| r128_n1_ct10/sample_000 | DLR D-Flow | 1.28 | 0.6823 | 6.6457 | 2 | benchmark128_cap/benchmark.json |
| r128_n1_ct10/sample_000 | DLR Gradient Flow (corrected) | 1.28 | 0.6757 | 1.3744 | 2 | benchmark128_cap/benchmark.json |
| r128_n1_ct10/sample_000 | DLR Gradient Flow (Jacobian secant) | 1.28 | 1.3495 | 6.3441 | 2 | benchmark128_cap/benchmark.json |
| r250_n1_ct10/sample_000 | D-Flow | 2.5 | 1.4440 | 0.0000 | 2 | benchmark250_cap/benchmark.json |
| r250_n1_ct10/sample_000 | DLR D-Flow | 2.5 | 1.6049 | 5.5115 | 2 | benchmark250_cap/benchmark.json |
| r250_n1_ct10/sample_000 | DLR Gradient Flow (corrected) | 2.5 | 1.7236 | 0.3048 | 2 | benchmark250_cap/benchmark.json |
| r250_n1_ct10/sample_000 | DLR Gradient Flow (Jacobian secant) | 2.5 | 3.4587 | 4.4032 | 2 | benchmark250_cap/benchmark.json |
| r250_n1_ct10/sample_000 | DLR Gradient Flow (corrected) | 2.5 | 1.7314 | 0.3048 | 3 | benchmark250_width4_4/benchmark.json |
| r250_n1_ct10/sample_000 | DLR Gradient Flow (corrected) | 2.5 | 1.7524 | 0.3048 | 3 | benchmark250_width6_4/benchmark.json |
| r250_n1_ct10/sample_000 | DLR Gradient Flow (corrected) | 2.5 | 1.7173 | 0.3048 | 3 | benchmark250_width8_4/benchmark.json |
| r64_n1_ct10/sample_000 | D-Flow | 0.064 | 0.9981 | 0.0000 | 2 | benchmark64/benchmark.json |
| r64_n1_ct10/sample_000 | DLR D-Flow | 0.064 | 0.4689 | 7.1264 | 2 | benchmark64/benchmark.json |
| r64_n1_ct10/sample_000 | DLR Gradient Flow (corrected) | 0.064 | 0.4277 | 0.4298 | 2 | benchmark64/benchmark.json |
| r64_n1_ct10/sample_000 | D-Flow | 0.64 | 1.0064 | 0.0000 | 2 | benchmark64_cap/benchmark.json |
| r64_n1_ct10/sample_000 | DLR D-Flow | 0.64 | 0.4644 | 7.0084 | 2 | benchmark64_cap/benchmark.json |
| r64_n1_ct10/sample_000 | DLR Gradient Flow (corrected) | 0.64 | 0.4235 | 0.4285 | 2 | benchmark64_cap/benchmark.json |
| r64_n1_ct10/sample_000 | DLR Gradient Flow (Jacobian secant) | 0.64 | 0.8451 | 4.8573 | 2 | benchmark64_cap/benchmark.json |
| r64_n1_ct10/sample_000 | DLR Gradient Flow (literal) | 0.064 | 58.9426 | 4.9584 | 1 | benchmark64_literal/benchmark.json |

Figures use fixed measurement-residual color scales: CT 0–1, MRI 0–25, `amp`. Companion figures explicitly replace only the fourth column with the named experimental method. CSV/JSON include retrospective best-image metrics and residuals at those same iterates.
