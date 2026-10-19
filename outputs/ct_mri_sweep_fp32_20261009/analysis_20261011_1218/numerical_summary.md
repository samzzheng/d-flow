# Interim CT/MRI numerical summary — 2026-10-11 12:19 EDT

504 paired terminal cases out of 540 planned. Final saved checkpoints, not retrospectively selected best images. No measurement noise; FP32; 100 Euler steps; up to 100 optimizer attempts with common LBFGS/geometric Armijo. Sixteen acquisitions complete. The 250-square three-circle CT group has 24/30 paired cases; three-circle MRI has no paired cases.

## Median final relative image L2 error (%)

| Resolution | Problem | Paired cases | D-Flow | P-Flow | DLR D-Flow |
| --- | --- | --- | --- | --- | --- |
| 64² | CT | 90 | 16.87 | 23.40 | 24.51 |
| 64² | MRI | 90 | 2.09 | 1.04 | 19.38 |
| 128² | CT | 90 | 39.14 | 20.32 | 52.18 |
| 128² | MRI | 90 | 48.58 | 0.92 | 61.68 |
| 250² | CT | 84 | 5.50 | 12.28 | 14.96 |
| 250² | MRI | 60 | 1.81 | 0.76 | 8.57 |

## Median relative measurement residual (%)

No noise: observed and clean measurement references coincide.

| Resolution | Problem | Paired cases | D-Flow | P-Flow | DLR D-Flow |
| --- | --- | --- | --- | --- | --- |
| 64² | CT | 90 | 6.64 | 9.10 | 9.93 |
| 64² | MRI | 90 | 1.79 | 0.57 | 15.91 |
| 128² | CT | 90 | 17.40 | 6.61 | 28.69 |
| 128² | MRI | 90 | 44.74 | 0.52 | 56.52 |
| 250² | CT | 84 | 1.36 | 2.59 | 3.88 |
| 250² | MRI | 60 | 1.15 | 0.43 | 5.95 |

## Overall descriptive counts

| Method | Image error ≤5% | Image error >20% | Lowest image error |
| --- | --- | --- | --- |
| D-Flow | 230/504 | 246/504 | 153/504 |
| P-Flow | 309/504 | 159/504 | 304/504 |
| DLR D-Flow | 182/504 | 280/504 | 47/504 |

The 5% and 20% cutoffs are descriptive thresholds, not stopping criteria. All methods are compared on identical completed cases, but this incomplete cohort underweights 250-square three-circle problems. Cross-resolution comparisons use separately trained models and do not isolate spatial discretization. A terminal status does not establish convergence.

DLR beats D-Flow on 144/504 cases. At the >20% threshold: both fail on 216, D-Flow alone on 30, DLR alone on 64, neither on 194.

See report.md for the full circle-count breakdown and cases.csv for per-case results.
