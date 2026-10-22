# Separated-shapes adaptive-rank pilot
Updated 2026-10-22T08:19-04:00.

Paired noiseless CT10/MRI4× cases; 117k models, five Euler steps, analytical alpha=1-t, native forward normalization, zero enrichment, and the same LBFGS/Armijo settings and 1000-attempt limit.
Adaptive bounds: 32–128; retain core singular values at least 0.001 times the largest, subject to these bounds. This cutoff is not a bound on gradient error.
The maximum-rank initialization sketch is rebuilt at each gradient evaluation. With zero enrichment, rank can only decrease within one flow solve; it can increase again at the next gradient evaluation.
Fixed-rank canonical controls use equal rank bounds in the adaptive code path. They help distinguish rank selection from changes in factor arithmetic and execution.

Errors below use the last saved source. Partial results are explicitly marked; these selected cases do not estimate population performance.

| Case | Variant | Status | Attempts | Relative image error | Clean measurement error |
| --- | --- | --- | ---: | ---: | ---: |
| r250_shapes_mri4/sample_002 | adaptive | Complete (stalled) | 240 | 99.768% | 99.659% |

Completed 1/1 method runs.
All available variants within each case have matching initial-source and checkpoint hashes.

## r250_shapes_mri4/sample_002

Last recorded flow ranks: 0 → 128 → 128 → 42 → 32 → 32.
![Rank history](ranks_r250_shapes_mri4_sample_002.png)
