# Five-step optimizer controls

Two selected failed cases, unchanged original models, source/acquisition seeds, five Euler steps and rank 32. History five gives short-history LBFGS, compared with the existing history-100 baseline. All retain geometric halving, first-attempt scaling, the 0.01*sqrt(d) cap and a 1000-attempt budget. Settings are exploratory, not independently validated. Timings include GPU sharing.

| Optimizer | Case | Method | Status | Attempts | Image error % | Measurement error % | Seconds |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| lbfgs_h5 | r64_shapes_mri4/sample_001 | D-Flow | step_limit | 1000 | 100.066 | 100.079 | 76.8 |
| lbfgs_h5 | r64_shapes_mri4/sample_001 | P-Flow | stalled | 61 | 3.417 | 1.754 | 19.6 |
| lbfgs_h5 | r64_shapes_mri4/sample_001 | DLR D-Flow | step_limit | 1000 | 99.735 | 99.381 | 307.9 |
| lbfgs_h5 | r128_shapes_ct10/sample_000 | D-Flow | step_limit | 1000 | 90.951 | 69.782 | 106.4 |
| lbfgs_h5 | r128_shapes_ct10/sample_000 | P-Flow | step_limit | 1000 | 3.879 | 0.318 | 105.1 |
| lbfgs_h5 | r128_shapes_ct10/sample_000 | DLR D-Flow | stalled | 329 | 99.742 | 89.893 | 75.9 |
