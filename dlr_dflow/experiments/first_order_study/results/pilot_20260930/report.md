# First-order study at 64×64

CPU pilot: 100 Euler steps, non-attention epoch-100 model, 15-angle sparse CT, 1% relative noise. Analytical alpha = 1 − t, zero trace probes, fresh DLR factors for every gradient.

All image errors are raw relative L2 errors (no clipping). Ground-truth best checkpoints are retrospective diagnostics.

## Optimizer comparison

Common-prefix errors compare the same number of attempted updates across both estimators and all three test images. Stalled runs remain in the comparison; final errors may use different completed budgets. Timing excludes metric collection and file I/O.

| Constraint | Optimizer | Estimator | Common updates | Mean common error | Mean final error | Gradient s | Iteration excluding trials s | Total iteration s | Stalled/failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Free | gd | exact | 21 | 0.5453 | 0.5529 | 2.721 | 2.723 | 7.290 | 0/3 |
| Free | gd | dlr | 21 | 0.5059 | 0.5059 | 10.475 | 10.475 | 14.996 | 0/3 |
| Free | adam | exact | 21 | 0.9168 | 0.9170 | 2.705 | 2.707 | 4.231 | 0/3 |
| Free | adam | dlr | 21 | 0.8703 | 0.8703 | 10.443 | 10.443 | 11.956 | 0/3 |
| Free | lbfgs | exact | 21 | 0.2821 | 0.2765 | 2.695 | 2.697 | 4.520 | 0/3 |
| Free | lbfgs | dlr | 21 | 0.2445 | 0.2445 | 10.418 | 10.419 | 12.119 | 0/3 |
| Sphere | gd | exact | 17 | 0.5292 | 0.5273 | 2.676 | 2.678 | 8.241 | 1/3 |
| Sphere | gd | dlr | 17 | 0.5758 | 0.5660 | 10.411 | 10.412 | 15.023 | 0/3 |
| Sphere | adam | exact | 21 | 0.9016 | 0.9016 | 2.674 | 2.676 | 4.222 | 0/3 |
| Sphere | adam | dlr | 21 | 0.4876 | 0.4876 | 10.399 | 10.399 | 11.905 | 0/3 |
| Sphere | lbfgs | exact | 21 | 0.2576 | 0.2576 | 2.673 | 2.675 | 4.569 | 0/3 |
| Sphere | lbfgs | dlr | 21 | 0.4417 | 0.4417 | 10.415 | 10.415 | 12.748 | 0/3 |

## Rank sweep

Settings are fixed across ranks and the exact control within each constraint.

| Constraint | Rank | Updates | Final image error | Best image error | Final noisy residual | Residual at best image |
| --- | --- | --- | --- | --- | --- | --- |
| Free | 1 | 10 | 0.7059 | 0.6855 | 2.20494 | 2.17229 |
| Free | 2 | 10 | 0.6745 | 0.6745 | 2.20229 | 2.20229 |
| Free | 4 | 10 | 0.6563 | 0.6563 | 2.05427 | 2.05427 |
| Free | 8 | 10 | 0.6628 | 0.6628 | 2.13097 | 2.13097 |
| Free | 16 | 10 | 0.6679 | 0.6679 | 2.17415 | 2.17415 |
| Free | 32 | 10 | 0.6741 | 0.6741 | 2.21649 | 2.21649 |
| Free | exact | 10 | 0.6758 | 0.6758 | 2.22765 | 2.22765 |
| Sphere | 1 | 10 | 0.4857 | 0.4857 | 1.39384 | 1.39384 |
| Sphere | 2 | 10 | 0.6311 | 0.6311 | 2.00633 | 2.00633 |
| Sphere | 4 | 10 | 0.6691 | 0.6691 | 2.18527 | 2.18527 |
| Sphere | 8 | 10 | 0.6530 | 0.6530 | 2.09262 | 2.09262 |
| Sphere | 16 | 10 | 0.6615 | 0.6615 | 2.15010 | 2.15010 |
| Sphere | 32 | 10 | 0.6679 | 0.6679 | 2.18961 | 2.18961 |
| Sphere | exact | 10 | 0.6652 | 0.6652 | 2.17414 | 2.17414 |

## Gradient diagnostics

Spectral references differentiate the full discrete Euler map independently of DLR. Certification is numerical; unresolved references are labeled.

| Constraint | Iteration | DLR relative error | DLR cosine | Subspace relative error | Rank-8 reference relative error | Spectral status | Clean ratio R (trajectory noise) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Free | 0 | 0.0593 | 0.9987 | 0.0433 | 0.0031 | certified_numerical | 68.6666 |
| Free | 11 | 0.0450 | 0.9993 | 0.0313 | 0.0023 | certified_numerical | 5.5811 |
| Free | 22 | 0.0370 | 0.9995 | 0.0313 | 0.0007 | ambiguous_cutoff | 2.7034 |
| Sphere | 0 | 0.0586 | 0.9987 | 0.0431 | 0.0024 | certified_numerical | 63.9506 |
| Sphere | 11 | 0.0301 | 0.9997 | 0.0230 | 0.0004 | ambiguous_cutoff | 7.0237 |
| Sphere | 22 | 0.0405 | 0.9994 | 0.0351 | 0.0046 | ambiguous_cutoff | 0.9570 |

## Limits and completion

Elapsed experimental time: 219.9 minutes. Three held-out images and one validation image constitute a pilot, not a population-level accuracy result.
Fresh-factor DLR timing is not comparable to earlier factor-reuse benchmarks without accounting for the different refresh policy.
The ten extra noise realizations per saved source are conditional diagnostics, not independent reconstruction samples.
Strong Wolfe results are condition checks at selected trial rates using true derivatives, not a tuned Strong Wolfe reconstruction sweep.

GPU 0 became available after the CPU reconstruction runs. Supplemental spectral references use batched GPU products with TF32 disabled, CPU/GPU endpoint and gradient parity checks, and final CPU singular-triplet residual verification. Only certified supplemental references replace the initial CPU references in this report; optimizer timings remain CPU-only.

- comparisons: phase deadline

## Figures

![Convergence](convergence.png)

![Iteration timing](iteration_timing.png)

![Learning-rate comparison](learning_rates.png)

![Rank and residual comparison](rank_residual.png)

![Clean-gradient ratios](clean_gradient_ratios.png)
