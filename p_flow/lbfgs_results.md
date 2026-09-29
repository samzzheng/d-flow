# P-Flow with projected LBFGS + Armijo

Run: 2026-09-29T00:15:26.489445-04:00. Same three images, measurements, normalized initial sources, non-attention circles-64 epoch-100 model and 100-step Euler integrator as the fixed-step pilot.

All methods use learning rate 1, history 100, Armijo c1=1e-4, at most 30 trials, and displacement cap 0.01 sqrt(d). Budget: 15 steps; stop early after two failed searches. Initial L1 scaling occurs only on the first attempted search. Directions are projected onto the source sphere's tangent plane, trials are projected to the sphere BEFORE Armijo checks the true objective, and curvature pairs use actual displacements. This is projected ambient LBFGS, not a full Riemannian BFGS implementation.

Fast DLR refreshes rank-8 factors every 5 steps, clearing curvature history when the approximation changes while preserving lifetime initial-step scaling.

| Method | Mean relative image error | Raw PSNR | Mean step incl. Armijo | Mean step excluding extra trial flows | Mean total solve time |
| --- | ---: | ---: | ---: | ---: | ---: |
| P-Flow | 7.63% | 34.69 dB | 3.100 s | 1.587 s | 46.49 s |
| D-Flow | 94.32% | 12.23 dB | 7.578 s | 2.897 s | 113.68 s |
| Fast DLR | 111.33% | 10.70 dB | 10.566 s | 4.054 s | 101.12 s |

Relative error is ||reconstruction − truth||₂ / ||truth||₂ on unclipped images, averaged over test images. Accepted-trial forward solves reused for P-Flow/DLR gradients remain counted in the column excluding extra trial flows; only additional search flows are removed. Gradient evaluation, factor refreshes, LBFGS arithmetic and projection remain counted. Metrics and plotting are excluded. The last accepted state is reported.

## Per-image errors and search behavior

| Sample | Method | Fixed-step GD error | LBFGS + Armijo error | Steps | Trial evaluations | Failed searches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | P-Flow | 9.03% | 7.28% | 15 | 16 | 0 |
| 0 | D-Flow | 71.09% | 79.43% | 15 | 15 | 0 |
| 0 | Fast DLR | 74.44% | 107.33% | 15 | 47 | 1 |
| 1 | P-Flow | 2.44% | 11.46% | 15 | 51 | 1 |
| 1 | D-Flow | 141.86% | 115.14% | 15 | 105 | 0 |
| 1 | Fast DLR | 122.16% | 120.99% | 8 | 29 | 2 |
| 2 | P-Flow | 2.18% | 4.14% | 15 | 23 | 0 |
| 2 | D-Flow | 111.65% | 88.40% | 15 | 20 | 0 |
| 2 | Fast DLR | 114.81% | 105.67% | 8 | 71 | 2 |

![Per-iteration speed](outputs/lbfgs_64_2026_09_29/iteration_speed.png)

![Reconstructions](outputs/lbfgs_64_2026_09_29/reconstructions.png)

[Full metrics and settings](outputs/lbfgs_64_2026_09_29/results.json)

This pilot is limited to three 64×64 noisy-blur images and a short common iteration budget. All methods retain the sphere constraint for a controlled comparison; this differs from an unconstrained production D-Flow solve. The positive-definite LBFGS model uses approximate gradients for P-Flow/DLR; Armijo enforces true loss decrease but does not prove that those gradients approximate the exact derivative. No GPU benchmark or 250×250 conclusion follows.
