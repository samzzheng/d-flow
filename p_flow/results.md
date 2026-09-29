# P-Flow CPU pilot

Run: 2026-09-29T00:02:50.155088-04:00

Non-attention circles-64, epoch 100; 100 Euler steps, 15 projected-gradient updates, 3 paired test images, 2 CPU threads. Blur sigma 1.5 pixels, measurement-noise standard deviation 0.05.

Learning rates were selected independently from a common grid using only validation image 0 after 5 updates. Reported accuracy is the final iterate, without choosing the best test iteration. PSNR uses data range 1; raw and clipped metrics are separate.

Relative error means ||reconstruction − truth||₂ / ||truth||₂, averaged across images. These values use unclipped reconstructions.

| Method | Selected learning rate | Mean iteration (s) | Relative error | Raw MSE | Raw PSNR (dB) | Clipped PSNR (dB) | Measurement MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P-Flow | 1 | 1.479 | 4.55% | 0.00024884 | 40.403 | 41.505 | 0.00252923 |
| D-Flow (projected GD) | 0.01 | 2.700 | 108.20% | 0.0809208 | 11.273 | 11.291 | 0.0642906 |
| Fast DLR (projected GD) | 0.01 | 3.642 | 103.80% | 0.0740262 | 11.492 | 11.514 | 0.0591241 |

Iteration timings exclude diagnostics and the extra final-reporting forward solve. DLR refreshes rank-8 factors every 5 updates; its timings include those refreshes. There is no Armijo in any of these runs.

## Per-image relative reconstruction errors (%)

| Sample | Initial | P-Flow | D-Flow | Fast DLR |
| --- | ---: | ---: | ---: | ---: |
| 0 | 131.13 | 9.03 | 71.09 | 74.44 |
| 1 | 150.05 | 2.44 | 141.86 | 122.16 |
| 2 | 117.66 | 2.18 | 111.65 | 114.81 |

## Shared initial-source gradients

| Sample | Approximation | Cosine with exact | Relative gradient error | Norm / exact norm | Tangent cosine | True derivative along projected update |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | P-Flow | 0.4404 | 0.9671 | 0.0810 | 0.4404 | -2396.8 |
| 0 | Fast DLR (projected GD) | 0.9999 | 0.0161 | 0.9928 | 0.9999 | -66750 |
| 1 | P-Flow | 0.4278 | 0.9688 | 0.7765 | 0.4280 | -215.9 |
| 1 | Fast DLR (projected GD) | 0.9991 | 0.0547 | 0.9638 | 0.9991 | -625.99 |
| 2 | P-Flow | 0.4685 | 0.8948 | 0.6105 | 0.4689 | -211.97 |
| 2 | Fast DLR (projected GD) | 0.9982 | 0.0649 | 0.9722 | 0.9982 | -718.87 |

Negative directional derivative means local descent on the sphere. Positive cosine does not mean an accurate Jacobian approximation, nor does it guarantee descent for a finite learning rate.

## Figures and evidence

![Reconstructions](outputs/pilot_64_2026_09_29/reconstructions.png)

![Convergence](outputs/pilot_64_2026_09_29/convergence.png)

[Full settings and histories](outputs/pilot_64_2026_09_29/results.json)

## Limits

This is a small single-model, single-task pilot. Three images and one short validation run cannot establish a general ranking. The grid may not contain the best learning rate. D-Flow here uses matched projected gradient descent, not the project's tuned LBFGS + Armijo optimizer. P-Flow includes spherical projection; this is more than just replacing a gradient in an otherwise unchanged production D-Flow solve. GPU speed and 250×250 performance are untested.
