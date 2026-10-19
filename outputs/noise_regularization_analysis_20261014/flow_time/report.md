# Jacobian evolution along flow time

Single-circle64px epoch200 model. Native partial Euler derivatives use the same dt and arithmetic as the sampler. DLR uses alpha=1-t, rank8 residual factors, and the existing fast profile. All diagnostics are CPU only.

J_t = derivative of x_t with respect to source z. This is flow time, not optimizer iteration. The approximation includes its identity baseline: Jhat_t=(1-t)I+U_t S_t V_t^T. Its full singular spectrum is calculated exactly using a <=16-dimensional block plus the remaining identity singular values.

Stable rank = ||J||_F^2 / ||J||_2^2. Energy rank is the number of singular values needed for99.9% of squared-singular-value energy. Neither is algebraic rank. Native spectra are bounded randomized estimates with independent omitted-energy probes; unresolved energy cutoffs are not assigned a rank. Stable-rank estimates also depend on convergence of the leading singular value.

At intermediate times, noise-gradient diagnostics concern hypothetical measurements A(x_t). Only t=1 is the reconstruction objective. The100-step companion uses the same initial source but a different, more finely integrated generator; it is not a rerun of the five-step optimizer.

Terminal table errors use the low-rank Frobenius identity with native JVPs. Plot lines keep the common probe estimator at every time, and hollow diamonds show refined terminal errors. Leading-norm refinements update stable rank independently of unresolved energy cutoffs.

## CT, d_flow_final, 5 Euler steps

Basis initialization: fresh sketch. Fast/native endpoint discrepancy: 0.00171.

| t | Native stable rank estimate | DLR full stable rank | Native99.9% energy rank | DLR full99.9% energy rank | Relative Frobenius error estimate | Noise-gradient norm ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0.00 | 4096.000 | 4096.000 | 4092 | 4092 | 0.00% | 1.000 |
| 0.20 | 3196.904 | 3480.993 | unresolved | 4092 | 0.69% | 0.999 |
| 0.40 | 141.253 | 255.980 | unresolved | 4092 | 6.69% | 0.974 |
| 0.60 | 2.316 | 12.156 | unresolved | 4090 | 64.22% | 0.819 |
| 0.80 | 1.006 | 1.047 | 3 | 3596 | 99.81% | 0.275 |
| 1.00 | 1.000 | 1.001 | 1 | 1 | 97.02% | 0.215 |

## CT, dlr_final, 5 Euler steps

Basis initialization: saved terminal V used for a new evolution; not original gradient replay. Fast/native endpoint discrepancy: 3.74e-06.

| t | Native stable rank estimate | DLR full stable rank | Native99.9% energy rank | DLR full99.9% energy rank | Relative Frobenius error estimate | Noise-gradient norm ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0.00 | 4096.000 | 4096.000 | 4092 | 4092 | 0.00% | 1.000 |
| 0.20 | 3192.100 | 3196.664 | unresolved | 4092 | 0.64% | 0.997 |
| 0.40 | 57.482 | 67.574 | unresolved | 4092 | 5.90% | 0.967 |
| 0.60 | 1.233 | 1.297 | unresolved | 4076 | 42.30% | 0.826 |
| 0.80 | 1.046 | 1.031 | unresolved | 3952 | 44.01% | 0.605 |
| 1.00 | 1.015 | 1.001 | 3 | 1 | 47.23% | 0.806 |

## CT, initial, 100 Euler steps

Basis initialization: fresh sketch. Fast/native endpoint discrepancy: 4.61e-07.

| t | Native stable rank estimate | DLR full stable rank | Native99.9% energy rank | DLR full99.9% energy rank | Relative Frobenius error estimate | Noise-gradient norm ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0.00 | 4096.000 | 4096.000 | 4092 | 4092 | 0.00% | 1.000 |
| 0.10 | 1655.579 | 1670.927 | unresolved | 4092 | 0.27% | 0.998 |
| 0.25 | 45.566 | 46.138 | unresolved | 4092 | 0.94% | 0.994 |
| 0.50 | 1.649 | 1.660 | unresolved | 4085 | 2.65% | 0.992 |
| 0.75 | 1.026 | 1.027 | unresolved | 3911 | 3.17% | 0.993 |
| 0.90 | 1.005 | 1.005 | unresolved | 1687 | 3.17% | 0.993 |
| 0.95 | 1.004 | 1.004 | 2 | 2 | 3.16% | 0.993 |
| 0.99 | 1.003 | 1.003 | 2 | 2 | 3.16% | 0.993 |
| 1.00 | 1.003 | 1.003 | 2 | 2 | 2.41% | 0.993 |

## CT, initial, 5 Euler steps

Basis initialization: fresh sketch. Fast/native endpoint discrepancy: 5.09e-07.

| t | Native stable rank estimate | DLR full stable rank | Native99.9% energy rank | DLR full99.9% energy rank | Relative Frobenius error estimate | Noise-gradient norm ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0.00 | 4096.000 | 4096.000 | 4092 | 4092 | 0.00% | 1.000 |
| 0.20 | 3193.203 | 3516.473 | unresolved | 4092 | 0.76% | 1.002 |
| 0.40 | 370.341 | 841.279 | unresolved | 4091 | 3.71% | 0.973 |
| 0.60 | 73.670 | 208.864 | unresolved | 4091 | 11.40% | 0.855 |
| 0.80 | 6.804 | 22.006 | unresolved | 4090 | 40.51% | 0.604 |
| 1.00 | 1.946 | 1.718 | 3 | 4 | 76.72% | 0.473 |

## MRI, d_flow_final, 5 Euler steps

Basis initialization: fresh sketch. Fast/native endpoint discrepancy: 2.62e-07.

| t | Native stable rank estimate | DLR full stable rank | Native99.9% energy rank | DLR full99.9% energy rank | Relative Frobenius error estimate | Noise-gradient norm ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0.00 | 4096.000 | 4096.000 | 4092 | 4092 | 0.00% | 1.000 |
| 0.20 | 3092.997 | 3574.409 | unresolved | 4092 | 2.59% | 0.925 |
| 0.40 | 194.401 | 788.667 | unresolved | 4092 | 7.92% | 0.880 |
| 0.60 | 97.655 | 350.584 | unresolved | 4092 | 13.70% | 0.853 |
| 0.80 | 48.720 | 107.693 | unresolved | 4091 | 24.55% | 0.836 |
| 1.00 | 2.915 | 1.933 | unresolved | 8 | 77.93% | 0.846 |

## MRI, dlr_final, 5 Euler steps

Basis initialization: saved terminal V used for a new evolution; not original gradient replay. Fast/native endpoint discrepancy: 2.79e-06.

| t | Native stable rank estimate | DLR full stable rank | Native99.9% energy rank | DLR full99.9% energy rank | Relative Frobenius error estimate | Noise-gradient norm ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0.00 | 4096.000 | 4096.000 | 4092 | 4092 | 0.00% | 1.000 |
| 0.20 | 3108.965 | 3148.910 | unresolved | 4092 | 0.66% | 1.000 |
| 0.40 | 131.459 | 288.167 | unresolved | 4091 | 6.73% | 0.991 |
| 0.60 | 1.993 | 3.454 | unresolved | 4089 | 59.21% | 0.725 |
| 0.80 | 1.016 | 1.024 | unresolved | 3910 | 83.44% | 0.619 |
| 1.00 | 1.003 | 1.000 | 2 | 1 | 78.98% | 0.619 |

## MRI, initial, 5 Euler steps

Basis initialization: fresh sketch. Fast/native endpoint discrepancy: 4.12e-07.

| t | Native stable rank estimate | DLR full stable rank | Native99.9% energy rank | DLR full99.9% energy rank | Relative Frobenius error estimate | Noise-gradient norm ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0.00 | 4096.000 | 4096.000 | 4092 | 4092 | 0.00% | 1.000 |
| 0.20 | 3192.721 | 3397.413 | unresolved | 4092 | 0.70% | 1.001 |
| 0.40 | 267.143 | 770.101 | unresolved | 4092 | 5.21% | 0.988 |
| 0.60 | 37.402 | 124.154 | unresolved | 4091 | 14.17% | 0.933 |
| 0.80 | 3.393 | 12.419 | unresolved | 4090 | 49.43% | 0.721 |
| 1.00 | 1.129 | 1.288 | 3 | 3 | 83.79% | 0.355 |

