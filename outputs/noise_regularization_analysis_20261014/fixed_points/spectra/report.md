# Native Jacobian spectral diagnostic

CT10, 5% noise, 64² single-circle sample 15; Jacobian of the actual five-step Euler map. CPU only.

Randomized width 32, at most three refinements, 16 independent omitted-energy probes. Energy ranks refer to cumulative squared singular values, not algebraic rank. The minimum singular value and full condition number are not measured.

| Source | Largest singular value | 99.9% energy rank estimate | 99.99% energy rank estimate | Omitted-energy fraction estimate | Relevant relative triplet residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial | 9.948 | 3 | unresolved | 0.000468 | 0.000643 |
| d_flow_final | 20679.438 | 1 | 1 | 1.22e-09 | 3.18e-07 |
| dlr_final | 176.050 | 3 | 3 | 7.12e-06 | 8.67e-07 |

In this selected failure case, the largest singular value is much larger at the failed D-Flow source even though its energy-defined effective rank is smaller. This is a cross-source association, not proof that sensitivity growth caused failure. The comparison does not establish the rank of other datasets or a universal conditioning result.
