# Five-step Euler pilot

Noiseless CT10 and Cartesian MRI4; one single-circle test sample at each resolution. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank8, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error. One sample cannot establish a population ranking.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_n1_ct10 | D-Flow | 1000 | 1000 | 1.3476 | 0.0545 | step_limit |
| r64_n1_ct10 | P-Flow | 632 | 510 | 21.9887 | 9.3186 | stalled |
| r64_n1_ct10 | DLR D-Flow | 486 | 472 | 8.9559 | 5.0516 | stalled |
| r64_n1_mri4 | D-Flow | 1000 | 1000 | 1.2504 | 0.2476 | step_limit |
| r64_n1_mri4 | P-Flow | 155 | 133 | 1.8743 | 0.9098 | stalled |
| r64_n1_mri4 | DLR D-Flow | 49 | 46 | 77.4629 | 77.0613 | stalled |
| r96_n1_ct10 | D-Flow | 1000 | 1000 | 1.2799 | 0.0302 | step_limit |
| r96_n1_ct10 | P-Flow | 39 | 34 | 1.9267 | 0.4696 | stalled |
| r96_n1_ct10 | DLR D-Flow | 307 | 297 | 1.8544 | 0.4358 | stalled |
| r96_n1_mri4 | D-Flow | 1000 | 1000 | 1.0285 | 0.1456 | step_limit |
| r96_n1_mri4 | P-Flow | 161 | 137 | 1.5582 | 0.8709 | stalled |
| r96_n1_mri4 | DLR D-Flow | 903 | 842 | 1.7550 | 1.0089 | stalled |
| r128_n1_ct10 | D-Flow | 737 | 735 | 1.8300 | 0.0469 | stalled |
| r128_n1_ct10 | P-Flow | 1000 | 975 | 2.4324 | 0.4319 | step_limit |
| r128_n1_ct10 | DLR D-Flow | 1000 | 969 | 3.1071 | 0.7178 | step_limit |
| r128_n1_mri4 | D-Flow | 1000 | 1000 | 1.2367 | 0.0521 | step_limit |
| r128_n1_mri4 | P-Flow | 19 | 15 | 3.9274 | 2.7569 | stalled |
| r128_n1_mri4 | DLR D-Flow | 1000 | 964 | 2.3134 | 1.5070 | step_limit |
| r160_n1_ct10 | D-Flow | 1000 | 1000 | 1.3190 | 0.0573 | step_limit |
| r160_n1_ct10 | P-Flow | 195 | 162 | 1.9656 | 0.3030 | stalled |
| r160_n1_ct10 | DLR D-Flow | 299 | 289 | 2.5484 | 0.3848 | stalled |
| r160_n1_mri4 | D-Flow | 1000 | 1000 | 1.2025 | 0.4963 | step_limit |
| r160_n1_mri4 | P-Flow | 120 | 106 | 1.5327 | 0.7061 | stalled |
| r160_n1_mri4 | DLR D-Flow | 26 | 22 | 96.9544 | 96.7167 | stalled |
| r200_n1_ct10 | D-Flow | 865 | 862 | 0.9181 | 0.0133 | stalled |
| r200_n1_ct10 | P-Flow | 37 | 30 | 8.0119 | 2.0394 | stalled |
| r200_n1_ct10 | DLR D-Flow | 516 | 494 | 1.3229 | 0.1512 | stalled |
| r200_n1_mri4 | D-Flow | 1000 | 1000 | 0.7011 | 0.1589 | step_limit |
| r200_n1_mri4 | P-Flow | 137 | 119 | 1.0685 | 0.5562 | stalled |
| r200_n1_mri4 | DLR D-Flow | 410 | 391 | 1.2709 | 0.6833 | stalled |
| r250_n1_ct10 | D-Flow | 1000 | 1000 | 1.0715 | 0.0275 | step_limit |
| r250_n1_ct10 | P-Flow | 1000 | 849 | 1.1990 | 0.1334 | step_limit |
| r250_n1_ct10 | DLR D-Flow | 267 | 261 | 1.9001 | 0.3155 | stalled |
| r250_n1_mri4 | D-Flow | 1000 | 1000 | 0.7124 | 0.2356 | step_limit |
| r250_n1_mri4 | P-Flow | 49 | 41 | 1.0478 | 0.5517 | stalled |
| r250_n1_mri4 | DLR D-Flow | 225 | 214 | 1.5472 | 0.9065 | stalled |

Terminal method runs: 36/36. Running rows are provisional.

Each production acquisition saves `sample_000/comparison.png` and `.pdf`, including image/residual panels and error versus optimizer attempts.
