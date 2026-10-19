# Five-step Euler pilot

0% relative measurement noise; CT10 and Cartesian MRI4; 5 test samples per acquisition at each resolution; separated shapes: 1–4 circles/triangles. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 32, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_shapes_ct10/sample_000 | DLR D-Flow | 621 | 612 | 4.0542 | 1.0812 | stalled |
| r64_shapes_ct10/sample_001 | DLR D-Flow | 1000 | 976 | 108.3392 | 91.9162 | step_limit |
| r64_shapes_ct10/sample_002 | DLR D-Flow | 1000 | 972 | 101.0140 | 94.6368 | step_limit |
| r64_shapes_ct10/sample_003 | DLR D-Flow | 1000 | 977 | 105.9947 | 90.3439 | step_limit |
| r64_shapes_ct10/sample_004 | DLR D-Flow | 647 | 628 | 2.9808 | 0.6941 | stalled |
| r64_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 977 | 50.6564 | 48.9093 | step_limit |
| r64_shapes_mri4/sample_001 | DLR D-Flow | 660 | 644 | 99.7729 | 99.2839 | stalled |
| r64_shapes_mri4/sample_002 | DLR D-Flow | 380 | 369 | 102.1207 | 97.6805 | stalled |
| r64_shapes_mri4/sample_003 | DLR D-Flow | 734 | 714 | 100.9090 | 98.5700 | stalled |
| r64_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 980 | 2.0142 | 1.0677 | step_limit |
| r96_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 973 | 83.5400 | 72.8556 | step_limit |
| r96_shapes_ct10/sample_001 | DLR D-Flow | 99 | 95 | 102.3828 | 95.1805 | stalled |
| r96_shapes_ct10/sample_002 | DLR D-Flow | 136 | 131 | 100.5931 | 94.8947 | stalled |
| r96_shapes_ct10/sample_003 | DLR D-Flow | 1000 | 983 | 53.7922 | 41.2971 | step_limit |
| r96_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 975 | 23.9724 | 9.1371 | step_limit |
| r96_shapes_mri4/sample_000 | DLR D-Flow | 36 | 32 | 99.8321 | 99.0350 | stalled |
| r96_shapes_mri4/sample_001 | DLR D-Flow | 413 | 396 | 99.9117 | 98.9964 | stalled |
| r96_shapes_mri4/sample_002 | DLR D-Flow | 388 | 368 | 65.4748 | 57.8644 | stalled |
| r96_shapes_mri4/sample_003 | DLR D-Flow | 189 | 178 | 100.1169 | 98.8342 | stalled |
| r96_shapes_mri4/sample_004 | DLR D-Flow | 486 | 464 | 71.6527 | 67.8797 | stalled |
| r128_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 979 | 100.1152 | 88.1003 | step_limit |
| r128_shapes_ct10/sample_001 | DLR D-Flow | 387 | 372 | 99.8498 | 94.2221 | stalled |
| r128_shapes_ct10/sample_002 | DLR D-Flow | 1000 | 970 | 6.2898 | 1.6875 | step_limit |
| r128_shapes_ct10/sample_003 | DLR D-Flow | 447 | 431 | 2.3243 | 0.6272 | stalled |
| r128_shapes_ct10/sample_004 | DLR D-Flow | 284 | 277 | 84.2287 | 58.8609 | stalled |
| r128_shapes_mri4/sample_000 | DLR D-Flow | 648 | 635 | 99.0368 | 98.5389 | stalled |
| r128_shapes_mri4/sample_001 | DLR D-Flow | 1000 | 919 | 99.8179 | 98.9256 | step_limit |
| r128_shapes_mri4/sample_002 | DLR D-Flow | 105 | 97 | 100.0634 | 98.8329 | stalled |
| r128_shapes_mri4/sample_003 | DLR D-Flow | 137 | 130 | 99.6616 | 99.0200 | stalled |
| r128_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 983 | 73.2320 | 71.8960 | step_limit |
| r160_shapes_ct10/sample_000 | DLR D-Flow | 110 | 104 | 101.5684 | 95.4124 | stalled |
| r160_shapes_ct10/sample_001 | DLR D-Flow | 11 | 9 | 100.3402 | 96.0531 | stalled |
| r160_shapes_ct10/sample_002 | DLR D-Flow | 196 | 186 | 100.4997 | 94.0581 | stalled |
| r160_shapes_ct10/sample_003 | DLR D-Flow | 67 | 63 | 99.8737 | 97.4752 | stalled |
| r160_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 978 | 2.1110 | 0.3743 | step_limit |
| r160_shapes_mri4/sample_000 | DLR D-Flow | 173 | 166 | 99.2313 | 98.7451 | stalled |
| r160_shapes_mri4/sample_001 | DLR D-Flow | 330 | 319 | 1.9251 | 1.1079 | stalled |
| r160_shapes_mri4/sample_002 | DLR D-Flow | 90 | 85 | 100.1702 | 98.8974 | stalled |
| r160_shapes_mri4/sample_003 | DLR D-Flow | 190 | 183 | 100.8825 | 97.7893 | stalled |
| r160_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 984 | 69.2980 | 66.2511 | step_limit |
| r200_shapes_ct10/sample_000 | DLR D-Flow | 296 | 287 | 101.0164 | 79.3571 | stalled |
| r200_shapes_ct10/sample_001 | DLR D-Flow | 659 | 639 | 101.5045 | 94.0227 | stalled |
| r200_shapes_ct10/sample_002 | DLR D-Flow | 47 | 43 | 100.9475 | 94.3319 | stalled |
| r200_shapes_ct10/sample_003 | DLR D-Flow | 556 | 537 | 99.9581 | 93.7395 | stalled |
| r200_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 986 | 80.8365 | 54.1926 | step_limit |
| r200_shapes_mri4/sample_000 | DLR D-Flow | 104 | 98 | 99.4796 | 99.3747 | stalled |
| r200_shapes_mri4/sample_001 | DLR D-Flow | 400 | 386 | 99.4663 | 99.2297 | stalled |
| r200_shapes_mri4/sample_002 | DLR D-Flow | 589 | 562 | 99.7565 | 99.4791 | stalled |
| r200_shapes_mri4/sample_003 | DLR D-Flow | 618 | 596 | 99.5564 | 99.3870 | stalled |
| r200_shapes_mri4/sample_004 | DLR D-Flow | 112 | 104 | 99.3316 | 99.1441 | stalled |
| r250_shapes_ct10/sample_000 | DLR D-Flow | 281 | 273 | 94.0639 | 73.0961 | stalled |
| r250_shapes_ct10/sample_001 | DLR D-Flow | 57 | 51 | 99.5727 | 94.7920 | stalled |
| r250_shapes_ct10/sample_002 | DLR D-Flow | 324 | 308 | 99.7257 | 94.5074 | stalled |
| r250_shapes_ct10/sample_003 | DLR D-Flow | 14 | 11 | 101.6431 | 95.9837 | stalled |
| r250_shapes_ct10/sample_004 | DLR D-Flow | 460 | 451 | 31.8479 | 18.5667 | stalled |
| r250_shapes_mri4/sample_000 | DLR D-Flow | 253 | 247 | 99.0204 | 98.6806 | stalled |
| r250_shapes_mri4/sample_001 | DLR D-Flow | 534 | 509 | 99.4042 | 99.3546 | stalled |
| r250_shapes_mri4/sample_002 | DLR D-Flow | 857 | 823 | 99.6603 | 99.5129 | stalled |
| r250_shapes_mri4/sample_003 | DLR D-Flow | 398 | 384 | 99.5025 | 99.3669 | stalled |
| r250_shapes_mri4/sample_004 | DLR D-Flow | 257 | 249 | 31.2079 | 30.1254 | stalled |

Terminal method runs: 60/60. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.

Original D/P controls and rank-8 results: /home/samzheng/d-flow/outputs/separated_shapes_noiseless_20261016. Completed rank pairs: 60.

| Paired case | Rank 8 image error % | Rank 32 image error % |
| --- | ---: | ---: |
| r64_shapes_ct10/sample_000 | 78.8474 | 4.0542 |
| r64_shapes_ct10/sample_001 | 102.8367 | 108.3392 |
| r64_shapes_ct10/sample_002 | 100.8987 | 101.0140 |
| r64_shapes_ct10/sample_003 | 103.1740 | 105.9947 |
| r64_shapes_ct10/sample_004 | 30.2782 | 2.9808 |
| r64_shapes_mri4/sample_000 | 76.3618 | 50.6564 |
| r64_shapes_mri4/sample_001 | 99.8533 | 99.7729 |
| r64_shapes_mri4/sample_002 | 101.7113 | 102.1207 |
| r64_shapes_mri4/sample_003 | 99.8358 | 100.9090 |
| r64_shapes_mri4/sample_004 | 73.9069 | 2.0142 |
| r96_shapes_ct10/sample_000 | 84.8165 | 83.5400 |
| r96_shapes_ct10/sample_001 | 102.3571 | 102.3828 |
| r96_shapes_ct10/sample_002 | 100.4289 | 100.5931 |
| r96_shapes_ct10/sample_003 | 100.3618 | 53.7922 |
| r96_shapes_ct10/sample_004 | 77.9043 | 23.9724 |
| r96_shapes_mri4/sample_000 | 82.2297 | 99.8321 |
| r96_shapes_mri4/sample_001 | 99.5716 | 99.9117 |
| r96_shapes_mri4/sample_002 | 99.6326 | 65.4748 |
| r96_shapes_mri4/sample_003 | 99.4967 | 100.1169 |
| r96_shapes_mri4/sample_004 | 97.8279 | 71.6527 |
| r128_shapes_ct10/sample_000 | 99.5985 | 100.1152 |
| r128_shapes_ct10/sample_001 | 99.7100 | 99.8498 |
| r128_shapes_ct10/sample_002 | 104.3587 | 6.2898 |
| r128_shapes_ct10/sample_003 | 104.8848 | 2.3243 |
| r128_shapes_ct10/sample_004 | 79.6018 | 84.2287 |
| r128_shapes_mri4/sample_000 | 99.3421 | 99.0368 |
| r128_shapes_mri4/sample_001 | 99.9842 | 99.8179 |
| r128_shapes_mri4/sample_002 | 100.0172 | 100.0634 |
| r128_shapes_mri4/sample_003 | 99.5989 | 99.6616 |
| r128_shapes_mri4/sample_004 | 73.4243 | 73.2320 |
| r160_shapes_ct10/sample_000 | 101.5283 | 101.5684 |
| r160_shapes_ct10/sample_001 | 100.3453 | 100.3402 |
| r160_shapes_ct10/sample_002 | 99.9719 | 100.4997 |
| r160_shapes_ct10/sample_003 | 101.9155 | 99.8737 |
| r160_shapes_ct10/sample_004 | 102.8103 | 2.1110 |
| r160_shapes_mri4/sample_000 | 99.6357 | 99.2313 |
| r160_shapes_mri4/sample_001 | 99.4535 | 1.9251 |
| r160_shapes_mri4/sample_002 | 100.2464 | 100.1702 |
| r160_shapes_mri4/sample_003 | 23.7943 | 100.8825 |
| r160_shapes_mri4/sample_004 | 73.1857 | 69.2980 |
| r200_shapes_ct10/sample_000 | 92.2966 | 101.0164 |
| r200_shapes_ct10/sample_001 | 100.3014 | 101.5045 |
| r200_shapes_ct10/sample_002 | 101.2903 | 100.9475 |
| r200_shapes_ct10/sample_003 | 100.4612 | 99.9581 |
| r200_shapes_ct10/sample_004 | 31.6637 | 80.8365 |
| r200_shapes_mri4/sample_000 | 99.6888 | 99.4796 |
| r200_shapes_mri4/sample_001 | 99.6028 | 99.4663 |
| r200_shapes_mri4/sample_002 | 97.9975 | 99.7565 |
| r200_shapes_mri4/sample_003 | 99.6854 | 99.5564 |
| r200_shapes_mri4/sample_004 | 99.5719 | 99.3316 |
| r250_shapes_ct10/sample_000 | 82.8371 | 94.0639 |
| r250_shapes_ct10/sample_001 | 99.9850 | 99.5727 |
| r250_shapes_ct10/sample_002 | 100.0471 | 99.7257 |
| r250_shapes_ct10/sample_003 | 111.5932 | 101.6431 |
| r250_shapes_ct10/sample_004 | 46.6633 | 31.8479 |
| r250_shapes_mri4/sample_000 | 99.2466 | 99.0204 |
| r250_shapes_mri4/sample_001 | 99.4823 | 99.4042 |
| r250_shapes_mri4/sample_002 | 99.6921 | 99.6603 |
| r250_shapes_mri4/sample_003 | 99.6892 | 99.5025 |
| r250_shapes_mri4/sample_004 | 88.5921 | 31.2079 |
