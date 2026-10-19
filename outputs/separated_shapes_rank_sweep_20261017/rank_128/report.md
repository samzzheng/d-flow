# Five-step Euler pilot

0% relative measurement noise; CT10 and Cartesian MRI4; 5 test samples per acquisition at each resolution; separated shapes: 1–4 circles/triangles. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 128, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_shapes_ct10/sample_000 | DLR D-Flow | 486 | 478 | 2.6040 | 0.4955 | stalled |
| r64_shapes_ct10/sample_001 | DLR D-Flow | 770 | 759 | 1.9005 | 0.4365 | stalled |
| r64_shapes_ct10/sample_002 | DLR D-Flow | 699 | 688 | 3.7301 | 0.8216 | stalled |
| r64_shapes_ct10/sample_003 | DLR D-Flow | 1000 | 950 | 6.0697 | 3.6795 | step_limit |
| r64_shapes_ct10/sample_004 | DLR D-Flow | 537 | 532 | 1.6663 | 0.3257 | stalled |
| r64_shapes_mri4/sample_000 | DLR D-Flow | 750 | 737 | 2.2878 | 1.0723 | stalled |
| r64_shapes_mri4/sample_001 | DLR D-Flow | 989 | 971 | 99.7793 | 99.1810 | stalled |
| r64_shapes_mri4/sample_002 | DLR D-Flow | 341 | 333 | 102.0144 | 97.7804 | stalled |
| r64_shapes_mri4/sample_003 | DLR D-Flow | 271 | 268 | 2.3423 | 0.8443 | stalled |
| r64_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 985 | 1.7430 | 0.6727 | step_limit |
| r96_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 978 | 83.0935 | 72.3479 | step_limit |
| r96_shapes_ct10/sample_001 | DLR D-Flow | 164 | 158 | 102.3271 | 94.9642 | stalled |
| r96_shapes_ct10/sample_002 | DLR D-Flow | 453 | 443 | 105.2308 | 87.9328 | stalled |
| r96_shapes_ct10/sample_003 | DLR D-Flow | 924 | 904 | 2.5586 | 0.6170 | stalled |
| r96_shapes_ct10/sample_004 | DLR D-Flow | 727 | 713 | 1.6664 | 0.2922 | stalled |
| r96_shapes_mri4/sample_000 | DLR D-Flow | 63 | 60 | 99.7857 | 98.9887 | stalled |
| r96_shapes_mri4/sample_001 | DLR D-Flow | 519 | 491 | 99.8659 | 99.0810 | stalled |
| r96_shapes_mri4/sample_002 | DLR D-Flow | 756 | 744 | 3.8814 | 1.4566 | stalled |
| r96_shapes_mri4/sample_003 | DLR D-Flow | 228 | 221 | 100.1226 | 98.7988 | stalled |
| r96_shapes_mri4/sample_004 | DLR D-Flow | 840 | 817 | 70.7249 | 66.8937 | stalled |
| r128_shapes_ct10/sample_000 | DLR D-Flow | 110 | 106 | 99.7497 | 89.7757 | stalled |
| r128_shapes_ct10/sample_001 | DLR D-Flow | 133 | 124 | 100.1924 | 98.6893 | stalled |
| r128_shapes_ct10/sample_002 | DLR D-Flow | 790 | 770 | 3.6041 | 0.7025 | stalled |
| r128_shapes_ct10/sample_003 | DLR D-Flow | 1000 | 990 | 2.1937 | 0.5904 | step_limit |
| r128_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 981 | 1.3882 | 0.2231 | step_limit |
| r128_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 986 | 45.2164 | 41.4926 | step_limit |
| r128_shapes_mri4/sample_001 | DLR D-Flow | 224 | 211 | 99.6423 | 99.5063 | stalled |
| r128_shapes_mri4/sample_002 | DLR D-Flow | 771 | 754 | 2.6452 | 1.4268 | stalled |
| r128_shapes_mri4/sample_003 | DLR D-Flow | 182 | 176 | 99.6607 | 99.0105 | stalled |
| r128_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 985 | 67.2285 | 65.7767 | step_limit |
| r160_shapes_ct10/sample_000 | DLR D-Flow | 52 | 49 | 101.3344 | 94.9818 | stalled |
| r160_shapes_ct10/sample_001 | DLR D-Flow | 79 | 75 | 100.2689 | 95.5721 | stalled |
| r160_shapes_ct10/sample_002 | DLR D-Flow | 99 | 93 | 99.9099 | 94.3181 | stalled |
| r160_shapes_ct10/sample_003 | DLR D-Flow | 137 | 134 | 101.5952 | 96.4764 | stalled |
| r160_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 980 | 2.2567 | 0.4808 | step_limit |
| r160_shapes_mri4/sample_000 | DLR D-Flow | 87 | 82 | 99.2782 | 98.8518 | stalled |
| r160_shapes_mri4/sample_001 | DLR D-Flow | 1000 | 976 | 1.7661 | 0.9215 | step_limit |
| r160_shapes_mri4/sample_002 | DLR D-Flow | 149 | 142 | 100.2399 | 98.8746 | stalled |
| r160_shapes_mri4/sample_003 | DLR D-Flow | 586 | 569 | 1.7961 | 1.1868 | stalled |
| r160_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 990 | 8.0001 | 7.3196 | step_limit |
| r200_shapes_ct10/sample_000 | DLR D-Flow | 271 | 264 | 56.2549 | 42.7254 | stalled |
| r200_shapes_ct10/sample_001 | DLR D-Flow | 135 | 128 | 100.1516 | 96.1120 | stalled |
| r200_shapes_ct10/sample_002 | DLR D-Flow | 86 | 82 | 101.0481 | 94.9866 | stalled |
| r200_shapes_ct10/sample_003 | DLR D-Flow | 379 | 374 | 99.8228 | 93.1461 | stalled |
| r200_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 973 | 1.8411 | 0.3300 | step_limit |
| r200_shapes_mri4/sample_000 | DLR D-Flow | 59 | 50 | 99.3501 | 99.2171 | stalled |
| r200_shapes_mri4/sample_001 | DLR D-Flow | 326 | 318 | 99.4696 | 99.2293 | stalled |
| r200_shapes_mri4/sample_002 | DLR D-Flow | 244 | 233 | 99.7485 | 99.4096 | stalled |
| r200_shapes_mri4/sample_003 | DLR D-Flow | 171 | 166 | 99.5514 | 99.4119 | stalled |
| r200_shapes_mri4/sample_004 | DLR D-Flow | 548 | 536 | 99.3197 | 99.0662 | stalled |
| r250_shapes_ct10/sample_000 | DLR D-Flow | 410 | 401 | 90.7321 | 69.9334 | stalled |
| r250_shapes_ct10/sample_001 | DLR D-Flow | 478 | 467 | 99.5785 | 94.2458 | stalled |
| r250_shapes_ct10/sample_002 | DLR D-Flow | 97 | 88 | 100.0336 | 97.4403 | stalled |
| r250_shapes_ct10/sample_003 | DLR D-Flow | 22 | 19 | 100.9508 | 94.6580 | stalled |
| r250_shapes_ct10/sample_004 | DLR D-Flow | 237 | 232 | 33.6601 | 18.1199 | stalled |
| r250_shapes_mri4/sample_000 | DLR D-Flow | 211 | 204 | 99.1434 | 98.8741 | stalled |
| r250_shapes_mri4/sample_001 | DLR D-Flow | 165 | 160 | 99.4675 | 99.3696 | stalled |
| r250_shapes_mri4/sample_002 | DLR D-Flow | 855 | 840 | 99.6991 | 99.4544 | stalled |
| r250_shapes_mri4/sample_003 | DLR D-Flow | 237 | 229 | 99.4939 | 99.2138 | stalled |
| r250_shapes_mri4/sample_004 | DLR D-Flow | 492 | 486 | 31.1955 | 30.1204 | stalled |

Terminal method runs: 60/60. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.

Original D/P controls and rank-8 results: /home/samzheng/d-flow/outputs/separated_shapes_noiseless_20261016. Completed rank pairs: 60.

| Paired case | Rank 8 image error % | Rank 128 image error % |
| --- | ---: | ---: |
| r64_shapes_ct10/sample_000 | 78.8474 | 2.6040 |
| r64_shapes_ct10/sample_001 | 102.8367 | 1.9005 |
| r64_shapes_ct10/sample_002 | 100.8987 | 3.7301 |
| r64_shapes_ct10/sample_003 | 103.1740 | 6.0697 |
| r64_shapes_ct10/sample_004 | 30.2782 | 1.6663 |
| r64_shapes_mri4/sample_000 | 76.3618 | 2.2878 |
| r64_shapes_mri4/sample_001 | 99.8533 | 99.7793 |
| r64_shapes_mri4/sample_002 | 101.7113 | 102.0144 |
| r64_shapes_mri4/sample_003 | 99.8358 | 2.3423 |
| r64_shapes_mri4/sample_004 | 73.9069 | 1.7430 |
| r96_shapes_ct10/sample_000 | 84.8165 | 83.0935 |
| r96_shapes_ct10/sample_001 | 102.3571 | 102.3271 |
| r96_shapes_ct10/sample_002 | 100.4289 | 105.2308 |
| r96_shapes_ct10/sample_003 | 100.3618 | 2.5586 |
| r96_shapes_ct10/sample_004 | 77.9043 | 1.6664 |
| r96_shapes_mri4/sample_000 | 82.2297 | 99.7857 |
| r96_shapes_mri4/sample_001 | 99.5716 | 99.8659 |
| r96_shapes_mri4/sample_002 | 99.6326 | 3.8814 |
| r96_shapes_mri4/sample_003 | 99.4967 | 100.1226 |
| r96_shapes_mri4/sample_004 | 97.8279 | 70.7249 |
| r128_shapes_ct10/sample_000 | 99.5985 | 99.7497 |
| r128_shapes_ct10/sample_001 | 99.7100 | 100.1924 |
| r128_shapes_ct10/sample_002 | 104.3587 | 3.6041 |
| r128_shapes_ct10/sample_003 | 104.8848 | 2.1937 |
| r128_shapes_ct10/sample_004 | 79.6018 | 1.3882 |
| r128_shapes_mri4/sample_000 | 99.3421 | 45.2164 |
| r128_shapes_mri4/sample_001 | 99.9842 | 99.6423 |
| r128_shapes_mri4/sample_002 | 100.0172 | 2.6452 |
| r128_shapes_mri4/sample_003 | 99.5989 | 99.6607 |
| r128_shapes_mri4/sample_004 | 73.4243 | 67.2285 |
| r160_shapes_ct10/sample_000 | 101.5283 | 101.3344 |
| r160_shapes_ct10/sample_001 | 100.3453 | 100.2689 |
| r160_shapes_ct10/sample_002 | 99.9719 | 99.9099 |
| r160_shapes_ct10/sample_003 | 101.9155 | 101.5952 |
| r160_shapes_ct10/sample_004 | 102.8103 | 2.2567 |
| r160_shapes_mri4/sample_000 | 99.6357 | 99.2782 |
| r160_shapes_mri4/sample_001 | 99.4535 | 1.7661 |
| r160_shapes_mri4/sample_002 | 100.2464 | 100.2399 |
| r160_shapes_mri4/sample_003 | 23.7943 | 1.7961 |
| r160_shapes_mri4/sample_004 | 73.1857 | 8.0001 |
| r200_shapes_ct10/sample_000 | 92.2966 | 56.2549 |
| r200_shapes_ct10/sample_001 | 100.3014 | 100.1516 |
| r200_shapes_ct10/sample_002 | 101.2903 | 101.0481 |
| r200_shapes_ct10/sample_003 | 100.4612 | 99.8228 |
| r200_shapes_ct10/sample_004 | 31.6637 | 1.8411 |
| r200_shapes_mri4/sample_000 | 99.6888 | 99.3501 |
| r200_shapes_mri4/sample_001 | 99.6028 | 99.4696 |
| r200_shapes_mri4/sample_002 | 97.9975 | 99.7485 |
| r200_shapes_mri4/sample_003 | 99.6854 | 99.5514 |
| r200_shapes_mri4/sample_004 | 99.5719 | 99.3197 |
| r250_shapes_ct10/sample_000 | 82.8371 | 90.7321 |
| r250_shapes_ct10/sample_001 | 99.9850 | 99.5785 |
| r250_shapes_ct10/sample_002 | 100.0471 | 100.0336 |
| r250_shapes_ct10/sample_003 | 111.5932 | 100.9508 |
| r250_shapes_ct10/sample_004 | 46.6633 | 33.6601 |
| r250_shapes_mri4/sample_000 | 99.2466 | 99.1434 |
| r250_shapes_mri4/sample_001 | 99.4823 | 99.4675 |
| r250_shapes_mri4/sample_002 | 99.6921 | 99.6991 |
| r250_shapes_mri4/sample_003 | 99.6892 | 99.4939 |
| r250_shapes_mri4/sample_004 | 88.5921 | 31.1955 |
