# Five-step Euler pilot

0% relative measurement noise; CT10 and Cartesian MRI4; 5 test samples per acquisition at each resolution; separated shapes: 1–4 circles/triangles. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 64, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 968 | 52.1872 | 41.8668 | step_limit |
| r64_shapes_ct10/sample_001 | DLR D-Flow | 891 | 873 | 2.3720 | 0.7073 | stalled |
| r64_shapes_ct10/sample_002 | DLR D-Flow | 626 | 616 | 3.7484 | 1.1567 | stalled |
| r64_shapes_ct10/sample_003 | DLR D-Flow | 1000 | 963 | 4.9019 | 3.5035 | step_limit |
| r64_shapes_ct10/sample_004 | DLR D-Flow | 369 | 364 | 2.0114 | 0.4275 | stalled |
| r64_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 972 | 50.7108 | 48.6178 | step_limit |
| r64_shapes_mri4/sample_001 | DLR D-Flow | 738 | 720 | 99.7117 | 99.2571 | stalled |
| r64_shapes_mri4/sample_002 | DLR D-Flow | 218 | 209 | 101.8176 | 97.9697 | stalled |
| r64_shapes_mri4/sample_003 | DLR D-Flow | 527 | 512 | 2.3219 | 0.9783 | stalled |
| r64_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 971 | 2.5298 | 1.4391 | step_limit |
| r96_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 970 | 83.1113 | 72.5177 | step_limit |
| r96_shapes_ct10/sample_001 | DLR D-Flow | 170 | 164 | 102.2003 | 94.7929 | stalled |
| r96_shapes_ct10/sample_002 | DLR D-Flow | 1000 | 971 | 6.8821 | 2.0672 | step_limit |
| r96_shapes_ct10/sample_003 | DLR D-Flow | 997 | 978 | 3.1673 | 0.7719 | stalled |
| r96_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 983 | 72.4899 | 51.9692 | step_limit |
| r96_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 962 | 82.0123 | 81.6930 | step_limit |
| r96_shapes_mri4/sample_001 | DLR D-Flow | 292 | 276 | 99.8786 | 99.0591 | stalled |
| r96_shapes_mri4/sample_002 | DLR D-Flow | 1000 | 968 | 6.9481 | 4.0408 | step_limit |
| r96_shapes_mri4/sample_003 | DLR D-Flow | 195 | 190 | 100.1208 | 98.8483 | stalled |
| r96_shapes_mri4/sample_004 | DLR D-Flow | 617 | 591 | 70.4570 | 66.7262 | stalled |
| r128_shapes_ct10/sample_000 | DLR D-Flow | 168 | 161 | 99.8216 | 90.3504 | stalled |
| r128_shapes_ct10/sample_001 | DLR D-Flow | 94 | 87 | 100.3415 | 99.2668 | stalled |
| r128_shapes_ct10/sample_002 | DLR D-Flow | 544 | 522 | 3.2794 | 0.8808 | stalled |
| r128_shapes_ct10/sample_003 | DLR D-Flow | 258 | 251 | 104.7736 | 92.3261 | stalled |
| r128_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 984 | 66.7394 | 40.6193 | step_limit |
| r128_shapes_mri4/sample_000 | DLR D-Flow | 233 | 228 | 99.1677 | 98.7486 | stalled |
| r128_shapes_mri4/sample_001 | DLR D-Flow | 43 | 39 | 99.9897 | 99.9557 | stalled |
| r128_shapes_mri4/sample_002 | DLR D-Flow | 114 | 108 | 100.0788 | 98.7116 | stalled |
| r128_shapes_mri4/sample_003 | DLR D-Flow | 39 | 34 | 99.7037 | 99.4941 | stalled |
| r128_shapes_mri4/sample_004 | DLR D-Flow | 650 | 633 | 67.3068 | 65.9127 | stalled |
| r160_shapes_ct10/sample_000 | DLR D-Flow | 45 | 40 | 101.5178 | 95.6075 | stalled |
| r160_shapes_ct10/sample_001 | DLR D-Flow | 8 | 6 | 99.6654 | 96.2980 | stalled |
| r160_shapes_ct10/sample_002 | DLR D-Flow | 163 | 158 | 100.9294 | 93.9017 | stalled |
| r160_shapes_ct10/sample_003 | DLR D-Flow | 71 | 67 | 101.0974 | 95.1482 | stalled |
| r160_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 984 | 1.8549 | 0.3376 | step_limit |
| r160_shapes_mri4/sample_000 | DLR D-Flow | 363 | 351 | 99.1369 | 98.5678 | stalled |
| r160_shapes_mri4/sample_001 | DLR D-Flow | 661 | 644 | 1.8294 | 1.0260 | stalled |
| r160_shapes_mri4/sample_002 | DLR D-Flow | 61 | 58 | 100.1772 | 98.9124 | stalled |
| r160_shapes_mri4/sample_003 | DLR D-Flow | 917 | 885 | 1.8376 | 1.2333 | stalled |
| r160_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 981 | 68.2764 | 65.4104 | step_limit |
| r200_shapes_ct10/sample_000 | DLR D-Flow | 683 | 664 | 52.2608 | 42.1366 | stalled |
| r200_shapes_ct10/sample_001 | DLR D-Flow | 350 | 342 | 101.5731 | 94.2109 | stalled |
| r200_shapes_ct10/sample_002 | DLR D-Flow | 47 | 42 | 101.0051 | 94.3798 | stalled |
| r200_shapes_ct10/sample_003 | DLR D-Flow | 177 | 171 | 99.9046 | 93.2721 | stalled |
| r200_shapes_ct10/sample_004 | DLR D-Flow | 56 | 52 | 100.6281 | 96.8446 | stalled |
| r200_shapes_mri4/sample_000 | DLR D-Flow | 718 | 704 | 81.6042 | 81.6448 | stalled |
| r200_shapes_mri4/sample_001 | DLR D-Flow | 340 | 330 | 99.4682 | 99.2294 | stalled |
| r200_shapes_mri4/sample_002 | DLR D-Flow | 289 | 276 | 99.7240 | 99.4554 | stalled |
| r200_shapes_mri4/sample_003 | DLR D-Flow | 534 | 519 | 99.5592 | 99.3849 | stalled |
| r200_shapes_mri4/sample_004 | DLR D-Flow | 119 | 114 | 99.2137 | 98.9765 | stalled |
| r250_shapes_ct10/sample_000 | DLR D-Flow | 210 | 202 | 96.6251 | 75.2994 | stalled |
| r250_shapes_ct10/sample_001 | DLR D-Flow | 186 | 179 | 100.1468 | 96.9937 | stalled |
| r250_shapes_ct10/sample_002 | DLR D-Flow | 80 | 76 | 100.4253 | 95.0388 | stalled |
| r250_shapes_ct10/sample_003 | DLR D-Flow | 41 | 37 | 100.3024 | 93.2754 | stalled |
| r250_shapes_ct10/sample_004 | DLR D-Flow | 263 | 255 | 32.6332 | 18.6756 | stalled |
| r250_shapes_mri4/sample_000 | DLR D-Flow | 38 | 35 | 99.3298 | 99.1423 | stalled |
| r250_shapes_mri4/sample_001 | DLR D-Flow | 279 | 272 | 99.3990 | 99.3528 | stalled |
| r250_shapes_mri4/sample_002 | DLR D-Flow | 79 | 74 | 99.6670 | 99.5329 | stalled |
| r250_shapes_mri4/sample_003 | DLR D-Flow | 152 | 144 | 99.4794 | 99.3854 | stalled |
| r250_shapes_mri4/sample_004 | DLR D-Flow | 187 | 182 | 31.2137 | 30.1296 | stalled |

Terminal method runs: 60/60. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.

Original D/P controls and rank-8 results: /home/samzheng/d-flow/outputs/separated_shapes_noiseless_20261016. Completed rank pairs: 60.

| Paired case | Rank 8 image error % | Rank 64 image error % |
| --- | ---: | ---: |
| r64_shapes_ct10/sample_000 | 78.8474 | 52.1872 |
| r64_shapes_ct10/sample_001 | 102.8367 | 2.3720 |
| r64_shapes_ct10/sample_002 | 100.8987 | 3.7484 |
| r64_shapes_ct10/sample_003 | 103.1740 | 4.9019 |
| r64_shapes_ct10/sample_004 | 30.2782 | 2.0114 |
| r64_shapes_mri4/sample_000 | 76.3618 | 50.7108 |
| r64_shapes_mri4/sample_001 | 99.8533 | 99.7117 |
| r64_shapes_mri4/sample_002 | 101.7113 | 101.8176 |
| r64_shapes_mri4/sample_003 | 99.8358 | 2.3219 |
| r64_shapes_mri4/sample_004 | 73.9069 | 2.5298 |
| r96_shapes_ct10/sample_000 | 84.8165 | 83.1113 |
| r96_shapes_ct10/sample_001 | 102.3571 | 102.2003 |
| r96_shapes_ct10/sample_002 | 100.4289 | 6.8821 |
| r96_shapes_ct10/sample_003 | 100.3618 | 3.1673 |
| r96_shapes_ct10/sample_004 | 77.9043 | 72.4899 |
| r96_shapes_mri4/sample_000 | 82.2297 | 82.0123 |
| r96_shapes_mri4/sample_001 | 99.5716 | 99.8786 |
| r96_shapes_mri4/sample_002 | 99.6326 | 6.9481 |
| r96_shapes_mri4/sample_003 | 99.4967 | 100.1208 |
| r96_shapes_mri4/sample_004 | 97.8279 | 70.4570 |
| r128_shapes_ct10/sample_000 | 99.5985 | 99.8216 |
| r128_shapes_ct10/sample_001 | 99.7100 | 100.3415 |
| r128_shapes_ct10/sample_002 | 104.3587 | 3.2794 |
| r128_shapes_ct10/sample_003 | 104.8848 | 104.7736 |
| r128_shapes_ct10/sample_004 | 79.6018 | 66.7394 |
| r128_shapes_mri4/sample_000 | 99.3421 | 99.1677 |
| r128_shapes_mri4/sample_001 | 99.9842 | 99.9897 |
| r128_shapes_mri4/sample_002 | 100.0172 | 100.0788 |
| r128_shapes_mri4/sample_003 | 99.5989 | 99.7037 |
| r128_shapes_mri4/sample_004 | 73.4243 | 67.3068 |
| r160_shapes_ct10/sample_000 | 101.5283 | 101.5178 |
| r160_shapes_ct10/sample_001 | 100.3453 | 99.6654 |
| r160_shapes_ct10/sample_002 | 99.9719 | 100.9294 |
| r160_shapes_ct10/sample_003 | 101.9155 | 101.0974 |
| r160_shapes_ct10/sample_004 | 102.8103 | 1.8549 |
| r160_shapes_mri4/sample_000 | 99.6357 | 99.1369 |
| r160_shapes_mri4/sample_001 | 99.4535 | 1.8294 |
| r160_shapes_mri4/sample_002 | 100.2464 | 100.1772 |
| r160_shapes_mri4/sample_003 | 23.7943 | 1.8376 |
| r160_shapes_mri4/sample_004 | 73.1857 | 68.2764 |
| r200_shapes_ct10/sample_000 | 92.2966 | 52.2608 |
| r200_shapes_ct10/sample_001 | 100.3014 | 101.5731 |
| r200_shapes_ct10/sample_002 | 101.2903 | 101.0051 |
| r200_shapes_ct10/sample_003 | 100.4612 | 99.9046 |
| r200_shapes_ct10/sample_004 | 31.6637 | 100.6281 |
| r200_shapes_mri4/sample_000 | 99.6888 | 81.6042 |
| r200_shapes_mri4/sample_001 | 99.6028 | 99.4682 |
| r200_shapes_mri4/sample_002 | 97.9975 | 99.7240 |
| r200_shapes_mri4/sample_003 | 99.6854 | 99.5592 |
| r200_shapes_mri4/sample_004 | 99.5719 | 99.2137 |
| r250_shapes_ct10/sample_000 | 82.8371 | 96.6251 |
| r250_shapes_ct10/sample_001 | 99.9850 | 100.1468 |
| r250_shapes_ct10/sample_002 | 100.0471 | 100.4253 |
| r250_shapes_ct10/sample_003 | 111.5932 | 100.3024 |
| r250_shapes_ct10/sample_004 | 46.6633 | 32.6332 |
| r250_shapes_mri4/sample_000 | 99.2466 | 99.3298 |
| r250_shapes_mri4/sample_001 | 99.4823 | 99.3990 |
| r250_shapes_mri4/sample_002 | 99.6921 | 99.6670 |
| r250_shapes_mri4/sample_003 | 99.6892 | 99.4794 |
| r250_shapes_mri4/sample_004 | 88.5921 | 31.2137 |
