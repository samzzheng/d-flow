# Five-step Euler pilot

0% relative measurement noise; CT10 and Cartesian MRI4; 5 test samples per acquisition at each resolution; separated shapes: 1–4 circles/triangles. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 512, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 975 | 2.5116 | 0.6461 | step_limit |
| r64_shapes_ct10/sample_001 | DLR D-Flow | 908 | 901 | 1.9450 | 0.1824 | stalled |
| r64_shapes_ct10/sample_002 | DLR D-Flow | 491 | 486 | 3.0198 | 0.5240 | stalled |
| r64_shapes_ct10/sample_003 | DLR D-Flow | 1000 | 964 | 4.9003 | 1.8594 | step_limit |
| r64_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 996 | 2.3465 | 0.4569 | step_limit |
| r64_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 985 | 2.1358 | 0.6006 | step_limit |
| r64_shapes_mri4/sample_001 | DLR D-Flow | 1000 | 978 | 99.8451 | 99.2366 | step_limit |
| r64_shapes_mri4/sample_002 | DLR D-Flow | 73 | 69 | 101.7328 | 98.1006 | stalled |
| r64_shapes_mri4/sample_003 | DLR D-Flow | 1000 | 993 | 5.2900 | 0.3532 | step_limit |
| r64_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 985 | 1.5647 | 0.4551 | step_limit |
| r96_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 975 | 52.4305 | 41.6600 | step_limit |
| r96_shapes_ct10/sample_001 | DLR D-Flow | 265 | 255 | 102.2424 | 94.7532 | stalled |
| r96_shapes_ct10/sample_002 | DLR D-Flow | 1000 | 992 | 3.5987 | 0.9404 | step_limit |
| r96_shapes_ct10/sample_003 | DLR D-Flow | 959 | 944 | 2.3554 | 0.4330 | stalled |
| r96_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 992 | 1.5725 | 0.2279 | step_limit |
| r96_shapes_mri4/sample_000 | DLR D-Flow | 737 | 711 | 50.8089 | 49.2889 | stalled |
| r96_shapes_mri4/sample_001 | DLR D-Flow | 639 | 615 | 100.7318 | 98.3285 | stalled |
| r96_shapes_mri4/sample_002 | DLR D-Flow | 1000 | 976 | 3.1759 | 1.2683 | step_limit |
| r96_shapes_mri4/sample_003 | DLR D-Flow | 1000 | 988 | 2.4438 | 0.8536 | step_limit |
| r96_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 977 | 1.6963 | 0.8931 | step_limit |
| r128_shapes_ct10/sample_000 | DLR D-Flow | 571 | 560 | 103.9026 | 85.9683 | stalled |
| r128_shapes_ct10/sample_001 | DLR D-Flow | 142 | 138 | 99.8081 | 96.6485 | stalled |
| r128_shapes_ct10/sample_002 | DLR D-Flow | 1000 | 982 | 3.2574 | 0.8041 | step_limit |
| r128_shapes_ct10/sample_003 | DLR D-Flow | 391 | 381 | 2.3365 | 0.6431 | stalled |
| r128_shapes_ct10/sample_004 | DLR D-Flow | 234 | 229 | 78.0790 | 53.8309 | stalled |
| r128_shapes_mri4/sample_000 | DLR D-Flow | 537 | 527 | 99.0653 | 98.5556 | stalled |
| r128_shapes_mri4/sample_001 | DLR D-Flow | 1000 | 924 | 99.7563 | 98.8857 | step_limit |
| r128_shapes_mri4/sample_002 | DLR D-Flow | 1000 | 974 | 2.3284 | 1.1800 | step_limit |
| r128_shapes_mri4/sample_003 | DLR D-Flow | 126 | 122 | 99.6630 | 98.9930 | stalled |
| r128_shapes_mri4/sample_004 | DLR D-Flow | 309 | 301 | 73.2097 | 71.8064 | stalled |
| r160_shapes_ct10/sample_000 | DLR D-Flow | 137 | 131 | 101.4874 | 94.5730 | stalled |
| r160_shapes_ct10/sample_001 | DLR D-Flow | 10 | 8 | 99.6651 | 96.2992 | stalled |
| r160_shapes_ct10/sample_002 | DLR D-Flow | 88 | 83 | 100.5497 | 93.9074 | stalled |
| r160_shapes_ct10/sample_003 | DLR D-Flow | 426 | 419 | 104.3569 | 91.4715 | stalled |
| r160_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 986 | 1.3445 | 0.1741 | step_limit |
| r160_shapes_mri4/sample_000 | DLR D-Flow | 52 | 49 | 99.3419 | 98.9384 | stalled |
| r160_shapes_mri4/sample_001 | DLR D-Flow | 771 | 756 | 1.6093 | 0.7959 | stalled |
| r160_shapes_mri4/sample_002 | DLR D-Flow | 98 | 93 | 100.1726 | 98.8443 | stalled |
| r160_shapes_mri4/sample_003 | DLR D-Flow | 1000 | 944 | 3.1522 | 2.3541 | step_limit |
| r160_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 989 | 1.7821 | 1.0672 | step_limit |
| r200_shapes_ct10/sample_000 | DLR D-Flow | 456 | 445 | 52.0455 | 42.1416 | stalled |
| r200_shapes_ct10/sample_001 | DLR D-Flow | 275 | 270 | 101.4998 | 94.2768 | stalled |
| r200_shapes_ct10/sample_002 | DLR D-Flow | 61 | 57 | 100.9072 | 94.2989 | stalled |
| r200_shapes_ct10/sample_003 | DLR D-Flow | 218 | 211 | 99.8987 | 93.4338 | stalled |
| r200_shapes_ct10/sample_004 | DLR D-Flow | 17 | 15 | 100.2185 | 98.7098 | stalled |
| r200_shapes_mri4/sample_000 | DLR D-Flow | 111 | 103 | 99.2956 | 99.1483 | stalled |
| r200_shapes_mri4/sample_001 | DLR D-Flow | 658 | 644 | 99.4379 | 99.2079 | stalled |
| r200_shapes_mri4/sample_002 | DLR D-Flow | 242 | 228 | 99.8364 | 99.1949 | stalled |
| r200_shapes_mri4/sample_003 | DLR D-Flow | 226 | 221 | 99.5625 | 99.3774 | stalled |
| r200_shapes_mri4/sample_004 | DLR D-Flow | 115 | 111 | 99.0032 | 98.7651 | stalled |
| r250_shapes_ct10/sample_000 | DLR D-Flow | 398 | 388 | 86.9173 | 68.0270 | stalled |
| r250_shapes_ct10/sample_001 | DLR D-Flow | 80 | 74 | 99.9963 | 97.8940 | stalled |
| r250_shapes_ct10/sample_002 | DLR D-Flow | 30 | 27 | 99.9162 | 97.9676 | stalled |
| r250_shapes_ct10/sample_003 | DLR D-Flow | 18 | 16 | 101.4963 | 95.9578 | stalled |
| r250_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 983 | 2.4522 | 0.4738 | step_limit |
| r250_shapes_mri4/sample_000 | DLR D-Flow | 258 | 250 | 99.1457 | 98.8816 | stalled |
| r250_shapes_mri4/sample_001 | DLR D-Flow | 234 | 227 | 99.4152 | 99.3497 | stalled |
| r250_shapes_mri4/sample_002 | DLR D-Flow | 367 | 357 | 99.7242 | 99.4628 | stalled |
| r250_shapes_mri4/sample_003 | DLR D-Flow | 73 | 69 | 99.6352 | 99.5636 | stalled |
| r250_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 990 | 31.3417 | 29.9997 | step_limit |

Terminal method runs: 60/60. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.

Original D/P controls and rank-8 results: /home/samzheng/d-flow/outputs/separated_shapes_noiseless_20261016. Completed rank pairs: 60.

| Paired case | Rank 8 image error % | Rank 512 image error % |
| --- | ---: | ---: |
| r64_shapes_ct10/sample_000 | 78.8474 | 2.5116 |
| r64_shapes_ct10/sample_001 | 102.8367 | 1.9450 |
| r64_shapes_ct10/sample_002 | 100.8987 | 3.0198 |
| r64_shapes_ct10/sample_003 | 103.1740 | 4.9003 |
| r64_shapes_ct10/sample_004 | 30.2782 | 2.3465 |
| r64_shapes_mri4/sample_000 | 76.3618 | 2.1358 |
| r64_shapes_mri4/sample_001 | 99.8533 | 99.8451 |
| r64_shapes_mri4/sample_002 | 101.7113 | 101.7328 |
| r64_shapes_mri4/sample_003 | 99.8358 | 5.2900 |
| r64_shapes_mri4/sample_004 | 73.9069 | 1.5647 |
| r96_shapes_ct10/sample_000 | 84.8165 | 52.4305 |
| r96_shapes_ct10/sample_001 | 102.3571 | 102.2424 |
| r96_shapes_ct10/sample_002 | 100.4289 | 3.5987 |
| r96_shapes_ct10/sample_003 | 100.3618 | 2.3554 |
| r96_shapes_ct10/sample_004 | 77.9043 | 1.5725 |
| r96_shapes_mri4/sample_000 | 82.2297 | 50.8089 |
| r96_shapes_mri4/sample_001 | 99.5716 | 100.7318 |
| r96_shapes_mri4/sample_002 | 99.6326 | 3.1759 |
| r96_shapes_mri4/sample_003 | 99.4967 | 2.4438 |
| r96_shapes_mri4/sample_004 | 97.8279 | 1.6963 |
| r128_shapes_ct10/sample_000 | 99.5985 | 103.9026 |
| r128_shapes_ct10/sample_001 | 99.7100 | 99.8081 |
| r128_shapes_ct10/sample_002 | 104.3587 | 3.2574 |
| r128_shapes_ct10/sample_003 | 104.8848 | 2.3365 |
| r128_shapes_ct10/sample_004 | 79.6018 | 78.0790 |
| r128_shapes_mri4/sample_000 | 99.3421 | 99.0653 |
| r128_shapes_mri4/sample_001 | 99.9842 | 99.7563 |
| r128_shapes_mri4/sample_002 | 100.0172 | 2.3284 |
| r128_shapes_mri4/sample_003 | 99.5989 | 99.6630 |
| r128_shapes_mri4/sample_004 | 73.4243 | 73.2097 |
| r160_shapes_ct10/sample_000 | 101.5283 | 101.4874 |
| r160_shapes_ct10/sample_001 | 100.3453 | 99.6651 |
| r160_shapes_ct10/sample_002 | 99.9719 | 100.5497 |
| r160_shapes_ct10/sample_003 | 101.9155 | 104.3569 |
| r160_shapes_ct10/sample_004 | 102.8103 | 1.3445 |
| r160_shapes_mri4/sample_000 | 99.6357 | 99.3419 |
| r160_shapes_mri4/sample_001 | 99.4535 | 1.6093 |
| r160_shapes_mri4/sample_002 | 100.2464 | 100.1726 |
| r160_shapes_mri4/sample_003 | 23.7943 | 3.1522 |
| r160_shapes_mri4/sample_004 | 73.1857 | 1.7821 |
| r200_shapes_ct10/sample_000 | 92.2966 | 52.0455 |
| r200_shapes_ct10/sample_001 | 100.3014 | 101.4998 |
| r200_shapes_ct10/sample_002 | 101.2903 | 100.9072 |
| r200_shapes_ct10/sample_003 | 100.4612 | 99.8987 |
| r200_shapes_ct10/sample_004 | 31.6637 | 100.2185 |
| r200_shapes_mri4/sample_000 | 99.6888 | 99.2956 |
| r200_shapes_mri4/sample_001 | 99.6028 | 99.4379 |
| r200_shapes_mri4/sample_002 | 97.9975 | 99.8364 |
| r200_shapes_mri4/sample_003 | 99.6854 | 99.5625 |
| r200_shapes_mri4/sample_004 | 99.5719 | 99.0032 |
| r250_shapes_ct10/sample_000 | 82.8371 | 86.9173 |
| r250_shapes_ct10/sample_001 | 99.9850 | 99.9963 |
| r250_shapes_ct10/sample_002 | 100.0471 | 99.9162 |
| r250_shapes_ct10/sample_003 | 111.5932 | 101.4963 |
| r250_shapes_ct10/sample_004 | 46.6633 | 2.4522 |
| r250_shapes_mri4/sample_000 | 99.2466 | 99.1457 |
| r250_shapes_mri4/sample_001 | 99.4823 | 99.4152 |
| r250_shapes_mri4/sample_002 | 99.6921 | 99.7242 |
| r250_shapes_mri4/sample_003 | 99.6892 | 99.6352 |
| r250_shapes_mri4/sample_004 | 88.5921 | 31.3417 |
