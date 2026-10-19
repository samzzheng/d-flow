# Five-step Euler pilot

0% relative measurement noise; CT10 and Cartesian MRI4; 5 test samples per acquisition at each resolution; separated shapes: 1–4 circles/triangles. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 256, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 978 | 2.0829 | 0.5155 | step_limit |
| r64_shapes_ct10/sample_001 | DLR D-Flow | 775 | 764 | 1.8798 | 0.5184 | stalled |
| r64_shapes_ct10/sample_002 | DLR D-Flow | 831 | 818 | 3.0766 | 0.5731 | stalled |
| r64_shapes_ct10/sample_003 | DLR D-Flow | 1000 | 971 | 9.1671 | 6.5048 | step_limit |
| r64_shapes_ct10/sample_004 | DLR D-Flow | 400 | 397 | 1.5023 | 0.2314 | stalled |
| r64_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 988 | 2.1521 | 0.7480 | step_limit |
| r64_shapes_mri4/sample_001 | DLR D-Flow | 1000 | 978 | 99.7869 | 99.2300 | step_limit |
| r64_shapes_mri4/sample_002 | DLR D-Flow | 100 | 97 | 101.9604 | 97.8568 | stalled |
| r64_shapes_mri4/sample_003 | DLR D-Flow | 538 | 533 | 3.5065 | 0.6556 | stalled |
| r64_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 965 | 3.3708 | 1.7734 | step_limit |
| r96_shapes_ct10/sample_000 | DLR D-Flow | 731 | 713 | 52.9588 | 41.6922 | stalled |
| r96_shapes_ct10/sample_001 | DLR D-Flow | 213 | 205 | 102.2107 | 94.6997 | stalled |
| r96_shapes_ct10/sample_002 | DLR D-Flow | 719 | 704 | 3.4016 | 0.6421 | stalled |
| r96_shapes_ct10/sample_003 | DLR D-Flow | 696 | 682 | 2.5423 | 0.5811 | stalled |
| r96_shapes_ct10/sample_004 | DLR D-Flow | 687 | 673 | 1.4479 | 0.2269 | stalled |
| r96_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 966 | 2.5640 | 1.3627 | step_limit |
| r96_shapes_mri4/sample_001 | DLR D-Flow | 546 | 521 | 100.2308 | 98.6645 | stalled |
| r96_shapes_mri4/sample_002 | DLR D-Flow | 1000 | 981 | 4.1488 | 1.2773 | step_limit |
| r96_shapes_mri4/sample_003 | DLR D-Flow | 613 | 602 | 100.4707 | 98.4473 | stalled |
| r96_shapes_mri4/sample_004 | DLR D-Flow | 633 | 614 | 71.2270 | 67.3405 | stalled |
| r128_shapes_ct10/sample_000 | DLR D-Flow | 629 | 616 | 100.3218 | 89.0274 | stalled |
| r128_shapes_ct10/sample_001 | DLR D-Flow | 48 | 45 | 100.3199 | 99.2964 | stalled |
| r128_shapes_ct10/sample_002 | DLR D-Flow | 1000 | 985 | 3.2606 | 0.8523 | step_limit |
| r128_shapes_ct10/sample_003 | DLR D-Flow | 547 | 527 | 2.0819 | 0.4985 | stalled |
| r128_shapes_ct10/sample_004 | DLR D-Flow | 434 | 426 | 77.4001 | 53.8665 | stalled |
| r128_shapes_mri4/sample_000 | DLR D-Flow | 262 | 255 | 99.0965 | 98.6649 | stalled |
| r128_shapes_mri4/sample_001 | DLR D-Flow | 1000 | 926 | 99.8779 | 98.7630 | step_limit |
| r128_shapes_mri4/sample_002 | DLR D-Flow | 147 | 141 | 100.1033 | 98.5217 | stalled |
| r128_shapes_mri4/sample_003 | DLR D-Flow | 175 | 168 | 99.6614 | 98.9973 | stalled |
| r128_shapes_mri4/sample_004 | DLR D-Flow | 429 | 420 | 73.1844 | 71.7421 | stalled |
| r160_shapes_ct10/sample_000 | DLR D-Flow | 151 | 146 | 101.3204 | 93.2756 | stalled |
| r160_shapes_ct10/sample_001 | DLR D-Flow | 13 | 11 | 100.3335 | 96.0617 | stalled |
| r160_shapes_ct10/sample_002 | DLR D-Flow | 179 | 172 | 100.6450 | 93.6723 | stalled |
| r160_shapes_ct10/sample_003 | DLR D-Flow | 429 | 421 | 105.4311 | 91.8348 | stalled |
| r160_shapes_ct10/sample_004 | DLR D-Flow | 996 | 979 | 1.6304 | 0.2498 | stalled |
| r160_shapes_mri4/sample_000 | DLR D-Flow | 326 | 320 | 99.1295 | 98.5699 | stalled |
| r160_shapes_mri4/sample_001 | DLR D-Flow | 1000 | 969 | 1.6693 | 0.8571 | step_limit |
| r160_shapes_mri4/sample_002 | DLR D-Flow | 110 | 105 | 100.1380 | 98.8416 | stalled |
| r160_shapes_mri4/sample_003 | DLR D-Flow | 814 | 789 | 1.7575 | 1.0937 | stalled |
| r160_shapes_mri4/sample_004 | DLR D-Flow | 756 | 744 | 66.3377 | 63.4747 | stalled |
| r200_shapes_ct10/sample_000 | DLR D-Flow | 516 | 503 | 52.2297 | 42.1399 | stalled |
| r200_shapes_ct10/sample_001 | DLR D-Flow | 226 | 219 | 99.9319 | 95.3221 | stalled |
| r200_shapes_ct10/sample_002 | DLR D-Flow | 684 | 669 | 100.4750 | 93.6342 | stalled |
| r200_shapes_ct10/sample_003 | DLR D-Flow | 304 | 298 | 99.7867 | 93.1330 | stalled |
| r200_shapes_ct10/sample_004 | DLR D-Flow | 209 | 206 | 82.1497 | 54.1050 | stalled |
| r200_shapes_mri4/sample_000 | DLR D-Flow | 194 | 183 | 99.2912 | 99.1431 | stalled |
| r200_shapes_mri4/sample_001 | DLR D-Flow | 436 | 427 | 99.4164 | 99.2104 | stalled |
| r200_shapes_mri4/sample_002 | DLR D-Flow | 133 | 128 | 99.7273 | 99.3846 | stalled |
| r200_shapes_mri4/sample_003 | DLR D-Flow | 211 | 206 | 99.5779 | 99.3865 | stalled |
| r200_shapes_mri4/sample_004 | DLR D-Flow | 413 | 400 | 99.2240 | 99.0117 | stalled |
| r250_shapes_ct10/sample_000 | DLR D-Flow | 251 | 245 | 92.3214 | 71.6169 | stalled |
| r250_shapes_ct10/sample_001 | DLR D-Flow | 348 | 340 | 99.5553 | 94.2315 | stalled |
| r250_shapes_ct10/sample_002 | DLR D-Flow | 308 | 304 | 100.8493 | 94.9900 | stalled |
| r250_shapes_ct10/sample_003 | DLR D-Flow | 89 | 85 | 99.9830 | 93.1526 | stalled |
| r250_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 987 | 37.6912 | 19.2364 | step_limit |
| r250_shapes_mri4/sample_000 | DLR D-Flow | 233 | 226 | 99.1136 | 98.8323 | stalled |
| r250_shapes_mri4/sample_001 | DLR D-Flow | 120 | 116 | 99.4550 | 99.3695 | stalled |
| r250_shapes_mri4/sample_002 | DLR D-Flow | 347 | 340 | 99.6846 | 99.4887 | stalled |
| r250_shapes_mri4/sample_003 | DLR D-Flow | 30 | 27 | 99.5446 | 99.4320 | stalled |
| r250_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 990 | 31.2798 | 30.0423 | step_limit |

Terminal method runs: 60/60. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.

Original D/P controls and rank-8 results: /home/samzheng/d-flow/outputs/separated_shapes_noiseless_20261016. Completed rank pairs: 60.

| Paired case | Rank 8 image error % | Rank 256 image error % |
| --- | ---: | ---: |
| r64_shapes_ct10/sample_000 | 78.8474 | 2.0829 |
| r64_shapes_ct10/sample_001 | 102.8367 | 1.8798 |
| r64_shapes_ct10/sample_002 | 100.8987 | 3.0766 |
| r64_shapes_ct10/sample_003 | 103.1740 | 9.1671 |
| r64_shapes_ct10/sample_004 | 30.2782 | 1.5023 |
| r64_shapes_mri4/sample_000 | 76.3618 | 2.1521 |
| r64_shapes_mri4/sample_001 | 99.8533 | 99.7869 |
| r64_shapes_mri4/sample_002 | 101.7113 | 101.9604 |
| r64_shapes_mri4/sample_003 | 99.8358 | 3.5065 |
| r64_shapes_mri4/sample_004 | 73.9069 | 3.3708 |
| r96_shapes_ct10/sample_000 | 84.8165 | 52.9588 |
| r96_shapes_ct10/sample_001 | 102.3571 | 102.2107 |
| r96_shapes_ct10/sample_002 | 100.4289 | 3.4016 |
| r96_shapes_ct10/sample_003 | 100.3618 | 2.5423 |
| r96_shapes_ct10/sample_004 | 77.9043 | 1.4479 |
| r96_shapes_mri4/sample_000 | 82.2297 | 2.5640 |
| r96_shapes_mri4/sample_001 | 99.5716 | 100.2308 |
| r96_shapes_mri4/sample_002 | 99.6326 | 4.1488 |
| r96_shapes_mri4/sample_003 | 99.4967 | 100.4707 |
| r96_shapes_mri4/sample_004 | 97.8279 | 71.2270 |
| r128_shapes_ct10/sample_000 | 99.5985 | 100.3218 |
| r128_shapes_ct10/sample_001 | 99.7100 | 100.3199 |
| r128_shapes_ct10/sample_002 | 104.3587 | 3.2606 |
| r128_shapes_ct10/sample_003 | 104.8848 | 2.0819 |
| r128_shapes_ct10/sample_004 | 79.6018 | 77.4001 |
| r128_shapes_mri4/sample_000 | 99.3421 | 99.0965 |
| r128_shapes_mri4/sample_001 | 99.9842 | 99.8779 |
| r128_shapes_mri4/sample_002 | 100.0172 | 100.1033 |
| r128_shapes_mri4/sample_003 | 99.5989 | 99.6614 |
| r128_shapes_mri4/sample_004 | 73.4243 | 73.1844 |
| r160_shapes_ct10/sample_000 | 101.5283 | 101.3204 |
| r160_shapes_ct10/sample_001 | 100.3453 | 100.3335 |
| r160_shapes_ct10/sample_002 | 99.9719 | 100.6450 |
| r160_shapes_ct10/sample_003 | 101.9155 | 105.4311 |
| r160_shapes_ct10/sample_004 | 102.8103 | 1.6304 |
| r160_shapes_mri4/sample_000 | 99.6357 | 99.1295 |
| r160_shapes_mri4/sample_001 | 99.4535 | 1.6693 |
| r160_shapes_mri4/sample_002 | 100.2464 | 100.1380 |
| r160_shapes_mri4/sample_003 | 23.7943 | 1.7575 |
| r160_shapes_mri4/sample_004 | 73.1857 | 66.3377 |
| r200_shapes_ct10/sample_000 | 92.2966 | 52.2297 |
| r200_shapes_ct10/sample_001 | 100.3014 | 99.9319 |
| r200_shapes_ct10/sample_002 | 101.2903 | 100.4750 |
| r200_shapes_ct10/sample_003 | 100.4612 | 99.7867 |
| r200_shapes_ct10/sample_004 | 31.6637 | 82.1497 |
| r200_shapes_mri4/sample_000 | 99.6888 | 99.2912 |
| r200_shapes_mri4/sample_001 | 99.6028 | 99.4164 |
| r200_shapes_mri4/sample_002 | 97.9975 | 99.7273 |
| r200_shapes_mri4/sample_003 | 99.6854 | 99.5779 |
| r200_shapes_mri4/sample_004 | 99.5719 | 99.2240 |
| r250_shapes_ct10/sample_000 | 82.8371 | 92.3214 |
| r250_shapes_ct10/sample_001 | 99.9850 | 99.5553 |
| r250_shapes_ct10/sample_002 | 100.0471 | 100.8493 |
| r250_shapes_ct10/sample_003 | 111.5932 | 99.9830 |
| r250_shapes_ct10/sample_004 | 46.6633 | 37.6912 |
| r250_shapes_mri4/sample_000 | 99.2466 | 99.1136 |
| r250_shapes_mri4/sample_001 | 99.4823 | 99.4550 |
| r250_shapes_mri4/sample_002 | 99.6921 | 99.6846 |
| r250_shapes_mri4/sample_003 | 99.6892 | 99.5446 |
| r250_shapes_mri4/sample_004 | 88.5921 | 31.2798 |
