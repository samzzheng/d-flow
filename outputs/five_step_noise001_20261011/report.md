# Five-step Euler pilot

1% relative measurement noise; CT10 and Cartesian MRI4; 30 single-circle test samples at each resolution. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank8, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 2.7033 | 0.5725 | step_limit |
| r64_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 3.1022 | 0.3111 | step_limit |
| r64_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 4.3836 | 0.1729 | step_limit |
| r64_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 5.2625 | 0.1343 | step_limit |
| r64_n1_ct10/sample_004 | D-Flow | 539 | 536 | 99.8740 | 87.1134 | stalled |
| r64_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 3.0732 | 0.3660 | step_limit |
| r64_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 3.0093 | 0.4723 | step_limit |
| r64_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 3.2564 | 0.3226 | step_limit |
| r64_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 3.8728 | 0.2003 | step_limit |
| r64_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 3.6429 | 0.2275 | step_limit |
| r64_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 4.8597 | 0.1529 | step_limit |
| r64_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 4.4748 | 0.2184 | step_limit |
| r64_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 7.5699 | 0.0638 | step_limit |
| r64_n1_ct10/sample_013 | D-Flow | 1000 | 1000 | 4.0619 | 0.2476 | step_limit |
| r64_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 3.3847 | 0.2468 | step_limit |
| r64_n1_ct10/sample_015 | D-Flow | 1000 | 1000 | 2.6120 | 0.4235 | step_limit |
| r64_n1_ct10/sample_016 | D-Flow | 1000 | 1000 | 2.8799 | 0.3878 | step_limit |
| r64_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 3.2881 | 0.1902 | step_limit |
| r64_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 5.9048 | 0.1314 | step_limit |
| r64_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 5.5456 | 0.2294 | step_limit |
| r64_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 3.6935 | 0.2400 | step_limit |
| r64_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 2.7039 | 0.5073 | step_limit |
| r64_n1_ct10/sample_022 | D-Flow | 1000 | 1000 | 8.1337 | 0.0554 | step_limit |
| r64_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 9.9325 | 0.0634 | step_limit |
| r64_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 10.4722 | 0.0906 | step_limit |
| r64_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 5.3957 | 0.2151 | step_limit |
| r64_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 4.6636 | 0.1956 | step_limit |
| r64_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 3.1942 | 0.2560 | step_limit |
| r64_n1_ct10/sample_028 | D-Flow | 1000 | 1000 | 4.3033 | 0.2755 | step_limit |
| r64_n1_ct10/sample_029 | D-Flow | 235 | 233 | 1.8621 | 0.9700 | stalled |
| r64_n1_ct10/sample_000 | P-Flow | 508 | 411 | 23.6532 | 10.2463 | stalled |
| r64_n1_ct10/sample_001 | P-Flow | 96 | 79 | 3.1968 | 1.4595 | stalled |
| r64_n1_ct10/sample_002 | P-Flow | 1000 | 807 | 2.2478 | 1.0394 | step_limit |
| r64_n1_ct10/sample_003 | P-Flow | 80 | 67 | 2.4982 | 1.1603 | stalled |
| r64_n1_ct10/sample_004 | P-Flow | 505 | 492 | 5.2738 | 0.1426 | stalled |
| r64_n1_ct10/sample_005 | P-Flow | 35 | 28 | 4.2429 | 1.6627 | stalled |
| r64_n1_ct10/sample_006 | P-Flow | 48 | 41 | 1.6860 | 1.0530 | stalled |
| r64_n1_ct10/sample_007 | P-Flow | 674 | 611 | 5.4266 | 0.1840 | stalled |
| r64_n1_ct10/sample_008 | P-Flow | 38 | 32 | 3.6653 | 1.6159 | stalled |
| r64_n1_ct10/sample_009 | P-Flow | 1000 | 825 | 4.6332 | 0.3477 | step_limit |
| r64_n1_ct10/sample_010 | P-Flow | 76 | 63 | 7.6862 | 3.5700 | stalled |
| r64_n1_ct10/sample_011 | P-Flow | 1000 | 936 | 5.5860 | 0.1767 | step_limit |
| r64_n1_ct10/sample_012 | P-Flow | 98 | 77 | 2.7867 | 1.2522 | stalled |
| r64_n1_ct10/sample_013 | P-Flow | 1000 | 800 | 2.3700 | 1.0943 | step_limit |
| r64_n1_ct10/sample_014 | P-Flow | 64 | 57 | 5.2463 | 3.3446 | stalled |
| r64_n1_ct10/sample_015 | P-Flow | 88 | 78 | 2.3431 | 1.2064 | stalled |
| r64_n1_ct10/sample_016 | P-Flow | 79 | 65 | 15.7585 | 7.9755 | stalled |
| r64_n1_ct10/sample_017 | P-Flow | 92 | 81 | 2.1795 | 1.0992 | stalled |
| r64_n1_ct10/sample_018 | P-Flow | 67 | 55 | 4.5633 | 2.1977 | stalled |
| r64_n1_ct10/sample_019 | P-Flow | 73 | 63 | 2.2836 | 1.4715 | stalled |
| r64_n1_ct10/sample_020 | P-Flow | 149 | 132 | 2.0998 | 1.0881 | stalled |
| r64_n1_ct10/sample_021 | P-Flow | 163 | 139 | 2.9198 | 1.2558 | stalled |
| r64_n1_ct10/sample_022 | P-Flow | 67 | 54 | 9.8911 | 4.3543 | stalled |
| r64_n1_ct10/sample_023 | P-Flow | 1000 | 928 | 5.6621 | 0.1884 | step_limit |
| r64_n1_ct10/sample_024 | P-Flow | 18 | 15 | 17.9708 | 6.5564 | stalled |
| r64_n1_ct10/sample_025 | P-Flow | 123 | 107 | 2.3759 | 0.9890 | stalled |
| r64_n1_ct10/sample_026 | P-Flow | 138 | 121 | 2.2134 | 1.1462 | stalled |
| r64_n1_ct10/sample_027 | P-Flow | 33 | 28 | 3.8649 | 1.7358 | stalled |
| r64_n1_ct10/sample_028 | P-Flow | 72 | 62 | 1.9180 | 1.0747 | stalled |
| r64_n1_ct10/sample_029 | P-Flow | 154 | 126 | 1.9549 | 1.0736 | stalled |
| r64_n1_ct10/sample_000 | DLR D-Flow | 1000 | 971 | 18.9215 | 8.4280 | step_limit |
| r64_n1_ct10/sample_001 | DLR D-Flow | 408 | 381 | 2.5632 | 1.1751 | stalled |
| r64_n1_ct10/sample_002 | DLR D-Flow | 262 | 246 | 2.5164 | 1.1428 | stalled |
| r64_n1_ct10/sample_003 | DLR D-Flow | 265 | 251 | 2.1514 | 1.1354 | stalled |
| r64_n1_ct10/sample_004 | DLR D-Flow | 17 | 14 | 99.7724 | 90.1761 | stalled |
| r64_n1_ct10/sample_005 | DLR D-Flow | 793 | 750 | 2.9989 | 1.4398 | stalled |
| r64_n1_ct10/sample_006 | DLR D-Flow | 124 | 121 | 1.7033 | 1.0440 | stalled |
| r64_n1_ct10/sample_007 | DLR D-Flow | 195 | 183 | 1.6627 | 1.0504 | stalled |
| r64_n1_ct10/sample_008 | DLR D-Flow | 424 | 402 | 1.7212 | 1.0488 | stalled |
| r64_n1_ct10/sample_009 | DLR D-Flow | 394 | 375 | 1.8803 | 1.0526 | stalled |
| r64_n1_ct10/sample_010 | DLR D-Flow | 235 | 224 | 1.6894 | 1.0384 | stalled |
| r64_n1_ct10/sample_011 | DLR D-Flow | 98 | 94 | 2.0434 | 1.0407 | stalled |
| r64_n1_ct10/sample_012 | DLR D-Flow | 244 | 236 | 3.7307 | 1.2234 | stalled |
| r64_n1_ct10/sample_013 | DLR D-Flow | 199 | 193 | 2.3055 | 1.0867 | stalled |
| r64_n1_ct10/sample_014 | DLR D-Flow | 452 | 440 | 10.3372 | 5.0127 | stalled |
| r64_n1_ct10/sample_015 | DLR D-Flow | 398 | 384 | 2.4237 | 1.2299 | stalled |
| r64_n1_ct10/sample_016 | DLR D-Flow | 1000 | 940 | 3.2988 | 1.3410 | step_limit |
| r64_n1_ct10/sample_017 | DLR D-Flow | 1000 | 946 | 2.1177 | 1.0840 | step_limit |
| r64_n1_ct10/sample_018 | DLR D-Flow | 484 | 472 | 6.3813 | 4.3375 | stalled |
| r64_n1_ct10/sample_019 | DLR D-Flow | 208 | 198 | 1.7647 | 1.0459 | stalled |
| r64_n1_ct10/sample_020 | DLR D-Flow | 274 | 264 | 1.8347 | 1.0539 | stalled |
| r64_n1_ct10/sample_021 | DLR D-Flow | 45 | 41 | 100.3741 | 89.1745 | stalled |
| r64_n1_ct10/sample_022 | DLR D-Flow | 121 | 116 | 2.0431 | 1.0575 | stalled |
| r64_n1_ct10/sample_023 | DLR D-Flow | 117 | 113 | 2.2067 | 1.0855 | stalled |
| r64_n1_ct10/sample_024 | DLR D-Flow | 213 | 198 | 1.6926 | 1.0470 | stalled |
| r64_n1_ct10/sample_025 | DLR D-Flow | 155 | 146 | 2.0045 | 1.0855 | stalled |
| r64_n1_ct10/sample_026 | DLR D-Flow | 1000 | 937 | 2.2504 | 1.1765 | step_limit |
| r64_n1_ct10/sample_027 | DLR D-Flow | 141 | 134 | 1.4647 | 1.0513 | stalled |
| r64_n1_ct10/sample_028 | DLR D-Flow | 217 | 211 | 4.0835 | 1.5183 | stalled |
| r64_n1_ct10/sample_029 | DLR D-Flow | 153 | 144 | 1.7738 | 1.0754 | stalled |
| r64_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.4882 | 0.7006 | step_limit |
| r64_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.6799 | 0.5550 | step_limit |
| r64_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.3608 | 0.4716 | step_limit |
| r64_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.4582 | 0.5161 | step_limit |
| r64_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.6993 | 1.0718 | step_limit |
| r64_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.2453 | 0.5194 | step_limit |
| r64_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.4961 | 0.5397 | step_limit |
| r64_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.3036 | 0.5400 | step_limit |
| r64_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.3233 | 0.7721 | step_limit |
| r64_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.4563 | 0.7400 | step_limit |
| r64_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.3790 | 0.5174 | step_limit |
| r64_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.6422 | 0.4155 | step_limit |
| r64_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.5673 | 0.5667 | step_limit |
| r64_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 1.7664 | 0.6046 | step_limit |
| r64_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 1.5322 | 0.6465 | step_limit |
| r64_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 1.6958 | 0.7207 | step_limit |
| r64_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 1.6664 | 0.5531 | step_limit |
| r64_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 1.4431 | 0.6210 | step_limit |
| r64_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 1.2648 | 0.6514 | step_limit |
| r64_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 1.4687 | 0.6518 | step_limit |
| r64_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 1.3668 | 0.5911 | step_limit |
| r64_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 1.4062 | 0.4833 | step_limit |
| r64_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 1.4379 | 0.6083 | step_limit |
| r64_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 1.5301 | 0.5780 | step_limit |
| r64_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 1.2892 | 0.5687 | step_limit |
| r64_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 1.3107 | 0.4795 | step_limit |
| r64_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 1.5616 | 0.5204 | step_limit |
| r64_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 1.3301 | 0.6162 | step_limit |
| r64_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 1.4787 | 0.6538 | step_limit |
| r64_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 1.3365 | 0.6675 | step_limit |
| r64_n1_mri4/sample_000 | P-Flow | 408 | 360 | 2.7380 | 0.5861 | stalled |
| r64_n1_mri4/sample_001 | P-Flow | 214 | 182 | 2.6247 | 1.3115 | stalled |
| r64_n1_mri4/sample_002 | P-Flow | 193 | 174 | 2.4567 | 1.0637 | stalled |
| r64_n1_mri4/sample_003 | P-Flow | 115 | 104 | 2.0313 | 1.4319 | stalled |
| r64_n1_mri4/sample_004 | P-Flow | 121 | 113 | 3.1263 | 1.2654 | stalled |
| r64_n1_mri4/sample_005 | P-Flow | 86 | 76 | 1.4341 | 1.2790 | stalled |
| r64_n1_mri4/sample_006 | P-Flow | 173 | 154 | 2.0136 | 1.1233 | stalled |
| r64_n1_mri4/sample_007 | P-Flow | 237 | 209 | 2.6307 | 0.7510 | stalled |
| r64_n1_mri4/sample_008 | P-Flow | 23 | 19 | 2.0577 | 1.6380 | stalled |
| r64_n1_mri4/sample_009 | P-Flow | 245 | 217 | 2.8629 | 1.1001 | stalled |
| r64_n1_mri4/sample_010 | P-Flow | 71 | 61 | 1.5448 | 1.2503 | stalled |
| r64_n1_mri4/sample_011 | P-Flow | 278 | 250 | 3.0083 | 0.7679 | stalled |
| r64_n1_mri4/sample_012 | P-Flow | 36 | 30 | 1.7919 | 1.3720 | stalled |
| r64_n1_mri4/sample_013 | P-Flow | 366 | 320 | 2.9849 | 0.7659 | stalled |
| r64_n1_mri4/sample_014 | P-Flow | 46 | 39 | 2.9327 | 2.3553 | stalled |
| r64_n1_mri4/sample_015 | P-Flow | 150 | 129 | 2.0452 | 1.5388 | stalled |
| r64_n1_mri4/sample_016 | P-Flow | 346 | 310 | 3.1626 | 0.6782 | stalled |
| r64_n1_mri4/sample_017 | P-Flow | 176 | 152 | 1.9238 | 1.1268 | stalled |
| r64_n1_mri4/sample_018 | P-Flow | 64 | 58 | 1.3452 | 1.2022 | stalled |
| r64_n1_mri4/sample_019 | P-Flow | 342 | 284 | 2.1665 | 1.1234 | stalled |
| r64_n1_mri4/sample_020 | P-Flow | 119 | 101 | 1.7185 | 1.3401 | stalled |
| r64_n1_mri4/sample_021 | P-Flow | 290 | 266 | 3.1038 | 0.4928 | stalled |
| r64_n1_mri4/sample_022 | P-Flow | 404 | 350 | 2.4282 | 1.0844 | stalled |
| r64_n1_mri4/sample_023 | P-Flow | 462 | 417 | 3.4321 | 0.5844 | stalled |
| r64_n1_mri4/sample_024 | P-Flow | 252 | 223 | 2.1910 | 0.9490 | stalled |
| r64_n1_mri4/sample_025 | P-Flow | 129 | 113 | 1.7315 | 1.4182 | stalled |
| r64_n1_mri4/sample_026 | P-Flow | 229 | 203 | 2.5745 | 1.1542 | stalled |
| r64_n1_mri4/sample_027 | P-Flow | 100 | 85 | 1.4036 | 1.2894 | stalled |
| r64_n1_mri4/sample_028 | P-Flow | 255 | 212 | 2.0992 | 1.2182 | stalled |
| r64_n1_mri4/sample_029 | P-Flow | 112 | 92 | 1.6277 | 1.3140 | stalled |
| r64_n1_mri4/sample_000 | DLR D-Flow | 63 | 60 | 77.2923 | 76.8089 | stalled |
| r64_n1_mri4/sample_001 | DLR D-Flow | 18 | 16 | 12.0636 | 10.5319 | stalled |
| r64_n1_mri4/sample_002 | DLR D-Flow | 305 | 293 | 1.8431 | 1.5270 | stalled |
| r64_n1_mri4/sample_003 | DLR D-Flow | 570 | 540 | 2.0081 | 1.6581 | stalled |
| r64_n1_mri4/sample_004 | DLR D-Flow | 103 | 98 | 100.3971 | 95.7189 | stalled |
| r64_n1_mri4/sample_005 | DLR D-Flow | 946 | 909 | 1.5396 | 1.4041 | stalled |
| r64_n1_mri4/sample_006 | DLR D-Flow | 160 | 157 | 2.0782 | 1.5745 | stalled |
| r64_n1_mri4/sample_007 | DLR D-Flow | 484 | 452 | 1.5969 | 1.3360 | stalled |
| r64_n1_mri4/sample_008 | DLR D-Flow | 1000 | 959 | 1.6558 | 1.3098 | step_limit |
| r64_n1_mri4/sample_009 | DLR D-Flow | 897 | 865 | 1.8010 | 1.5136 | stalled |
| r64_n1_mri4/sample_010 | DLR D-Flow | 128 | 120 | 3.8367 | 3.4438 | stalled |
| r64_n1_mri4/sample_011 | DLR D-Flow | 289 | 275 | 1.6582 | 1.3876 | stalled |
| r64_n1_mri4/sample_012 | DLR D-Flow | 315 | 304 | 2.1046 | 1.5749 | stalled |
| r64_n1_mri4/sample_013 | DLR D-Flow | 339 | 322 | 4.7183 | 4.3283 | stalled |
| r64_n1_mri4/sample_014 | DLR D-Flow | 172 | 165 | 3.3747 | 2.6564 | stalled |
| r64_n1_mri4/sample_015 | DLR D-Flow | 1000 | 964 | 2.2715 | 1.7335 | step_limit |
| r64_n1_mri4/sample_016 | DLR D-Flow | 1000 | 960 | 3.3181 | 2.6128 | step_limit |
| r64_n1_mri4/sample_017 | DLR D-Flow | 1000 | 948 | 2.0452 | 1.5511 | step_limit |
| r64_n1_mri4/sample_018 | DLR D-Flow | 321 | 308 | 2.4991 | 1.7006 | stalled |
| r64_n1_mri4/sample_019 | DLR D-Flow | 195 | 187 | 1.8177 | 1.3960 | stalled |
| r64_n1_mri4/sample_020 | DLR D-Flow | 133 | 122 | 1.7890 | 1.4111 | stalled |
| r64_n1_mri4/sample_021 | DLR D-Flow | 458 | 442 | 2.0222 | 1.5561 | stalled |
| r64_n1_mri4/sample_022 | DLR D-Flow | 45 | 43 | 2.2310 | 1.6024 | stalled |
| r64_n1_mri4/sample_023 | DLR D-Flow | 100 | 94 | 1.9768 | 1.5173 | stalled |
| r64_n1_mri4/sample_024 | DLR D-Flow | 467 | 446 | 1.6905 | 1.4338 | stalled |
| r64_n1_mri4/sample_025 | DLR D-Flow | 194 | 184 | 1.8285 | 1.5109 | stalled |
| r64_n1_mri4/sample_026 | DLR D-Flow | 992 | 942 | 2.1228 | 1.6993 | stalled |
| r64_n1_mri4/sample_027 | DLR D-Flow | 596 | 570 | 1.3844 | 1.2717 | stalled |
| r64_n1_mri4/sample_028 | DLR D-Flow | 792 | 760 | 2.0308 | 1.4634 | stalled |
| r64_n1_mri4/sample_029 | DLR D-Flow | 159 | 150 | 1.7705 | 1.4564 | stalled |
| r160_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 4.6765 | 0.2890 | step_limit |
| r160_n1_ct10/sample_001 | D-Flow | 81 | 79 | 1.8641 | 1.0357 | stalled |
| r160_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 3.9452 | 0.3086 | step_limit |
| r160_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 3.2382 | 0.5796 | step_limit |
| r160_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 4.2300 | 0.2046 | step_limit |
| r160_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 4.2384 | 0.1048 | step_limit |
| r160_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 5.4669 | 0.0606 | step_limit |
| r160_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 6.1434 | 0.0984 | step_limit |
| r160_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 3.7096 | 0.2117 | step_limit |
| r160_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 2.2646 | 0.9090 | step_limit |
| r160_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 4.4310 | 0.2134 | step_limit |
| r160_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 4.5780 | 0.1477 | step_limit |
| r160_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 5.2269 | 0.1211 | step_limit |
| r160_n1_ct10/sample_013 | D-Flow | 1000 | 1000 | 4.2631 | 0.1671 | step_limit |
| r160_n1_ct10/sample_014 | D-Flow | 160 | 158 | 2.0622 | 1.0547 | stalled |
| r160_n1_ct10/sample_015 | D-Flow | 182 | 180 | 99.9383 | 86.8888 | stalled |
| r160_n1_ct10/sample_016 | D-Flow | 1000 | 1000 | 3.9709 | 0.2344 | step_limit |
| r160_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 3.5644 | 0.2491 | step_limit |
| r160_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 4.2081 | 0.0941 | step_limit |
| r160_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 6.1809 | 0.1585 | step_limit |
| r160_n1_ct10/sample_020 | D-Flow | 20 | 17 | 100.2087 | 90.1328 | stalled |
| r160_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 7.7664 | 0.0853 | step_limit |
| r160_n1_ct10/sample_022 | D-Flow | 89 | 84 | 93.3758 | 78.8102 | stalled |
| r160_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 4.3927 | 0.2485 | step_limit |
| r160_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 5.1420 | 0.3137 | step_limit |
| r160_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 5.5190 | 0.0898 | step_limit |
| r160_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 2.4298 | 0.8493 | step_limit |
| r160_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 5.1513 | 0.1220 | step_limit |
| r160_n1_ct10/sample_028 | D-Flow | 234 | 231 | 94.9744 | 79.7118 | stalled |
| r160_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 4.4206 | 0.4293 | step_limit |
| r160_n1_ct10/sample_000 | P-Flow | 88 | 72 | 3.9334 | 1.8586 | stalled |
| r160_n1_ct10/sample_001 | P-Flow | 47 | 39 | 9.4493 | 3.9985 | stalled |
| r160_n1_ct10/sample_002 | P-Flow | 1000 | 820 | 2.5104 | 1.5140 | step_limit |
| r160_n1_ct10/sample_003 | P-Flow | 1000 | 936 | 6.0633 | 0.1136 | step_limit |
| r160_n1_ct10/sample_004 | P-Flow | 21 | 17 | 9.5450 | 2.9834 | stalled |
| r160_n1_ct10/sample_005 | P-Flow | 51 | 42 | 6.7445 | 2.7794 | stalled |
| r160_n1_ct10/sample_006 | P-Flow | 22 | 18 | 4.5525 | 1.6320 | stalled |
| r160_n1_ct10/sample_007 | P-Flow | 74 | 64 | 1.3028 | 1.0141 | stalled |
| r160_n1_ct10/sample_008 | P-Flow | 590 | 472 | 2.6322 | 1.1223 | stalled |
| r160_n1_ct10/sample_009 | P-Flow | 295 | 237 | 1.3822 | 1.0228 | stalled |
| r160_n1_ct10/sample_010 | P-Flow | 38 | 31 | 1.3833 | 1.0155 | stalled |
| r160_n1_ct10/sample_011 | P-Flow | 117 | 101 | 1.8605 | 1.0645 | stalled |
| r160_n1_ct10/sample_012 | P-Flow | 1000 | 788 | 1.4051 | 1.0068 | step_limit |
| r160_n1_ct10/sample_013 | P-Flow | 45 | 40 | 3.0511 | 1.1921 | stalled |
| r160_n1_ct10/sample_014 | P-Flow | 287 | 236 | 3.2557 | 2.2958 | stalled |
| r160_n1_ct10/sample_015 | P-Flow | 1000 | 990 | 2.3886 | 1.1048 | step_limit |
| r160_n1_ct10/sample_016 | P-Flow | 39 | 34 | 2.3805 | 1.1007 | stalled |
| r160_n1_ct10/sample_017 | P-Flow | 217 | 178 | 4.3995 | 1.3612 | stalled |
| r160_n1_ct10/sample_018 | P-Flow | 558 | 449 | 3.1124 | 1.1857 | stalled |
| r160_n1_ct10/sample_019 | P-Flow | 42 | 36 | 4.7153 | 1.8518 | stalled |
| r160_n1_ct10/sample_020 | P-Flow | 29 | 23 | 2.6118 | 1.1629 | stalled |
| r160_n1_ct10/sample_021 | P-Flow | 604 | 493 | 1.5279 | 1.0235 | stalled |
| r160_n1_ct10/sample_022 | P-Flow | 105 | 87 | 1.6241 | 1.0115 | stalled |
| r160_n1_ct10/sample_023 | P-Flow | 90 | 75 | 3.6690 | 1.3421 | stalled |
| r160_n1_ct10/sample_024 | P-Flow | 577 | 450 | 2.5091 | 1.2293 | stalled |
| r160_n1_ct10/sample_025 | P-Flow | 99 | 81 | 1.7813 | 1.0483 | stalled |
| r160_n1_ct10/sample_026 | P-Flow | 27 | 23 | 2.1727 | 1.1223 | stalled |
| r160_n1_ct10/sample_027 | P-Flow | 588 | 474 | 1.4587 | 1.0115 | stalled |
| r160_n1_ct10/sample_028 | P-Flow | 1000 | 964 | 6.1017 | 0.1064 | step_limit |
| r160_n1_ct10/sample_029 | P-Flow | 811 | 625 | 1.2990 | 1.0181 | stalled |
| r160_n1_ct10/sample_000 | DLR D-Flow | 148 | 141 | 92.5842 | 75.2545 | stalled |
| r160_n1_ct10/sample_001 | DLR D-Flow | 262 | 250 | 2.0534 | 1.0331 | stalled |
| r160_n1_ct10/sample_002 | DLR D-Flow | 34 | 31 | 96.9845 | 83.3800 | stalled |
| r160_n1_ct10/sample_003 | DLR D-Flow | 269 | 262 | 1.4434 | 1.0218 | stalled |
| r160_n1_ct10/sample_004 | DLR D-Flow | 414 | 395 | 1.4322 | 1.0170 | stalled |
| r160_n1_ct10/sample_005 | DLR D-Flow | 866 | 835 | 1.8567 | 1.0244 | stalled |
| r160_n1_ct10/sample_006 | DLR D-Flow | 353 | 330 | 2.6641 | 1.1077 | stalled |
| r160_n1_ct10/sample_007 | DLR D-Flow | 302 | 290 | 2.7019 | 1.0345 | stalled |
| r160_n1_ct10/sample_008 | DLR D-Flow | 110 | 106 | 2.2991 | 1.0691 | stalled |
| r160_n1_ct10/sample_009 | DLR D-Flow | 327 | 315 | 6.4950 | 1.3523 | stalled |
| r160_n1_ct10/sample_010 | DLR D-Flow | 257 | 244 | 1.3767 | 1.0047 | stalled |
| r160_n1_ct10/sample_011 | DLR D-Flow | 465 | 446 | 1.3261 | 1.0070 | stalled |
| r160_n1_ct10/sample_012 | DLR D-Flow | 209 | 199 | 1.3821 | 1.0089 | stalled |
| r160_n1_ct10/sample_013 | DLR D-Flow | 543 | 517 | 2.7999 | 1.1019 | stalled |
| r160_n1_ct10/sample_014 | DLR D-Flow | 372 | 353 | 4.3779 | 3.3091 | stalled |
| r160_n1_ct10/sample_015 | DLR D-Flow | 22 | 20 | 107.0318 | 91.7774 | stalled |
| r160_n1_ct10/sample_016 | DLR D-Flow | 302 | 293 | 1.7497 | 1.0243 | stalled |
| r160_n1_ct10/sample_017 | DLR D-Flow | 523 | 500 | 4.1030 | 1.3033 | stalled |
| r160_n1_ct10/sample_018 | DLR D-Flow | 369 | 357 | 5.6329 | 1.9036 | stalled |
| r160_n1_ct10/sample_019 | DLR D-Flow | 285 | 275 | 3.4640 | 1.1676 | stalled |
| r160_n1_ct10/sample_020 | DLR D-Flow | 30 | 27 | 114.5774 | 98.8255 | stalled |
| r160_n1_ct10/sample_021 | DLR D-Flow | 404 | 395 | 1.8715 | 1.1064 | stalled |
| r160_n1_ct10/sample_022 | DLR D-Flow | 27 | 21 | 94.1775 | 81.3763 | stalled |
| r160_n1_ct10/sample_023 | DLR D-Flow | 20 | 17 | 96.4926 | 81.5482 | stalled |
| r160_n1_ct10/sample_024 | DLR D-Flow | 356 | 336 | 1.5182 | 1.0202 | stalled |
| r160_n1_ct10/sample_025 | DLR D-Flow | 396 | 369 | 1.3437 | 1.0092 | stalled |
| r160_n1_ct10/sample_026 | DLR D-Flow | 267 | 250 | 1.4872 | 1.0231 | stalled |
| r160_n1_ct10/sample_027 | DLR D-Flow | 554 | 539 | 1.4496 | 1.0181 | stalled |
| r160_n1_ct10/sample_028 | DLR D-Flow | 285 | 266 | 1.4367 | 1.0109 | stalled |
| r160_n1_ct10/sample_029 | DLR D-Flow | 353 | 341 | 1.3118 | 1.0115 | stalled |
| r160_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.2920 | 0.7829 | step_limit |
| r160_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.1748 | 1.0682 | step_limit |
| r160_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 9.4979 | 0.6659 | step_limit |
| r160_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.1095 | 0.7838 | step_limit |
| r160_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.9991 | 0.7411 | step_limit |
| r160_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.2165 | 0.6850 | step_limit |
| r160_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.1144 | 0.7313 | step_limit |
| r160_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.9783 | 0.7439 | step_limit |
| r160_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.4432 | 0.7945 | step_limit |
| r160_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.9297 | 0.9085 | step_limit |
| r160_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.0343 | 0.7390 | step_limit |
| r160_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.0788 | 0.8079 | step_limit |
| r160_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 119.8018 | 0.9670 | step_limit |
| r160_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 1.4136 | 0.7066 | step_limit |
| r160_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 1.3754 | 0.7919 | step_limit |
| r160_n1_mri4/sample_015 | D-Flow | 54 | 52 | 100.6540 | 100.1885 | stalled |
| r160_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 1.2886 | 0.6576 | step_limit |
| r160_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 1.6689 | 0.6359 | step_limit |
| r160_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 1.5607 | 0.6488 | step_limit |
| r160_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 1.0319 | 0.7236 | step_limit |
| r160_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 1.3759 | 0.6724 | step_limit |
| r160_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 1.0519 | 0.8225 | step_limit |
| r160_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 1.1274 | 0.7835 | step_limit |
| r160_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 1.1493 | 0.8369 | step_limit |
| r160_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 1.0451 | 0.7380 | step_limit |
| r160_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 1.0688 | 0.8602 | step_limit |
| r160_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 1.0430 | 0.7845 | step_limit |
| r160_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 0.9857 | 0.9987 | step_limit |
| r160_n1_mri4/sample_028 | D-Flow | 227 | 225 | 90.4112 | 89.4509 | stalled |
| r160_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 1.0195 | 0.9457 | step_limit |
| r160_n1_mri4/sample_000 | P-Flow | 524 | 450 | 2.6685 | 0.5983 | stalled |
| r160_n1_mri4/sample_001 | P-Flow | 60 | 51 | 1.3888 | 1.2084 | stalled |
| r160_n1_mri4/sample_002 | P-Flow | 418 | 364 | 2.5340 | 0.6595 | stalled |
| r160_n1_mri4/sample_003 | P-Flow | 76 | 63 | 1.2936 | 1.2071 | stalled |
| r160_n1_mri4/sample_004 | P-Flow | 307 | 270 | 2.2233 | 0.7470 | stalled |
| r160_n1_mri4/sample_005 | P-Flow | 207 | 178 | 1.3854 | 1.2062 | stalled |
| r160_n1_mri4/sample_006 | P-Flow | 89 | 74 | 1.3375 | 1.2500 | stalled |
| r160_n1_mri4/sample_007 | P-Flow | 547 | 478 | 2.3408 | 0.6156 | stalled |
| r160_n1_mri4/sample_008 | P-Flow | 48 | 40 | 2.1209 | 1.6112 | stalled |
| r160_n1_mri4/sample_009 | P-Flow | 325 | 285 | 2.3251 | 0.7861 | stalled |
| r160_n1_mri4/sample_010 | P-Flow | 555 | 492 | 2.5276 | 0.5975 | stalled |
| r160_n1_mri4/sample_011 | P-Flow | 415 | 359 | 2.1729 | 0.6175 | stalled |
| r160_n1_mri4/sample_012 | P-Flow | 244 | 217 | 2.3587 | 0.7850 | stalled |
| r160_n1_mri4/sample_013 | P-Flow | 596 | 496 | 2.7924 | 0.7853 | stalled |
| r160_n1_mri4/sample_014 | P-Flow | 17 | 13 | 3.5116 | 3.2068 | stalled |
| r160_n1_mri4/sample_015 | P-Flow | 87 | 75 | 2.0076 | 1.4844 | stalled |
| r160_n1_mri4/sample_016 | P-Flow | 201 | 173 | 2.2572 | 1.0222 | stalled |
| r160_n1_mri4/sample_017 | P-Flow | 45 | 37 | 2.3627 | 1.6075 | stalled |
| r160_n1_mri4/sample_018 | P-Flow | 141 | 127 | 2.7005 | 1.3706 | stalled |
| r160_n1_mri4/sample_019 | P-Flow | 428 | 367 | 2.3578 | 0.6839 | stalled |
| r160_n1_mri4/sample_020 | P-Flow | 96 | 84 | 1.6336 | 1.3423 | stalled |
| r160_n1_mri4/sample_021 | P-Flow | 15 | 11 | 82.0377 | 80.3055 | stalled |
| r160_n1_mri4/sample_022 | P-Flow | 308 | 271 | 2.3702 | 0.6276 | stalled |
| r160_n1_mri4/sample_023 | P-Flow | 279 | 250 | 2.3205 | 0.7928 | stalled |
| r160_n1_mri4/sample_024 | P-Flow | 349 | 306 | 2.4095 | 0.6143 | stalled |
| r160_n1_mri4/sample_025 | P-Flow | 308 | 281 | 2.6006 | 0.5195 | stalled |
| r160_n1_mri4/sample_026 | P-Flow | 345 | 308 | 2.2195 | 0.7673 | stalled |
| r160_n1_mri4/sample_027 | P-Flow | 53 | 44 | 1.2096 | 1.2411 | stalled |
| r160_n1_mri4/sample_028 | P-Flow | 388 | 350 | 2.3366 | 0.5613 | stalled |
| r160_n1_mri4/sample_029 | P-Flow | 386 | 329 | 2.2353 | 0.6455 | stalled |
| r160_n1_mri4/sample_000 | DLR D-Flow | 421 | 403 | 4.9898 | 3.5502 | stalled |
| r160_n1_mri4/sample_001 | DLR D-Flow | 144 | 135 | 106.7093 | 106.2622 | stalled |
| r160_n1_mri4/sample_002 | DLR D-Flow | 97 | 92 | 96.6748 | 96.0251 | stalled |
| r160_n1_mri4/sample_003 | DLR D-Flow | 564 | 545 | 1.3685 | 1.2419 | stalled |
| r160_n1_mri4/sample_004 | DLR D-Flow | 543 | 510 | 1.3485 | 1.3395 | stalled |
| r160_n1_mri4/sample_005 | DLR D-Flow | 146 | 137 | 7.9551 | 6.0606 | stalled |
| r160_n1_mri4/sample_006 | DLR D-Flow | 242 | 230 | 1.5755 | 1.4129 | stalled |
| r160_n1_mri4/sample_007 | DLR D-Flow | 268 | 257 | 1.5270 | 1.3117 | stalled |
| r160_n1_mri4/sample_008 | DLR D-Flow | 1000 | 965 | 2.2704 | 1.7041 | step_limit |
| r160_n1_mri4/sample_009 | DLR D-Flow | 404 | 387 | 4.5086 | 3.9387 | stalled |
| r160_n1_mri4/sample_010 | DLR D-Flow | 369 | 356 | 1.1983 | 1.1890 | stalled |
| r160_n1_mri4/sample_011 | DLR D-Flow | 118 | 113 | 2.3582 | 2.0883 | stalled |
| r160_n1_mri4/sample_012 | DLR D-Flow | 514 | 498 | 126.9482 | 48.7494 | stalled |
| r160_n1_mri4/sample_013 | DLR D-Flow | 45 | 41 | 17.0568 | 12.1419 | stalled |
| r160_n1_mri4/sample_014 | DLR D-Flow | 582 | 556 | 4.3926 | 3.7339 | stalled |
| r160_n1_mri4/sample_015 | DLR D-Flow | 61 | 57 | 98.6152 | 98.0571 | stalled |
| r160_n1_mri4/sample_016 | DLR D-Flow | 387 | 364 | 1.7522 | 1.4573 | stalled |
| r160_n1_mri4/sample_017 | DLR D-Flow | 48 | 44 | 99.4482 | 98.4796 | stalled |
| r160_n1_mri4/sample_018 | DLR D-Flow | 461 | 443 | 2.3792 | 1.8521 | stalled |
| r160_n1_mri4/sample_019 | DLR D-Flow | 929 | 883 | 1.3704 | 1.3328 | stalled |
| r160_n1_mri4/sample_020 | DLR D-Flow | 25 | 21 | 111.5325 | 110.5024 | stalled |
| r160_n1_mri4/sample_021 | DLR D-Flow | 205 | 199 | 77.4886 | 75.5001 | stalled |
| r160_n1_mri4/sample_022 | DLR D-Flow | 91 | 83 | 94.0805 | 93.8284 | stalled |
| r160_n1_mri4/sample_023 | DLR D-Flow | 16 | 13 | 103.0721 | 103.7785 | stalled |
| r160_n1_mri4/sample_024 | DLR D-Flow | 509 | 485 | 1.3492 | 1.2976 | stalled |
| r160_n1_mri4/sample_025 | DLR D-Flow | 187 | 177 | 1.3613 | 1.2362 | stalled |
| r160_n1_mri4/sample_026 | DLR D-Flow | 600 | 581 | 1.4313 | 1.3426 | stalled |
| r160_n1_mri4/sample_027 | DLR D-Flow | 422 | 407 | 6.2713 | 4.1098 | stalled |
| r160_n1_mri4/sample_028 | DLR D-Flow | 82 | 78 | 94.8254 | 94.4790 | stalled |
| r160_n1_mri4/sample_029 | DLR D-Flow | 432 | 418 | 1.2385 | 1.2108 | stalled |
| r250_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 6.3796 | 0.0655 | step_limit |
| r250_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 6.5054 | 0.5069 | step_limit |
| r250_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 6.5756 | 0.1169 | step_limit |
| r250_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 6.4801 | 0.2292 | step_limit |
| r250_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 7.4430 | 0.0550 | step_limit |
| r250_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 4.0195 | 0.3018 | step_limit |
| r250_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 8.3335 | 0.0359 | step_limit |
| r250_n1_ct10/sample_007 | D-Flow | 26 | 24 | 99.7901 | 97.3546 | stalled |
| r250_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 5.7956 | 0.0764 | step_limit |
| r250_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 3.8542 | 0.3839 | step_limit |
| r250_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 5.9893 | 0.0984 | step_limit |
| r250_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 4.1772 | 0.2790 | step_limit |
| r250_n1_ct10/sample_012 | D-Flow | 12 | 9 | 99.7252 | 97.6929 | stalled |
| r250_n1_ct10/sample_013 | D-Flow | 1000 | 1000 | 6.2960 | 0.0824 | step_limit |
| r250_n1_ct10/sample_014 | D-Flow | 33 | 31 | 99.8693 | 98.9452 | stalled |
| r250_n1_ct10/sample_015 | D-Flow | 1000 | 1000 | 7.3310 | 0.0561 | step_limit |
| r250_n1_ct10/sample_016 | D-Flow | 1000 | 1000 | 6.3006 | 0.3349 | step_limit |
| r250_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 2.4025 | 0.9021 | step_limit |
| r250_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 1.7383 | 1.0035 | step_limit |
| r250_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 4.0401 | 0.2086 | step_limit |
| r250_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 4.2338 | 0.1321 | step_limit |
| r250_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 4.0427 | 0.1584 | step_limit |
| r250_n1_ct10/sample_022 | D-Flow | 1000 | 1000 | 5.1908 | 0.4136 | step_limit |
| r250_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 3.7884 | 0.1466 | step_limit |
| r250_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 4.3024 | 0.2929 | step_limit |
| r250_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 4.1632 | 0.1340 | step_limit |
| r250_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 6.2596 | 0.2406 | step_limit |
| r250_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 4.0381 | 0.0812 | step_limit |
| r250_n1_ct10/sample_028 | D-Flow | 1000 | 1000 | 5.0093 | 0.5572 | step_limit |
| r250_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 6.8742 | 0.0459 | step_limit |
| r250_n1_ct10/sample_000 | P-Flow | 1000 | 849 | 5.9072 | 0.1122 | step_limit |
| r250_n1_ct10/sample_001 | P-Flow | 382 | 307 | 1.4823 | 1.0232 | stalled |
| r250_n1_ct10/sample_002 | P-Flow | 34 | 30 | 6.0685 | 2.2177 | stalled |
| r250_n1_ct10/sample_003 | P-Flow | 411 | 344 | 2.3048 | 1.1121 | stalled |
| r250_n1_ct10/sample_004 | P-Flow | 1000 | 880 | 5.7218 | 0.1236 | step_limit |
| r250_n1_ct10/sample_005 | P-Flow | 8 | 6 | 112.0439 | 95.6934 | stalled |
| r250_n1_ct10/sample_006 | P-Flow | 173 | 142 | 1.2561 | 1.0046 | stalled |
| r250_n1_ct10/sample_007 | P-Flow | 61 | 51 | 3.6442 | 3.9357 | stalled |
| r250_n1_ct10/sample_008 | P-Flow | 96 | 78 | 2.3226 | 1.1128 | stalled |
| r250_n1_ct10/sample_009 | P-Flow | 112 | 88 | 2.3904 | 1.0874 | stalled |
| r250_n1_ct10/sample_010 | P-Flow | 34 | 28 | 2.3230 | 1.0765 | stalled |
| r250_n1_ct10/sample_011 | P-Flow | 31 | 26 | 4.5482 | 3.2567 | stalled |
| r250_n1_ct10/sample_012 | P-Flow | 33 | 27 | 2.2616 | 1.1093 | stalled |
| r250_n1_ct10/sample_013 | P-Flow | 235 | 222 | 1.3578 | 1.0130 | stalled |
| r250_n1_ct10/sample_014 | P-Flow | 56 | 48 | 1.8520 | 1.0600 | stalled |
| r250_n1_ct10/sample_015 | P-Flow | 148 | 122 | 1.4074 | 1.0249 | stalled |
| r250_n1_ct10/sample_016 | P-Flow | 318 | 261 | 1.1659 | 1.0076 | stalled |
| r250_n1_ct10/sample_017 | P-Flow | 179 | 143 | 1.6763 | 1.0441 | stalled |
| r250_n1_ct10/sample_018 | P-Flow | 251 | 199 | 2.5674 | 1.1152 | stalled |
| r250_n1_ct10/sample_019 | P-Flow | 13 | 11 | 5.8312 | 2.0922 | stalled |
| r250_n1_ct10/sample_020 | P-Flow | 26 | 22 | 2.3823 | 1.3315 | stalled |
| r250_n1_ct10/sample_021 | P-Flow | 26 | 22 | 2.5636 | 1.1359 | stalled |
| r250_n1_ct10/sample_022 | P-Flow | 544 | 438 | 1.3360 | 1.0063 | stalled |
| r250_n1_ct10/sample_023 | P-Flow | 91 | 74 | 1.8864 | 1.0760 | stalled |
| r250_n1_ct10/sample_024 | P-Flow | 90 | 79 | 1.3173 | 1.0168 | stalled |
| r250_n1_ct10/sample_025 | P-Flow | 142 | 113 | 2.5909 | 1.5221 | stalled |
| r250_n1_ct10/sample_026 | P-Flow | 61 | 50 | 2.9208 | 2.0648 | stalled |
| r250_n1_ct10/sample_027 | P-Flow | 214 | 173 | 3.6187 | 1.5614 | stalled |
| r250_n1_ct10/sample_028 | P-Flow | 46 | 38 | 11.6714 | 3.4478 | stalled |
| r250_n1_ct10/sample_029 | P-Flow | 50 | 42 | 1.2475 | 1.0143 | stalled |
| r250_n1_ct10/sample_000 | DLR D-Flow | 112 | 107 | 8.2255 | 1.7016 | stalled |
| r250_n1_ct10/sample_001 | DLR D-Flow | 125 | 119 | 7.3933 | 1.2321 | stalled |
| r250_n1_ct10/sample_002 | DLR D-Flow | 404 | 390 | 1.2783 | 1.0051 | stalled |
| r250_n1_ct10/sample_003 | DLR D-Flow | 154 | 149 | 1.6083 | 1.0196 | stalled |
| r250_n1_ct10/sample_004 | DLR D-Flow | 314 | 296 | 1.5957 | 1.0140 | stalled |
| r250_n1_ct10/sample_005 | DLR D-Flow | 384 | 369 | 1.3088 | 1.0065 | stalled |
| r250_n1_ct10/sample_006 | DLR D-Flow | 254 | 243 | 1.1573 | 1.0053 | stalled |
| r250_n1_ct10/sample_007 | DLR D-Flow | 20 | 16 | 99.6348 | 96.9325 | stalled |
| r250_n1_ct10/sample_008 | DLR D-Flow | 1000 | 952 | 1.7556 | 1.0537 | step_limit |
| r250_n1_ct10/sample_009 | DLR D-Flow | 352 | 336 | 1.3559 | 1.0125 | stalled |
| r250_n1_ct10/sample_010 | DLR D-Flow | 772 | 739 | 1.6430 | 1.0174 | stalled |
| r250_n1_ct10/sample_011 | DLR D-Flow | 314 | 302 | 1.3247 | 1.0137 | stalled |
| r250_n1_ct10/sample_012 | DLR D-Flow | 11 | 9 | 99.7309 | 97.6920 | stalled |
| r250_n1_ct10/sample_013 | DLR D-Flow | 186 | 175 | 1.3619 | 1.0154 | stalled |
| r250_n1_ct10/sample_014 | DLR D-Flow | 20 | 18 | 103.3548 | 98.0564 | stalled |
| r250_n1_ct10/sample_015 | DLR D-Flow | 201 | 191 | 3.2900 | 1.0608 | stalled |
| r250_n1_ct10/sample_016 | DLR D-Flow | 846 | 843 | 9.9186 | 2.6259 | stalled |
| r250_n1_ct10/sample_017 | DLR D-Flow | 115 | 112 | 1.8279 | 1.0363 | stalled |
| r250_n1_ct10/sample_018 | DLR D-Flow | 63 | 60 | 108.2009 | 93.4895 | stalled |
| r250_n1_ct10/sample_019 | DLR D-Flow | 295 | 286 | 1.3033 | 1.0167 | stalled |
| r250_n1_ct10/sample_020 | DLR D-Flow | 419 | 402 | 1.4319 | 1.0115 | stalled |
| r250_n1_ct10/sample_021 | DLR D-Flow | 181 | 176 | 2.1012 | 1.0552 | stalled |
| r250_n1_ct10/sample_022 | DLR D-Flow | 456 | 436 | 1.3625 | 1.0131 | stalled |
| r250_n1_ct10/sample_023 | DLR D-Flow | 68 | 63 | 111.0631 | 91.4279 | stalled |
| r250_n1_ct10/sample_024 | DLR D-Flow | 224 | 214 | 1.2197 | 1.0076 | stalled |
| r250_n1_ct10/sample_025 | DLR D-Flow | 267 | 255 | 1.5080 | 1.0146 | stalled |
| r250_n1_ct10/sample_026 | DLR D-Flow | 479 | 459 | 1.3933 | 1.0213 | stalled |
| r250_n1_ct10/sample_027 | DLR D-Flow | 117 | 110 | 18.7174 | 7.3602 | stalled |
| r250_n1_ct10/sample_028 | DLR D-Flow | 721 | 699 | 1.8347 | 1.0248 | stalled |
| r250_n1_ct10/sample_029 | DLR D-Flow | 582 | 556 | 1.7588 | 1.0255 | stalled |
| r250_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.9525 | 0.9162 | step_limit |
| r250_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.9866 | 0.7908 | step_limit |
| r250_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.0261 | 0.8023 | step_limit |
| r250_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.9621 | 0.8031 | step_limit |
| r250_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.9741 | 0.7402 | step_limit |
| r250_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.9698 | 0.8734 | step_limit |
| r250_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.9660 | 0.7838 | step_limit |
| r250_n1_mri4/sample_007 | D-Flow | 31 | 29 | 99.6306 | 99.5894 | stalled |
| r250_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.1371 | 0.9841 | step_limit |
| r250_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.9368 | 1.0042 | step_limit |
| r250_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.0934 | 0.7750 | step_limit |
| r250_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.0321 | 1.0880 | step_limit |
| r250_n1_mri4/sample_012 | D-Flow | 13 | 11 | 99.5923 | 99.5374 | stalled |
| r250_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 0.9947 | 0.9856 | step_limit |
| r250_n1_mri4/sample_014 | D-Flow | 240 | 236 | 98.3401 | 97.7008 | stalled |
| r250_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 0.9111 | 0.8628 | step_limit |
| r250_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 0.9585 | 0.7836 | step_limit |
| r250_n1_mri4/sample_017 | D-Flow | 39 | 37 | 99.4724 | 99.4298 | stalled |
| r250_n1_mri4/sample_018 | D-Flow | 84 | 81 | 99.4674 | 99.4180 | stalled |
| r250_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 0.8955 | 1.0193 | step_limit |
| r250_n1_mri4/sample_020 | D-Flow | 14 | 12 | 99.3686 | 99.3361 | stalled |
| r250_n1_mri4/sample_021 | D-Flow | 13 | 11 | 99.5006 | 99.4704 | stalled |
| r250_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 1.0963 | 0.8326 | step_limit |
| r250_n1_mri4/sample_023 | D-Flow | 40 | 38 | 99.5307 | 99.5042 | stalled |
| r250_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 0.9471 | 0.8629 | step_limit |
| r250_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 1.0787 | 0.8969 | step_limit |
| r250_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 1.0569 | 0.8217 | step_limit |
| r250_n1_mri4/sample_027 | D-Flow | 271 | 269 | 99.2195 | 99.1657 | stalled |
| r250_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 1.0619 | 0.8035 | step_limit |
| r250_n1_mri4/sample_029 | D-Flow | 12 | 10 | 99.5093 | 99.3397 | stalled |
| r250_n1_mri4/sample_000 | P-Flow | 57 | 48 | 1.0613 | 1.1403 | stalled |
| r250_n1_mri4/sample_001 | P-Flow | 533 | 460 | 2.1197 | 0.5680 | stalled |
| r250_n1_mri4/sample_002 | P-Flow | 21 | 17 | 1.6443 | 1.5356 | stalled |
| r250_n1_mri4/sample_003 | P-Flow | 446 | 390 | 1.9203 | 0.6345 | stalled |
| r250_n1_mri4/sample_004 | P-Flow | 81 | 70 | 1.0401 | 1.1681 | stalled |
| r250_n1_mri4/sample_005 | P-Flow | 31 | 26 | 1.2045 | 1.2238 | stalled |
| r250_n1_mri4/sample_006 | P-Flow | 123 | 103 | 0.9864 | 1.1222 | stalled |
| r250_n1_mri4/sample_007 | P-Flow | 631 | 548 | 3.1434 | 0.6513 | stalled |
| r250_n1_mri4/sample_008 | P-Flow | 83 | 70 | 1.3472 | 1.2255 | stalled |
| r250_n1_mri4/sample_009 | P-Flow | 16 | 13 | 1.5723 | 1.5253 | stalled |
| r250_n1_mri4/sample_010 | P-Flow | 29 | 26 | 1.5326 | 1.4034 | stalled |
| r250_n1_mri4/sample_011 | P-Flow | 820 | 725 | 2.3732 | 0.5748 | stalled |
| r250_n1_mri4/sample_012 | P-Flow | 729 | 637 | 2.8408 | 0.5582 | stalled |
| r250_n1_mri4/sample_013 | P-Flow | 54 | 45 | 1.1918 | 1.2307 | stalled |
| r250_n1_mri4/sample_014 | P-Flow | 129 | 114 | 1.5719 | 1.3218 | stalled |
| r250_n1_mri4/sample_015 | P-Flow | 42 | 36 | 1.0825 | 1.1868 | stalled |
| r250_n1_mri4/sample_016 | P-Flow | 278 | 239 | 1.4728 | 0.8720 | stalled |
| r250_n1_mri4/sample_017 | P-Flow | 83 | 70 | 1.2846 | 1.2264 | stalled |
| r250_n1_mri4/sample_018 | P-Flow | 70 | 56 | 1.5845 | 1.3131 | stalled |
| r250_n1_mri4/sample_019 | P-Flow | 96 | 82 | 1.0548 | 1.1666 | stalled |
| r250_n1_mri4/sample_020 | P-Flow | 82 | 71 | 1.2778 | 1.2126 | stalled |
| r250_n1_mri4/sample_021 | P-Flow | 61 | 52 | 1.5347 | 1.3076 | stalled |
| r250_n1_mri4/sample_022 | P-Flow | 17 | 14 | 1.6587 | 1.5898 | stalled |
| r250_n1_mri4/sample_023 | P-Flow | 129 | 112 | 1.4090 | 1.1993 | stalled |
| r250_n1_mri4/sample_024 | P-Flow | 87 | 71 | 1.0355 | 1.1470 | stalled |
| r250_n1_mri4/sample_025 | P-Flow | 15 | 12 | 1.8202 | 1.6010 | stalled |
| r250_n1_mri4/sample_026 | P-Flow | 670 | 572 | 2.2824 | 0.5292 | stalled |
| r250_n1_mri4/sample_027 | P-Flow | 620 | 540 | 3.0826 | 0.5419 | stalled |
| r250_n1_mri4/sample_028 | P-Flow | 622 | 547 | 2.3796 | 0.5055 | stalled |
| r250_n1_mri4/sample_029 | P-Flow | 395 | 351 | 2.0820 | 0.5252 | stalled |
| r250_n1_mri4/sample_000 | DLR D-Flow | 635 | 610 | 1.3545 | 1.2718 | stalled |
| r250_n1_mri4/sample_001 | DLR D-Flow | 481 | 467 | 1.3450 | 1.2614 | stalled |
| r250_n1_mri4/sample_002 | DLR D-Flow | 403 | 391 | 1.1939 | 1.2068 | stalled |
| r250_n1_mri4/sample_003 | DLR D-Flow | 563 | 537 | 1.5523 | 1.3700 | stalled |
| r250_n1_mri4/sample_004 | DLR D-Flow | 288 | 278 | 1.1876 | 1.2476 | stalled |
| r250_n1_mri4/sample_005 | DLR D-Flow | 538 | 522 | 1.1742 | 1.1861 | stalled |
| r250_n1_mri4/sample_006 | DLR D-Flow | 72 | 67 | 1.0948 | 1.1675 | stalled |
| r250_n1_mri4/sample_007 | DLR D-Flow | 15 | 13 | 99.4731 | 99.4189 | stalled |
| r250_n1_mri4/sample_008 | DLR D-Flow | 474 | 454 | 1.4975 | 1.2885 | stalled |
| r250_n1_mri4/sample_009 | DLR D-Flow | 433 | 414 | 1.3494 | 1.2424 | stalled |
| r250_n1_mri4/sample_010 | DLR D-Flow | 278 | 270 | 3.3101 | 2.9313 | stalled |
| r250_n1_mri4/sample_011 | DLR D-Flow | 797 | 768 | 1.2860 | 1.2368 | stalled |
| r250_n1_mri4/sample_012 | DLR D-Flow | 17 | 15 | 99.5921 | 99.5372 | stalled |
| r250_n1_mri4/sample_013 | DLR D-Flow | 320 | 303 | 1.2683 | 1.2766 | stalled |
| r250_n1_mri4/sample_014 | DLR D-Flow | 36 | 33 | 99.7461 | 99.5517 | stalled |
| r250_n1_mri4/sample_015 | DLR D-Flow | 400 | 386 | 1.0869 | 1.1681 | stalled |
| r250_n1_mri4/sample_016 | DLR D-Flow | 608 | 579 | 2.2136 | 1.7808 | stalled |
| r250_n1_mri4/sample_017 | DLR D-Flow | 22 | 19 | 99.4647 | 99.4235 | stalled |
| r250_n1_mri4/sample_018 | DLR D-Flow | 22 | 18 | 109.6724 | 109.1164 | stalled |
| r250_n1_mri4/sample_019 | DLR D-Flow | 524 | 505 | 1.1918 | 1.2518 | stalled |
| r250_n1_mri4/sample_020 | DLR D-Flow | 603 | 580 | 1.2886 | 1.2161 | stalled |
| r250_n1_mri4/sample_021 | DLR D-Flow | 19 | 15 | 97.8138 | 97.6252 | stalled |
| r250_n1_mri4/sample_022 | DLR D-Flow | 488 | 474 | 1.2239 | 1.2009 | stalled |
| r250_n1_mri4/sample_023 | DLR D-Flow | 26 | 24 | 99.1516 | 99.1071 | stalled |
| r250_n1_mri4/sample_024 | DLR D-Flow | 293 | 283 | 1.1773 | 1.2051 | stalled |
| r250_n1_mri4/sample_025 | DLR D-Flow | 33 | 31 | 1.9171 | 1.7138 | stalled |
| r250_n1_mri4/sample_026 | DLR D-Flow | 33 | 30 | 1.3903 | 1.3051 | stalled |
| r250_n1_mri4/sample_027 | DLR D-Flow | 23 | 20 | 99.6259 | 99.5980 | stalled |
| r250_n1_mri4/sample_028 | DLR D-Flow | 81 | 79 | 1.9947 | 1.8262 | stalled |
| r250_n1_mri4/sample_029 | DLR D-Flow | 11 | 9 | 99.4977 | 99.3245 | stalled |

Terminal method runs: 540/540. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including image/residual panels and error versus optimizer attempts.
