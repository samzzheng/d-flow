# Five-step Euler pilot

1% relative measurement noise; CT10 and Cartesian MRI4; 30 test samples per acquisition at each resolution; circle count in [1, 2, 4]. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 8, analytical alpha=1-t, no trace probes/enrichment and previous V.

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
| r64_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 5.2965 | 0.1335 | step_limit |
| r64_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 6.4082 | 0.1375 | step_limit |
| r64_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 3.3627 | 0.3710 | step_limit |
| r64_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 2.8015 | 0.5061 | step_limit |
| r64_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 2.2774 | 0.7114 | step_limit |
| r64_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 4.5515 | 0.2118 | step_limit |
| r64_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 2.8307 | 0.4750 | step_limit |
| r64_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 2.6090 | 0.8114 | step_limit |
| r64_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 2.9234 | 0.4570 | step_limit |
| r64_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 3.3442 | 0.2226 | step_limit |
| r64_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 2.3705 | 0.7217 | step_limit |
| r64_n2_ct10/sample_000 | P-Flow | 1000 | 876 | 6.0514 | 0.1880 | step_limit |
| r64_n2_ct10/sample_001 | P-Flow | 337 | 291 | 4.7289 | 0.7119 | stalled |
| r64_n2_ct10/sample_002 | P-Flow | 292 | 262 | 5.0046 | 0.3250 | stalled |
| r64_n2_ct10/sample_003 | P-Flow | 12 | 10 | 111.7452 | 70.2052 | stalled |
| r64_n2_ct10/sample_004 | P-Flow | 17 | 15 | 72.0862 | 39.6867 | stalled |
| r64_n2_ct10/sample_005 | P-Flow | 44 | 36 | 60.0434 | 32.7086 | stalled |
| r64_n2_ct10/sample_006 | P-Flow | 48 | 41 | 3.4278 | 1.3559 | stalled |
| r64_n2_ct10/sample_007 | P-Flow | 88 | 75 | 3.6504 | 1.4425 | stalled |
| r64_n2_ct10/sample_008 | P-Flow | 1000 | 946 | 5.9302 | 0.1723 | step_limit |
| r64_n2_ct10/sample_009 | P-Flow | 279 | 231 | 2.3042 | 1.1232 | stalled |
| r64_n2_ct10/sample_010 | P-Flow | 369 | 333 | 5.3488 | 0.4017 | stalled |
| r64_n2_ct10/sample_000 | DLR D-Flow | 592 | 579 | 1.8513 | 1.0779 | stalled |
| r64_n2_ct10/sample_001 | DLR D-Flow | 861 | 819 | 4.5254 | 1.5571 | stalled |
| r64_n2_ct10/sample_002 | DLR D-Flow | 165 | 160 | 2.9112 | 1.2657 | stalled |
| r64_n2_ct10/sample_003 | DLR D-Flow | 353 | 344 | 24.6988 | 10.9337 | stalled |
| r64_n2_ct10/sample_004 | DLR D-Flow | 1000 | 964 | 2.7007 | 1.3146 | step_limit |
| r64_n2_ct10/sample_005 | DLR D-Flow | 823 | 804 | 51.8476 | 31.9086 | stalled |
| r64_n2_ct10/sample_006 | DLR D-Flow | 188 | 179 | 74.8639 | 55.5682 | stalled |
| r64_n2_ct10/sample_007 | DLR D-Flow | 379 | 365 | 5.1214 | 1.6783 | stalled |
| r64_n2_ct10/sample_008 | DLR D-Flow | 173 | 164 | 75.6701 | 48.2032 | stalled |
| r64_n2_ct10/sample_009 | DLR D-Flow | 260 | 245 | 2.1177 | 1.1169 | stalled |
| r64_n2_ct10/sample_010 | DLR D-Flow | 654 | 634 | 2.4707 | 1.2481 | stalled |
| r64_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 2.6330 | 0.5472 | step_limit |
| r64_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 4.9885 | 0.1761 | step_limit |
| r64_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.4702 | 0.6670 | step_limit |
| r64_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 2.6948 | 0.7987 | step_limit |
| r64_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 3.7994 | 0.5738 | step_limit |
| r64_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 4.3830 | 0.2027 | step_limit |
| r64_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 3.8667 | 0.5446 | step_limit |
| r64_n4_ct10/sample_007 | D-Flow | 516 | 514 | 73.0976 | 39.7179 | stalled |
| r64_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 2.4682 | 0.6543 | step_limit |
| r64_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 4.6237 | 0.3295 | step_limit |
| r64_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 6.3164 | 0.1705 | step_limit |
| r64_n4_ct10/sample_000 | P-Flow | 356 | 313 | 4.8468 | 0.6713 | stalled |
| r64_n4_ct10/sample_001 | P-Flow | 1000 | 845 | 5.7934 | 0.2769 | step_limit |
| r64_n4_ct10/sample_002 | P-Flow | 1000 | 819 | 2.1498 | 1.0299 | step_limit |
| r64_n4_ct10/sample_003 | P-Flow | 13 | 11 | 59.9486 | 30.2331 | stalled |
| r64_n4_ct10/sample_004 | P-Flow | 25 | 22 | 42.5722 | 18.3779 | stalled |
| r64_n4_ct10/sample_005 | P-Flow | 1000 | 956 | 5.8613 | 0.1994 | step_limit |
| r64_n4_ct10/sample_006 | P-Flow | 1000 | 802 | 2.2655 | 1.0351 | step_limit |
| r64_n4_ct10/sample_007 | P-Flow | 1000 | 933 | 5.7626 | 0.2022 | step_limit |
| r64_n4_ct10/sample_008 | P-Flow | 1000 | 867 | 6.1930 | 0.1885 | step_limit |
| r64_n4_ct10/sample_009 | P-Flow | 20 | 16 | 57.5594 | 27.8122 | stalled |
| r64_n4_ct10/sample_010 | P-Flow | 109 | 94 | 2.1956 | 1.0296 | stalled |
| r64_n4_ct10/sample_000 | DLR D-Flow | 218 | 209 | 33.8589 | 12.2205 | stalled |
| r64_n4_ct10/sample_001 | DLR D-Flow | 977 | 938 | 6.6735 | 2.3281 | stalled |
| r64_n4_ct10/sample_002 | DLR D-Flow | 1000 | 974 | 68.2627 | 36.7110 | step_limit |
| r64_n4_ct10/sample_003 | DLR D-Flow | 328 | 325 | 22.3734 | 9.9172 | stalled |
| r64_n4_ct10/sample_004 | DLR D-Flow | 262 | 256 | 14.4705 | 4.9525 | stalled |
| r64_n4_ct10/sample_005 | DLR D-Flow | 1000 | 970 | 7.8590 | 2.9626 | step_limit |
| r64_n4_ct10/sample_006 | DLR D-Flow | 1000 | 972 | 86.8840 | 42.8982 | step_limit |
| r64_n4_ct10/sample_007 | DLR D-Flow | 1000 | 972 | 15.5750 | 5.2842 | step_limit |
| r64_n4_ct10/sample_008 | DLR D-Flow | 628 | 614 | 73.0071 | 38.0858 | stalled |
| r64_n4_ct10/sample_009 | DLR D-Flow | 619 | 594 | 30.4133 | 12.9418 | stalled |
| r64_n4_ct10/sample_010 | DLR D-Flow | 117 | 115 | 15.2127 | 4.4933 | stalled |
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
| r64_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.2881 | 0.7701 | step_limit |
| r64_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.2151 | 0.6432 | step_limit |
| r64_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.4210 | 0.6010 | step_limit |
| r64_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.3827 | 0.5332 | step_limit |
| r64_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.4020 | 0.8830 | step_limit |
| r64_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.1912 | 0.7245 | step_limit |
| r64_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.9192 | 0.5490 | step_limit |
| r64_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.5374 | 0.6108 | step_limit |
| r64_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.2398 | 0.6698 | step_limit |
| r64_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.4636 | 0.6002 | step_limit |
| r64_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.2428 | 0.5648 | step_limit |
| r64_n2_mri4/sample_000 | P-Flow | 98 | 83 | 1.7316 | 1.2665 | stalled |
| r64_n2_mri4/sample_001 | P-Flow | 247 | 215 | 1.8070 | 1.1566 | stalled |
| r64_n2_mri4/sample_002 | P-Flow | 89 | 73 | 1.7931 | 1.3472 | stalled |
| r64_n2_mri4/sample_003 | P-Flow | 8 | 6 | 108.9225 | 108.8176 | stalled |
| r64_n2_mri4/sample_004 | P-Flow | 81 | 70 | 1.5894 | 1.3439 | stalled |
| r64_n2_mri4/sample_005 | P-Flow | 328 | 281 | 1.9172 | 1.0832 | stalled |
| r64_n2_mri4/sample_006 | P-Flow | 50 | 41 | 2.8111 | 1.8859 | stalled |
| r64_n2_mri4/sample_007 | P-Flow | 132 | 114 | 2.1289 | 1.4614 | stalled |
| r64_n2_mri4/sample_008 | P-Flow | 271 | 233 | 2.0092 | 1.0853 | stalled |
| r64_n2_mri4/sample_009 | P-Flow | 97 | 83 | 2.0089 | 1.4865 | stalled |
| r64_n2_mri4/sample_010 | P-Flow | 10 | 7 | 112.4291 | 108.4765 | stalled |
| r64_n2_mri4/sample_000 | DLR D-Flow | 709 | 683 | 1.8057 | 1.4155 | stalled |
| r64_n2_mri4/sample_001 | DLR D-Flow | 512 | 496 | 66.3635 | 58.1433 | stalled |
| r64_n2_mri4/sample_002 | DLR D-Flow | 347 | 320 | 2.0864 | 1.6112 | stalled |
| r64_n2_mri4/sample_003 | DLR D-Flow | 1000 | 964 | 69.4882 | 67.1999 | step_limit |
| r64_n2_mri4/sample_004 | DLR D-Flow | 1000 | 983 | 40.3347 | 33.0533 | step_limit |
| r64_n2_mri4/sample_005 | DLR D-Flow | 848 | 820 | 2.1279 | 1.8555 | stalled |
| r64_n2_mri4/sample_006 | DLR D-Flow | 812 | 777 | 2.6684 | 1.7883 | stalled |
| r64_n2_mri4/sample_007 | DLR D-Flow | 752 | 731 | 71.5264 | 71.6015 | stalled |
| r64_n2_mri4/sample_008 | DLR D-Flow | 835 | 807 | 1.7244 | 1.4232 | stalled |
| r64_n2_mri4/sample_009 | DLR D-Flow | 533 | 511 | 2.1194 | 1.5914 | stalled |
| r64_n2_mri4/sample_010 | DLR D-Flow | 375 | 355 | 91.2062 | 89.8731 | stalled |
| r64_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.4914 | 0.5937 | step_limit |
| r64_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.3227 | 0.6134 | step_limit |
| r64_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.3778 | 0.7724 | step_limit |
| r64_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.3774 | 0.6871 | step_limit |
| r64_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.5231 | 0.5709 | step_limit |
| r64_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.6019 | 0.6233 | step_limit |
| r64_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.5716 | 0.7469 | step_limit |
| r64_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.4940 | 0.9376 | step_limit |
| r64_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.3726 | 0.6924 | step_limit |
| r64_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.3411 | 0.6554 | step_limit |
| r64_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.5988 | 0.4821 | step_limit |
| r64_n4_mri4/sample_000 | P-Flow | 86 | 74 | 1.8301 | 1.4169 | stalled |
| r64_n4_mri4/sample_001 | P-Flow | 72 | 62 | 1.6050 | 1.3329 | stalled |
| r64_n4_mri4/sample_002 | P-Flow | 115 | 97 | 1.8455 | 1.4088 | stalled |
| r64_n4_mri4/sample_003 | P-Flow | 135 | 117 | 1.9243 | 1.2468 | stalled |
| r64_n4_mri4/sample_004 | P-Flow | 907 | 792 | 2.9743 | 0.5228 | stalled |
| r64_n4_mri4/sample_005 | P-Flow | 83 | 70 | 2.0072 | 1.3994 | stalled |
| r64_n4_mri4/sample_006 | P-Flow | 444 | 390 | 3.0946 | 0.5856 | stalled |
| r64_n4_mri4/sample_007 | P-Flow | 1000 | 870 | 3.0712 | 0.3971 | step_limit |
| r64_n4_mri4/sample_008 | P-Flow | 1000 | 837 | 2.8359 | 0.5840 | step_limit |
| r64_n4_mri4/sample_009 | P-Flow | 237 | 200 | 1.9894 | 1.2940 | stalled |
| r64_n4_mri4/sample_010 | P-Flow | 401 | 357 | 2.6911 | 0.4430 | stalled |
| r64_n4_mri4/sample_000 | DLR D-Flow | 1000 | 951 | 81.6779 | 74.3863 | step_limit |
| r64_n4_mri4/sample_001 | DLR D-Flow | 253 | 241 | 15.4868 | 11.7260 | stalled |
| r64_n4_mri4/sample_002 | DLR D-Flow | 594 | 574 | 16.5567 | 11.8861 | stalled |
| r64_n4_mri4/sample_003 | DLR D-Flow | 125 | 120 | 23.1802 | 18.0880 | stalled |
| r64_n4_mri4/sample_004 | DLR D-Flow | 178 | 171 | 7.3508 | 5.3698 | stalled |
| r64_n4_mri4/sample_005 | DLR D-Flow | 351 | 333 | 80.7494 | 78.4594 | stalled |
| r64_n4_mri4/sample_006 | DLR D-Flow | 343 | 334 | 88.8248 | 83.8156 | stalled |
| r64_n4_mri4/sample_007 | DLR D-Flow | 564 | 541 | 65.0693 | 58.3752 | stalled |
| r64_n4_mri4/sample_008 | DLR D-Flow | 230 | 221 | 92.4415 | 89.3306 | stalled |
| r64_n4_mri4/sample_009 | DLR D-Flow | 1000 | 961 | 62.7337 | 53.0389 | step_limit |
| r64_n4_mri4/sample_010 | DLR D-Flow | 160 | 154 | 21.2459 | 16.6348 | stalled |
| r96_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 2.8434 | 0.4592 | step_limit |
| r96_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 2.9173 | 0.6228 | step_limit |
| r96_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 3.0213 | 0.4669 | step_limit |
| r96_n1_ct10/sample_003 | D-Flow | 434 | 432 | 2.1397 | 0.8506 | stalled |
| r96_n1_ct10/sample_004 | D-Flow | 81 | 79 | 105.2348 | 108.4149 | stalled |
| r96_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 7.4999 | 0.0671 | step_limit |
| r96_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 2.9797 | 0.4044 | step_limit |
| r96_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 5.1262 | 0.2089 | step_limit |
| r96_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 2.9596 | 0.4384 | step_limit |
| r96_n1_ct10/sample_009 | D-Flow | 78 | 75 | 114.9660 | 103.7221 | stalled |
| r96_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 6.0165 | 0.0947 | step_limit |
| r96_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 5.5569 | 0.1674 | step_limit |
| r96_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 3.6308 | 0.1845 | step_limit |
| r96_n1_ct10/sample_013 | D-Flow | 106 | 104 | 104.7769 | 107.9403 | stalled |
| r96_n1_ct10/sample_014 | D-Flow | 190 | 187 | 115.3253 | 105.5936 | stalled |
| r96_n1_ct10/sample_015 | D-Flow | 1000 | 999 | 3.7407 | 0.2945 | step_limit |
| r96_n1_ct10/sample_016 | D-Flow | 1000 | 1000 | 3.1584 | 0.4006 | step_limit |
| r96_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 5.2251 | 0.1610 | step_limit |
| r96_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 3.5373 | 0.3061 | step_limit |
| r96_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 2.3989 | 0.7240 | step_limit |
| r96_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 3.3034 | 0.3875 | step_limit |
| r96_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 3.2869 | 0.3484 | step_limit |
| r96_n1_ct10/sample_022 | D-Flow | 1000 | 1000 | 5.2708 | 0.1449 | step_limit |
| r96_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 3.1117 | 0.4806 | step_limit |
| r96_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 2.8872 | 0.3044 | step_limit |
| r96_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 2.8066 | 0.4906 | step_limit |
| r96_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 3.8580 | 0.3273 | step_limit |
| r96_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 5.4518 | 0.1114 | step_limit |
| r96_n1_ct10/sample_028 | D-Flow | 1000 | 1000 | 2.4682 | 0.8836 | step_limit |
| r96_n1_ct10/sample_029 | D-Flow | 496 | 493 | 2.3947 | 0.7907 | stalled |
| r96_n1_ct10/sample_000 | P-Flow | 31 | 27 | 2.0645 | 1.1044 | stalled |
| r96_n1_ct10/sample_001 | P-Flow | 151 | 126 | 3.1350 | 1.5019 | stalled |
| r96_n1_ct10/sample_002 | P-Flow | 33 | 28 | 4.0068 | 1.9214 | stalled |
| r96_n1_ct10/sample_003 | P-Flow | 77 | 70 | 1.5872 | 1.0325 | stalled |
| r96_n1_ct10/sample_004 | P-Flow | 83 | 69 | 3.5611 | 1.3980 | stalled |
| r96_n1_ct10/sample_005 | P-Flow | 1000 | 818 | 2.1831 | 1.2176 | step_limit |
| r96_n1_ct10/sample_006 | P-Flow | 1000 | 970 | 4.9969 | 0.1502 | step_limit |
| r96_n1_ct10/sample_007 | P-Flow | 38 | 32 | 4.2130 | 1.6195 | stalled |
| r96_n1_ct10/sample_008 | P-Flow | 434 | 410 | 5.3030 | 0.1318 | stalled |
| r96_n1_ct10/sample_009 | P-Flow | 106 | 84 | 2.1843 | 1.1347 | stalled |
| r96_n1_ct10/sample_010 | P-Flow | 674 | 610 | 5.4560 | 0.1376 | stalled |
| r96_n1_ct10/sample_011 | P-Flow | 39 | 32 | 2.4300 | 1.3598 | stalled |
| r96_n1_ct10/sample_012 | P-Flow | 1000 | 815 | 4.0414 | 0.4955 | step_limit |
| r96_n1_ct10/sample_013 | P-Flow | 519 | 486 | 5.3325 | 0.1643 | stalled |
| r96_n1_ct10/sample_014 | P-Flow | 572 | 512 | 5.3880 | 0.1437 | stalled |
| r96_n1_ct10/sample_015 | P-Flow | 1000 | 812 | 3.5658 | 2.5621 | step_limit |
| r96_n1_ct10/sample_016 | P-Flow | 1000 | 809 | 2.1231 | 1.1156 | step_limit |
| r96_n1_ct10/sample_017 | P-Flow | 164 | 136 | 1.4165 | 1.0297 | stalled |
| r96_n1_ct10/sample_018 | P-Flow | 76 | 67 | 2.4650 | 1.1306 | stalled |
| r96_n1_ct10/sample_019 | P-Flow | 1000 | 883 | 5.5493 | 0.1716 | step_limit |
| r96_n1_ct10/sample_020 | P-Flow | 1000 | 979 | 5.0618 | 0.1212 | step_limit |
| r96_n1_ct10/sample_021 | P-Flow | 572 | 453 | 2.1292 | 1.1363 | stalled |
| r96_n1_ct10/sample_022 | P-Flow | 22 | 18 | 4.4126 | 1.8240 | stalled |
| r96_n1_ct10/sample_023 | P-Flow | 29 | 25 | 2.7502 | 1.8832 | stalled |
| r96_n1_ct10/sample_024 | P-Flow | 34 | 28 | 14.3377 | 6.3876 | stalled |
| r96_n1_ct10/sample_025 | P-Flow | 529 | 433 | 2.5128 | 1.1925 | stalled |
| r96_n1_ct10/sample_026 | P-Flow | 47 | 40 | 3.2176 | 1.4107 | stalled |
| r96_n1_ct10/sample_027 | P-Flow | 27 | 23 | 1.8690 | 1.3546 | stalled |
| r96_n1_ct10/sample_028 | P-Flow | 1000 | 813 | 2.0445 | 1.2092 | step_limit |
| r96_n1_ct10/sample_029 | P-Flow | 86 | 68 | 2.3720 | 1.1774 | stalled |
| r96_n1_ct10/sample_000 | DLR D-Flow | 115 | 109 | 1.8903 | 1.0908 | stalled |
| r96_n1_ct10/sample_001 | DLR D-Flow | 49 | 45 | 113.9047 | 100.9381 | stalled |
| r96_n1_ct10/sample_002 | DLR D-Flow | 887 | 840 | 1.7675 | 1.0423 | stalled |
| r96_n1_ct10/sample_003 | DLR D-Flow | 197 | 189 | 2.0053 | 1.2892 | stalled |
| r96_n1_ct10/sample_004 | DLR D-Flow | 813 | 766 | 3.3237 | 1.1973 | stalled |
| r96_n1_ct10/sample_005 | DLR D-Flow | 337 | 322 | 1.7467 | 1.0334 | stalled |
| r96_n1_ct10/sample_006 | DLR D-Flow | 489 | 474 | 1.8874 | 1.0509 | stalled |
| r96_n1_ct10/sample_007 | DLR D-Flow | 1000 | 893 | 2.4423 | 1.1995 | step_limit |
| r96_n1_ct10/sample_008 | DLR D-Flow | 34 | 30 | 114.2432 | 102.6509 | stalled |
| r96_n1_ct10/sample_009 | DLR D-Flow | 284 | 273 | 118.7680 | 104.4094 | stalled |
| r96_n1_ct10/sample_010 | DLR D-Flow | 256 | 254 | 14.8142 | 4.6992 | stalled |
| r96_n1_ct10/sample_011 | DLR D-Flow | 327 | 316 | 1.5168 | 1.0373 | stalled |
| r96_n1_ct10/sample_012 | DLR D-Flow | 245 | 232 | 1.5038 | 1.0096 | stalled |
| r96_n1_ct10/sample_013 | DLR D-Flow | 28 | 24 | 123.3987 | 111.8955 | stalled |
| r96_n1_ct10/sample_014 | DLR D-Flow | 162 | 159 | 105.0814 | 107.8407 | stalled |
| r96_n1_ct10/sample_015 | DLR D-Flow | 1000 | 962 | 2.4439 | 1.1400 | step_limit |
| r96_n1_ct10/sample_016 | DLR D-Flow | 1000 | 952 | 2.2340 | 1.1857 | step_limit |
| r96_n1_ct10/sample_017 | DLR D-Flow | 499 | 496 | 1.6144 | 1.0413 | stalled |
| r96_n1_ct10/sample_018 | DLR D-Flow | 387 | 367 | 4.5897 | 1.3449 | stalled |
| r96_n1_ct10/sample_019 | DLR D-Flow | 567 | 535 | 1.7388 | 1.0434 | stalled |
| r96_n1_ct10/sample_020 | DLR D-Flow | 269 | 257 | 1.8745 | 1.0589 | stalled |
| r96_n1_ct10/sample_021 | DLR D-Flow | 196 | 192 | 3.7386 | 2.3557 | stalled |
| r96_n1_ct10/sample_022 | DLR D-Flow | 395 | 375 | 1.5819 | 1.0269 | stalled |
| r96_n1_ct10/sample_023 | DLR D-Flow | 681 | 654 | 1.5121 | 1.0281 | stalled |
| r96_n1_ct10/sample_024 | DLR D-Flow | 1000 | 968 | 3.6100 | 2.6888 | step_limit |
| r96_n1_ct10/sample_025 | DLR D-Flow | 940 | 902 | 2.7230 | 1.2748 | stalled |
| r96_n1_ct10/sample_026 | DLR D-Flow | 305 | 296 | 119.8792 | 99.1975 | stalled |
| r96_n1_ct10/sample_027 | DLR D-Flow | 215 | 208 | 1.3746 | 1.0070 | stalled |
| r96_n1_ct10/sample_028 | DLR D-Flow | 33 | 29 | 119.4465 | 101.6918 | stalled |
| r96_n1_ct10/sample_029 | DLR D-Flow | 207 | 197 | 1.7466 | 1.0262 | stalled |
| r96_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 2.5023 | 1.0985 | step_limit |
| r96_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 4.7800 | 0.3002 | step_limit |
| r96_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 4.6174 | 0.2420 | step_limit |
| r96_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 4.7117 | 0.2225 | step_limit |
| r96_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 2.3344 | 0.6613 | step_limit |
| r96_n2_ct10/sample_005 | D-Flow | 300 | 298 | 2.0779 | 0.9407 | stalled |
| r96_n2_ct10/sample_006 | D-Flow | 471 | 469 | 72.9751 | 55.7960 | stalled |
| r96_n2_ct10/sample_007 | D-Flow | 1000 | 999 | 2.4568 | 0.7503 | step_limit |
| r96_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 3.2031 | 0.3619 | step_limit |
| r96_n2_ct10/sample_009 | D-Flow | 51 | 46 | 83.0551 | 63.7522 | stalled |
| r96_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 4.0607 | 0.5733 | step_limit |
| r96_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 3.7095 | 0.6332 | step_limit |
| r96_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 2.8370 | 0.7673 | step_limit |
| r96_n2_ct10/sample_000 | P-Flow | 64 | 55 | 2.2109 | 1.1348 | stalled |
| r96_n2_ct10/sample_001 | P-Flow | 772 | 632 | 1.9058 | 1.0452 | stalled |
| r96_n2_ct10/sample_002 | P-Flow | 270 | 220 | 3.5670 | 1.9141 | stalled |
| r96_n2_ct10/sample_003 | P-Flow | 50 | 43 | 57.3851 | 32.6320 | stalled |
| r96_n2_ct10/sample_004 | P-Flow | 199 | 166 | 1.8318 | 1.0931 | stalled |
| r96_n2_ct10/sample_005 | P-Flow | 291 | 235 | 1.9032 | 1.0525 | stalled |
| r96_n2_ct10/sample_006 | P-Flow | 1000 | 798 | 4.9332 | 2.8214 | step_limit |
| r96_n2_ct10/sample_007 | P-Flow | 18 | 15 | 36.3774 | 17.2734 | stalled |
| r96_n2_ct10/sample_008 | P-Flow | 161 | 131 | 3.5552 | 1.4269 | stalled |
| r96_n2_ct10/sample_009 | P-Flow | 1000 | 816 | 2.0775 | 1.1606 | step_limit |
| r96_n2_ct10/sample_010 | P-Flow | 93 | 78 | 2.2428 | 1.4383 | stalled |
| r96_n2_ct10/sample_011 | P-Flow | 1000 | 950 | 5.8978 | 0.1571 | step_limit |
| r96_n2_ct10/sample_012 | P-Flow | 1000 | 929 | 5.6859 | 0.1825 | step_limit |
| r96_n2_ct10/sample_000 | DLR D-Flow | 698 | 669 | 83.0999 | 56.2143 | stalled |
| r96_n2_ct10/sample_001 | DLR D-Flow | 227 | 219 | 9.2687 | 2.2519 | stalled |
| r96_n2_ct10/sample_002 | DLR D-Flow | 675 | 658 | 3.0710 | 1.1453 | stalled |
| r96_n2_ct10/sample_003 | DLR D-Flow | 980 | 951 | 56.4369 | 31.4305 | stalled |
| r96_n2_ct10/sample_004 | DLR D-Flow | 279 | 268 | 2.6642 | 1.0750 | stalled |
| r96_n2_ct10/sample_005 | DLR D-Flow | 614 | 584 | 2.0691 | 1.0586 | stalled |
| r96_n2_ct10/sample_006 | DLR D-Flow | 43 | 39 | 103.1096 | 72.1320 | stalled |
| r96_n2_ct10/sample_007 | DLR D-Flow | 505 | 489 | 61.4787 | 36.3061 | stalled |
| r96_n2_ct10/sample_008 | DLR D-Flow | 1000 | 948 | 3.1050 | 1.5594 | step_limit |
| r96_n2_ct10/sample_009 | DLR D-Flow | 417 | 402 | 76.7034 | 63.5922 | stalled |
| r96_n2_ct10/sample_010 | DLR D-Flow | 340 | 315 | 2.7413 | 1.1241 | stalled |
| r96_n2_ct10/sample_011 | DLR D-Flow | 1000 | 963 | 25.6366 | 8.9535 | step_limit |
| r96_n2_ct10/sample_012 | DLR D-Flow | 354 | 344 | 6.6748 | 2.3682 | stalled |
| r96_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 6.3095 | 0.1494 | step_limit |
| r96_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 5.0424 | 0.3444 | step_limit |
| r96_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 4.7325 | 0.1709 | step_limit |
| r96_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 5.1684 | 0.2283 | step_limit |
| r96_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.8775 | 1.0267 | step_limit |
| r96_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 6.5578 | 0.0863 | step_limit |
| r96_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 5.8228 | 0.1520 | step_limit |
| r96_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 7.3820 | 0.1415 | step_limit |
| r96_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 5.8929 | 0.1438 | step_limit |
| r96_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 2.9900 | 0.7503 | step_limit |
| r96_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 3.9402 | 0.5051 | step_limit |
| r96_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 3.1274 | 0.7640 | step_limit |
| r96_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 3.4903 | 0.7830 | step_limit |
| r96_n4_ct10/sample_000 | P-Flow | 128 | 113 | 1.7662 | 1.0433 | stalled |
| r96_n4_ct10/sample_001 | P-Flow | 1000 | 965 | 5.6024 | 0.1598 | step_limit |
| r96_n4_ct10/sample_002 | P-Flow | 262 | 214 | 2.6646 | 1.2835 | stalled |
| r96_n4_ct10/sample_003 | P-Flow | 717 | 594 | 2.4996 | 1.1073 | stalled |
| r96_n4_ct10/sample_004 | P-Flow | 221 | 185 | 1.9835 | 1.0548 | stalled |
| r96_n4_ct10/sample_005 | P-Flow | 1000 | 951 | 5.4944 | 0.1462 | step_limit |
| r96_n4_ct10/sample_006 | P-Flow | 1000 | 924 | 5.4529 | 0.1637 | step_limit |
| r96_n4_ct10/sample_007 | P-Flow | 1000 | 950 | 5.3641 | 0.1883 | step_limit |
| r96_n4_ct10/sample_008 | P-Flow | 848 | 759 | 5.6555 | 0.1376 | stalled |
| r96_n4_ct10/sample_009 | P-Flow | 691 | 624 | 5.4290 | 0.1388 | stalled |
| r96_n4_ct10/sample_010 | P-Flow | 858 | 804 | 4.9922 | 0.1362 | stalled |
| r96_n4_ct10/sample_011 | P-Flow | 290 | 235 | 2.0681 | 1.0293 | stalled |
| r96_n4_ct10/sample_012 | P-Flow | 689 | 641 | 5.9717 | 0.1691 | stalled |
| r96_n4_ct10/sample_000 | DLR D-Flow | 296 | 289 | 24.6260 | 6.9370 | stalled |
| r96_n4_ct10/sample_001 | DLR D-Flow | 1000 | 967 | 33.9815 | 13.8849 | step_limit |
| r96_n4_ct10/sample_002 | DLR D-Flow | 329 | 315 | 70.1131 | 39.2158 | stalled |
| r96_n4_ct10/sample_003 | DLR D-Flow | 1000 | 975 | 72.7422 | 38.6581 | step_limit |
| r96_n4_ct10/sample_004 | DLR D-Flow | 436 | 417 | 92.0823 | 48.4388 | stalled |
| r96_n4_ct10/sample_005 | DLR D-Flow | 340 | 316 | 18.6390 | 5.8432 | stalled |
| r96_n4_ct10/sample_006 | DLR D-Flow | 1000 | 968 | 17.5270 | 5.7764 | step_limit |
| r96_n4_ct10/sample_007 | DLR D-Flow | 431 | 418 | 4.9101 | 1.4329 | stalled |
| r96_n4_ct10/sample_008 | DLR D-Flow | 319 | 302 | 34.3060 | 14.8723 | stalled |
| r96_n4_ct10/sample_009 | DLR D-Flow | 639 | 617 | 45.3330 | 24.8886 | stalled |
| r96_n4_ct10/sample_010 | DLR D-Flow | 1000 | 962 | 91.8122 | 49.2031 | step_limit |
| r96_n4_ct10/sample_011 | DLR D-Flow | 565 | 539 | 19.9021 | 7.1838 | stalled |
| r96_n4_ct10/sample_012 | DLR D-Flow | 560 | 538 | 44.3158 | 19.0909 | stalled |
| r96_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.3684 | 0.6057 | step_limit |
| r96_n1_mri4/sample_001 | D-Flow | 229 | 227 | 102.6975 | 102.4752 | stalled |
| r96_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.3622 | 0.6548 | step_limit |
| r96_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.2478 | 0.8337 | step_limit |
| r96_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.8861 | 0.5775 | step_limit |
| r96_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.2694 | 0.6510 | step_limit |
| r96_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.3343 | 0.7001 | step_limit |
| r96_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.0683 | 0.6522 | step_limit |
| r96_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.5106 | 0.9326 | step_limit |
| r96_n1_mri4/sample_009 | D-Flow | 27 | 25 | 103.5456 | 102.2011 | stalled |
| r96_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.1456 | 0.7612 | step_limit |
| r96_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.1949 | 0.7594 | step_limit |
| r96_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.2508 | 0.5902 | step_limit |
| r96_n1_mri4/sample_013 | D-Flow | 328 | 325 | 102.4318 | 101.9118 | stalled |
| r96_n1_mri4/sample_014 | D-Flow | 503 | 499 | 103.4075 | 100.9466 | stalled |
| r96_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 1.2524 | 0.7557 | step_limit |
| r96_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 1.3697 | 0.7775 | step_limit |
| r96_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 1.1240 | 0.6454 | step_limit |
| r96_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 1.2328 | 0.6443 | step_limit |
| r96_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 1.2555 | 0.6476 | step_limit |
| r96_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 1.4573 | 0.5552 | step_limit |
| r96_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 1.4445 | 0.6580 | step_limit |
| r96_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 1.3500 | 0.6578 | step_limit |
| r96_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 1.1722 | 0.6871 | step_limit |
| r96_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 1.7124 | 0.6236 | step_limit |
| r96_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 1.6289 | 0.5849 | step_limit |
| r96_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 1.5541 | 0.8210 | step_limit |
| r96_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 1.2247 | 0.6439 | step_limit |
| r96_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 2.2369 | 1.7098 | step_limit |
| r96_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 1.2505 | 0.5472 | step_limit |
| r96_n1_mri4/sample_000 | P-Flow | 449 | 373 | 1.9950 | 1.0576 | stalled |
| r96_n1_mri4/sample_001 | P-Flow | 12 | 8 | 117.3890 | 116.0087 | stalled |
| r96_n1_mri4/sample_002 | P-Flow | 29 | 24 | 1.8159 | 1.4479 | stalled |
| r96_n1_mri4/sample_003 | P-Flow | 234 | 196 | 1.5684 | 1.1820 | stalled |
| r96_n1_mri4/sample_004 | P-Flow | 19 | 15 | 5.3022 | 3.5721 | stalled |
| r96_n1_mri4/sample_005 | P-Flow | 321 | 277 | 2.2035 | 0.8291 | stalled |
| r96_n1_mri4/sample_006 | P-Flow | 239 | 217 | 3.0500 | 0.6974 | stalled |
| r96_n1_mri4/sample_007 | P-Flow | 23 | 20 | 1.7699 | 1.6504 | stalled |
| r96_n1_mri4/sample_008 | P-Flow | 196 | 172 | 2.5230 | 1.1076 | stalled |
| r96_n1_mri4/sample_009 | P-Flow | 12 | 9 | 120.4536 | 111.7319 | stalled |
| r96_n1_mri4/sample_010 | P-Flow | 369 | 326 | 2.2442 | 0.5223 | stalled |
| r96_n1_mri4/sample_011 | P-Flow | 49 | 40 | 1.3996 | 1.2420 | stalled |
| r96_n1_mri4/sample_012 | P-Flow | 320 | 296 | 2.3761 | 0.4328 | stalled |
| r96_n1_mri4/sample_013 | P-Flow | 260 | 225 | 3.2977 | 1.3172 | stalled |
| r96_n1_mri4/sample_014 | P-Flow | 9 | 7 | 135.3198 | 127.9697 | stalled |
| r96_n1_mri4/sample_015 | P-Flow | 12 | 10 | 112.8804 | 112.7143 | stalled |
| r96_n1_mri4/sample_016 | P-Flow | 95 | 80 | 1.7564 | 1.3673 | stalled |
| r96_n1_mri4/sample_017 | P-Flow | 85 | 73 | 1.3295 | 1.2854 | stalled |
| r96_n1_mri4/sample_018 | P-Flow | 75 | 61 | 1.6319 | 1.3523 | stalled |
| r96_n1_mri4/sample_019 | P-Flow | 564 | 493 | 2.4486 | 0.5016 | stalled |
| r96_n1_mri4/sample_020 | P-Flow | 243 | 223 | 2.5973 | 0.5442 | stalled |
| r96_n1_mri4/sample_021 | P-Flow | 60 | 49 | 1.8488 | 1.4244 | stalled |
| r96_n1_mri4/sample_022 | P-Flow | 321 | 284 | 2.5396 | 0.6724 | stalled |
| r96_n1_mri4/sample_023 | P-Flow | 130 | 110 | 1.4301 | 1.3273 | stalled |
| r96_n1_mri4/sample_024 | P-Flow | 45 | 38 | 3.4049 | 2.7139 | stalled |
| r96_n1_mri4/sample_025 | P-Flow | 451 | 391 | 2.8220 | 0.7875 | stalled |
| r96_n1_mri4/sample_026 | P-Flow | 267 | 233 | 3.1296 | 0.8405 | stalled |
| r96_n1_mri4/sample_027 | P-Flow | 12 | 10 | 1.4847 | 1.4066 | stalled |
| r96_n1_mri4/sample_028 | P-Flow | 25 | 20 | 2.0939 | 1.6654 | stalled |
| r96_n1_mri4/sample_029 | P-Flow | 287 | 255 | 2.2992 | 0.6110 | stalled |
| r96_n1_mri4/sample_000 | DLR D-Flow | 383 | 364 | 1.7795 | 1.4452 | stalled |
| r96_n1_mri4/sample_001 | DLR D-Flow | 120 | 116 | 102.5211 | 102.2930 | stalled |
| r96_n1_mri4/sample_002 | DLR D-Flow | 1000 | 931 | 1.6997 | 1.3482 | step_limit |
| r96_n1_mri4/sample_003 | DLR D-Flow | 114 | 107 | 1.8289 | 1.5140 | stalled |
| r96_n1_mri4/sample_004 | DLR D-Flow | 1000 | 914 | 3.6787 | 2.6225 | step_limit |
| r96_n1_mri4/sample_005 | DLR D-Flow | 90 | 86 | 1.7691 | 1.5681 | stalled |
| r96_n1_mri4/sample_006 | DLR D-Flow | 265 | 256 | 1.7324 | 1.4858 | stalled |
| r96_n1_mri4/sample_007 | DLR D-Flow | 634 | 618 | 1.3077 | 1.2972 | stalled |
| r96_n1_mri4/sample_008 | DLR D-Flow | 145 | 136 | 102.8958 | 102.8831 | stalled |
| r96_n1_mri4/sample_009 | DLR D-Flow | 169 | 163 | 103.7544 | 101.9786 | stalled |
| r96_n1_mri4/sample_010 | DLR D-Flow | 382 | 366 | 1.5384 | 1.3377 | stalled |
| r96_n1_mri4/sample_011 | DLR D-Flow | 298 | 284 | 1.4850 | 1.2808 | stalled |
| r96_n1_mri4/sample_012 | DLR D-Flow | 435 | 414 | 1.6556 | 1.3756 | stalled |
| r96_n1_mri4/sample_013 | DLR D-Flow | 56 | 52 | 103.1454 | 102.8660 | stalled |
| r96_n1_mri4/sample_014 | DLR D-Flow | 23 | 20 | 127.5350 | 119.4833 | stalled |
| r96_n1_mri4/sample_015 | DLR D-Flow | 620 | 605 | 2.0964 | 1.7042 | stalled |
| r96_n1_mri4/sample_016 | DLR D-Flow | 227 | 214 | 4.0406 | 2.8422 | stalled |
| r96_n1_mri4/sample_017 | DLR D-Flow | 206 | 201 | 1.4503 | 1.3608 | stalled |
| r96_n1_mri4/sample_018 | DLR D-Flow | 315 | 301 | 1.6620 | 1.3890 | stalled |
| r96_n1_mri4/sample_019 | DLR D-Flow | 54 | 50 | 102.4263 | 102.7424 | stalled |
| r96_n1_mri4/sample_020 | DLR D-Flow | 333 | 321 | 1.8421 | 1.3813 | stalled |
| r96_n1_mri4/sample_021 | DLR D-Flow | 393 | 380 | 1.7943 | 1.4077 | stalled |
| r96_n1_mri4/sample_022 | DLR D-Flow | 422 | 409 | 1.5823 | 1.3016 | stalled |
| r96_n1_mri4/sample_023 | DLR D-Flow | 1000 | 961 | 1.5155 | 1.4184 | step_limit |
| r96_n1_mri4/sample_024 | DLR D-Flow | 422 | 404 | 10.4762 | 7.9094 | stalled |
| r96_n1_mri4/sample_025 | DLR D-Flow | 687 | 647 | 2.4775 | 1.9665 | stalled |
| r96_n1_mri4/sample_026 | DLR D-Flow | 86 | 81 | 118.4270 | 112.1704 | stalled |
| r96_n1_mri4/sample_027 | DLR D-Flow | 165 | 155 | 1.2495 | 1.1940 | stalled |
| r96_n1_mri4/sample_028 | DLR D-Flow | 59 | 53 | 118.3152 | 115.5517 | stalled |
| r96_n1_mri4/sample_029 | DLR D-Flow | 165 | 157 | 1.5239 | 1.4236 | stalled |
| r96_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.2046 | 0.7656 | step_limit |
| r96_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.1672 | 0.7013 | step_limit |
| r96_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.2148 | 0.9324 | step_limit |
| r96_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.1926 | 0.7932 | step_limit |
| r96_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.1128 | 0.7167 | step_limit |
| r96_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.3295 | 0.5892 | step_limit |
| r96_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.4638 | 0.7505 | step_limit |
| r96_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.3132 | 1.2002 | step_limit |
| r96_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.5621 | 0.8152 | step_limit |
| r96_n2_mri4/sample_009 | D-Flow | 382 | 380 | 68.4351 | 68.2879 | stalled |
| r96_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.1135 | 0.6780 | step_limit |
| r96_n2_mri4/sample_011 | D-Flow | 624 | 622 | 41.6464 | 39.9725 | stalled |
| r96_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.1068 | 0.8913 | step_limit |
| r96_n2_mri4/sample_000 | P-Flow | 155 | 132 | 1.6760 | 1.4442 | stalled |
| r96_n2_mri4/sample_001 | P-Flow | 507 | 458 | 2.4174 | 0.5254 | stalled |
| r96_n2_mri4/sample_002 | P-Flow | 538 | 475 | 2.5290 | 0.4053 | stalled |
| r96_n2_mri4/sample_003 | P-Flow | 298 | 269 | 2.5648 | 0.5146 | stalled |
| r96_n2_mri4/sample_004 | P-Flow | 386 | 342 | 2.4274 | 0.5236 | stalled |
| r96_n2_mri4/sample_005 | P-Flow | 314 | 290 | 2.5143 | 0.5154 | stalled |
| r96_n2_mri4/sample_006 | P-Flow | 59 | 53 | 5.4667 | 4.0763 | stalled |
| r96_n2_mri4/sample_007 | P-Flow | 513 | 449 | 2.4596 | 0.4335 | stalled |
| r96_n2_mri4/sample_008 | P-Flow | 32 | 28 | 2.9933 | 2.2808 | stalled |
| r96_n2_mri4/sample_009 | P-Flow | 66 | 57 | 1.5577 | 1.2979 | stalled |
| r96_n2_mri4/sample_010 | P-Flow | 559 | 504 | 2.4218 | 0.5609 | stalled |
| r96_n2_mri4/sample_011 | P-Flow | 395 | 359 | 2.1648 | 0.4839 | stalled |
| r96_n2_mri4/sample_012 | P-Flow | 392 | 361 | 2.4533 | 0.3764 | stalled |
| r96_n2_mri4/sample_000 | DLR D-Flow | 482 | 463 | 2.1931 | 1.9327 | stalled |
| r96_n2_mri4/sample_001 | DLR D-Flow | 898 | 851 | 1.6705 | 1.4790 | stalled |
| r96_n2_mri4/sample_002 | DLR D-Flow | 665 | 642 | 2.8651 | 2.5458 | stalled |
| r96_n2_mri4/sample_003 | DLR D-Flow | 1000 | 972 | 40.5635 | 34.8064 | step_limit |
| r96_n2_mri4/sample_004 | DLR D-Flow | 582 | 546 | 2.1673 | 1.8690 | stalled |
| r96_n2_mri4/sample_005 | DLR D-Flow | 318 | 299 | 1.6640 | 1.4001 | stalled |
| r96_n2_mri4/sample_006 | DLR D-Flow | 523 | 505 | 68.0130 | 67.8729 | stalled |
| r96_n2_mri4/sample_007 | DLR D-Flow | 1000 | 968 | 52.6763 | 51.1345 | step_limit |
| r96_n2_mri4/sample_008 | DLR D-Flow | 1000 | 970 | 68.4968 | 66.2551 | step_limit |
| r96_n2_mri4/sample_009 | DLR D-Flow | 114 | 107 | 76.1888 | 74.3702 | stalled |
| r96_n2_mri4/sample_010 | DLR D-Flow | 542 | 528 | 1.8071 | 1.5806 | stalled |
| r96_n2_mri4/sample_011 | DLR D-Flow | 1000 | 966 | 2.3164 | 2.1530 | step_limit |
| r96_n2_mri4/sample_012 | DLR D-Flow | 1000 | 976 | 101.0573 | 99.2928 | step_limit |
| r96_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.2479 | 0.5159 | step_limit |
| r96_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.3458 | 0.7606 | step_limit |
| r96_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.4316 | 0.7585 | step_limit |
| r96_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.3650 | 0.7168 | step_limit |
| r96_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 42.5035 | 40.5613 | step_limit |
| r96_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.3613 | 0.7980 | step_limit |
| r96_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.2380 | 0.6742 | step_limit |
| r96_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.3065 | 0.6653 | step_limit |
| r96_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.2710 | 0.8381 | step_limit |
| r96_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.4031 | 0.5618 | step_limit |
| r96_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.4592 | 0.8059 | step_limit |
| r96_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.3496 | 0.6098 | step_limit |
| r96_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.2549 | 0.6919 | step_limit |
| r96_n4_mri4/sample_000 | P-Flow | 934 | 811 | 2.3526 | 0.3760 | stalled |
| r96_n4_mri4/sample_001 | P-Flow | 688 | 598 | 2.7480 | 0.5174 | stalled |
| r96_n4_mri4/sample_002 | P-Flow | 89 | 76 | 1.7114 | 1.3319 | stalled |
| r96_n4_mri4/sample_003 | P-Flow | 638 | 570 | 2.4600 | 0.5127 | stalled |
| r96_n4_mri4/sample_004 | P-Flow | 78 | 67 | 1.6812 | 1.3538 | stalled |
| r96_n4_mri4/sample_005 | P-Flow | 4 | 2 | 105.7076 | 89.8567 | stalled |
| r96_n4_mri4/sample_006 | P-Flow | 511 | 454 | 2.4056 | 0.3907 | stalled |
| r96_n4_mri4/sample_007 | P-Flow | 851 | 744 | 2.3356 | 0.5667 | stalled |
| r96_n4_mri4/sample_008 | P-Flow | 595 | 530 | 2.4343 | 0.6665 | stalled |
| r96_n4_mri4/sample_009 | P-Flow | 453 | 411 | 2.3840 | 0.4255 | stalled |
| r96_n4_mri4/sample_010 | P-Flow | 178 | 152 | 1.6676 | 1.2774 | stalled |
| r96_n4_mri4/sample_011 | P-Flow | 565 | 501 | 2.4122 | 0.5504 | stalled |
| r96_n4_mri4/sample_012 | P-Flow | 524 | 468 | 2.3982 | 0.4790 | stalled |
| r96_n4_mri4/sample_000 | DLR D-Flow | 237 | 228 | 15.2394 | 11.2985 | stalled |
| r96_n4_mri4/sample_001 | DLR D-Flow | 1000 | 965 | 38.8893 | 34.3141 | step_limit |
| r96_n4_mri4/sample_002 | DLR D-Flow | 1000 | 974 | 75.8110 | 69.6590 | step_limit |
| r96_n4_mri4/sample_003 | DLR D-Flow | 924 | 890 | 45.2617 | 43.8400 | stalled |
| r96_n4_mri4/sample_004 | DLR D-Flow | 1000 | 975 | 88.4339 | 77.2545 | step_limit |
| r96_n4_mri4/sample_005 | DLR D-Flow | 309 | 292 | 4.1467 | 3.0477 | stalled |
| r96_n4_mri4/sample_006 | DLR D-Flow | 1000 | 972 | 13.6617 | 10.3648 | step_limit |
| r96_n4_mri4/sample_007 | DLR D-Flow | 676 | 657 | 78.2281 | 75.3300 | stalled |
| r96_n4_mri4/sample_008 | DLR D-Flow | 1000 | 959 | 8.2718 | 6.1061 | step_limit |
| r96_n4_mri4/sample_009 | DLR D-Flow | 218 | 209 | 39.5031 | 38.4421 | stalled |
| r96_n4_mri4/sample_010 | DLR D-Flow | 237 | 229 | 19.9309 | 16.3521 | stalled |
| r96_n4_mri4/sample_011 | DLR D-Flow | 1000 | 970 | 57.2033 | 53.7660 | step_limit |
| r96_n4_mri4/sample_012 | DLR D-Flow | 1000 | 958 | 77.0187 | 73.4914 | step_limit |
| r128_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 3.0469 | 0.3426 | step_limit |
| r128_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 3.8100 | 0.4723 | step_limit |
| r128_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 3.7948 | 0.3626 | step_limit |
| r128_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 5.1498 | 0.1296 | step_limit |
| r128_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 4.2037 | 0.2575 | step_limit |
| r128_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 4.1163 | 0.2421 | step_limit |
| r128_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 6.0462 | 0.1106 | step_limit |
| r128_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 5.6270 | 0.1643 | step_limit |
| r128_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 3.9258 | 0.2385 | step_limit |
| r128_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 3.6408 | 0.2950 | step_limit |
| r128_n1_ct10/sample_010 | D-Flow | 58 | 54 | 119.0651 | 112.5838 | stalled |
| r128_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 4.8632 | 0.1208 | step_limit |
| r128_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 4.0188 | 0.1790 | step_limit |
| r128_n1_ct10/sample_013 | D-Flow | 1000 | 1000 | 3.5491 | 0.4269 | step_limit |
| r128_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 3.5473 | 0.2469 | step_limit |
| r128_n1_ct10/sample_015 | D-Flow | 1000 | 1000 | 3.8730 | 0.2496 | step_limit |
| r128_n1_ct10/sample_016 | D-Flow | 1000 | 1000 | 4.6597 | 0.1297 | step_limit |
| r128_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 6.2076 | 0.0746 | step_limit |
| r128_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 4.5689 | 0.1977 | step_limit |
| r128_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 5.3230 | 0.1411 | step_limit |
| r128_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 5.3687 | 0.1772 | step_limit |
| r128_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 5.3468 | 0.1117 | step_limit |
| r128_n1_ct10/sample_022 | D-Flow | 1000 | 1000 | 3.7022 | 0.1447 | step_limit |
| r128_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 3.0816 | 0.7048 | step_limit |
| r128_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 4.5625 | 0.5007 | step_limit |
| r128_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 2.7898 | 0.4857 | step_limit |
| r128_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 3.4695 | 0.2346 | step_limit |
| r128_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 4.1937 | 0.1968 | step_limit |
| r128_n1_ct10/sample_028 | D-Flow | 1000 | 1000 | 5.0180 | 0.1807 | step_limit |
| r128_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 4.1613 | 0.2179 | step_limit |
| r128_n1_ct10/sample_000 | P-Flow | 98 | 82 | 3.5045 | 1.3861 | stalled |
| r128_n1_ct10/sample_001 | P-Flow | 1000 | 766 | 2.5528 | 1.2159 | step_limit |
| r128_n1_ct10/sample_002 | P-Flow | 62 | 56 | 4.3349 | 1.5528 | stalled |
| r128_n1_ct10/sample_003 | P-Flow | 54 | 48 | 2.4387 | 1.1772 | stalled |
| r128_n1_ct10/sample_004 | P-Flow | 584 | 554 | 6.0382 | 0.1126 | stalled |
| r128_n1_ct10/sample_005 | P-Flow | 530 | 488 | 5.6571 | 0.1640 | stalled |
| r128_n1_ct10/sample_006 | P-Flow | 1000 | 967 | 6.1796 | 0.1402 | step_limit |
| r128_n1_ct10/sample_007 | P-Flow | 56 | 48 | 2.3310 | 1.1315 | stalled |
| r128_n1_ct10/sample_008 | P-Flow | 1000 | 968 | 6.2354 | 0.1427 | step_limit |
| r128_n1_ct10/sample_009 | P-Flow | 30 | 26 | 3.2072 | 1.7955 | stalled |
| r128_n1_ct10/sample_010 | P-Flow | 91 | 75 | 3.4667 | 1.3613 | stalled |
| r128_n1_ct10/sample_011 | P-Flow | 125 | 103 | 3.4925 | 2.0879 | stalled |
| r128_n1_ct10/sample_012 | P-Flow | 185 | 153 | 2.6114 | 1.3496 | stalled |
| r128_n1_ct10/sample_013 | P-Flow | 51 | 42 | 15.2572 | 4.9665 | stalled |
| r128_n1_ct10/sample_014 | P-Flow | 48 | 42 | 10.2643 | 3.8167 | stalled |
| r128_n1_ct10/sample_015 | P-Flow | 49 | 43 | 3.4898 | 1.3870 | stalled |
| r128_n1_ct10/sample_016 | P-Flow | 55 | 46 | 2.3528 | 1.1686 | stalled |
| r128_n1_ct10/sample_017 | P-Flow | 1000 | 964 | 6.4631 | 0.1221 | step_limit |
| r128_n1_ct10/sample_018 | P-Flow | 35 | 29 | 2.5599 | 1.1923 | stalled |
| r128_n1_ct10/sample_019 | P-Flow | 86 | 78 | 2.1598 | 1.1508 | stalled |
| r128_n1_ct10/sample_020 | P-Flow | 46 | 39 | 4.2830 | 1.5486 | stalled |
| r128_n1_ct10/sample_021 | P-Flow | 495 | 396 | 2.2162 | 1.2393 | stalled |
| r128_n1_ct10/sample_022 | P-Flow | 672 | 643 | 5.8851 | 0.1238 | stalled |
| r128_n1_ct10/sample_023 | P-Flow | 47 | 41 | 12.4225 | 3.4269 | stalled |
| r128_n1_ct10/sample_024 | P-Flow | 60 | 54 | 2.0322 | 1.1086 | stalled |
| r128_n1_ct10/sample_025 | P-Flow | 34 | 30 | 5.7811 | 2.1426 | stalled |
| r128_n1_ct10/sample_026 | P-Flow | 310 | 248 | 2.9685 | 1.2478 | stalled |
| r128_n1_ct10/sample_027 | P-Flow | 296 | 242 | 1.8318 | 1.1549 | stalled |
| r128_n1_ct10/sample_028 | P-Flow | 238 | 195 | 1.6258 | 1.0306 | stalled |
| r128_n1_ct10/sample_029 | P-Flow | 55 | 49 | 1.6478 | 1.0735 | stalled |
| r128_n1_ct10/sample_000 | DLR D-Flow | 239 | 227 | 47.7159 | 29.4301 | stalled |
| r128_n1_ct10/sample_001 | DLR D-Flow | 497 | 467 | 1.7521 | 1.0455 | stalled |
| r128_n1_ct10/sample_002 | DLR D-Flow | 532 | 515 | 11.4672 | 3.8753 | stalled |
| r128_n1_ct10/sample_003 | DLR D-Flow | 178 | 173 | 2.1692 | 1.0744 | stalled |
| r128_n1_ct10/sample_004 | DLR D-Flow | 146 | 141 | 3.0866 | 1.2490 | stalled |
| r128_n1_ct10/sample_005 | DLR D-Flow | 312 | 303 | 2.1140 | 1.0784 | stalled |
| r128_n1_ct10/sample_006 | DLR D-Flow | 141 | 138 | 1.9137 | 1.0683 | stalled |
| r128_n1_ct10/sample_007 | DLR D-Flow | 243 | 231 | 1.4920 | 1.0136 | stalled |
| r128_n1_ct10/sample_008 | DLR D-Flow | 199 | 185 | 113.5908 | 93.4950 | stalled |
| r128_n1_ct10/sample_009 | DLR D-Flow | 571 | 544 | 1.4179 | 1.0302 | stalled |
| r128_n1_ct10/sample_010 | DLR D-Flow | 87 | 81 | 117.3379 | 114.3802 | stalled |
| r128_n1_ct10/sample_011 | DLR D-Flow | 187 | 176 | 2.5380 | 1.5868 | stalled |
| r128_n1_ct10/sample_012 | DLR D-Flow | 527 | 502 | 2.0780 | 1.0479 | stalled |
| r128_n1_ct10/sample_013 | DLR D-Flow | 165 | 156 | 3.7155 | 1.2175 | stalled |
| r128_n1_ct10/sample_014 | DLR D-Flow | 635 | 608 | 8.7006 | 3.3512 | stalled |
| r128_n1_ct10/sample_015 | DLR D-Flow | 1000 | 949 | 2.5808 | 1.2211 | step_limit |
| r128_n1_ct10/sample_016 | DLR D-Flow | 139 | 134 | 1.8230 | 1.0741 | stalled |
| r128_n1_ct10/sample_017 | DLR D-Flow | 313 | 304 | 1.6841 | 1.0393 | stalled |
| r128_n1_ct10/sample_018 | DLR D-Flow | 399 | 380 | 1.4166 | 1.0182 | stalled |
| r128_n1_ct10/sample_019 | DLR D-Flow | 441 | 423 | 1.5033 | 1.0308 | stalled |
| r128_n1_ct10/sample_020 | DLR D-Flow | 1000 | 974 | 1.6550 | 1.0304 | step_limit |
| r128_n1_ct10/sample_021 | DLR D-Flow | 343 | 327 | 1.5108 | 1.0315 | stalled |
| r128_n1_ct10/sample_022 | DLR D-Flow | 127 | 121 | 3.8418 | 1.5498 | stalled |
| r128_n1_ct10/sample_023 | DLR D-Flow | 151 | 146 | 5.7169 | 1.8499 | stalled |
| r128_n1_ct10/sample_024 | DLR D-Flow | 109 | 107 | 1.9134 | 1.0429 | stalled |
| r128_n1_ct10/sample_025 | DLR D-Flow | 450 | 432 | 26.6259 | 14.7684 | stalled |
| r128_n1_ct10/sample_026 | DLR D-Flow | 1000 | 947 | 2.4733 | 1.2050 | step_limit |
| r128_n1_ct10/sample_027 | DLR D-Flow | 366 | 346 | 1.6542 | 1.0334 | stalled |
| r128_n1_ct10/sample_028 | DLR D-Flow | 375 | 358 | 1.7003 | 1.0218 | stalled |
| r128_n1_ct10/sample_029 | DLR D-Flow | 81 | 76 | 1.5245 | 1.0171 | stalled |
| r128_n2_ct10/sample_000 | D-Flow | 725 | 721 | 16.2334 | 5.4993 | stalled |
| r128_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 5.1938 | 0.1529 | step_limit |
| r128_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.1512 | 0.8086 | step_limit |
| r128_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 2.7381 | 0.8282 | step_limit |
| r128_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 2.8252 | 0.6057 | step_limit |
| r128_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 4.0569 | 0.5956 | step_limit |
| r128_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 4.5557 | 0.4024 | step_limit |
| r128_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 5.5599 | 0.1337 | step_limit |
| r128_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 5.7227 | 0.1274 | step_limit |
| r128_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.7494 | 0.9078 | step_limit |
| r128_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 4.6591 | 0.4283 | step_limit |
| r128_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.2843 | 1.0019 | step_limit |
| r128_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 5.1847 | 0.1123 | step_limit |
| r128_n2_ct10/sample_000 | P-Flow | 463 | 376 | 2.3202 | 1.1351 | stalled |
| r128_n2_ct10/sample_001 | P-Flow | 145 | 128 | 1.9334 | 1.0406 | stalled |
| r128_n2_ct10/sample_002 | P-Flow | 36 | 30 | 6.7832 | 2.2880 | stalled |
| r128_n2_ct10/sample_003 | P-Flow | 243 | 205 | 1.5349 | 0.9785 | stalled |
| r128_n2_ct10/sample_004 | P-Flow | 38 | 32 | 3.8816 | 2.3117 | stalled |
| r128_n2_ct10/sample_005 | P-Flow | 1000 | 820 | 3.7435 | 2.0162 | step_limit |
| r128_n2_ct10/sample_006 | P-Flow | 1000 | 931 | 5.5810 | 0.1697 | step_limit |
| r128_n2_ct10/sample_007 | P-Flow | 1000 | 936 | 5.8923 | 0.1330 | step_limit |
| r128_n2_ct10/sample_008 | P-Flow | 1000 | 942 | 5.8737 | 0.1499 | step_limit |
| r128_n2_ct10/sample_009 | P-Flow | 42 | 34 | 7.2062 | 2.5629 | stalled |
| r128_n2_ct10/sample_010 | P-Flow | 275 | 250 | 5.9638 | 0.1474 | stalled |
| r128_n2_ct10/sample_011 | P-Flow | 53 | 46 | 5.2358 | 1.7727 | stalled |
| r128_n2_ct10/sample_012 | P-Flow | 1000 | 924 | 5.9656 | 0.1217 | step_limit |
| r128_n2_ct10/sample_000 | DLR D-Flow | 71 | 67 | 114.2255 | 88.8924 | stalled |
| r128_n2_ct10/sample_001 | DLR D-Flow | 385 | 370 | 4.3565 | 1.4711 | stalled |
| r128_n2_ct10/sample_002 | DLR D-Flow | 464 | 443 | 97.9448 | 66.4981 | stalled |
| r128_n2_ct10/sample_003 | DLR D-Flow | 191 | 187 | 119.1350 | 86.0027 | stalled |
| r128_n2_ct10/sample_004 | DLR D-Flow | 1000 | 968 | 7.9205 | 4.0840 | step_limit |
| r128_n2_ct10/sample_005 | DLR D-Flow | 94 | 91 | 6.2185 | 1.6036 | stalled |
| r128_n2_ct10/sample_006 | DLR D-Flow | 457 | 436 | 1.6963 | 1.0284 | stalled |
| r128_n2_ct10/sample_007 | DLR D-Flow | 948 | 926 | 46.0360 | 20.9269 | stalled |
| r128_n2_ct10/sample_008 | DLR D-Flow | 1000 | 971 | 1.7861 | 1.2443 | step_limit |
| r128_n2_ct10/sample_009 | DLR D-Flow | 158 | 153 | 101.4606 | 71.2707 | stalled |
| r128_n2_ct10/sample_010 | DLR D-Flow | 757 | 716 | 3.1821 | 1.1797 | stalled |
| r128_n2_ct10/sample_011 | DLR D-Flow | 689 | 665 | 86.5017 | 57.6707 | stalled |
| r128_n2_ct10/sample_012 | DLR D-Flow | 82 | 78 | 3.4635 | 1.6967 | stalled |
| r128_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 2.4176 | 0.8091 | step_limit |
| r128_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 3.6236 | 0.5343 | step_limit |
| r128_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.3170 | 0.8982 | step_limit |
| r128_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 2.6851 | 0.8021 | step_limit |
| r128_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 2.6088 | 0.8312 | step_limit |
| r128_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 2.6913 | 0.8478 | step_limit |
| r128_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 37.4290 | 20.7628 | step_limit |
| r128_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 46.7627 | 25.4491 | step_limit |
| r128_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 2.4955 | 0.6599 | step_limit |
| r128_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 2.0212 | 0.9024 | step_limit |
| r128_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 2.4982 | 0.7011 | step_limit |
| r128_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.9836 | 0.9187 | step_limit |
| r128_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 1.8535 | 0.9598 | step_limit |
| r128_n4_ct10/sample_000 | P-Flow | 79 | 69 | 2.3798 | 1.0637 | stalled |
| r128_n4_ct10/sample_001 | P-Flow | 1000 | 890 | 5.1997 | 0.1464 | step_limit |
| r128_n4_ct10/sample_002 | P-Flow | 1000 | 908 | 5.3427 | 0.1797 | step_limit |
| r128_n4_ct10/sample_003 | P-Flow | 1000 | 821 | 2.4909 | 1.0826 | step_limit |
| r128_n4_ct10/sample_004 | P-Flow | 111 | 96 | 1.9273 | 0.9976 | stalled |
| r128_n4_ct10/sample_005 | P-Flow | 1000 | 950 | 5.4748 | 0.1470 | step_limit |
| r128_n4_ct10/sample_006 | P-Flow | 1000 | 930 | 5.3767 | 0.1425 | step_limit |
| r128_n4_ct10/sample_007 | P-Flow | 1000 | 861 | 5.3287 | 0.2405 | step_limit |
| r128_n4_ct10/sample_008 | P-Flow | 79 | 68 | 1.5818 | 1.0122 | stalled |
| r128_n4_ct10/sample_009 | P-Flow | 1000 | 926 | 5.4406 | 0.1106 | step_limit |
| r128_n4_ct10/sample_010 | P-Flow | 1000 | 916 | 5.2488 | 0.1564 | step_limit |
| r128_n4_ct10/sample_011 | P-Flow | 1000 | 958 | 5.4326 | 0.1410 | step_limit |
| r128_n4_ct10/sample_012 | P-Flow | 790 | 724 | 5.3068 | 0.1545 | stalled |
| r128_n4_ct10/sample_000 | DLR D-Flow | 495 | 478 | 86.4071 | 43.2415 | stalled |
| r128_n4_ct10/sample_001 | DLR D-Flow | 509 | 492 | 29.2016 | 14.0149 | stalled |
| r128_n4_ct10/sample_002 | DLR D-Flow | 30 | 26 | 94.8984 | 52.9670 | stalled |
| r128_n4_ct10/sample_003 | DLR D-Flow | 402 | 387 | 27.3032 | 9.2061 | stalled |
| r128_n4_ct10/sample_004 | DLR D-Flow | 552 | 530 | 28.1627 | 9.6455 | stalled |
| r128_n4_ct10/sample_005 | DLR D-Flow | 534 | 511 | 90.5455 | 47.5693 | stalled |
| r128_n4_ct10/sample_006 | DLR D-Flow | 400 | 393 | 13.3710 | 4.7657 | stalled |
| r128_n4_ct10/sample_007 | DLR D-Flow | 185 | 178 | 67.1573 | 31.4055 | stalled |
| r128_n4_ct10/sample_008 | DLR D-Flow | 50 | 48 | 80.5655 | 42.6484 | stalled |
| r128_n4_ct10/sample_009 | DLR D-Flow | 325 | 315 | 46.4818 | 19.1649 | stalled |
| r128_n4_ct10/sample_010 | DLR D-Flow | 1000 | 955 | 80.2785 | 43.3518 | step_limit |
| r128_n4_ct10/sample_011 | DLR D-Flow | 1000 | 960 | 93.8708 | 51.8121 | step_limit |
| r128_n4_ct10/sample_012 | DLR D-Flow | 118 | 111 | 63.4850 | 39.8949 | stalled |
| r128_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.5712 | 0.4997 | step_limit |
| r128_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.0862 | 0.8960 | step_limit |
| r128_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.2483 | 0.6228 | step_limit |
| r128_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.1489 | 0.6751 | step_limit |
| r128_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.5079 | 0.7136 | step_limit |
| r128_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.3546 | 0.6999 | step_limit |
| r128_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.2365 | 0.5981 | step_limit |
| r128_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.1635 | 0.7104 | step_limit |
| r128_n1_mri4/sample_008 | D-Flow | 323 | 320 | 103.7707 | 101.0897 | stalled |
| r128_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.1554 | 0.6933 | step_limit |
| r128_n1_mri4/sample_010 | D-Flow | 398 | 394 | 104.8788 | 101.8406 | stalled |
| r128_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.0611 | 0.6727 | step_limit |
| r128_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.1236 | 0.6907 | step_limit |
| r128_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 1.1028 | 0.7542 | step_limit |
| r128_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 1.6236 | 0.5077 | step_limit |
| r128_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 1.1776 | 0.8476 | step_limit |
| r128_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 1.3354 | 0.6392 | step_limit |
| r128_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 1.1379 | 0.6722 | step_limit |
| r128_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 1.0594 | 0.7694 | step_limit |
| r128_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 1.1028 | 0.6526 | step_limit |
| r128_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 1.0679 | 0.9724 | step_limit |
| r128_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 1.1570 | 0.7650 | step_limit |
| r128_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 1.4946 | 0.5997 | step_limit |
| r128_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 1.0964 | 0.8342 | step_limit |
| r128_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 1.0800 | 0.7308 | step_limit |
| r128_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 1.3838 | 0.6744 | step_limit |
| r128_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 1.6343 | 0.5286 | step_limit |
| r128_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 1.1181 | 0.7797 | step_limit |
| r128_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 1.1463 | 0.6938 | step_limit |
| r128_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 1.0881 | 0.7616 | step_limit |
| r128_n1_mri4/sample_000 | P-Flow | 44 | 38 | 3.0748 | 2.2983 | stalled |
| r128_n1_mri4/sample_001 | P-Flow | 16 | 12 | 1.6590 | 1.5247 | stalled |
| r128_n1_mri4/sample_002 | P-Flow | 14 | 12 | 2.2490 | 1.7455 | stalled |
| r128_n1_mri4/sample_003 | P-Flow | 618 | 518 | 2.6357 | 0.6084 | stalled |
| r128_n1_mri4/sample_004 | P-Flow | 192 | 170 | 2.5496 | 1.0374 | stalled |
| r128_n1_mri4/sample_005 | P-Flow | 126 | 105 | 1.6686 | 1.2807 | stalled |
| r128_n1_mri4/sample_006 | P-Flow | 351 | 309 | 2.5760 | 0.4955 | stalled |
| r128_n1_mri4/sample_007 | P-Flow | 112 | 93 | 1.3792 | 1.2391 | stalled |
| r128_n1_mri4/sample_008 | P-Flow | 561 | 512 | 3.0612 | 0.5696 | stalled |
| r128_n1_mri4/sample_009 | P-Flow | 31 | 26 | 1.5374 | 1.4280 | stalled |
| r128_n1_mri4/sample_010 | P-Flow | 107 | 93 | 2.4025 | 1.6645 | stalled |
| r128_n1_mri4/sample_011 | P-Flow | 631 | 555 | 2.4742 | 0.5684 | stalled |
| r128_n1_mri4/sample_012 | P-Flow | 78 | 66 | 1.3121 | 1.2239 | stalled |
| r128_n1_mri4/sample_013 | P-Flow | 35 | 30 | 65.8473 | 62.6397 | stalled |
| r128_n1_mri4/sample_014 | P-Flow | 641 | 568 | 3.7918 | 0.5719 | stalled |
| r128_n1_mri4/sample_015 | P-Flow | 113 | 98 | 1.5382 | 1.3906 | stalled |
| r128_n1_mri4/sample_016 | P-Flow | 114 | 94 | 1.7089 | 1.2573 | stalled |
| r128_n1_mri4/sample_017 | P-Flow | 240 | 222 | 2.4298 | 0.5046 | stalled |
| r128_n1_mri4/sample_018 | P-Flow | 434 | 377 | 2.4869 | 0.5647 | stalled |
| r128_n1_mri4/sample_019 | P-Flow | 166 | 145 | 1.4341 | 1.2247 | stalled |
| r128_n1_mri4/sample_020 | P-Flow | 67 | 60 | 1.4191 | 1.3009 | stalled |
| r128_n1_mri4/sample_021 | P-Flow | 26 | 21 | 1.6731 | 1.4561 | stalled |
| r128_n1_mri4/sample_022 | P-Flow | 88 | 76 | 2.0063 | 1.5972 | stalled |
| r128_n1_mri4/sample_023 | P-Flow | 109 | 92 | 1.2811 | 1.1650 | stalled |
| r128_n1_mri4/sample_024 | P-Flow | 72 | 59 | 1.2871 | 1.2201 | stalled |
| r128_n1_mri4/sample_025 | P-Flow | 48 | 39 | 2.2886 | 1.8777 | stalled |
| r128_n1_mri4/sample_026 | P-Flow | 74 | 63 | 2.0941 | 1.5519 | stalled |
| r128_n1_mri4/sample_027 | P-Flow | 26 | 22 | 1.4369 | 1.3149 | stalled |
| r128_n1_mri4/sample_028 | P-Flow | 371 | 325 | 2.4648 | 0.5421 | stalled |
| r128_n1_mri4/sample_029 | P-Flow | 452 | 406 | 2.4783 | 0.4732 | stalled |
| r128_n1_mri4/sample_000 | DLR D-Flow | 346 | 325 | 2.3881 | 1.8098 | stalled |
| r128_n1_mri4/sample_001 | DLR D-Flow | 374 | 357 | 1.3512 | 1.2629 | stalled |
| r128_n1_mri4/sample_002 | DLR D-Flow | 132 | 123 | 51.2280 | 42.0609 | stalled |
| r128_n1_mri4/sample_003 | DLR D-Flow | 901 | 870 | 100.5989 | 100.2177 | stalled |
| r128_n1_mri4/sample_004 | DLR D-Flow | 464 | 446 | 2.1622 | 1.7714 | stalled |
| r128_n1_mri4/sample_005 | DLR D-Flow | 17 | 14 | 114.3850 | 109.7362 | stalled |
| r128_n1_mri4/sample_006 | DLR D-Flow | 169 | 166 | 6.8246 | 5.6370 | stalled |
| r128_n1_mri4/sample_007 | DLR D-Flow | 352 | 333 | 1.3587 | 1.2617 | stalled |
| r128_n1_mri4/sample_008 | DLR D-Flow | 14 | 12 | 104.4684 | 104.4038 | stalled |
| r128_n1_mri4/sample_009 | DLR D-Flow | 384 | 373 | 2.8710 | 2.4351 | stalled |
| r128_n1_mri4/sample_010 | DLR D-Flow | 76 | 72 | 106.2631 | 104.8470 | stalled |
| r128_n1_mri4/sample_011 | DLR D-Flow | 299 | 291 | 1.4832 | 1.3698 | stalled |
| r128_n1_mri4/sample_012 | DLR D-Flow | 442 | 431 | 39.0113 | 33.6378 | stalled |
| r128_n1_mri4/sample_013 | DLR D-Flow | 15 | 13 | 89.6690 | 89.7667 | stalled |
| r128_n1_mri4/sample_014 | DLR D-Flow | 336 | 324 | 4.6281 | 4.0834 | stalled |
| r128_n1_mri4/sample_015 | DLR D-Flow | 969 | 902 | 1.9616 | 1.6969 | stalled |
| r128_n1_mri4/sample_016 | DLR D-Flow | 195 | 188 | 1.7582 | 1.4084 | stalled |
| r128_n1_mri4/sample_017 | DLR D-Flow | 145 | 142 | 4.7272 | 3.7021 | stalled |
| r128_n1_mri4/sample_018 | DLR D-Flow | 554 | 529 | 1.1794 | 1.2070 | stalled |
| r128_n1_mri4/sample_019 | DLR D-Flow | 313 | 299 | 1.4971 | 1.3998 | stalled |
| r128_n1_mri4/sample_020 | DLR D-Flow | 552 | 524 | 2.0418 | 1.7498 | stalled |
| r128_n1_mri4/sample_021 | DLR D-Flow | 541 | 506 | 2.1326 | 1.8857 | stalled |
| r128_n1_mri4/sample_022 | DLR D-Flow | 101 | 93 | 2.1265 | 1.7740 | stalled |
| r128_n1_mri4/sample_023 | DLR D-Flow | 188 | 180 | 2.2731 | 2.0242 | stalled |
| r128_n1_mri4/sample_024 | DLR D-Flow | 172 | 167 | 1.3776 | 1.2913 | stalled |
| r128_n1_mri4/sample_025 | DLR D-Flow | 195 | 183 | 45.5992 | 41.1267 | stalled |
| r128_n1_mri4/sample_026 | DLR D-Flow | 143 | 137 | 2.1207 | 1.5981 | stalled |
| r128_n1_mri4/sample_027 | DLR D-Flow | 603 | 580 | 1.6122 | 1.3812 | stalled |
| r128_n1_mri4/sample_028 | DLR D-Flow | 374 | 351 | 102.9420 | 102.6573 | stalled |
| r128_n1_mri4/sample_029 | DLR D-Flow | 418 | 403 | 1.7481 | 1.6253 | stalled |
| r128_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 5.5478 | 4.0773 | step_limit |
| r128_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.0852 | 0.7893 | step_limit |
| r128_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.0370 | 0.9652 | step_limit |
| r128_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.1530 | 1.1601 | step_limit |
| r128_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.6886 | 0.9514 | step_limit |
| r128_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.1164 | 0.8476 | step_limit |
| r128_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.9985 | 0.9439 | step_limit |
| r128_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.1193 | 1.0819 | step_limit |
| r128_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.0480 | 0.9039 | step_limit |
| r128_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.2024 | 0.9462 | step_limit |
| r128_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.0305 | 0.8537 | step_limit |
| r128_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.0449 | 1.1082 | step_limit |
| r128_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 0.9989 | 0.8239 | step_limit |
| r128_n2_mri4/sample_000 | P-Flow | 69 | 58 | 1.6559 | 1.3363 | stalled |
| r128_n2_mri4/sample_001 | P-Flow | 417 | 371 | 2.0892 | 0.4502 | stalled |
| r128_n2_mri4/sample_002 | P-Flow | 753 | 656 | 2.1039 | 0.5622 | stalled |
| r128_n2_mri4/sample_003 | P-Flow | 387 | 347 | 2.1656 | 0.4882 | stalled |
| r128_n2_mri4/sample_004 | P-Flow | 473 | 412 | 3.4928 | 0.6229 | stalled |
| r128_n2_mri4/sample_005 | P-Flow | 178 | 167 | 2.1787 | 0.5992 | stalled |
| r128_n2_mri4/sample_006 | P-Flow | 833 | 706 | 2.1006 | 0.4867 | stalled |
| r128_n2_mri4/sample_007 | P-Flow | 343 | 307 | 2.3142 | 0.4915 | stalled |
| r128_n2_mri4/sample_008 | P-Flow | 665 | 572 | 2.0912 | 0.4903 | stalled |
| r128_n2_mri4/sample_009 | P-Flow | 152 | 128 | 1.4772 | 1.2446 | stalled |
| r128_n2_mri4/sample_010 | P-Flow | 715 | 628 | 2.1459 | 0.4926 | stalled |
| r128_n2_mri4/sample_011 | P-Flow | 86 | 73 | 1.1120 | 1.1420 | stalled |
| r128_n2_mri4/sample_012 | P-Flow | 9 | 7 | 96.2519 | 62.9852 | stalled |
| r128_n2_mri4/sample_000 | DLR D-Flow | 398 | 384 | 79.4172 | 78.4316 | stalled |
| r128_n2_mri4/sample_001 | DLR D-Flow | 861 | 825 | 2.3214 | 2.0327 | stalled |
| r128_n2_mri4/sample_002 | DLR D-Flow | 63 | 58 | 89.3991 | 88.0809 | stalled |
| r128_n2_mri4/sample_003 | DLR D-Flow | 43 | 40 | 119.1472 | 118.7195 | stalled |
| r128_n2_mri4/sample_004 | DLR D-Flow | 291 | 280 | 74.7572 | 64.2902 | stalled |
| r128_n2_mri4/sample_005 | DLR D-Flow | 234 | 227 | 69.2401 | 65.2842 | stalled |
| r128_n2_mri4/sample_006 | DLR D-Flow | 115 | 109 | 1.9437 | 1.6542 | stalled |
| r128_n2_mri4/sample_007 | DLR D-Flow | 158 | 154 | 68.1221 | 66.6326 | stalled |
| r128_n2_mri4/sample_008 | DLR D-Flow | 905 | 867 | 1.4773 | 1.3218 | stalled |
| r128_n2_mri4/sample_009 | DLR D-Flow | 134 | 125 | 123.5382 | 118.8918 | stalled |
| r128_n2_mri4/sample_010 | DLR D-Flow | 564 | 541 | 2.3446 | 1.8052 | stalled |
| r128_n2_mri4/sample_011 | DLR D-Flow | 171 | 162 | 86.3548 | 84.9907 | stalled |
| r128_n2_mri4/sample_012 | DLR D-Flow | 189 | 183 | 2.6525 | 2.3445 | stalled |
| r128_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 42.7833 | 40.3576 | step_limit |
| r128_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.3371 | 0.8081 | step_limit |
| r128_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.5694 | 1.2940 | step_limit |
| r128_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 37.6674 | 37.3944 | step_limit |
| r128_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.1984 | 0.8595 | step_limit |
| r128_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.3317 | 1.1632 | step_limit |
| r128_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.1942 | 0.6560 | step_limit |
| r128_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 6.8895 | 6.2067 | step_limit |
| r128_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.1216 | 0.7686 | step_limit |
| r128_n4_mri4/sample_009 | D-Flow | 437 | 433 | 35.7295 | 34.1641 | stalled |
| r128_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.3123 | 1.1185 | step_limit |
| r128_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.2115 | 0.9447 | step_limit |
| r128_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.2383 | 0.9589 | step_limit |
| r128_n4_mri4/sample_000 | P-Flow | 358 | 328 | 2.2221 | 0.4122 | stalled |
| r128_n4_mri4/sample_001 | P-Flow | 409 | 368 | 2.4317 | 0.5197 | stalled |
| r128_n4_mri4/sample_002 | P-Flow | 408 | 372 | 2.2951 | 0.5168 | stalled |
| r128_n4_mri4/sample_003 | P-Flow | 319 | 300 | 2.3141 | 0.5729 | stalled |
| r128_n4_mri4/sample_004 | P-Flow | 462 | 410 | 2.2569 | 0.4860 | stalled |
| r128_n4_mri4/sample_005 | P-Flow | 426 | 385 | 2.1827 | 0.5689 | stalled |
| r128_n4_mri4/sample_006 | P-Flow | 291 | 270 | 2.2294 | 0.4164 | stalled |
| r128_n4_mri4/sample_007 | P-Flow | 300 | 280 | 2.2793 | 0.5518 | stalled |
| r128_n4_mri4/sample_008 | P-Flow | 430 | 389 | 2.0842 | 0.5535 | stalled |
| r128_n4_mri4/sample_009 | P-Flow | 291 | 264 | 2.1879 | 0.5157 | stalled |
| r128_n4_mri4/sample_010 | P-Flow | 90 | 73 | 1.5839 | 1.3299 | stalled |
| r128_n4_mri4/sample_011 | P-Flow | 475 | 427 | 2.1596 | 0.4547 | stalled |
| r128_n4_mri4/sample_012 | P-Flow | 355 | 321 | 2.2103 | 0.5465 | stalled |
| r128_n4_mri4/sample_000 | DLR D-Flow | 154 | 147 | 71.7589 | 65.6360 | stalled |
| r128_n4_mri4/sample_001 | DLR D-Flow | 598 | 579 | 30.5176 | 26.5812 | stalled |
| r128_n4_mri4/sample_002 | DLR D-Flow | 139 | 131 | 105.1867 | 103.7033 | stalled |
| r128_n4_mri4/sample_003 | DLR D-Flow | 17 | 14 | 106.1066 | 104.7218 | stalled |
| r128_n4_mri4/sample_004 | DLR D-Flow | 1000 | 968 | 59.5366 | 55.2351 | step_limit |
| r128_n4_mri4/sample_005 | DLR D-Flow | 830 | 799 | 84.0310 | 82.2160 | stalled |
| r128_n4_mri4/sample_006 | DLR D-Flow | 358 | 342 | 37.4722 | 34.7153 | stalled |
| r128_n4_mri4/sample_007 | DLR D-Flow | 724 | 691 | 47.5560 | 45.1993 | stalled |
| r128_n4_mri4/sample_008 | DLR D-Flow | 227 | 213 | 39.6234 | 38.0164 | stalled |
| r128_n4_mri4/sample_009 | DLR D-Flow | 1000 | 971 | 35.2697 | 29.8755 | step_limit |
| r128_n4_mri4/sample_010 | DLR D-Flow | 150 | 144 | 79.8126 | 76.0348 | stalled |
| r128_n4_mri4/sample_011 | DLR D-Flow | 946 | 909 | 6.8886 | 5.3031 | stalled |
| r128_n4_mri4/sample_012 | DLR D-Flow | 1000 | 966 | 64.8487 | 63.2495 | step_limit |
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
| r160_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 2.7068 | 0.7464 | step_limit |
| r160_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 5.9612 | 0.2119 | step_limit |
| r160_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.3974 | 0.7891 | step_limit |
| r160_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 3.7069 | 0.6705 | step_limit |
| r160_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.4032 | 1.0158 | step_limit |
| r160_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 3.0905 | 0.5585 | step_limit |
| r160_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.5180 | 0.9708 | step_limit |
| r160_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 2.9240 | 0.7400 | step_limit |
| r160_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 2.0923 | 0.8851 | step_limit |
| r160_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 2.9849 | 0.6499 | step_limit |
| r160_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 5.9092 | 0.3197 | step_limit |
| r160_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 3.0902 | 0.8178 | step_limit |
| r160_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 2.4144 | 0.9368 | step_limit |
| r160_n2_ct10/sample_000 | P-Flow | 851 | 807 | 6.1231 | 0.1280 | stalled |
| r160_n2_ct10/sample_001 | P-Flow | 21 | 17 | 43.3818 | 18.4855 | stalled |
| r160_n2_ct10/sample_002 | P-Flow | 212 | 177 | 1.3965 | 1.0156 | stalled |
| r160_n2_ct10/sample_003 | P-Flow | 1000 | 793 | 1.5116 | 1.0209 | step_limit |
| r160_n2_ct10/sample_004 | P-Flow | 69 | 61 | 1.3397 | 1.0253 | stalled |
| r160_n2_ct10/sample_005 | P-Flow | 1000 | 902 | 6.0147 | 0.1498 | step_limit |
| r160_n2_ct10/sample_006 | P-Flow | 1000 | 790 | 2.3462 | 1.0893 | step_limit |
| r160_n2_ct10/sample_007 | P-Flow | 91 | 83 | 1.2938 | 1.0155 | stalled |
| r160_n2_ct10/sample_008 | P-Flow | 4 | 2 | 98.5152 | 65.0309 | stalled |
| r160_n2_ct10/sample_009 | P-Flow | 385 | 342 | 5.9731 | 0.0915 | stalled |
| r160_n2_ct10/sample_010 | P-Flow | 1000 | 851 | 6.0831 | 0.0796 | step_limit |
| r160_n2_ct10/sample_011 | P-Flow | 1000 | 929 | 5.8007 | 0.1004 | step_limit |
| r160_n2_ct10/sample_012 | P-Flow | 66 | 57 | 6.6128 | 2.8781 | stalled |
| r160_n2_ct10/sample_000 | DLR D-Flow | 285 | 276 | 8.3120 | 1.6436 | stalled |
| r160_n2_ct10/sample_001 | DLR D-Flow | 590 | 567 | 2.5445 | 1.0783 | stalled |
| r160_n2_ct10/sample_002 | DLR D-Flow | 180 | 171 | 60.9861 | 41.1330 | stalled |
| r160_n2_ct10/sample_003 | DLR D-Flow | 34 | 31 | 76.4692 | 42.4553 | stalled |
| r160_n2_ct10/sample_004 | DLR D-Flow | 64 | 60 | 80.1150 | 50.5245 | stalled |
| r160_n2_ct10/sample_005 | DLR D-Flow | 1000 | 976 | 6.2682 | 1.5912 | step_limit |
| r160_n2_ct10/sample_006 | DLR D-Flow | 1000 | 978 | 11.6119 | 3.0166 | step_limit |
| r160_n2_ct10/sample_007 | DLR D-Flow | 791 | 762 | 69.1614 | 36.1912 | stalled |
| r160_n2_ct10/sample_008 | DLR D-Flow | 47 | 44 | 92.6919 | 58.7715 | stalled |
| r160_n2_ct10/sample_009 | DLR D-Flow | 238 | 228 | 1.8428 | 1.1156 | stalled |
| r160_n2_ct10/sample_010 | DLR D-Flow | 426 | 405 | 1.9236 | 1.0357 | stalled |
| r160_n2_ct10/sample_011 | DLR D-Flow | 592 | 571 | 47.6206 | 28.8321 | stalled |
| r160_n2_ct10/sample_012 | DLR D-Flow | 50 | 48 | 127.6041 | 108.5202 | stalled |
| r160_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 3.1795 | 0.7001 | step_limit |
| r160_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 5.1200 | 0.5135 | step_limit |
| r160_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.5566 | 0.8348 | step_limit |
| r160_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 4.6768 | 0.6612 | step_limit |
| r160_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 6.3495 | 0.1326 | step_limit |
| r160_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 5.6387 | 0.1884 | step_limit |
| r160_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 4.9450 | 0.4121 | step_limit |
| r160_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.5754 | 0.9750 | step_limit |
| r160_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 3.1362 | 0.8229 | step_limit |
| r160_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 2.2012 | 0.9060 | step_limit |
| r160_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 2.8692 | 0.8621 | step_limit |
| r160_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 2.0887 | 0.8657 | step_limit |
| r160_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 3.2233 | 0.7184 | step_limit |
| r160_n4_ct10/sample_000 | P-Flow | 184 | 151 | 1.8353 | 1.0261 | stalled |
| r160_n4_ct10/sample_001 | P-Flow | 1000 | 945 | 5.4018 | 0.1793 | step_limit |
| r160_n4_ct10/sample_002 | P-Flow | 1000 | 879 | 5.4165 | 0.1273 | step_limit |
| r160_n4_ct10/sample_003 | P-Flow | 1000 | 919 | 5.8323 | 0.1012 | step_limit |
| r160_n4_ct10/sample_004 | P-Flow | 1000 | 956 | 5.6670 | 0.1231 | step_limit |
| r160_n4_ct10/sample_005 | P-Flow | 1000 | 968 | 5.5054 | 0.1368 | step_limit |
| r160_n4_ct10/sample_006 | P-Flow | 1000 | 939 | 5.3142 | 0.1388 | step_limit |
| r160_n4_ct10/sample_007 | P-Flow | 1000 | 920 | 5.9072 | 0.1271 | step_limit |
| r160_n4_ct10/sample_008 | P-Flow | 1000 | 806 | 1.3596 | 1.0005 | step_limit |
| r160_n4_ct10/sample_009 | P-Flow | 1000 | 934 | 5.7006 | 0.1537 | step_limit |
| r160_n4_ct10/sample_010 | P-Flow | 3 | 1 | 89.0217 | 47.4439 | stalled |
| r160_n4_ct10/sample_011 | P-Flow | 236 | 199 | 1.6620 | 0.9755 | stalled |
| r160_n4_ct10/sample_012 | P-Flow | 1000 | 935 | 5.2217 | 0.1613 | step_limit |
| r160_n4_ct10/sample_000 | DLR D-Flow | 1000 | 967 | 13.7507 | 3.5029 | step_limit |
| r160_n4_ct10/sample_001 | DLR D-Flow | 1000 | 954 | 33.9720 | 14.6905 | step_limit |
| r160_n4_ct10/sample_002 | DLR D-Flow | 706 | 675 | 49.5638 | 22.4912 | stalled |
| r160_n4_ct10/sample_003 | DLR D-Flow | 397 | 379 | 9.0033 | 2.3308 | stalled |
| r160_n4_ct10/sample_004 | DLR D-Flow | 703 | 673 | 8.9858 | 2.2444 | stalled |
| r160_n4_ct10/sample_005 | DLR D-Flow | 330 | 313 | 41.5214 | 15.1185 | stalled |
| r160_n4_ct10/sample_006 | DLR D-Flow | 1000 | 962 | 58.3756 | 28.8676 | step_limit |
| r160_n4_ct10/sample_007 | DLR D-Flow | 1000 | 968 | 66.7852 | 30.8929 | step_limit |
| r160_n4_ct10/sample_008 | DLR D-Flow | 475 | 458 | 42.4890 | 17.6685 | stalled |
| r160_n4_ct10/sample_009 | DLR D-Flow | 1000 | 965 | 41.2949 | 19.8875 | step_limit |
| r160_n4_ct10/sample_010 | DLR D-Flow | 1000 | 960 | 12.0407 | 3.3214 | step_limit |
| r160_n4_ct10/sample_011 | DLR D-Flow | 300 | 283 | 67.8180 | 28.9278 | stalled |
| r160_n4_ct10/sample_012 | DLR D-Flow | 547 | 533 | 25.9313 | 9.4684 | stalled |
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
| r160_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.0565 | 0.7213 | step_limit |
| r160_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.9749 | 0.8116 | step_limit |
| r160_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.2195 | 1.2393 | step_limit |
| r160_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.9806 | 1.0514 | step_limit |
| r160_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.9890 | 0.8693 | step_limit |
| r160_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.9648 | 1.0374 | step_limit |
| r160_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.9508 | 0.8459 | step_limit |
| r160_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.0174 | 0.7720 | step_limit |
| r160_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 67.1218 | 66.8677 | step_limit |
| r160_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.0224 | 0.7664 | step_limit |
| r160_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.0151 | 0.6551 | step_limit |
| r160_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.9496 | 0.9503 | step_limit |
| r160_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.2941 | 0.9896 | step_limit |
| r160_n2_mri4/sample_000 | P-Flow | 265 | 239 | 2.1209 | 0.5296 | stalled |
| r160_n2_mri4/sample_001 | P-Flow | 314 | 279 | 1.9252 | 0.5227 | stalled |
| r160_n2_mri4/sample_002 | P-Flow | 663 | 594 | 2.0473 | 0.4021 | stalled |
| r160_n2_mri4/sample_003 | P-Flow | 357 | 315 | 2.2241 | 0.5020 | stalled |
| r160_n2_mri4/sample_004 | P-Flow | 293 | 267 | 2.1485 | 0.5152 | stalled |
| r160_n2_mri4/sample_005 | P-Flow | 481 | 443 | 2.0562 | 0.4905 | stalled |
| r160_n2_mri4/sample_006 | P-Flow | 448 | 382 | 1.9754 | 0.5436 | stalled |
| r160_n2_mri4/sample_007 | P-Flow | 524 | 468 | 2.0808 | 0.4587 | stalled |
| r160_n2_mri4/sample_008 | P-Flow | 670 | 599 | 2.3309 | 0.5110 | stalled |
| r160_n2_mri4/sample_009 | P-Flow | 473 | 416 | 2.1615 | 0.5200 | stalled |
| r160_n2_mri4/sample_010 | P-Flow | 446 | 386 | 1.9453 | 0.4985 | stalled |
| r160_n2_mri4/sample_011 | P-Flow | 929 | 788 | 2.0804 | 0.5837 | stalled |
| r160_n2_mri4/sample_012 | P-Flow | 533 | 474 | 2.4483 | 0.4746 | stalled |
| r160_n2_mri4/sample_000 | DLR D-Flow | 80 | 76 | 53.8726 | 51.1473 | stalled |
| r160_n2_mri4/sample_001 | DLR D-Flow | 777 | 751 | 2.2116 | 1.7779 | stalled |
| r160_n2_mri4/sample_002 | DLR D-Flow | 388 | 377 | 54.6511 | 53.8573 | stalled |
| r160_n2_mri4/sample_003 | DLR D-Flow | 1000 | 960 | 2.4511 | 1.7878 | step_limit |
| r160_n2_mri4/sample_004 | DLR D-Flow | 878 | 855 | 63.4066 | 61.4299 | stalled |
| r160_n2_mri4/sample_005 | DLR D-Flow | 806 | 785 | 1.6488 | 1.4637 | stalled |
| r160_n2_mri4/sample_006 | DLR D-Flow | 219 | 212 | 12.2653 | 10.5256 | stalled |
| r160_n2_mri4/sample_007 | DLR D-Flow | 475 | 457 | 4.5493 | 3.3456 | stalled |
| r160_n2_mri4/sample_008 | DLR D-Flow | 158 | 152 | 48.6347 | 43.9915 | stalled |
| r160_n2_mri4/sample_009 | DLR D-Flow | 521 | 494 | 2.5114 | 2.1889 | stalled |
| r160_n2_mri4/sample_010 | DLR D-Flow | 203 | 191 | 1.3965 | 1.3728 | stalled |
| r160_n2_mri4/sample_011 | DLR D-Flow | 467 | 453 | 44.9639 | 43.4234 | stalled |
| r160_n2_mri4/sample_012 | DLR D-Flow | 144 | 136 | 124.3403 | 123.0120 | stalled |
| r160_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.9547 | 0.7620 | step_limit |
| r160_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.0031 | 0.8707 | step_limit |
| r160_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.0053 | 0.8482 | step_limit |
| r160_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.0171 | 0.7595 | step_limit |
| r160_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.9841 | 0.8439 | step_limit |
| r160_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.0181 | 0.9274 | step_limit |
| r160_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.1336 | 1.1295 | step_limit |
| r160_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.0830 | 1.1640 | step_limit |
| r160_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.9966 | 1.0762 | step_limit |
| r160_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 32.0691 | 29.4834 | step_limit |
| r160_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.9816 | 0.8866 | step_limit |
| r160_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.9530 | 0.8977 | step_limit |
| r160_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.0317 | 0.8826 | step_limit |
| r160_n4_mri4/sample_000 | P-Flow | 578 | 500 | 1.8897 | 0.4435 | stalled |
| r160_n4_mri4/sample_001 | P-Flow | 342 | 307 | 1.8504 | 0.5446 | stalled |
| r160_n4_mri4/sample_002 | P-Flow | 485 | 435 | 2.0170 | 0.5122 | stalled |
| r160_n4_mri4/sample_003 | P-Flow | 380 | 344 | 1.8762 | 0.5499 | stalled |
| r160_n4_mri4/sample_004 | P-Flow | 667 | 585 | 1.9092 | 0.5354 | stalled |
| r160_n4_mri4/sample_005 | P-Flow | 701 | 603 | 2.0010 | 0.5305 | stalled |
| r160_n4_mri4/sample_006 | P-Flow | 494 | 449 | 2.1337 | 0.5529 | stalled |
| r160_n4_mri4/sample_007 | P-Flow | 776 | 671 | 1.9473 | 0.5097 | stalled |
| r160_n4_mri4/sample_008 | P-Flow | 282 | 258 | 1.8701 | 0.5362 | stalled |
| r160_n4_mri4/sample_009 | P-Flow | 643 | 565 | 1.8468 | 0.5387 | stalled |
| r160_n4_mri4/sample_010 | P-Flow | 548 | 500 | 1.9457 | 0.5877 | stalled |
| r160_n4_mri4/sample_011 | P-Flow | 609 | 548 | 1.8934 | 0.5557 | stalled |
| r160_n4_mri4/sample_012 | P-Flow | 486 | 422 | 1.9148 | 0.5061 | stalled |
| r160_n4_mri4/sample_000 | DLR D-Flow | 682 | 645 | 6.4816 | 4.7462 | stalled |
| r160_n4_mri4/sample_001 | DLR D-Flow | 927 | 892 | 65.5511 | 56.2167 | stalled |
| r160_n4_mri4/sample_002 | DLR D-Flow | 1000 | 971 | 25.5989 | 22.2697 | step_limit |
| r160_n4_mri4/sample_003 | DLR D-Flow | 217 | 205 | 74.6829 | 68.6072 | stalled |
| r160_n4_mri4/sample_004 | DLR D-Flow | 1000 | 971 | 47.7528 | 41.4289 | step_limit |
| r160_n4_mri4/sample_005 | DLR D-Flow | 966 | 931 | 84.0156 | 81.8644 | stalled |
| r160_n4_mri4/sample_006 | DLR D-Flow | 1000 | 978 | 67.0015 | 65.2814 | step_limit |
| r160_n4_mri4/sample_007 | DLR D-Flow | 1000 | 963 | 69.5199 | 66.3869 | step_limit |
| r160_n4_mri4/sample_008 | DLR D-Flow | 63 | 60 | 65.1243 | 56.6356 | stalled |
| r160_n4_mri4/sample_009 | DLR D-Flow | 1000 | 974 | 48.0298 | 44.2079 | step_limit |
| r160_n4_mri4/sample_010 | DLR D-Flow | 1000 | 964 | 48.8911 | 46.2184 | step_limit |
| r160_n4_mri4/sample_011 | DLR D-Flow | 1000 | 953 | 57.2983 | 52.9772 | step_limit |
| r160_n4_mri4/sample_012 | DLR D-Flow | 559 | 535 | 7.5666 | 6.3396 | stalled |
| r200_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 3.8661 | 0.2826 | step_limit |
| r200_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 4.4873 | 0.1632 | step_limit |
| r200_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 4.9258 | 0.1028 | step_limit |
| r200_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 4.1527 | 0.5729 | step_limit |
| r200_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 4.3443 | 0.1546 | step_limit |
| r200_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 4.5873 | 0.1198 | step_limit |
| r200_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 3.8605 | 0.4579 | step_limit |
| r200_n1_ct10/sample_007 | D-Flow | 68 | 65 | 109.6591 | 95.5148 | stalled |
| r200_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 4.2553 | 0.5617 | step_limit |
| r200_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 4.9649 | 0.2355 | step_limit |
| r200_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 6.4638 | 0.0906 | step_limit |
| r200_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 6.3529 | 0.1055 | step_limit |
| r200_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 4.5665 | 0.1097 | step_limit |
| r200_n1_ct10/sample_013 | D-Flow | 1000 | 1000 | 7.1029 | 0.0646 | step_limit |
| r200_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 2.5389 | 0.8164 | step_limit |
| r200_n1_ct10/sample_015 | D-Flow | 1000 | 1000 | 7.2885 | 0.0375 | step_limit |
| r200_n1_ct10/sample_016 | D-Flow | 1000 | 1000 | 6.1483 | 0.1234 | step_limit |
| r200_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 6.0220 | 0.1002 | step_limit |
| r200_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 6.6611 | 0.1292 | step_limit |
| r200_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 4.2471 | 0.3554 | step_limit |
| r200_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 4.0922 | 0.2416 | step_limit |
| r200_n1_ct10/sample_021 | D-Flow | 83 | 79 | 106.8347 | 96.9980 | stalled |
| r200_n1_ct10/sample_022 | D-Flow | 1000 | 1000 | 4.2413 | 0.5035 | step_limit |
| r200_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 4.4809 | 0.1036 | step_limit |
| r200_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 4.4617 | 0.1299 | step_limit |
| r200_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 6.6215 | 0.1766 | step_limit |
| r200_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 8.2434 | 0.0633 | step_limit |
| r200_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 4.4915 | 0.1298 | step_limit |
| r200_n1_ct10/sample_028 | D-Flow | 1000 | 1000 | 5.3332 | 0.4371 | step_limit |
| r200_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 6.7935 | 0.0254 | step_limit |
| r200_n1_ct10/sample_000 | P-Flow | 85 | 73 | 1.3047 | 1.0082 | stalled |
| r200_n1_ct10/sample_001 | P-Flow | 16 | 13 | 4.6880 | 2.2446 | stalled |
| r200_n1_ct10/sample_002 | P-Flow | 54 | 47 | 1.2349 | 1.0107 | stalled |
| r200_n1_ct10/sample_003 | P-Flow | 148 | 125 | 1.5746 | 1.0238 | stalled |
| r200_n1_ct10/sample_004 | P-Flow | 1000 | 965 | 2.0583 | 1.0468 | step_limit |
| r200_n1_ct10/sample_005 | P-Flow | 42 | 36 | 2.5966 | 1.1839 | stalled |
| r200_n1_ct10/sample_006 | P-Flow | 37 | 31 | 12.9457 | 6.3849 | stalled |
| r200_n1_ct10/sample_007 | P-Flow | 17 | 14 | 110.5413 | 94.9409 | stalled |
| r200_n1_ct10/sample_008 | P-Flow | 74 | 62 | 1.2863 | 1.0202 | stalled |
| r200_n1_ct10/sample_009 | P-Flow | 1000 | 946 | 5.5628 | 0.1970 | step_limit |
| r200_n1_ct10/sample_010 | P-Flow | 23 | 20 | 7.0292 | 2.3664 | stalled |
| r200_n1_ct10/sample_011 | P-Flow | 250 | 212 | 1.2886 | 1.0071 | stalled |
| r200_n1_ct10/sample_012 | P-Flow | 202 | 161 | 7.2691 | 2.2660 | stalled |
| r200_n1_ct10/sample_013 | P-Flow | 138 | 118 | 1.1951 | 1.0140 | stalled |
| r200_n1_ct10/sample_014 | P-Flow | 38 | 31 | 8.1552 | 4.4384 | stalled |
| r200_n1_ct10/sample_015 | P-Flow | 45 | 38 | 1.3472 | 1.0227 | stalled |
| r200_n1_ct10/sample_016 | P-Flow | 7 | 5 | 104.7074 | 97.1832 | stalled |
| r200_n1_ct10/sample_017 | P-Flow | 116 | 96 | 1.4824 | 1.0170 | stalled |
| r200_n1_ct10/sample_018 | P-Flow | 512 | 418 | 1.3644 | 1.0140 | stalled |
| r200_n1_ct10/sample_019 | P-Flow | 25 | 21 | 21.0511 | 5.9463 | stalled |
| r200_n1_ct10/sample_020 | P-Flow | 15 | 12 | 9.1133 | 2.7555 | stalled |
| r200_n1_ct10/sample_021 | P-Flow | 11 | 9 | 24.0324 | 8.4412 | stalled |
| r200_n1_ct10/sample_022 | P-Flow | 46 | 39 | 1.5833 | 1.0378 | stalled |
| r200_n1_ct10/sample_023 | P-Flow | 212 | 173 | 9.9325 | 3.6709 | stalled |
| r200_n1_ct10/sample_024 | P-Flow | 5 | 3 | 108.7593 | 94.0510 | stalled |
| r200_n1_ct10/sample_025 | P-Flow | 248 | 207 | 1.4496 | 1.0256 | stalled |
| r200_n1_ct10/sample_026 | P-Flow | 257 | 205 | 1.3375 | 1.0156 | stalled |
| r200_n1_ct10/sample_027 | P-Flow | 278 | 229 | 4.6197 | 2.1232 | stalled |
| r200_n1_ct10/sample_028 | P-Flow | 1000 | 951 | 6.0411 | 0.1187 | step_limit |
| r200_n1_ct10/sample_029 | P-Flow | 61 | 52 | 8.7864 | 4.1644 | stalled |
| r200_n1_ct10/sample_000 | DLR D-Flow | 120 | 113 | 1.3148 | 1.0064 | stalled |
| r200_n1_ct10/sample_001 | DLR D-Flow | 388 | 374 | 2.2439 | 1.0364 | stalled |
| r200_n1_ct10/sample_002 | DLR D-Flow | 241 | 230 | 1.1419 | 1.0056 | stalled |
| r200_n1_ct10/sample_003 | DLR D-Flow | 283 | 270 | 1.2603 | 1.0105 | stalled |
| r200_n1_ct10/sample_004 | DLR D-Flow | 506 | 485 | 2.2326 | 1.0508 | stalled |
| r200_n1_ct10/sample_005 | DLR D-Flow | 269 | 256 | 1.7188 | 1.0445 | stalled |
| r200_n1_ct10/sample_006 | DLR D-Flow | 1000 | 956 | 2.2437 | 1.0436 | step_limit |
| r200_n1_ct10/sample_007 | DLR D-Flow | 294 | 278 | 109.2083 | 95.8617 | stalled |
| r200_n1_ct10/sample_008 | DLR D-Flow | 592 | 569 | 1.3203 | 1.0108 | stalled |
| r200_n1_ct10/sample_009 | DLR D-Flow | 23 | 20 | 56.2302 | 28.1150 | stalled |
| r200_n1_ct10/sample_010 | DLR D-Flow | 133 | 128 | 1.4916 | 1.0360 | stalled |
| r200_n1_ct10/sample_011 | DLR D-Flow | 133 | 125 | 1.2943 | 1.0063 | stalled |
| r200_n1_ct10/sample_012 | DLR D-Flow | 642 | 619 | 1.6794 | 1.0341 | stalled |
| r200_n1_ct10/sample_013 | DLR D-Flow | 330 | 316 | 1.1543 | 1.0140 | stalled |
| r200_n1_ct10/sample_014 | DLR D-Flow | 17 | 15 | 112.0192 | 95.8072 | stalled |
| r200_n1_ct10/sample_015 | DLR D-Flow | 215 | 207 | 1.3592 | 1.0222 | stalled |
| r200_n1_ct10/sample_016 | DLR D-Flow | 190 | 183 | 2.3760 | 1.0691 | stalled |
| r200_n1_ct10/sample_017 | DLR D-Flow | 251 | 240 | 1.4561 | 1.0226 | stalled |
| r200_n1_ct10/sample_018 | DLR D-Flow | 240 | 225 | 1.2691 | 1.0155 | stalled |
| r200_n1_ct10/sample_019 | DLR D-Flow | 763 | 737 | 1.4962 | 1.0129 | stalled |
| r200_n1_ct10/sample_020 | DLR D-Flow | 751 | 722 | 1.3011 | 1.0091 | stalled |
| r200_n1_ct10/sample_021 | DLR D-Flow | 16 | 14 | 105.9017 | 97.7250 | stalled |
| r200_n1_ct10/sample_022 | DLR D-Flow | 539 | 518 | 1.5500 | 1.0335 | stalled |
| r200_n1_ct10/sample_023 | DLR D-Flow | 1000 | 966 | 6.4996 | 3.2022 | step_limit |
| r200_n1_ct10/sample_024 | DLR D-Flow | 423 | 413 | 1.4852 | 1.0208 | stalled |
| r200_n1_ct10/sample_025 | DLR D-Flow | 331 | 316 | 1.4738 | 1.0107 | stalled |
| r200_n1_ct10/sample_026 | DLR D-Flow | 410 | 389 | 1.4178 | 1.0157 | stalled |
| r200_n1_ct10/sample_027 | DLR D-Flow | 463 | 445 | 1.4113 | 1.0223 | stalled |
| r200_n1_ct10/sample_028 | DLR D-Flow | 431 | 419 | 1.4507 | 1.0218 | stalled |
| r200_n1_ct10/sample_029 | DLR D-Flow | 440 | 428 | 1.7544 | 1.0512 | stalled |
| r200_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 2.4398 | 0.9183 | step_limit |
| r200_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 8.8450 | 2.7257 | step_limit |
| r200_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 4.3135 | 0.6722 | step_limit |
| r200_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.6707 | 0.9819 | step_limit |
| r200_n2_ct10/sample_004 | D-Flow | 331 | 327 | 64.6306 | 43.7126 | stalled |
| r200_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 2.5934 | 0.9134 | step_limit |
| r200_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 4.6666 | 0.3817 | step_limit |
| r200_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 5.3431 | 0.3012 | step_limit |
| r200_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 3.0825 | 0.7313 | step_limit |
| r200_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 6.2484 | 0.2123 | step_limit |
| r200_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 2.8187 | 0.6272 | step_limit |
| r200_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.1914 | 0.9999 | step_limit |
| r200_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 3.2531 | 0.7521 | step_limit |
| r200_n2_ct10/sample_000 | P-Flow | 57 | 51 | 10.9484 | 3.3132 | stalled |
| r200_n2_ct10/sample_001 | P-Flow | 579 | 459 | 1.1596 | 1.0101 | stalled |
| r200_n2_ct10/sample_002 | P-Flow | 43 | 37 | 32.9311 | 12.1569 | stalled |
| r200_n2_ct10/sample_003 | P-Flow | 552 | 435 | 3.0906 | 1.2401 | stalled |
| r200_n2_ct10/sample_004 | P-Flow | 575 | 465 | 2.4812 | 1.1283 | stalled |
| r200_n2_ct10/sample_005 | P-Flow | 1000 | 935 | 5.4729 | 0.1305 | step_limit |
| r200_n2_ct10/sample_006 | P-Flow | 147 | 120 | 3.3706 | 1.3124 | stalled |
| r200_n2_ct10/sample_007 | P-Flow | 1000 | 866 | 5.3777 | 0.0869 | step_limit |
| r200_n2_ct10/sample_008 | P-Flow | 200 | 170 | 2.0524 | 1.1234 | stalled |
| r200_n2_ct10/sample_009 | P-Flow | 1000 | 776 | 1.2106 | 1.0092 | step_limit |
| r200_n2_ct10/sample_010 | P-Flow | 508 | 419 | 2.5808 | 1.2623 | stalled |
| r200_n2_ct10/sample_011 | P-Flow | 125 | 107 | 1.3519 | 1.0093 | stalled |
| r200_n2_ct10/sample_012 | P-Flow | 587 | 476 | 2.0416 | 1.1647 | stalled |
| r200_n2_ct10/sample_000 | DLR D-Flow | 534 | 520 | 7.9910 | 2.1134 | stalled |
| r200_n2_ct10/sample_001 | DLR D-Flow | 522 | 500 | 1.8384 | 1.0580 | stalled |
| r200_n2_ct10/sample_002 | DLR D-Flow | 1000 | 980 | 1.7976 | 1.0584 | step_limit |
| r200_n2_ct10/sample_003 | DLR D-Flow | 209 | 197 | 89.9261 | 71.4735 | stalled |
| r200_n2_ct10/sample_004 | DLR D-Flow | 178 | 168 | 79.7072 | 53.4760 | stalled |
| r200_n2_ct10/sample_005 | DLR D-Flow | 182 | 178 | 7.5749 | 2.5791 | stalled |
| r200_n2_ct10/sample_006 | DLR D-Flow | 895 | 864 | 80.4219 | 57.4896 | stalled |
| r200_n2_ct10/sample_007 | DLR D-Flow | 158 | 151 | 2.5712 | 1.1637 | stalled |
| r200_n2_ct10/sample_008 | DLR D-Flow | 711 | 677 | 3.7107 | 1.3848 | stalled |
| r200_n2_ct10/sample_009 | DLR D-Flow | 59 | 55 | 73.0084 | 44.8635 | stalled |
| r200_n2_ct10/sample_010 | DLR D-Flow | 1000 | 971 | 3.9587 | 2.5046 | step_limit |
| r200_n2_ct10/sample_011 | DLR D-Flow | 121 | 118 | 10.5465 | 2.9223 | stalled |
| r200_n2_ct10/sample_012 | DLR D-Flow | 611 | 597 | 17.2060 | 4.6975 | stalled |
| r200_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 3.4983 | 0.8205 | step_limit |
| r200_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 3.8168 | 0.7725 | step_limit |
| r200_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 5.4857 | 0.6783 | step_limit |
| r200_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.7052 | 0.9463 | step_limit |
| r200_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 5.7265 | 0.4535 | step_limit |
| r200_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 2.7850 | 0.8932 | step_limit |
| r200_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.5963 | 0.9507 | step_limit |
| r200_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 7.1874 | 0.0816 | step_limit |
| r200_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 5.1892 | 0.6411 | step_limit |
| r200_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 2.8088 | 0.9023 | step_limit |
| r200_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 6.1317 | 0.2578 | step_limit |
| r200_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.5500 | 0.9954 | step_limit |
| r200_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 2.1291 | 0.9165 | step_limit |
| r200_n4_ct10/sample_000 | P-Flow | 1000 | 960 | 5.4527 | 0.1273 | step_limit |
| r200_n4_ct10/sample_001 | P-Flow | 1000 | 859 | 5.5747 | 0.1243 | step_limit |
| r200_n4_ct10/sample_002 | P-Flow | 1000 | 931 | 5.9464 | 0.0889 | step_limit |
| r200_n4_ct10/sample_003 | P-Flow | 1000 | 959 | 5.5890 | 0.1204 | step_limit |
| r200_n4_ct10/sample_004 | P-Flow | 1000 | 918 | 5.6916 | 0.1322 | step_limit |
| r200_n4_ct10/sample_005 | P-Flow | 1000 | 945 | 5.8398 | 0.1084 | step_limit |
| r200_n4_ct10/sample_006 | P-Flow | 697 | 613 | 5.8954 | 0.1323 | stalled |
| r200_n4_ct10/sample_007 | P-Flow | 1000 | 932 | 5.7604 | 0.1157 | step_limit |
| r200_n4_ct10/sample_008 | P-Flow | 1000 | 936 | 5.9251 | 0.0947 | step_limit |
| r200_n4_ct10/sample_009 | P-Flow | 1000 | 902 | 5.5475 | 0.1361 | step_limit |
| r200_n4_ct10/sample_010 | P-Flow | 681 | 614 | 5.7284 | 0.0982 | stalled |
| r200_n4_ct10/sample_011 | P-Flow | 98 | 87 | 1.6207 | 1.0115 | stalled |
| r200_n4_ct10/sample_012 | P-Flow | 1000 | 901 | 5.7554 | 0.1013 | step_limit |
| r200_n4_ct10/sample_000 | DLR D-Flow | 539 | 516 | 51.9794 | 22.6036 | stalled |
| r200_n4_ct10/sample_001 | DLR D-Flow | 1000 | 966 | 80.5286 | 44.6330 | step_limit |
| r200_n4_ct10/sample_002 | DLR D-Flow | 1000 | 959 | 52.5112 | 23.8219 | step_limit |
| r200_n4_ct10/sample_003 | DLR D-Flow | 138 | 131 | 75.6247 | 39.7095 | stalled |
| r200_n4_ct10/sample_004 | DLR D-Flow | 734 | 702 | 43.0803 | 16.9028 | stalled |
| r200_n4_ct10/sample_005 | DLR D-Flow | 1000 | 961 | 33.5197 | 11.3711 | step_limit |
| r200_n4_ct10/sample_006 | DLR D-Flow | 575 | 552 | 22.7194 | 6.2432 | stalled |
| r200_n4_ct10/sample_007 | DLR D-Flow | 1000 | 961 | 19.5429 | 5.1365 | step_limit |
| r200_n4_ct10/sample_008 | DLR D-Flow | 403 | 387 | 64.3193 | 28.7326 | stalled |
| r200_n4_ct10/sample_009 | DLR D-Flow | 741 | 712 | 55.0941 | 26.4755 | stalled |
| r200_n4_ct10/sample_010 | DLR D-Flow | 166 | 157 | 82.9737 | 45.4621 | stalled |
| r200_n4_ct10/sample_011 | DLR D-Flow | 1000 | 959 | 74.4049 | 36.7267 | step_limit |
| r200_n4_ct10/sample_012 | DLR D-Flow | 1000 | 960 | 68.4274 | 37.3939 | step_limit |
| r200_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.9906 | 0.8235 | step_limit |
| r200_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.2705 | 0.6571 | step_limit |
| r200_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.9681 | 0.7216 | step_limit |
| r200_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.0576 | 0.6858 | step_limit |
| r200_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.3264 | 0.7037 | step_limit |
| r200_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.2968 | 0.6930 | step_limit |
| r200_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.4211 | 0.7202 | step_limit |
| r200_n1_mri4/sample_007 | D-Flow | 63 | 59 | 106.1719 | 104.8370 | stalled |
| r200_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.9914 | 0.6502 | step_limit |
| r200_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.2691 | 0.8905 | step_limit |
| r200_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.9505 | 0.8390 | step_limit |
| r200_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.0171 | 0.7304 | step_limit |
| r200_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.1256 | 0.7824 | step_limit |
| r200_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 0.9339 | 0.7942 | step_limit |
| r200_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 1.4657 | 1.0834 | step_limit |
| r200_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 0.9733 | 0.7066 | step_limit |
| r200_n1_mri4/sample_016 | D-Flow | 63 | 59 | 100.3624 | 99.7413 | stalled |
| r200_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 1.0507 | 0.7711 | step_limit |
| r200_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 1.0456 | 0.6881 | step_limit |
| r200_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 0.9418 | 0.7488 | step_limit |
| r200_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 1.0083 | 0.7866 | step_limit |
| r200_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 1.1989 | 0.9517 | step_limit |
| r200_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 1.1641 | 0.7738 | step_limit |
| r200_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 1.3680 | 0.7390 | step_limit |
| r200_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 1.0071 | 0.7659 | step_limit |
| r200_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 1.0363 | 0.7065 | step_limit |
| r200_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 1.0070 | 0.7077 | step_limit |
| r200_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 1.1878 | 0.6871 | step_limit |
| r200_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 1.0825 | 0.6235 | step_limit |
| r200_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 1.0575 | 0.8770 | step_limit |
| r200_n1_mri4/sample_000 | P-Flow | 458 | 400 | 2.1275 | 0.5377 | stalled |
| r200_n1_mri4/sample_001 | P-Flow | 318 | 271 | 1.9193 | 0.9689 | stalled |
| r200_n1_mri4/sample_002 | P-Flow | 960 | 812 | 2.0060 | 0.5105 | stalled |
| r200_n1_mri4/sample_003 | P-Flow | 65 | 55 | 1.1256 | 1.1610 | stalled |
| r200_n1_mri4/sample_004 | P-Flow | 87 | 75 | 1.8035 | 1.4985 | stalled |
| r200_n1_mri4/sample_005 | P-Flow | 97 | 83 | 1.5601 | 1.3738 | stalled |
| r200_n1_mri4/sample_006 | P-Flow | 72 | 62 | 1.7483 | 1.3926 | stalled |
| r200_n1_mri4/sample_007 | P-Flow | 567 | 510 | 2.9392 | 0.5282 | stalled |
| r200_n1_mri4/sample_008 | P-Flow | 53 | 43 | 1.1256 | 1.2269 | stalled |
| r200_n1_mri4/sample_009 | P-Flow | 104 | 89 | 1.6401 | 1.3741 | stalled |
| r200_n1_mri4/sample_010 | P-Flow | 114 | 97 | 1.0841 | 1.1602 | stalled |
| r200_n1_mri4/sample_011 | P-Flow | 19 | 16 | 1.7476 | 1.6383 | stalled |
| r200_n1_mri4/sample_012 | P-Flow | 76 | 63 | 1.3130 | 1.2683 | stalled |
| r200_n1_mri4/sample_013 | P-Flow | 106 | 86 | 1.0847 | 1.1777 | stalled |
| r200_n1_mri4/sample_014 | P-Flow | 80 | 70 | 1.8066 | 1.3705 | stalled |
| r200_n1_mri4/sample_015 | P-Flow | 83 | 72 | 1.0488 | 1.1962 | stalled |
| r200_n1_mri4/sample_016 | P-Flow | 77 | 66 | 1.0454 | 1.1392 | stalled |
| r200_n1_mri4/sample_017 | P-Flow | 50 | 41 | 1.2475 | 1.1962 | stalled |
| r200_n1_mri4/sample_018 | P-Flow | 426 | 372 | 2.1373 | 0.4884 | stalled |
| r200_n1_mri4/sample_019 | P-Flow | 106 | 94 | 1.1846 | 1.3277 | stalled |
| r200_n1_mri4/sample_020 | P-Flow | 23 | 20 | 4.0654 | 3.2874 | stalled |
| r200_n1_mri4/sample_021 | P-Flow | 468 | 417 | 2.4156 | 0.4592 | stalled |
| r200_n1_mri4/sample_022 | P-Flow | 469 | 415 | 2.3205 | 0.5167 | stalled |
| r200_n1_mri4/sample_023 | P-Flow | 30 | 25 | 5.5765 | 4.8514 | stalled |
| r200_n1_mri4/sample_024 | P-Flow | 75 | 63 | 1.1503 | 1.2148 | stalled |
| r200_n1_mri4/sample_025 | P-Flow | 97 | 82 | 1.1040 | 1.1540 | stalled |
| r200_n1_mri4/sample_026 | P-Flow | 77 | 63 | 1.1000 | 1.1972 | stalled |
| r200_n1_mri4/sample_027 | P-Flow | 74 | 62 | 1.3193 | 1.2305 | stalled |
| r200_n1_mri4/sample_028 | P-Flow | 607 | 539 | 2.1452 | 0.5140 | stalled |
| r200_n1_mri4/sample_029 | P-Flow | 137 | 113 | 1.3286 | 1.2278 | stalled |
| r200_n1_mri4/sample_000 | DLR D-Flow | 423 | 402 | 1.1588 | 1.1794 | stalled |
| r200_n1_mri4/sample_001 | DLR D-Flow | 1000 | 961 | 1.5913 | 1.3369 | step_limit |
| r200_n1_mri4/sample_002 | DLR D-Flow | 296 | 279 | 1.1697 | 1.2510 | stalled |
| r200_n1_mri4/sample_003 | DLR D-Flow | 401 | 382 | 1.2120 | 1.1996 | stalled |
| r200_n1_mri4/sample_004 | DLR D-Flow | 432 | 411 | 42.8255 | 40.5868 | stalled |
| r200_n1_mri4/sample_005 | DLR D-Flow | 946 | 904 | 1.6600 | 1.4514 | stalled |
| r200_n1_mri4/sample_006 | DLR D-Flow | 310 | 298 | 7.6650 | 6.8000 | stalled |
| r200_n1_mri4/sample_007 | DLR D-Flow | 15 | 12 | 104.7125 | 103.0100 | stalled |
| r200_n1_mri4/sample_008 | DLR D-Flow | 886 | 866 | 1.3620 | 1.3423 | stalled |
| r200_n1_mri4/sample_009 | DLR D-Flow | 329 | 320 | 77.4581 | 76.7036 | stalled |
| r200_n1_mri4/sample_010 | DLR D-Flow | 494 | 470 | 1.0719 | 1.1627 | stalled |
| r200_n1_mri4/sample_011 | DLR D-Flow | 415 | 403 | 1.3891 | 1.2549 | stalled |
| r200_n1_mri4/sample_012 | DLR D-Flow | 27 | 22 | 100.6693 | 99.6459 | stalled |
| r200_n1_mri4/sample_013 | DLR D-Flow | 149 | 144 | 80.6402 | 81.2533 | stalled |
| r200_n1_mri4/sample_014 | DLR D-Flow | 206 | 195 | 87.0251 | 84.0096 | stalled |
| r200_n1_mri4/sample_015 | DLR D-Flow | 259 | 250 | 1.5762 | 1.4540 | stalled |
| r200_n1_mri4/sample_016 | DLR D-Flow | 442 | 426 | 1.2430 | 1.2168 | stalled |
| r200_n1_mri4/sample_017 | DLR D-Flow | 92 | 83 | 100.8079 | 100.1114 | stalled |
| r200_n1_mri4/sample_018 | DLR D-Flow | 602 | 578 | 1.1758 | 1.1735 | stalled |
| r200_n1_mri4/sample_019 | DLR D-Flow | 1000 | 976 | 20.1675 | 20.2107 | step_limit |
| r200_n1_mri4/sample_020 | DLR D-Flow | 720 | 687 | 1.3734 | 1.2866 | stalled |
| r200_n1_mri4/sample_021 | DLR D-Flow | 161 | 154 | 101.1208 | 99.9944 | stalled |
| r200_n1_mri4/sample_022 | DLR D-Flow | 58 | 55 | 100.9481 | 99.8971 | stalled |
| r200_n1_mri4/sample_023 | DLR D-Flow | 235 | 220 | 6.7090 | 6.1408 | stalled |
| r200_n1_mri4/sample_024 | DLR D-Flow | 405 | 390 | 1.2573 | 1.2661 | stalled |
| r200_n1_mri4/sample_025 | DLR D-Flow | 256 | 244 | 1.1428 | 1.1792 | stalled |
| r200_n1_mri4/sample_026 | DLR D-Flow | 888 | 809 | 1.2693 | 1.3131 | stalled |
| r200_n1_mri4/sample_027 | DLR D-Flow | 1000 | 964 | 1.4407 | 1.2710 | step_limit |
| r200_n1_mri4/sample_028 | DLR D-Flow | 311 | 300 | 42.9303 | 42.5282 | stalled |
| r200_n1_mri4/sample_029 | DLR D-Flow | 121 | 113 | 60.8016 | 61.4906 | stalled |
| r200_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.9494 | 0.9371 | step_limit |
| r200_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 50.9049 | 48.5644 | step_limit |
| r200_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.9746 | 0.8284 | step_limit |
| r200_n2_mri4/sample_003 | D-Flow | 369 | 366 | 58.9905 | 58.5435 | stalled |
| r200_n2_mri4/sample_004 | D-Flow | 718 | 716 | 60.7511 | 59.0954 | stalled |
| r200_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.9246 | 0.9444 | step_limit |
| r200_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.2260 | 0.8281 | step_limit |
| r200_n2_mri4/sample_007 | D-Flow | 396 | 392 | 57.2407 | 56.6756 | stalled |
| r200_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.0690 | 1.1387 | step_limit |
| r200_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.9562 | 0.8601 | step_limit |
| r200_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.4043 | 1.2180 | step_limit |
| r200_n2_mri4/sample_011 | D-Flow | 289 | 289 | 5.5572 | 5.5938 | paused |
| r200_n2_mri4/sample_000 | P-Flow | 409 | 370 | 2.2345 | 0.5243 | stalled |
| r200_n2_mri4/sample_001 | P-Flow | 371 | 331 | 2.0098 | 0.5462 | stalled |
| r200_n2_mri4/sample_002 | P-Flow | 95 | 75 | 1.0523 | 1.1463 | stalled |
| r200_n2_mri4/sample_003 | P-Flow | 149 | 128 | 1.4242 | 1.1281 | stalled |
| r200_n2_mri4/sample_004 | P-Flow | 788 | 715 | 2.4580 | 0.5456 | stalled |
| r200_n2_mri4/sample_005 | P-Flow | 412 | 379 | 2.3289 | 0.5059 | stalled |
| r200_n2_mri4/sample_006 | P-Flow | 794 | 713 | 2.8132 | 0.4765 | stalled |
| r200_n2_mri4/sample_007 | P-Flow | 599 | 538 | 2.2199 | 0.4856 | stalled |
| r200_n2_mri4/sample_008 | P-Flow | 823 | 715 | 2.1183 | 0.5614 | stalled |
| r200_n2_mri4/sample_009 | P-Flow | 383 | 337 | 2.1570 | 0.4966 | stalled |
| r200_n2_mri4/sample_010 | P-Flow | 586 | 508 | 2.6407 | 0.5033 | stalled |
| r200_n2_mri4/sample_000 | DLR D-Flow | 1000 | 974 | 2.1819 | 1.9348 | step_limit |
| r200_n2_mri4/sample_001 | DLR D-Flow | 881 | 859 | 51.4006 | 50.1041 | stalled |
| r200_n2_mri4/sample_002 | DLR D-Flow | 201 | 196 | 7.2252 | 5.1385 | stalled |
| r200_n2_mri4/sample_003 | DLR D-Flow | 573 | 558 | 66.1004 | 63.9764 | stalled |
| r200_n2_mri4/sample_004 | DLR D-Flow | 1000 | 968 | 2.1284 | 2.0993 | step_limit |
| r200_n2_mri4/sample_005 | DLR D-Flow | 1000 | 957 | 2.0929 | 1.9159 | step_limit |
| r200_n2_mri4/sample_006 | DLR D-Flow | 1000 | 969 | 67.4626 | 66.2987 | step_limit |
| r200_n2_mri4/sample_007 | DLR D-Flow | 259 | 253 | 84.7101 | 84.4351 | stalled |
| r200_n2_mri4/sample_008 | DLR D-Flow | 354 | 344 | 1.6847 | 1.4363 | stalled |
| r200_n2_mri4/sample_009 | DLR D-Flow | 144 | 134 | 2.8540 | 2.7009 | stalled |
| r200_n2_mri4/sample_010 | DLR D-Flow | 493 | 476 | 2.5650 | 2.0366 | stalled |
| r200_n2_mri4/sample_011 | DLR D-Flow | 424 | 407 | 2.4853 | 1.9965 | stalled |
| r200_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.0147 | 1.0214 | step_limit |
| r200_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.2314 | 1.1996 | step_limit |
| r200_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.9746 | 1.0233 | step_limit |
| r200_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.0721 | 0.9765 | step_limit |
| r200_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.9219 | 0.8877 | step_limit |
| r200_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.0040 | 1.0972 | step_limit |
| r200_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.9691 | 0.9691 | step_limit |
| r200_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.9904 | 0.8387 | step_limit |
| r200_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.9703 | 0.8046 | step_limit |
| r200_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.0845 | 1.0664 | step_limit |
| r200_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.9742 | 1.0130 | step_limit |
| r200_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.0859 | 0.9823 | step_limit |
| r200_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.2477 | 1.2140 | step_limit |
| r200_n4_mri4/sample_013 | D-Flow | 1000 | 1000 | 1.0064 | 0.8934 | step_limit |
| r200_n4_mri4/sample_000 | P-Flow | 596 | 537 | 2.0960 | 0.4962 | stalled |
| r200_n4_mri4/sample_001 | P-Flow | 1000 | 874 | 2.1753 | 0.4866 | step_limit |
| r200_n4_mri4/sample_002 | P-Flow | 514 | 453 | 2.0877 | 0.5156 | stalled |
| r200_n4_mri4/sample_003 | P-Flow | 524 | 458 | 2.1364 | 0.5927 | stalled |
| r200_n4_mri4/sample_004 | P-Flow | 303 | 274 | 2.0738 | 0.4474 | stalled |
| r200_n4_mri4/sample_005 | P-Flow | 697 | 612 | 2.0985 | 0.4767 | stalled |
| r200_n4_mri4/sample_006 | P-Flow | 632 | 545 | 2.0665 | 0.6285 | stalled |
| r200_n4_mri4/sample_007 | P-Flow | 791 | 701 | 2.1008 | 0.5056 | stalled |
| r200_n4_mri4/sample_008 | P-Flow | 500 | 442 | 1.9669 | 0.4625 | stalled |
| r200_n4_mri4/sample_009 | P-Flow | 865 | 758 | 2.1343 | 0.5889 | stalled |
| r200_n4_mri4/sample_010 | P-Flow | 546 | 492 | 2.1230 | 0.5020 | stalled |
| r200_n4_mri4/sample_011 | P-Flow | 415 | 372 | 2.2560 | 0.5435 | stalled |
| r200_n4_mri4/sample_012 | P-Flow | 647 | 573 | 2.1670 | 0.5298 | stalled |
| r200_n4_mri4/sample_013 | P-Flow | 483 | 425 | 2.1217 | 0.4624 | stalled |
| r200_n4_mri4/sample_000 | DLR D-Flow | 769 | 742 | 18.9424 | 14.0756 | stalled |
| r200_n4_mri4/sample_001 | DLR D-Flow | 961 | 918 | 6.4652 | 4.8018 | stalled |
| r200_n4_mri4/sample_002 | DLR D-Flow | 327 | 314 | 65.3552 | 62.4627 | stalled |
| r200_n4_mri4/sample_003 | DLR D-Flow | 1000 | 972 | 56.3180 | 55.4453 | step_limit |
| r200_n4_mri4/sample_004 | DLR D-Flow | 1000 | 968 | 35.9758 | 30.4079 | step_limit |
| r200_n4_mri4/sample_005 | DLR D-Flow | 418 | 402 | 29.9697 | 26.0374 | stalled |
| r200_n4_mri4/sample_006 | DLR D-Flow | 1000 | 968 | 43.2668 | 41.1499 | step_limit |
| r200_n4_mri4/sample_007 | DLR D-Flow | 1000 | 962 | 4.6227 | 3.1824 | step_limit |
| r200_n4_mri4/sample_008 | DLR D-Flow | 507 | 485 | 34.1091 | 31.3049 | stalled |
| r200_n4_mri4/sample_009 | DLR D-Flow | 1000 | 967 | 48.9646 | 47.3318 | step_limit |
| r200_n4_mri4/sample_010 | DLR D-Flow | 846 | 821 | 62.2826 | 59.1543 | stalled |
| r200_n4_mri4/sample_011 | DLR D-Flow | 557 | 537 | 82.9617 | 80.6137 | stalled |
| r200_n4_mri4/sample_012 | DLR D-Flow | 658 | 637 | 66.1914 | 65.2864 | stalled |
| r200_n4_mri4/sample_013 | DLR D-Flow | 673 | 646 | 58.3135 | 52.4181 | stalled |
| r200_n4_mri4/sample_014 | DLR D-Flow | 535 | 514 | 65.5989 | 64.5983 | paused |
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
| r250_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 8.2945 | 0.0311 | step_limit |
| r250_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 6.3099 | 0.0919 | step_limit |
| r250_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 7.1610 | 0.0586 | step_limit |
| r250_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 4.3910 | 0.4859 | step_limit |
| r250_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 7.2627 | 0.0739 | step_limit |
| r250_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 6.8627 | 0.1084 | step_limit |
| r250_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.3843 | 1.0052 | step_limit |
| r250_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 3.2509 | 0.6738 | step_limit |
| r250_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 7.1919 | 0.0876 | step_limit |
| r250_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 6.9590 | 0.1234 | step_limit |
| r250_n2_ct10/sample_000 | P-Flow | 71 | 61 | 1.4226 | 1.0203 | stalled |
| r250_n2_ct10/sample_001 | P-Flow | 480 | 396 | 1.3578 | 0.9887 | stalled |
| r250_n2_ct10/sample_002 | P-Flow | 774 | 693 | 5.9381 | 0.0968 | stalled |
| r250_n2_ct10/sample_003 | P-Flow | 1000 | 955 | 5.4884 | 0.1752 | step_limit |
| r250_n2_ct10/sample_004 | P-Flow | 1000 | 793 | 1.7535 | 1.0451 | step_limit |
| r250_n2_ct10/sample_005 | P-Flow | 1000 | 859 | 5.8717 | 0.1228 | step_limit |
| r250_n2_ct10/sample_006 | P-Flow | 112 | 98 | 1.3799 | 1.0004 | stalled |
| r250_n2_ct10/sample_007 | P-Flow | 153 | 133 | 1.7202 | 1.0381 | stalled |
| r250_n2_ct10/sample_008 | P-Flow | 120 | 98 | 1.2514 | 1.0164 | stalled |
| r250_n2_ct10/sample_009 | P-Flow | 839 | 754 | 6.0766 | 0.0943 | stalled |
| r250_n2_ct10/sample_000 | DLR D-Flow | 109 | 105 | 7.5476 | 3.1086 | stalled |
| r250_n2_ct10/sample_001 | DLR D-Flow | 286 | 275 | 5.5780 | 1.4758 | stalled |
| r250_n2_ct10/sample_002 | DLR D-Flow | 390 | 378 | 2.0889 | 1.1865 | stalled |
| r250_n2_ct10/sample_003 | DLR D-Flow | 1000 | 976 | 3.8370 | 1.3316 | step_limit |
| r250_n2_ct10/sample_004 | DLR D-Flow | 477 | 460 | 2.5295 | 1.0886 | stalled |
| r250_n2_ct10/sample_005 | DLR D-Flow | 268 | 257 | 61.7264 | 37.7534 | stalled |
| r250_n2_ct10/sample_006 | DLR D-Flow | 239 | 233 | 8.4204 | 3.5467 | stalled |
| r250_n2_ct10/sample_007 | DLR D-Flow | 272 | 260 | 75.5639 | 45.6449 | stalled |
| r250_n2_ct10/sample_008 | DLR D-Flow | 728 | 709 | 2.1154 | 1.0766 | stalled |
| r250_n2_ct10/sample_009 | DLR D-Flow | 433 | 418 | 1.7174 | 1.0244 | stalled |
| r250_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 2.9529 | 0.9086 | step_limit |
| r250_n4_ct10/sample_001 | D-Flow | 538 | 536 | 2.2862 | 0.9306 | stalled |
| r250_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 51.1606 | 19.9452 | step_limit |
| r250_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 5.7777 | 0.6343 | step_limit |
| r250_n4_ct10/sample_004 | D-Flow | 587 | 584 | 46.9524 | 20.9614 | stalled |
| r250_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 6.7424 | 0.6096 | step_limit |
| r250_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 2.7828 | 0.8011 | step_limit |
| r250_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 3.9669 | 0.8442 | step_limit |
| r250_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 6.4191 | 0.4299 | step_limit |
| r250_n4_ct10/sample_009 | D-Flow | 785 | 782 | 50.9032 | 22.0947 | stalled |
| r250_n4_ct10/sample_000 | P-Flow | 232 | 200 | 1.8919 | 1.0249 | stalled |
| r250_n4_ct10/sample_001 | P-Flow | 1000 | 957 | 5.9861 | 0.1312 | step_limit |
| r250_n4_ct10/sample_002 | P-Flow | 1000 | 927 | 6.2421 | 0.1161 | step_limit |
| r250_n4_ct10/sample_003 | P-Flow | 1000 | 936 | 6.0491 | 0.1054 | step_limit |
| r250_n4_ct10/sample_004 | P-Flow | 103 | 90 | 1.4591 | 1.0185 | stalled |
| r250_n4_ct10/sample_005 | P-Flow | 1000 | 956 | 6.1580 | 0.0752 | step_limit |
| r250_n4_ct10/sample_006 | P-Flow | 106 | 91 | 1.3148 | 1.0015 | stalled |
| r250_n4_ct10/sample_007 | P-Flow | 444 | 377 | 3.2715 | 0.6554 | stalled |
| r250_n4_ct10/sample_008 | P-Flow | 1000 | 962 | 6.0979 | 0.1068 | step_limit |
| r250_n4_ct10/sample_009 | P-Flow | 1000 | 982 | 6.1760 | 0.0787 | step_limit |
| r250_n4_ct10/sample_000 | DLR D-Flow | 1000 | 965 | 45.3789 | 16.7269 | step_limit |
| r250_n4_ct10/sample_001 | DLR D-Flow | 258 | 248 | 47.9642 | 24.9079 | stalled |
| r250_n4_ct10/sample_002 | DLR D-Flow | 1000 | 957 | 52.7140 | 22.2109 | step_limit |
| r250_n4_ct10/sample_003 | DLR D-Flow | 357 | 338 | 22.5761 | 6.4380 | stalled |
| r250_n4_ct10/sample_004 | DLR D-Flow | 639 | 614 | 46.9109 | 22.3443 | stalled |
| r250_n4_ct10/sample_005 | DLR D-Flow | 1000 | 966 | 45.4704 | 20.1304 | step_limit |
| r250_n4_ct10/sample_006 | DLR D-Flow | 1000 | 952 | 27.9732 | 8.6107 | step_limit |
| r250_n4_ct10/sample_007 | DLR D-Flow | 1000 | 963 | 85.3731 | 43.6512 | step_limit |
| r250_n4_ct10/sample_008 | DLR D-Flow | 220 | 214 | 86.6444 | 43.1175 | stalled |
| r250_n4_ct10/sample_009 | DLR D-Flow | 339 | 322 | 58.9923 | 25.7475 | stalled |
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
| r250_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.8430 | 0.9585 | step_limit |
| r250_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.9237 | 0.8077 | step_limit |
| r250_n2_mri4/sample_002 | D-Flow | 354 | 352 | 44.1367 | 42.7698 | stalled |
| r250_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.9661 | 0.7839 | step_limit |
| r250_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.8830 | 0.9013 | step_limit |
| r250_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.8608 | 0.7849 | step_limit |
| r250_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.8589 | 0.9353 | step_limit |
| r250_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.2417 | 0.7970 | step_limit |
| r250_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.8448 | 0.8855 | step_limit |
| r250_n2_mri4/sample_009 | D-Flow | 72 | 70 | 1.1122 | 1.1992 | stalled |
| r250_n2_mri4/sample_000 | P-Flow | 350 | 309 | 1.9293 | 0.5592 | stalled |
| r250_n2_mri4/sample_001 | P-Flow | 340 | 296 | 2.0215 | 0.5328 | stalled |
| r250_n2_mri4/sample_002 | P-Flow | 333 | 296 | 2.1129 | 0.5554 | stalled |
| r250_n2_mri4/sample_003 | P-Flow | 282 | 257 | 2.0713 | 0.5428 | stalled |
| r250_n2_mri4/sample_004 | P-Flow | 252 | 224 | 1.9126 | 0.6003 | stalled |
| r250_n2_mri4/sample_005 | P-Flow | 334 | 290 | 1.8110 | 0.6101 | stalled |
| r250_n2_mri4/sample_006 | P-Flow | 445 | 390 | 2.1377 | 0.5529 | stalled |
| r250_n2_mri4/sample_007 | P-Flow | 235 | 212 | 2.4102 | 0.6736 | stalled |
| r250_n2_mri4/sample_008 | P-Flow | 344 | 304 | 1.9074 | 0.5028 | stalled |
| r250_n2_mri4/sample_009 | P-Flow | 383 | 338 | 2.0836 | 0.5209 | stalled |
| r250_n2_mri4/sample_000 | DLR D-Flow | 559 | 545 | 31.0417 | 25.7772 | stalled |
| r250_n2_mri4/sample_001 | DLR D-Flow | 1000 | 950 | 16.6667 | 13.9622 | step_limit |
| r250_n2_mri4/sample_002 | DLR D-Flow | 850 | 831 | 44.9273 | 43.6912 | stalled |
| r250_n2_mri4/sample_003 | DLR D-Flow | 1000 | 964 | 48.3153 | 47.9739 | step_limit |
| r250_n2_mri4/sample_004 | DLR D-Flow | 329 | 315 | 2.5957 | 2.3430 | stalled |
| r250_n2_mri4/sample_005 | DLR D-Flow | 211 | 204 | 88.8135 | 86.3612 | stalled |
| r250_n2_mri4/sample_006 | DLR D-Flow | 1000 | 980 | 2.1374 | 1.6586 | step_limit |
| r250_n2_mri4/sample_007 | DLR D-Flow | 52 | 50 | 75.5527 | 72.3304 | stalled |
| r250_n2_mri4/sample_008 | DLR D-Flow | 226 | 215 | 3.0950 | 2.7486 | stalled |
| r250_n2_mri4/sample_009 | DLR D-Flow | 1000 | 951 | 5.8883 | 4.4501 | step_limit |
| r250_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.8884 | 0.8494 | step_limit |
| r250_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.9003 | 0.8678 | step_limit |
| r250_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.8381 | 1.7083 | step_limit |
| r250_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.9158 | 0.8343 | step_limit |
| r250_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.0462 | 1.1388 | step_limit |
| r250_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.8534 | 0.9348 | step_limit |
| r250_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.8905 | 0.9283 | step_limit |
| r250_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.9186 | 1.0141 | step_limit |
| r250_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.9252 | 0.9589 | step_limit |
| r250_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.8985 | 0.9759 | step_limit |
| r250_n4_mri4/sample_000 | P-Flow | 617 | 534 | 2.2694 | 0.6241 | stalled |
| r250_n4_mri4/sample_001 | P-Flow | 347 | 305 | 2.0239 | 0.6000 | stalled |
| r250_n4_mri4/sample_002 | P-Flow | 277 | 238 | 2.0282 | 0.7315 | stalled |
| r250_n4_mri4/sample_003 | P-Flow | 376 | 331 | 2.0779 | 0.6454 | stalled |
| r250_n4_mri4/sample_004 | P-Flow | 249 | 223 | 2.1672 | 0.6779 | stalled |
| r250_n4_mri4/sample_005 | P-Flow | 316 | 278 | 1.8715 | 0.7409 | stalled |
| r250_n4_mri4/sample_006 | P-Flow | 266 | 240 | 1.9907 | 0.6812 | stalled |
| r250_n4_mri4/sample_007 | P-Flow | 378 | 332 | 2.2251 | 0.5999 | stalled |
| r250_n4_mri4/sample_008 | P-Flow | 483 | 413 | 2.0245 | 0.6476 | stalled |
| r250_n4_mri4/sample_009 | P-Flow | 269 | 243 | 2.1543 | 0.6206 | stalled |
| r250_n4_mri4/sample_000 | DLR D-Flow | 1000 | 953 | 40.9044 | 37.3470 | step_limit |
| r250_n4_mri4/sample_001 | DLR D-Flow | 163 | 156 | 51.0761 | 46.8635 | stalled |
| r250_n4_mri4/sample_002 | DLR D-Flow | 327 | 313 | 51.4349 | 49.3898 | stalled |
| r250_n4_mri4/sample_003 | DLR D-Flow | 1000 | 959 | 42.1715 | 41.4776 | step_limit |
| r250_n4_mri4/sample_004 | DLR D-Flow | 1000 | 966 | 37.8794 | 36.6024 | step_limit |
| r250_n4_mri4/sample_005 | DLR D-Flow | 1000 | 962 | 29.8167 | 28.0271 | step_limit |
| r250_n4_mri4/sample_006 | DLR D-Flow | 1000 | 961 | 54.2885 | 50.1409 | step_limit |
| r250_n4_mri4/sample_007 | DLR D-Flow | 1000 | 967 | 58.6950 | 55.3333 | step_limit |
| r250_n4_mri4/sample_008 | DLR D-Flow | 1000 | 965 | 73.7435 | 71.6730 | step_limit |
| r250_n4_mri4/sample_009 | DLR D-Flow | 215 | 204 | 71.4697 | 69.3486 | stalled |

Terminal method runs: 1954/3240. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.
