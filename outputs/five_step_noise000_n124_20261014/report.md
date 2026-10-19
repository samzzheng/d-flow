# Five-step Euler pilot

0% relative measurement noise; CT10 and Cartesian MRI4; 30 test samples per acquisition at each resolution; circle count in [1, 2, 4]. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 8, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.3534 | 0.0592 | step_limit |
| r64_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.5340 | 0.0267 | step_limit |
| r64_n1_ct10/sample_002 | D-Flow | 813 | 811 | 1.2588 | 0.0192 | stalled |
| r64_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.3411 | 0.0279 | step_limit |
| r64_n1_ct10/sample_004 | D-Flow | 525 | 521 | 100.5092 | 87.0884 | stalled |
| r64_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 1.1073 | 0.0260 | step_limit |
| r64_n1_ct10/sample_006 | D-Flow | 567 | 565 | 1.1890 | 0.0146 | stalled |
| r64_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.0614 | 0.0140 | step_limit |
| r64_n1_ct10/sample_008 | D-Flow | 683 | 680 | 1.0940 | 0.0087 | stalled |
| r64_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.3394 | 0.0499 | step_limit |
| r64_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.1628 | 0.0199 | step_limit |
| r64_n1_ct10/sample_000 | P-Flow | 571 | 465 | 22.0285 | 9.3411 | stalled |
| r64_n1_ct10/sample_001 | P-Flow | 148 | 118 | 3.2374 | 1.2375 | stalled |
| r64_n1_ct10/sample_002 | P-Flow | 923 | 748 | 2.3248 | 0.5651 | stalled |
| r64_n1_ct10/sample_003 | P-Flow | 644 | 507 | 2.4040 | 0.5814 | stalled |
| r64_n1_ct10/sample_004 | P-Flow | 1000 | 966 | 2.8798 | 0.0645 | step_limit |
| r64_n1_ct10/sample_005 | P-Flow | 31 | 25 | 4.2249 | 1.2936 | stalled |
| r64_n1_ct10/sample_006 | P-Flow | 61 | 52 | 1.6010 | 0.2918 | stalled |
| r64_n1_ct10/sample_007 | P-Flow | 732 | 598 | 1.8794 | 0.2622 | stalled |
| r64_n1_ct10/sample_008 | P-Flow | 33 | 28 | 4.1550 | 1.5417 | stalled |
| r64_n1_ct10/sample_009 | P-Flow | 1000 | 895 | 1.8755 | 0.3738 | step_limit |
| r64_n1_ct10/sample_010 | P-Flow | 68 | 58 | 6.7740 | 3.0538 | stalled |
| r64_n1_ct10/sample_000 | DLR D-Flow | 958 | 924 | 9.1153 | 4.2665 | stalled |
| r64_n1_ct10/sample_001 | DLR D-Flow | 676 | 615 | 2.5510 | 0.6067 | stalled |
| r64_n1_ct10/sample_002 | DLR D-Flow | 172 | 169 | 2.1083 | 0.4659 | stalled |
| r64_n1_ct10/sample_003 | DLR D-Flow | 510 | 481 | 1.9925 | 0.4621 | stalled |
| r64_n1_ct10/sample_004 | DLR D-Flow | 14 | 12 | 99.7855 | 90.1628 | stalled |
| r64_n1_ct10/sample_005 | DLR D-Flow | 271 | 254 | 3.3523 | 1.1399 | stalled |
| r64_n1_ct10/sample_006 | DLR D-Flow | 187 | 180 | 2.2664 | 0.4278 | stalled |
| r64_n1_ct10/sample_007 | DLR D-Flow | 496 | 475 | 1.6499 | 0.3109 | stalled |
| r64_n1_ct10/sample_008 | DLR D-Flow | 1000 | 906 | 1.9193 | 0.4197 | step_limit |
| r64_n1_ct10/sample_009 | DLR D-Flow | 832 | 786 | 1.8947 | 0.3894 | stalled |
| r64_n1_ct10/sample_010 | DLR D-Flow | 327 | 313 | 1.6554 | 0.2941 | stalled |
| r64_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.1277 | 0.0211 | step_limit |
| r64_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.1363 | 0.0120 | step_limit |
| r64_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.1164 | 0.0096 | step_limit |
| r64_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.2449 | 0.0223 | step_limit |
| r64_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 8.0428 | 2.9179 | step_limit |
| r64_n2_ct10/sample_005 | D-Flow | 779 | 777 | 1.0108 | 0.0156 | stalled |
| r64_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 2.0331 | 0.0926 | step_limit |
| r64_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.5429 | 0.1039 | step_limit |
| r64_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 1.0915 | 0.0262 | step_limit |
| r64_n2_ct10/sample_009 | D-Flow | 988 | 986 | 1.2582 | 0.0069 | stalled |
| r64_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.1258 | 0.0679 | step_limit |
| r64_n2_ct10/sample_000 | P-Flow | 119 | 102 | 1.9982 | 0.3483 | stalled |
| r64_n2_ct10/sample_001 | P-Flow | 185 | 151 | 1.7465 | 0.3313 | stalled |
| r64_n2_ct10/sample_002 | P-Flow | 169 | 144 | 2.6989 | 0.6968 | stalled |
| r64_n2_ct10/sample_003 | P-Flow | 10 | 8 | 112.3092 | 70.2857 | stalled |
| r64_n2_ct10/sample_004 | P-Flow | 224 | 183 | 46.4797 | 23.6508 | stalled |
| r64_n2_ct10/sample_005 | P-Flow | 828 | 688 | 1.5888 | 0.3370 | stalled |
| r64_n2_ct10/sample_006 | P-Flow | 80 | 70 | 3.0073 | 0.7483 | stalled |
| r64_n2_ct10/sample_007 | P-Flow | 71 | 61 | 3.4060 | 0.8680 | stalled |
| r64_n2_ct10/sample_008 | P-Flow | 126 | 109 | 1.6975 | 0.3116 | stalled |
| r64_n2_ct10/sample_009 | P-Flow | 158 | 134 | 2.2447 | 0.4826 | stalled |
| r64_n2_ct10/sample_010 | P-Flow | 109 | 96 | 1.7086 | 0.3387 | stalled |
| r64_n2_ct10/sample_000 | DLR D-Flow | 586 | 562 | 43.0984 | 26.5739 | stalled |
| r64_n2_ct10/sample_001 | DLR D-Flow | 346 | 335 | 10.6348 | 3.0711 | stalled |
| r64_n2_ct10/sample_002 | DLR D-Flow | 94 | 91 | 4.6854 | 1.5526 | stalled |
| r64_n2_ct10/sample_003 | DLR D-Flow | 302 | 293 | 4.8236 | 1.5791 | stalled |
| r64_n2_ct10/sample_004 | DLR D-Flow | 1000 | 965 | 5.4601 | 1.9339 | step_limit |
| r64_n2_ct10/sample_005 | DLR D-Flow | 867 | 835 | 56.8032 | 30.0678 | stalled |
| r64_n2_ct10/sample_006 | DLR D-Flow | 90 | 81 | 73.8874 | 57.0416 | stalled |
| r64_n2_ct10/sample_007 | DLR D-Flow | 273 | 261 | 2.4347 | 0.6190 | stalled |
| r64_n2_ct10/sample_008 | DLR D-Flow | 170 | 160 | 75.9880 | 48.2457 | stalled |
| r64_n2_ct10/sample_009 | DLR D-Flow | 489 | 462 | 1.8702 | 0.3842 | stalled |
| r64_n2_ct10/sample_010 | DLR D-Flow | 265 | 253 | 4.0983 | 1.2513 | stalled |
| r64_n4_ct10/sample_000 | D-Flow | 651 | 649 | 1.2099 | 0.0118 | stalled |
| r64_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.1174 | 0.0180 | step_limit |
| r64_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.3695 | 0.0740 | step_limit |
| r64_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.2150 | 0.0288 | step_limit |
| r64_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.4226 | 0.0171 | step_limit |
| r64_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 1.4124 | 0.0154 | step_limit |
| r64_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.3363 | 0.0543 | step_limit |
| r64_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.3731 | 0.0543 | step_limit |
| r64_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 1.2243 | 0.0292 | step_limit |
| r64_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.1773 | 0.0611 | step_limit |
| r64_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.3149 | 0.0116 | step_limit |
| r64_n4_ct10/sample_000 | P-Flow | 1000 | 824 | 2.0778 | 0.3653 | step_limit |
| r64_n4_ct10/sample_001 | P-Flow | 361 | 303 | 1.8078 | 0.3024 | stalled |
| r64_n4_ct10/sample_002 | P-Flow | 945 | 745 | 1.9898 | 0.3482 | stalled |
| r64_n4_ct10/sample_003 | P-Flow | 13 | 11 | 59.8454 | 30.2670 | stalled |
| r64_n4_ct10/sample_004 | P-Flow | 29 | 25 | 43.4915 | 19.2917 | stalled |
| r64_n4_ct10/sample_005 | P-Flow | 449 | 365 | 2.1714 | 0.4089 | stalled |
| r64_n4_ct10/sample_006 | P-Flow | 1000 | 801 | 2.1802 | 0.4271 | step_limit |
| r64_n4_ct10/sample_007 | P-Flow | 141 | 119 | 2.0948 | 0.3537 | stalled |
| r64_n4_ct10/sample_008 | P-Flow | 1000 | 824 | 1.9217 | 0.3536 | step_limit |
| r64_n4_ct10/sample_009 | P-Flow | 21 | 17 | 60.0353 | 29.0193 | stalled |
| r64_n4_ct10/sample_010 | P-Flow | 159 | 134 | 2.0343 | 0.2856 | stalled |
| r64_n4_ct10/sample_000 | DLR D-Flow | 1000 | 976 | 76.0430 | 36.0872 | step_limit |
| r64_n4_ct10/sample_001 | DLR D-Flow | 217 | 203 | 19.2272 | 7.4754 | stalled |
| r64_n4_ct10/sample_002 | DLR D-Flow | 471 | 452 | 81.9632 | 46.5786 | stalled |
| r64_n4_ct10/sample_003 | DLR D-Flow | 150 | 147 | 32.4641 | 14.0010 | stalled |
| r64_n4_ct10/sample_004 | DLR D-Flow | 644 | 613 | 80.6846 | 46.5637 | stalled |
| r64_n4_ct10/sample_005 | DLR D-Flow | 408 | 390 | 77.1508 | 45.3300 | stalled |
| r64_n4_ct10/sample_006 | DLR D-Flow | 1000 | 963 | 46.3844 | 21.2738 | step_limit |
| r64_n4_ct10/sample_007 | DLR D-Flow | 569 | 548 | 101.2406 | 59.0727 | stalled |
| r64_n4_ct10/sample_008 | DLR D-Flow | 394 | 383 | 27.7091 | 9.7310 | stalled |
| r64_n4_ct10/sample_009 | DLR D-Flow | 1000 | 977 | 40.0842 | 16.6584 | step_limit |
| r64_n4_ct10/sample_010 | DLR D-Flow | 140 | 136 | 17.5226 | 6.1940 | stalled |
| r64_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.1697 | 0.1598 | step_limit |
| r64_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.4209 | 0.0173 | step_limit |
| r64_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.0223 | 0.0055 | step_limit |
| r64_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.1045 | 0.1149 | step_limit |
| r64_n1_mri4/sample_004 | D-Flow | 239 | 236 | 100.6279 | 95.3568 | stalled |
| r64_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.8666 | 0.0935 | step_limit |
| r64_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.9940 | 0.0060 | step_limit |
| r64_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.8723 | 0.0171 | step_limit |
| r64_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.9852 | 0.0801 | step_limit |
| r64_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.2030 | 0.0810 | step_limit |
| r64_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.0594 | 0.1758 | step_limit |
| r64_n1_mri4/sample_000 | P-Flow | 155 | 133 | 1.8743 | 0.9098 | stalled |
| r64_n1_mri4/sample_001 | P-Flow | 126 | 109 | 1.9162 | 1.1198 | stalled |
| r64_n1_mri4/sample_002 | P-Flow | 143 | 127 | 1.7981 | 0.8434 | stalled |
| r64_n1_mri4/sample_003 | P-Flow | 117 | 99 | 1.9271 | 1.0528 | stalled |
| r64_n1_mri4/sample_004 | P-Flow | 180 | 162 | 2.6272 | 0.9365 | stalled |
| r64_n1_mri4/sample_005 | P-Flow | 117 | 100 | 1.4049 | 0.8361 | stalled |
| r64_n1_mri4/sample_006 | P-Flow | 130 | 111 | 1.5265 | 0.7106 | stalled |
| r64_n1_mri4/sample_007 | P-Flow | 134 | 124 | 1.7442 | 0.7017 | stalled |
| r64_n1_mri4/sample_008 | P-Flow | 53 | 45 | 1.8779 | 1.1103 | stalled |
| r64_n1_mri4/sample_009 | P-Flow | 90 | 74 | 1.7419 | 1.0327 | stalled |
| r64_n1_mri4/sample_010 | P-Flow | 99 | 88 | 1.6303 | 0.8121 | stalled |
| r64_n1_mri4/sample_000 | DLR D-Flow | 39 | 36 | 77.6770 | 76.8877 | stalled |
| r64_n1_mri4/sample_001 | DLR D-Flow | 17 | 15 | 12.6189 | 11.1877 | stalled |
| r64_n1_mri4/sample_002 | DLR D-Flow | 180 | 172 | 1.9807 | 1.3961 | stalled |
| r64_n1_mri4/sample_003 | DLR D-Flow | 1000 | 919 | 1.9338 | 1.2402 | step_limit |
| r64_n1_mri4/sample_004 | DLR D-Flow | 165 | 158 | 100.4171 | 95.7153 | stalled |
| r64_n1_mri4/sample_005 | DLR D-Flow | 136 | 129 | 2.5585 | 1.8692 | stalled |
| r64_n1_mri4/sample_006 | DLR D-Flow | 154 | 151 | 1.8574 | 1.0846 | stalled |
| r64_n1_mri4/sample_007 | DLR D-Flow | 425 | 399 | 1.5970 | 0.9279 | stalled |
| r64_n1_mri4/sample_008 | DLR D-Flow | 453 | 443 | 8.5360 | 6.4245 | stalled |
| r64_n1_mri4/sample_009 | DLR D-Flow | 98 | 91 | 4.3833 | 3.7768 | stalled |
| r64_n1_mri4/sample_010 | DLR D-Flow | 229 | 220 | 1.8741 | 1.2355 | stalled |
| r64_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.0064 | 0.0304 | step_limit |
| r64_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.8660 | 0.0845 | step_limit |
| r64_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.0431 | 0.0204 | step_limit |
| r64_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.0766 | 0.0805 | step_limit |
| r64_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.1555 | 0.3139 | step_limit |
| r64_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.8606 | 0.1188 | step_limit |
| r64_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.7745 | 0.1422 | step_limit |
| r64_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.2488 | 0.1800 | step_limit |
| r64_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.0117 | 0.2425 | step_limit |
| r64_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.0603 | 0.0397 | step_limit |
| r64_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.8820 | 0.0945 | step_limit |
| r64_n2_mri4/sample_000 | P-Flow | 58 | 49 | 1.6714 | 0.8480 | stalled |
| r64_n2_mri4/sample_001 | P-Flow | 82 | 71 | 1.4288 | 0.8583 | stalled |
| r64_n2_mri4/sample_002 | P-Flow | 76 | 66 | 1.7190 | 0.8948 | stalled |
| r64_n2_mri4/sample_003 | P-Flow | 12 | 10 | 108.8043 | 108.6579 | stalled |
| r64_n2_mri4/sample_004 | P-Flow | 110 | 92 | 1.5788 | 0.8801 | stalled |
| r64_n2_mri4/sample_005 | P-Flow | 112 | 96 | 1.4410 | 0.8475 | stalled |
| r64_n2_mri4/sample_006 | P-Flow | 48 | 41 | 2.8727 | 1.6573 | stalled |
| r64_n2_mri4/sample_007 | P-Flow | 92 | 81 | 2.0529 | 1.0325 | stalled |
| r64_n2_mri4/sample_008 | P-Flow | 137 | 116 | 1.5434 | 0.8353 | stalled |
| r64_n2_mri4/sample_009 | P-Flow | 69 | 56 | 1.9648 | 1.1061 | stalled |
| r64_n2_mri4/sample_010 | P-Flow | 110 | 93 | 1.5161 | 0.8520 | stalled |
| r64_n2_mri4/sample_000 | DLR D-Flow | 1000 | 963 | 2.0684 | 1.2267 | step_limit |
| r64_n2_mri4/sample_001 | DLR D-Flow | 671 | 639 | 2.8080 | 1.8835 | stalled |
| r64_n2_mri4/sample_002 | DLR D-Flow | 120 | 116 | 3.1864 | 2.3708 | stalled |
| r64_n2_mri4/sample_003 | DLR D-Flow | 1000 | 968 | 72.7502 | 71.8668 | step_limit |
| r64_n2_mri4/sample_004 | DLR D-Flow | 1000 | 971 | 79.4489 | 77.0004 | step_limit |
| r64_n2_mri4/sample_005 | DLR D-Flow | 1000 | 974 | 2.0032 | 1.4071 | step_limit |
| r64_n2_mri4/sample_006 | DLR D-Flow | 1000 | 970 | 68.6145 | 68.0192 | step_limit |
| r64_n2_mri4/sample_007 | DLR D-Flow | 369 | 354 | 2.6798 | 1.9048 | stalled |
| r64_n2_mri4/sample_008 | DLR D-Flow | 835 | 807 | 53.9552 | 51.6933 | stalled |
| r64_n2_mri4/sample_009 | DLR D-Flow | 190 | 185 | 2.1857 | 1.2616 | stalled |
| r64_n2_mri4/sample_010 | DLR D-Flow | 427 | 391 | 91.7309 | 89.9381 | stalled |
| r64_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.1756 | 0.0042 | step_limit |
| r64_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.9576 | 0.1078 | step_limit |
| r64_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.2516 | 0.4051 | step_limit |
| r64_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.1121 | 0.0411 | step_limit |
| r64_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.1569 | 0.0771 | step_limit |
| r64_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.2295 | 0.0882 | step_limit |
| r64_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.3365 | 0.3493 | step_limit |
| r64_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.1965 | 0.2910 | step_limit |
| r64_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.0216 | 0.0854 | step_limit |
| r64_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.0579 | 0.2891 | step_limit |
| r64_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.0137 | 0.0728 | step_limit |
| r64_n4_mri4/sample_000 | P-Flow | 97 | 85 | 1.7939 | 1.0032 | stalled |
| r64_n4_mri4/sample_001 | P-Flow | 118 | 109 | 1.9718 | 0.7960 | stalled |
| r64_n4_mri4/sample_002 | P-Flow | 102 | 87 | 1.8210 | 1.0364 | stalled |
| r64_n4_mri4/sample_003 | P-Flow | 177 | 155 | 1.7272 | 0.8815 | stalled |
| r64_n4_mri4/sample_004 | P-Flow | 802 | 699 | 2.3351 | 0.1258 | stalled |
| r64_n4_mri4/sample_005 | P-Flow | 82 | 73 | 2.0200 | 0.9741 | stalled |
| r64_n4_mri4/sample_006 | P-Flow | 92 | 82 | 1.8090 | 0.8213 | stalled |
| r64_n4_mri4/sample_007 | P-Flow | 83 | 75 | 1.7374 | 0.8799 | stalled |
| r64_n4_mri4/sample_008 | P-Flow | 164 | 145 | 1.6671 | 0.8132 | stalled |
| r64_n4_mri4/sample_009 | P-Flow | 116 | 101 | 1.7656 | 0.9982 | stalled |
| r64_n4_mri4/sample_010 | P-Flow | 685 | 599 | 2.2001 | 0.2870 | stalled |
| r64_n4_mri4/sample_000 | DLR D-Flow | 256 | 250 | 15.7825 | 11.8158 | stalled |
| r64_n4_mri4/sample_001 | DLR D-Flow | 144 | 134 | 22.4710 | 18.2251 | stalled |
| r64_n4_mri4/sample_002 | DLR D-Flow | 850 | 827 | 16.7331 | 11.5867 | stalled |
| r64_n4_mri4/sample_003 | DLR D-Flow | 146 | 139 | 33.4693 | 29.5666 | stalled |
| r64_n4_mri4/sample_004 | DLR D-Flow | 957 | 918 | 48.9429 | 45.7843 | stalled |
| r64_n4_mri4/sample_005 | DLR D-Flow | 349 | 335 | 76.3259 | 74.2153 | stalled |
| r64_n4_mri4/sample_006 | DLR D-Flow | 1000 | 976 | 64.2386 | 60.0896 | step_limit |
| r64_n4_mri4/sample_007 | DLR D-Flow | 1000 | 967 | 82.7885 | 76.4386 | step_limit |
| r64_n4_mri4/sample_008 | DLR D-Flow | 476 | 456 | 34.9195 | 29.2178 | stalled |
| r64_n4_mri4/sample_009 | DLR D-Flow | 1000 | 965 | 62.9749 | 56.4573 | step_limit |
| r64_n4_mri4/sample_010 | DLR D-Flow | 252 | 245 | 15.8936 | 13.3957 | stalled |
| r96_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.3172 | 0.0434 | step_limit |
| r96_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.5109 | 0.1261 | step_limit |
| r96_n1_ct10/sample_002 | D-Flow | 101 | 98 | 1.7517 | 0.3068 | stalled |
| r96_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.5768 | 0.3095 | step_limit |
| r96_n1_ct10/sample_004 | D-Flow | 80 | 77 | 105.3769 | 108.5289 | stalled |
| r96_n1_ct10/sample_005 | D-Flow | 461 | 457 | 1.1927 | 0.0926 | stalled |
| r96_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.2920 | 0.0221 | step_limit |
| r96_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.0328 | 0.0150 | step_limit |
| r96_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 1.4654 | 0.0743 | step_limit |
| r96_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.7216 | 0.2906 | step_limit |
| r96_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.0330 | 0.0130 | step_limit |
| r96_n1_ct10/sample_000 | P-Flow | 41 | 35 | 1.9272 | 0.4687 | stalled |
| r96_n1_ct10/sample_001 | P-Flow | 88 | 75 | 3.2610 | 1.2151 | stalled |
| r96_n1_ct10/sample_002 | P-Flow | 42 | 37 | 4.3242 | 1.7379 | stalled |
| r96_n1_ct10/sample_003 | P-Flow | 135 | 114 | 1.9774 | 0.5563 | stalled |
| r96_n1_ct10/sample_004 | P-Flow | 52 | 44 | 3.5922 | 0.9934 | stalled |
| r96_n1_ct10/sample_005 | P-Flow | 1000 | 813 | 2.2167 | 0.8964 | step_limit |
| r96_n1_ct10/sample_006 | P-Flow | 1000 | 971 | 1.7579 | 0.3184 | step_limit |
| r96_n1_ct10/sample_007 | P-Flow | 29 | 25 | 2.8717 | 0.9316 | stalled |
| r96_n1_ct10/sample_008 | P-Flow | 52 | 49 | 2.0789 | 0.3630 | stalled |
| r96_n1_ct10/sample_009 | P-Flow | 61 | 52 | 2.1524 | 0.5207 | stalled |
| r96_n1_ct10/sample_010 | P-Flow | 28 | 24 | 2.1667 | 0.8929 | stalled |
| r96_n1_ct10/sample_000 | DLR D-Flow | 381 | 363 | 2.0137 | 0.4819 | stalled |
| r96_n1_ct10/sample_001 | DLR D-Flow | 41 | 38 | 116.5528 | 97.1737 | stalled |
| r96_n1_ct10/sample_002 | DLR D-Flow | 768 | 741 | 1.7534 | 0.3446 | stalled |
| r96_n1_ct10/sample_003 | DLR D-Flow | 183 | 176 | 1.9334 | 0.8078 | stalled |
| r96_n1_ct10/sample_004 | DLR D-Flow | 147 | 142 | 4.0021 | 1.2454 | stalled |
| r96_n1_ct10/sample_005 | DLR D-Flow | 495 | 473 | 1.5342 | 0.2673 | stalled |
| r96_n1_ct10/sample_006 | DLR D-Flow | 1000 | 980 | 1.8146 | 0.3531 | step_limit |
| r96_n1_ct10/sample_007 | DLR D-Flow | 1000 | 885 | 1.5597 | 0.2746 | step_limit |
| r96_n1_ct10/sample_008 | DLR D-Flow | 97 | 92 | 114.5252 | 102.5356 | stalled |
| r96_n1_ct10/sample_009 | DLR D-Flow | 105 | 100 | 118.6406 | 104.3676 | stalled |
| r96_n1_ct10/sample_010 | DLR D-Flow | 297 | 286 | 4.1033 | 1.0195 | stalled |
| r96_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.3394 | 0.0985 | step_limit |
| r96_n2_ct10/sample_001 | D-Flow | 807 | 804 | 1.0369 | 0.0209 | stalled |
| r96_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.2498 | 0.0235 | step_limit |
| r96_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.1146 | 0.0404 | step_limit |
| r96_n2_ct10/sample_004 | D-Flow | 993 | 990 | 1.0229 | 0.0238 | stalled |
| r96_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 1.2273 | 0.0123 | step_limit |
| r96_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.5965 | 0.1219 | step_limit |
| r96_n2_ct10/sample_007 | D-Flow | 159 | 157 | 61.0477 | 36.0048 | stalled |
| r96_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 1.5632 | 0.0581 | step_limit |
| r96_n2_ct10/sample_009 | D-Flow | 352 | 350 | 80.3094 | 62.2540 | stalled |
| r96_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.0116 | 0.0492 | step_limit |
| r96_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 0.9696 | 0.0790 | step_limit |
| r96_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 1.4003 | 0.1848 | step_limit |
| r96_n2_ct10/sample_000 | P-Flow | 1000 | 864 | 2.0229 | 0.2512 | step_limit |
| r96_n2_ct10/sample_001 | P-Flow | 20 | 17 | 18.6974 | 7.7260 | stalled |
| r96_n2_ct10/sample_002 | P-Flow | 266 | 219 | 3.5567 | 1.3398 | stalled |
| r96_n2_ct10/sample_003 | P-Flow | 61 | 49 | 69.7619 | 39.1936 | stalled |
| r96_n2_ct10/sample_004 | P-Flow | 16 | 14 | 9.7500 | 3.5285 | stalled |
| r96_n2_ct10/sample_005 | P-Flow | 1000 | 982 | 1.7608 | 0.2223 | step_limit |
| r96_n2_ct10/sample_006 | P-Flow | 1000 | 796 | 5.1913 | 2.7638 | step_limit |
| r96_n2_ct10/sample_007 | P-Flow | 18 | 15 | 37.4422 | 18.1841 | stalled |
| r96_n2_ct10/sample_008 | P-Flow | 373 | 306 | 3.3761 | 1.2625 | stalled |
| r96_n2_ct10/sample_009 | P-Flow | 196 | 163 | 1.7592 | 0.3540 | stalled |
| r96_n2_ct10/sample_010 | P-Flow | 31 | 26 | 5.1246 | 1.9143 | stalled |
| r96_n2_ct10/sample_011 | P-Flow | 167 | 136 | 1.3782 | 0.2427 | stalled |
| r96_n2_ct10/sample_012 | P-Flow | 1000 | 878 | 2.0470 | 0.1047 | step_limit |
| r96_n2_ct10/sample_000 | DLR D-Flow | 1000 | 964 | 66.0907 | 45.8276 | step_limit |
| r96_n2_ct10/sample_001 | DLR D-Flow | 978 | 944 | 12.9950 | 5.4638 | stalled |
| r96_n2_ct10/sample_002 | DLR D-Flow | 1000 | 972 | 16.3876 | 6.1967 | step_limit |
| r96_n2_ct10/sample_003 | DLR D-Flow | 746 | 719 | 54.8162 | 30.5658 | stalled |
| r96_n2_ct10/sample_004 | DLR D-Flow | 208 | 199 | 5.7478 | 1.7318 | stalled |
| r96_n2_ct10/sample_005 | DLR D-Flow | 514 | 489 | 2.4975 | 0.6502 | stalled |
| r96_n2_ct10/sample_006 | DLR D-Flow | 46 | 44 | 103.1572 | 72.1281 | stalled |
| r96_n2_ct10/sample_007 | DLR D-Flow | 778 | 756 | 62.4871 | 36.3378 | stalled |
| r96_n2_ct10/sample_008 | DLR D-Flow | 1000 | 983 | 7.0278 | 3.2740 | step_limit |
| r96_n2_ct10/sample_009 | DLR D-Flow | 319 | 310 | 78.7132 | 62.4488 | stalled |
| r96_n2_ct10/sample_010 | DLR D-Flow | 603 | 575 | 2.0787 | 0.3688 | stalled |
| r96_n2_ct10/sample_011 | DLR D-Flow | 555 | 532 | 1.8122 | 0.3708 | stalled |
| r96_n2_ct10/sample_012 | DLR D-Flow | 193 | 187 | 85.3573 | 52.7445 | stalled |
| r96_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.2081 | 0.0742 | step_limit |
| r96_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.2436 | 0.0321 | step_limit |
| r96_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.5983 | 0.0688 | step_limit |
| r96_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.7833 | 0.2630 | step_limit |
| r96_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 2.8893 | 0.7365 | step_limit |
| r96_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 1.3304 | 0.0220 | step_limit |
| r96_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.2380 | 0.0850 | step_limit |
| r96_n4_ct10/sample_007 | D-Flow | 793 | 791 | 1.1413 | 0.0064 | stalled |
| r96_n4_ct10/sample_008 | D-Flow | 444 | 442 | 1.1734 | 0.0146 | stalled |
| r96_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 20.1502 | 5.3767 | step_limit |
| r96_n4_ct10/sample_010 | D-Flow | 610 | 606 | 52.1193 | 30.4150 | stalled |
| r96_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.2499 | 0.0116 | step_limit |
| r96_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 1.3123 | 0.1052 | step_limit |
| r96_n4_ct10/sample_000 | P-Flow | 167 | 146 | 1.5425 | 0.2245 | stalled |
| r96_n4_ct10/sample_001 | P-Flow | 90 | 74 | 4.7891 | 1.2789 | stalled |
| r96_n4_ct10/sample_002 | P-Flow | 158 | 135 | 2.7308 | 0.7991 | stalled |
| r96_n4_ct10/sample_003 | P-Flow | 21 | 17 | 53.7662 | 24.1471 | stalled |
| r96_n4_ct10/sample_004 | P-Flow | 134 | 115 | 1.8615 | 0.2984 | stalled |
| r96_n4_ct10/sample_005 | P-Flow | 200 | 175 | 1.8000 | 0.2805 | stalled |
| r96_n4_ct10/sample_006 | P-Flow | 291 | 250 | 1.5824 | 0.2336 | stalled |
| r96_n4_ct10/sample_007 | P-Flow | 127 | 113 | 1.7838 | 0.2456 | stalled |
| r96_n4_ct10/sample_008 | P-Flow | 90 | 78 | 1.7010 | 0.2587 | stalled |
| r96_n4_ct10/sample_009 | P-Flow | 69 | 62 | 1.8454 | 0.2698 | stalled |
| r96_n4_ct10/sample_010 | P-Flow | 278 | 227 | 1.7364 | 0.2917 | stalled |
| r96_n4_ct10/sample_011 | P-Flow | 331 | 275 | 1.7837 | 0.2442 | stalled |
| r96_n4_ct10/sample_012 | P-Flow | 199 | 165 | 1.6475 | 0.2287 | stalled |
| r96_n4_ct10/sample_000 | DLR D-Flow | 191 | 185 | 20.0183 | 6.0053 | stalled |
| r96_n4_ct10/sample_001 | DLR D-Flow | 1000 | 970 | 45.9619 | 19.8246 | step_limit |
| r96_n4_ct10/sample_002 | DLR D-Flow | 1000 | 970 | 71.4112 | 41.0575 | step_limit |
| r96_n4_ct10/sample_003 | DLR D-Flow | 1000 | 968 | 64.8268 | 36.9827 | step_limit |
| r96_n4_ct10/sample_004 | DLR D-Flow | 84 | 75 | 90.8973 | 48.3261 | stalled |
| r96_n4_ct10/sample_005 | DLR D-Flow | 1000 | 969 | 32.3969 | 13.5334 | step_limit |
| r96_n4_ct10/sample_006 | DLR D-Flow | 1000 | 967 | 80.0338 | 39.6631 | step_limit |
| r96_n4_ct10/sample_007 | DLR D-Flow | 931 | 893 | 8.4922 | 2.1806 | stalled |
| r96_n4_ct10/sample_008 | DLR D-Flow | 178 | 175 | 20.6989 | 6.7030 | stalled |
| r96_n4_ct10/sample_009 | DLR D-Flow | 330 | 314 | 12.6260 | 3.5831 | stalled |
| r96_n4_ct10/sample_010 | DLR D-Flow | 698 | 678 | 83.7901 | 43.3825 | stalled |
| r96_n4_ct10/sample_011 | DLR D-Flow | 130 | 124 | 76.2872 | 37.1451 | stalled |
| r96_n4_ct10/sample_012 | DLR D-Flow | 1000 | 971 | 60.7329 | 23.9436 | step_limit |
| r96_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.0141 | 0.1196 | step_limit |
| r96_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.2272 | 0.2958 | step_limit |
| r96_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.0598 | 0.1439 | step_limit |
| r96_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.9420 | 0.0793 | step_limit |
| r96_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.6535 | 0.2739 | step_limit |
| r96_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.0225 | 0.1132 | step_limit |
| r96_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.0317 | 0.1803 | step_limit |
| r96_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.7897 | 0.0654 | step_limit |
| r96_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.6238 | 0.7270 | step_limit |
| r96_n1_mri4/sample_009 | D-Flow | 30 | 27 | 103.5482 | 102.2010 | stalled |
| r96_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.8389 | 0.0728 | step_limit |
| r96_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.8412 | 0.0962 | step_limit |
| r96_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 0.8486 | 0.0445 | step_limit |
| r96_n1_mri4/sample_000 | P-Flow | 161 | 137 | 1.5582 | 0.8709 | stalled |
| r96_n1_mri4/sample_001 | P-Flow | 12 | 9 | 117.3818 | 115.9990 | stalled |
| r96_n1_mri4/sample_002 | P-Flow | 29 | 24 | 1.8199 | 1.0567 | stalled |
| r96_n1_mri4/sample_003 | P-Flow | 71 | 60 | 1.5046 | 0.8118 | stalled |
| r96_n1_mri4/sample_004 | P-Flow | 16 | 12 | 5.2230 | 3.3913 | stalled |
| r96_n1_mri4/sample_005 | P-Flow | 188 | 163 | 1.4645 | 0.6300 | stalled |
| r96_n1_mri4/sample_006 | P-Flow | 201 | 174 | 1.7890 | 0.7254 | stalled |
| r96_n1_mri4/sample_007 | P-Flow | 37 | 30 | 1.5668 | 1.1195 | stalled |
| r96_n1_mri4/sample_008 | P-Flow | 120 | 105 | 1.9125 | 0.8729 | stalled |
| r96_n1_mri4/sample_009 | P-Flow | 15 | 12 | 120.4514 | 111.7023 | stalled |
| r96_n1_mri4/sample_010 | P-Flow | 158 | 138 | 1.3112 | 0.5472 | stalled |
| r96_n1_mri4/sample_011 | P-Flow | 177 | 153 | 1.2994 | 0.6100 | stalled |
| r96_n1_mri4/sample_012 | P-Flow | 26 | 23 | 2.4021 | 1.6461 | stalled |
| r96_n1_mri4/sample_000 | DLR D-Flow | 117 | 107 | 1.9963 | 1.1410 | stalled |
| r96_n1_mri4/sample_001 | DLR D-Flow | 103 | 99 | 102.6047 | 102.3828 | stalled |
| r96_n1_mri4/sample_002 | DLR D-Flow | 754 | 704 | 1.6534 | 0.8726 | stalled |
| r96_n1_mri4/sample_003 | DLR D-Flow | 179 | 170 | 1.5850 | 0.8693 | stalled |
| r96_n1_mri4/sample_004 | DLR D-Flow | 1000 | 916 | 3.4864 | 2.2221 | step_limit |
| r96_n1_mri4/sample_005 | DLR D-Flow | 883 | 834 | 1.4944 | 0.8871 | stalled |
| r96_n1_mri4/sample_006 | DLR D-Flow | 901 | 871 | 1.7539 | 1.1183 | stalled |
| r96_n1_mri4/sample_007 | DLR D-Flow | 637 | 615 | 1.3641 | 0.8869 | stalled |
| r96_n1_mri4/sample_008 | DLR D-Flow | 35 | 33 | 102.9233 | 102.9117 | stalled |
| r96_n1_mri4/sample_009 | DLR D-Flow | 173 | 164 | 103.7036 | 101.9685 | stalled |
| r96_n1_mri4/sample_010 | DLR D-Flow | 50 | 46 | 105.4287 | 103.0142 | stalled |
| r96_n1_mri4/sample_011 | DLR D-Flow | 285 | 272 | 1.4196 | 0.7804 | stalled |
| r96_n1_mri4/sample_012 | DLR D-Flow | 572 | 545 | 1.5457 | 0.8771 | stalled |
| r96_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.2563 | 0.6634 | step_limit |
| r96_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.8594 | 0.2226 | step_limit |
| r96_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.9051 | 0.2402 | step_limit |
| r96_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.9264 | 0.2058 | step_limit |
| r96_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.7744 | 0.1201 | step_limit |
| r96_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.9888 | 0.0386 | step_limit |
| r96_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.4693 | 0.6550 | step_limit |
| r96_n2_mri4/sample_007 | D-Flow | 388 | 385 | 51.6114 | 50.1733 | stalled |
| r96_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.3405 | 0.2255 | step_limit |
| r96_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.3624 | 0.6402 | step_limit |
| r96_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.7892 | 0.0693 | step_limit |
| r96_n2_mri4/sample_011 | D-Flow | 750 | 747 | 42.7633 | 41.0130 | stalled |
| r96_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.2505 | 0.6897 | step_limit |
| r96_n2_mri4/sample_000 | P-Flow | 152 | 130 | 1.6019 | 1.0259 | stalled |
| r96_n2_mri4/sample_001 | P-Flow | 104 | 87 | 1.3756 | 0.7223 | stalled |
| r96_n2_mri4/sample_002 | P-Flow | 127 | 107 | 1.4536 | 0.7833 | stalled |
| r96_n2_mri4/sample_003 | P-Flow | 480 | 427 | 1.9221 | 0.0626 | stalled |
| r96_n2_mri4/sample_004 | P-Flow | 327 | 292 | 1.7394 | 0.2100 | stalled |
| r96_n2_mri4/sample_005 | P-Flow | 85 | 70 | 1.4822 | 0.8396 | stalled |
| r96_n2_mri4/sample_006 | P-Flow | 61 | 49 | 5.4269 | 4.0517 | stalled |
| r96_n2_mri4/sample_007 | P-Flow | 141 | 122 | 1.5354 | 0.7944 | stalled |
| r96_n2_mri4/sample_008 | P-Flow | 26 | 22 | 2.9238 | 2.0233 | stalled |
| r96_n2_mri4/sample_009 | P-Flow | 75 | 62 | 1.6025 | 0.8398 | stalled |
| r96_n2_mri4/sample_010 | P-Flow | 72 | 58 | 1.3121 | 0.7497 | stalled |
| r96_n2_mri4/sample_011 | P-Flow | 378 | 338 | 1.5706 | 0.1791 | stalled |
| r96_n2_mri4/sample_012 | P-Flow | 380 | 341 | 1.8404 | 0.1361 | stalled |
| r96_n2_mri4/sample_000 | DLR D-Flow | 384 | 364 | 2.6037 | 2.1901 | stalled |
| r96_n2_mri4/sample_001 | DLR D-Flow | 1000 | 971 | 2.6469 | 2.0821 | step_limit |
| r96_n2_mri4/sample_002 | DLR D-Flow | 1000 | 970 | 2.7258 | 2.1794 | step_limit |
| r96_n2_mri4/sample_003 | DLR D-Flow | 1000 | 972 | 41.7337 | 36.1576 | step_limit |
| r96_n2_mri4/sample_004 | DLR D-Flow | 675 | 642 | 1.5638 | 1.0050 | stalled |
| r96_n2_mri4/sample_005 | DLR D-Flow | 1000 | 961 | 2.4446 | 1.9055 | step_limit |
| r96_n2_mri4/sample_006 | DLR D-Flow | 650 | 629 | 77.6882 | 74.4568 | stalled |
| r96_n2_mri4/sample_007 | DLR D-Flow | 606 | 576 | 58.9765 | 54.2333 | stalled |
| r96_n2_mri4/sample_008 | DLR D-Flow | 1000 | 961 | 66.7831 | 66.0967 | step_limit |
| r96_n2_mri4/sample_009 | DLR D-Flow | 88 | 83 | 76.7741 | 74.9249 | stalled |
| r96_n2_mri4/sample_010 | DLR D-Flow | 451 | 424 | 2.0067 | 1.2877 | stalled |
| r96_n2_mri4/sample_011 | DLR D-Flow | 1000 | 968 | 1.3935 | 0.9118 | step_limit |
| r96_n2_mri4/sample_012 | DLR D-Flow | 205 | 197 | 103.0060 | 101.5986 | stalled |
| r96_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.7772 | 0.0680 | step_limit |
| r96_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.0307 | 0.2408 | step_limit |
| r96_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.1938 | 0.2544 | step_limit |
| r96_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.0705 | 0.0634 | step_limit |
| r96_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.3347 | 0.5710 | step_limit |
| r96_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.1654 | 0.4155 | step_limit |
| r96_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.9696 | 0.2825 | step_limit |
| r96_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.9322 | 0.0411 | step_limit |
| r96_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.0337 | 0.1816 | step_limit |
| r96_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.9603 | 0.0394 | step_limit |
| r96_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.4876 | 0.6971 | step_limit |
| r96_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.9907 | 0.0793 | step_limit |
| r96_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.0130 | 0.3126 | step_limit |
| r96_n4_mri4/sample_000 | P-Flow | 1000 | 878 | 1.7461 | 0.0984 | step_limit |
| r96_n4_mri4/sample_001 | P-Flow | 985 | 866 | 2.4246 | 0.0455 | stalled |
| r96_n4_mri4/sample_002 | P-Flow | 94 | 81 | 1.7008 | 0.8762 | stalled |
| r96_n4_mri4/sample_003 | P-Flow | 187 | 163 | 1.8819 | 0.8685 | stalled |
| r96_n4_mri4/sample_004 | P-Flow | 97 | 82 | 1.6786 | 0.8859 | stalled |
| r96_n4_mri4/sample_005 | P-Flow | 5 | 3 | 105.7077 | 89.8669 | stalled |
| r96_n4_mri4/sample_006 | P-Flow | 257 | 220 | 1.4763 | 0.7730 | stalled |
| r96_n4_mri4/sample_007 | P-Flow | 453 | 390 | 1.7572 | 0.3303 | stalled |
| r96_n4_mri4/sample_008 | P-Flow | 140 | 123 | 1.5362 | 0.7649 | stalled |
| r96_n4_mri4/sample_009 | P-Flow | 694 | 623 | 1.7929 | 0.0434 | stalled |
| r96_n4_mri4/sample_010 | P-Flow | 88 | 75 | 1.6526 | 0.8671 | stalled |
| r96_n4_mri4/sample_011 | P-Flow | 1000 | 843 | 2.0155 | 0.1483 | step_limit |
| r96_n4_mri4/sample_012 | P-Flow | 1000 | 886 | 1.9899 | 0.0244 | step_limit |
| r96_n4_mri4/sample_000 | DLR D-Flow | 212 | 195 | 58.2758 | 53.6750 | stalled |
| r96_n4_mri4/sample_001 | DLR D-Flow | 1000 | 955 | 40.8232 | 36.1595 | step_limit |
| r96_n4_mri4/sample_002 | DLR D-Flow | 615 | 585 | 8.4101 | 5.3348 | stalled |
| r96_n4_mri4/sample_003 | DLR D-Flow | 914 | 870 | 7.1012 | 4.8424 | stalled |
| r96_n4_mri4/sample_004 | DLR D-Flow | 270 | 262 | 64.9411 | 57.4502 | stalled |
| r96_n4_mri4/sample_005 | DLR D-Flow | 241 | 232 | 25.5828 | 20.9193 | stalled |
| r96_n4_mri4/sample_006 | DLR D-Flow | 1000 | 959 | 17.7722 | 14.3156 | step_limit |
| r96_n4_mri4/sample_007 | DLR D-Flow | 167 | 163 | 8.2833 | 5.5271 | stalled |
| r96_n4_mri4/sample_008 | DLR D-Flow | 254 | 249 | 9.5209 | 7.1610 | stalled |
| r96_n4_mri4/sample_009 | DLR D-Flow | 618 | 593 | 46.2000 | 43.5119 | stalled |
| r96_n4_mri4/sample_010 | DLR D-Flow | 294 | 286 | 57.1912 | 55.4283 | stalled |
| r96_n4_mri4/sample_011 | DLR D-Flow | 575 | 557 | 70.4507 | 63.4641 | stalled |
| r96_n4_mri4/sample_012 | DLR D-Flow | 1000 | 965 | 66.7239 | 63.9560 | step_limit |
| r128_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.8135 | 0.0216 | step_limit |
| r128_n1_ct10/sample_001 | D-Flow | 754 | 751 | 1.0108 | 0.0337 | stalled |
| r128_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.2882 | 0.0186 | step_limit |
| r128_n1_ct10/sample_003 | D-Flow | 630 | 627 | 1.0959 | 0.0214 | stalled |
| r128_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.6204 | 0.0204 | step_limit |
| r128_n1_ct10/sample_005 | D-Flow | 286 | 284 | 1.4930 | 0.0777 | stalled |
| r128_n1_ct10/sample_006 | D-Flow | 355 | 353 | 1.1261 | 0.0374 | stalled |
| r128_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.0843 | 0.0169 | step_limit |
| r128_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 1.2605 | 0.0225 | step_limit |
| r128_n1_ct10/sample_009 | D-Flow | 168 | 166 | 1.2920 | 0.1225 | stalled |
| r128_n1_ct10/sample_010 | D-Flow | 152 | 148 | 117.9047 | 112.9234 | stalled |
| r128_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.0010 | 0.0190 | step_limit |
| r128_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 1.1197 | 0.0204 | step_limit |
| r128_n1_ct10/sample_000 | P-Flow | 35 | 30 | 3.9330 | 1.1353 | stalled |
| r128_n1_ct10/sample_001 | P-Flow | 81 | 72 | 1.3263 | 0.2134 | stalled |
| r128_n1_ct10/sample_002 | P-Flow | 39 | 36 | 4.4197 | 1.4012 | stalled |
| r128_n1_ct10/sample_003 | P-Flow | 50 | 43 | 2.3100 | 0.6390 | stalled |
| r128_n1_ct10/sample_004 | P-Flow | 73 | 63 | 2.5427 | 0.5724 | stalled |
| r128_n1_ct10/sample_005 | P-Flow | 83 | 74 | 1.7795 | 0.3225 | stalled |
| r128_n1_ct10/sample_006 | P-Flow | 128 | 112 | 1.5878 | 0.2415 | stalled |
| r128_n1_ct10/sample_007 | P-Flow | 85 | 74 | 1.4616 | 0.2516 | stalled |
| r128_n1_ct10/sample_008 | P-Flow | 97 | 88 | 2.1730 | 0.4861 | stalled |
| r128_n1_ct10/sample_009 | P-Flow | 23 | 19 | 3.2084 | 1.4686 | stalled |
| r128_n1_ct10/sample_010 | P-Flow | 46 | 38 | 3.7355 | 1.0911 | stalled |
| r128_n1_ct10/sample_011 | P-Flow | 123 | 104 | 3.5050 | 1.8358 | stalled |
| r128_n1_ct10/sample_012 | P-Flow | 76 | 71 | 2.1468 | 0.5579 | stalled |
| r128_n1_ct10/sample_000 | DLR D-Flow | 1000 | 962 | 5.5908 | 1.9316 | step_limit |
| r128_n1_ct10/sample_001 | DLR D-Flow | 448 | 421 | 1.3220 | 0.2213 | stalled |
| r128_n1_ct10/sample_002 | DLR D-Flow | 1000 | 965 | 2.2887 | 0.3526 | step_limit |
| r128_n1_ct10/sample_003 | DLR D-Flow | 349 | 338 | 1.7365 | 0.2990 | stalled |
| r128_n1_ct10/sample_004 | DLR D-Flow | 202 | 188 | 2.5266 | 0.5438 | stalled |
| r128_n1_ct10/sample_005 | DLR D-Flow | 127 | 122 | 1.8306 | 0.3495 | stalled |
| r128_n1_ct10/sample_006 | DLR D-Flow | 109 | 106 | 2.0544 | 0.4346 | stalled |
| r128_n1_ct10/sample_007 | DLR D-Flow | 580 | 548 | 1.4317 | 0.2266 | stalled |
| r128_n1_ct10/sample_008 | DLR D-Flow | 31 | 29 | 109.2481 | 99.2605 | stalled |
| r128_n1_ct10/sample_009 | DLR D-Flow | 601 | 578 | 2.8209 | 1.1580 | stalled |
| r128_n1_ct10/sample_010 | DLR D-Flow | 111 | 108 | 116.8637 | 107.3290 | stalled |
| r128_n1_ct10/sample_011 | DLR D-Flow | 583 | 559 | 2.2242 | 0.3134 | stalled |
| r128_n1_ct10/sample_012 | DLR D-Flow | 502 | 467 | 1.8511 | 0.2935 | stalled |
| r128_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.7275 | 0.2840 | step_limit |
| r128_n2_ct10/sample_001 | D-Flow | 917 | 915 | 1.0807 | 0.0449 | stalled |
| r128_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 0.9912 | 0.0595 | step_limit |
| r128_n2_ct10/sample_003 | D-Flow | 827 | 824 | 0.9785 | 0.0597 | stalled |
| r128_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.9796 | 0.1951 | step_limit |
| r128_n2_ct10/sample_005 | D-Flow | 509 | 506 | 1.1003 | 0.0226 | stalled |
| r128_n2_ct10/sample_006 | D-Flow | 660 | 658 | 0.9908 | 0.0497 | stalled |
| r128_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.1048 | 0.0644 | step_limit |
| r128_n2_ct10/sample_008 | D-Flow | 701 | 699 | 0.9422 | 0.0122 | stalled |
| r128_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.2769 | 0.0968 | step_limit |
| r128_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 0.9740 | 0.0469 | step_limit |
| r128_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.0807 | 0.1492 | step_limit |
| r128_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 0.9618 | 0.0194 | step_limit |
| r128_n2_ct10/sample_000 | P-Flow | 212 | 174 | 2.8248 | 1.3404 | stalled |
| r128_n2_ct10/sample_001 | P-Flow | 86 | 77 | 1.8834 | 0.3425 | stalled |
| r128_n2_ct10/sample_002 | P-Flow | 77 | 67 | 6.8911 | 1.6997 | stalled |
| r128_n2_ct10/sample_003 | P-Flow | 267 | 223 | 1.3090 | 0.1906 | stalled |
| r128_n2_ct10/sample_004 | P-Flow | 206 | 175 | 2.8239 | 0.6149 | stalled |
| r128_n2_ct10/sample_005 | P-Flow | 1000 | 817 | 3.6299 | 1.5801 | step_limit |
| r128_n2_ct10/sample_006 | P-Flow | 51 | 44 | 2.4895 | 0.7106 | stalled |
| r128_n2_ct10/sample_007 | P-Flow | 219 | 186 | 1.6297 | 0.2588 | stalled |
| r128_n2_ct10/sample_008 | P-Flow | 160 | 138 | 1.2373 | 0.1622 | stalled |
| r128_n2_ct10/sample_009 | P-Flow | 43 | 36 | 8.4258 | 2.6582 | stalled |
| r128_n2_ct10/sample_010 | P-Flow | 152 | 132 | 1.3149 | 0.1625 | stalled |
| r128_n2_ct10/sample_011 | P-Flow | 63 | 54 | 5.2549 | 1.4952 | stalled |
| r128_n2_ct10/sample_012 | P-Flow | 102 | 90 | 1.3661 | 0.2069 | stalled |
| r128_n2_ct10/sample_000 | DLR D-Flow | 122 | 116 | 114.1562 | 88.8023 | stalled |
| r128_n2_ct10/sample_001 | DLR D-Flow | 1000 | 978 | 42.9290 | 18.2335 | step_limit |
| r128_n2_ct10/sample_002 | DLR D-Flow | 623 | 609 | 18.8694 | 5.8845 | stalled |
| r128_n2_ct10/sample_003 | DLR D-Flow | 254 | 249 | 6.8345 | 1.5843 | stalled |
| r128_n2_ct10/sample_004 | DLR D-Flow | 905 | 871 | 3.6099 | 1.1776 | stalled |
| r128_n2_ct10/sample_005 | DLR D-Flow | 94 | 91 | 6.2717 | 1.6271 | stalled |
| r128_n2_ct10/sample_006 | DLR D-Flow | 479 | 452 | 1.9600 | 0.3197 | stalled |
| r128_n2_ct10/sample_007 | DLR D-Flow | 338 | 327 | 72.6796 | 41.9909 | stalled |
| r128_n2_ct10/sample_008 | DLR D-Flow | 486 | 457 | 1.2851 | 0.2736 | stalled |
| r128_n2_ct10/sample_009 | DLR D-Flow | 1000 | 973 | 4.9931 | 1.7486 | step_limit |
| r128_n2_ct10/sample_010 | DLR D-Flow | 195 | 187 | 7.3680 | 2.9713 | stalled |
| r128_n2_ct10/sample_011 | DLR D-Flow | 132 | 127 | 85.3081 | 56.0607 | stalled |
| r128_n2_ct10/sample_012 | DLR D-Flow | 125 | 119 | 3.0321 | 1.2511 | stalled |
| r128_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 48.9699 | 24.9036 | step_limit |
| r128_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.4718 | 0.0150 | step_limit |
| r128_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.2985 | 0.0126 | step_limit |
| r128_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.3631 | 0.0297 | step_limit |
| r128_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.2473 | 0.0479 | step_limit |
| r128_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 1.2372 | 0.0899 | step_limit |
| r128_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.3856 | 0.1140 | step_limit |
| r128_n4_ct10/sample_007 | D-Flow | 741 | 739 | 49.6025 | 25.6142 | stalled |
| r128_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 1.1198 | 0.0236 | step_limit |
| r128_n4_ct10/sample_009 | D-Flow | 669 | 667 | 1.1443 | 0.0171 | stalled |
| r128_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.3153 | 0.0547 | step_limit |
| r128_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.3661 | 0.1059 | step_limit |
| r128_n4_ct10/sample_012 | D-Flow | 619 | 617 | 53.8069 | 32.5795 | stalled |
| r128_n4_ct10/sample_000 | P-Flow | 1000 | 965 | 1.8042 | 0.2058 | step_limit |
| r128_n4_ct10/sample_001 | P-Flow | 1000 | 943 | 1.7104 | 0.2393 | step_limit |
| r128_n4_ct10/sample_002 | P-Flow | 1000 | 985 | 1.6058 | 0.2203 | step_limit |
| r128_n4_ct10/sample_003 | P-Flow | 145 | 128 | 1.6710 | 0.2288 | stalled |
| r128_n4_ct10/sample_004 | P-Flow | 194 | 169 | 1.7377 | 0.2339 | stalled |
| r128_n4_ct10/sample_005 | P-Flow | 92 | 81 | 1.5213 | 0.2172 | stalled |
| r128_n4_ct10/sample_006 | P-Flow | 92 | 81 | 1.6659 | 0.2265 | stalled |
| r128_n4_ct10/sample_007 | P-Flow | 1000 | 965 | 1.8960 | 0.1991 | step_limit |
| r128_n4_ct10/sample_008 | P-Flow | 191 | 165 | 1.4183 | 0.1894 | stalled |
| r128_n4_ct10/sample_009 | P-Flow | 1000 | 976 | 1.5206 | 0.1831 | step_limit |
| r128_n4_ct10/sample_010 | P-Flow | 1000 | 818 | 1.6858 | 0.2217 | step_limit |
| r128_n4_ct10/sample_011 | P-Flow | 65 | 56 | 1.6900 | 0.2272 | stalled |
| r128_n4_ct10/sample_012 | P-Flow | 100 | 84 | 1.6086 | 0.2160 | stalled |
| r128_n4_ct10/sample_000 | DLR D-Flow | 532 | 518 | 61.7432 | 29.8306 | stalled |
| r128_n4_ct10/sample_001 | DLR D-Flow | 504 | 485 | 23.1383 | 8.1143 | stalled |
| r128_n4_ct10/sample_002 | DLR D-Flow | 39 | 36 | 95.4366 | 52.4549 | stalled |
| r128_n4_ct10/sample_003 | DLR D-Flow | 1000 | 963 | 26.2558 | 8.9190 | step_limit |
| r128_n4_ct10/sample_004 | DLR D-Flow | 149 | 140 | 31.4303 | 11.4328 | stalled |
| r128_n4_ct10/sample_005 | DLR D-Flow | 233 | 224 | 90.7581 | 46.6190 | stalled |
| r128_n4_ct10/sample_006 | DLR D-Flow | 46 | 43 | 71.5995 | 34.1358 | stalled |
| r128_n4_ct10/sample_007 | DLR D-Flow | 1000 | 971 | 52.0561 | 26.5957 | step_limit |
| r128_n4_ct10/sample_008 | DLR D-Flow | 401 | 387 | 44.3833 | 23.5426 | stalled |
| r128_n4_ct10/sample_009 | DLR D-Flow | 249 | 240 | 33.3527 | 11.2785 | stalled |
| r128_n4_ct10/sample_010 | DLR D-Flow | 1000 | 965 | 57.6460 | 28.7378 | step_limit |
| r128_n4_ct10/sample_011 | DLR D-Flow | 282 | 273 | 96.1161 | 54.3889 | stalled |
| r128_n4_ct10/sample_012 | DLR D-Flow | 1000 | 967 | 19.1910 | 5.6759 | step_limit |
| r128_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.2791 | 0.2044 | step_limit |
| r128_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.7731 | 0.1263 | step_limit |
| r128_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.8514 | 0.0470 | step_limit |
| r128_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.0377 | 0.4011 | step_limit |
| r128_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.2314 | 0.0487 | step_limit |
| r128_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.0188 | 0.0576 | step_limit |
| r128_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.8387 | 0.0879 | step_limit |
| r128_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.8084 | 0.0590 | step_limit |
| r128_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.8629 | 0.1023 | step_limit |
| r128_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.8646 | 0.1989 | step_limit |
| r128_n1_mri4/sample_010 | D-Flow | 342 | 338 | 100.5629 | 99.1718 | stalled |
| r128_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.7661 | 0.0967 | step_limit |
| r128_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 0.8122 | 0.1031 | step_limit |
| r128_n1_mri4/sample_000 | P-Flow | 19 | 15 | 3.9274 | 2.7569 | stalled |
| r128_n1_mri4/sample_001 | P-Flow | 16 | 12 | 1.6407 | 1.1420 | stalled |
| r128_n1_mri4/sample_002 | P-Flow | 16 | 14 | 2.2070 | 1.4030 | stalled |
| r128_n1_mri4/sample_003 | P-Flow | 163 | 139 | 1.3961 | 0.6349 | stalled |
| r128_n1_mri4/sample_004 | P-Flow | 228 | 194 | 2.0536 | 0.8629 | stalled |
| r128_n1_mri4/sample_005 | P-Flow | 69 | 57 | 1.7014 | 0.9461 | stalled |
| r128_n1_mri4/sample_006 | P-Flow | 262 | 221 | 1.4323 | 0.4922 | stalled |
| r128_n1_mri4/sample_007 | P-Flow | 84 | 69 | 1.3551 | 0.7274 | stalled |
| r128_n1_mri4/sample_008 | P-Flow | 164 | 141 | 1.6912 | 0.7261 | stalled |
| r128_n1_mri4/sample_009 | P-Flow | 15 | 12 | 1.6312 | 1.1312 | stalled |
| r128_n1_mri4/sample_010 | P-Flow | 68 | 56 | 2.4270 | 1.3531 | stalled |
| r128_n1_mri4/sample_011 | P-Flow | 231 | 199 | 1.3047 | 0.5398 | stalled |
| r128_n1_mri4/sample_012 | P-Flow | 72 | 61 | 1.3028 | 0.7074 | stalled |
| r128_n1_mri4/sample_000 | DLR D-Flow | 1000 | 963 | 2.4159 | 1.5869 | step_limit |
| r128_n1_mri4/sample_001 | DLR D-Flow | 635 | 603 | 1.3369 | 0.7540 | stalled |
| r128_n1_mri4/sample_002 | DLR D-Flow | 720 | 693 | 1.6221 | 0.9098 | stalled |
| r128_n1_mri4/sample_003 | DLR D-Flow | 1000 | 973 | 1.4550 | 0.8528 | step_limit |
| r128_n1_mri4/sample_004 | DLR D-Flow | 465 | 444 | 2.2413 | 1.5325 | stalled |
| r128_n1_mri4/sample_005 | DLR D-Flow | 28 | 25 | 113.4167 | 108.7916 | stalled |
| r128_n1_mri4/sample_006 | DLR D-Flow | 143 | 139 | 4.4561 | 3.8297 | stalled |
| r128_n1_mri4/sample_007 | DLR D-Flow | 490 | 471 | 1.3737 | 0.7744 | stalled |
| r128_n1_mri4/sample_008 | DLR D-Flow | 18 | 16 | 104.4691 | 104.4044 | stalled |
| r128_n1_mri4/sample_009 | DLR D-Flow | 356 | 344 | 2.5140 | 1.8649 | stalled |
| r128_n1_mri4/sample_010 | DLR D-Flow | 525 | 485 | 106.2237 | 104.8187 | stalled |
| r128_n1_mri4/sample_011 | DLR D-Flow | 209 | 202 | 1.8926 | 1.2820 | stalled |
| r128_n1_mri4/sample_012 | DLR D-Flow | 41 | 34 | 108.7394 | 104.8252 | stalled |
| r128_n2_mri4/sample_000 | D-Flow | 638 | 634 | 76.4313 | 75.9813 | stalled |
| r128_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.8624 | 0.2938 | step_limit |
| r128_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.9252 | 0.3717 | step_limit |
| r128_n2_mri4/sample_003 | D-Flow | 349 | 347 | 72.1555 | 71.7317 | stalled |
| r128_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.8378 | 0.8523 | step_limit |
| r128_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.8625 | 0.1928 | step_limit |
| r128_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.7769 | 0.2292 | step_limit |
| r128_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.8362 | 0.2121 | step_limit |
| r128_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.7382 | 0.2186 | step_limit |
| r128_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.0470 | 0.3280 | step_limit |
| r128_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.7370 | 0.1665 | step_limit |
| r128_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.0463 | 0.5543 | step_limit |
| r128_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 0.7408 | 0.1880 | step_limit |
| r128_n2_mri4/sample_000 | P-Flow | 76 | 64 | 1.6693 | 0.8933 | stalled |
| r128_n2_mri4/sample_001 | P-Flow | 61 | 54 | 1.2464 | 0.7334 | stalled |
| r128_n2_mri4/sample_002 | P-Flow | 219 | 182 | 1.1382 | 0.5391 | stalled |
| r128_n2_mri4/sample_003 | P-Flow | 77 | 67 | 1.1410 | 0.5798 | stalled |
| r128_n2_mri4/sample_004 | P-Flow | 77 | 66 | 2.1835 | 1.2107 | stalled |
| r128_n2_mri4/sample_005 | P-Flow | 138 | 127 | 1.5124 | 0.5482 | stalled |
| r128_n2_mri4/sample_006 | P-Flow | 243 | 210 | 1.1576 | 0.5382 | stalled |
| r128_n2_mri4/sample_007 | P-Flow | 420 | 362 | 1.5123 | 0.4227 | stalled |
| r128_n2_mri4/sample_008 | P-Flow | 199 | 168 | 1.1496 | 0.4768 | stalled |
| r128_n2_mri4/sample_009 | P-Flow | 223 | 185 | 1.4650 | 0.7562 | stalled |
| r128_n2_mri4/sample_010 | P-Flow | 302 | 270 | 1.3238 | 0.3107 | stalled |
| r128_n2_mri4/sample_011 | P-Flow | 143 | 120 | 1.0772 | 0.5622 | stalled |
| r128_n2_mri4/sample_012 | P-Flow | 12 | 9 | 96.2513 | 62.9763 | stalled |
| r128_n2_mri4/sample_000 | DLR D-Flow | 928 | 890 | 80.7270 | 79.6315 | stalled |
| r128_n2_mri4/sample_001 | DLR D-Flow | 1000 | 974 | 12.7280 | 8.7037 | step_limit |
| r128_n2_mri4/sample_002 | DLR D-Flow | 53 | 48 | 88.0522 | 86.2221 | stalled |
| r128_n2_mri4/sample_003 | DLR D-Flow | 82 | 79 | 118.9317 | 118.4983 | stalled |
| r128_n2_mri4/sample_004 | DLR D-Flow | 1000 | 970 | 36.1693 | 28.9859 | step_limit |
| r128_n2_mri4/sample_005 | DLR D-Flow | 1000 | 973 | 53.0567 | 46.6003 | step_limit |
| r128_n2_mri4/sample_006 | DLR D-Flow | 315 | 299 | 2.9182 | 2.1018 | stalled |
| r128_n2_mri4/sample_007 | DLR D-Flow | 1000 | 961 | 1.9078 | 1.3270 | step_limit |
| r128_n2_mri4/sample_008 | DLR D-Flow | 923 | 891 | 1.2905 | 0.7452 | stalled |
| r128_n2_mri4/sample_009 | DLR D-Flow | 163 | 153 | 93.6373 | 92.4649 | stalled |
| r128_n2_mri4/sample_010 | DLR D-Flow | 361 | 350 | 6.9738 | 6.0423 | stalled |
| r128_n2_mri4/sample_011 | DLR D-Flow | 473 | 451 | 82.1863 | 81.1141 | stalled |
| r128_n2_mri4/sample_012 | DLR D-Flow | 76 | 72 | 89.6867 | 72.3703 | stalled |
| r128_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.0341 | 0.4410 | step_limit |
| r128_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.0777 | 0.1291 | step_limit |
| r128_n4_mri4/sample_002 | D-Flow | 919 | 917 | 56.2341 | 54.3136 | stalled |
| r128_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.4580 | 0.7088 | step_limit |
| r128_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.0399 | 0.3468 | step_limit |
| r128_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.0520 | 0.3782 | step_limit |
| r128_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.8935 | 0.0543 | step_limit |
| r128_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.5704 | 0.9133 | step_limit |
| r128_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.0036 | 0.4115 | step_limit |
| r128_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.3744 | 0.7302 | step_limit |
| r128_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.2338 | 0.5764 | step_limit |
| r128_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.9471 | 0.2740 | step_limit |
| r128_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.0567 | 0.3384 | step_limit |
| r128_n4_mri4/sample_000 | P-Flow | 353 | 323 | 1.5786 | 0.2601 | stalled |
| r128_n4_mri4/sample_001 | P-Flow | 599 | 545 | 1.9647 | 0.0506 | stalled |
| r128_n4_mri4/sample_002 | P-Flow | 548 | 489 | 1.7815 | 0.1048 | stalled |
| r128_n4_mri4/sample_003 | P-Flow | 504 | 458 | 1.7632 | 0.0222 | stalled |
| r128_n4_mri4/sample_004 | P-Flow | 557 | 503 | 1.8623 | 0.0824 | stalled |
| r128_n4_mri4/sample_005 | P-Flow | 454 | 412 | 1.9015 | 0.0819 | stalled |
| r128_n4_mri4/sample_006 | P-Flow | 433 | 391 | 1.9472 | 0.3041 | stalled |
| r128_n4_mri4/sample_007 | P-Flow | 569 | 518 | 1.8976 | 0.0511 | stalled |
| r128_n4_mri4/sample_008 | P-Flow | 394 | 349 | 1.7354 | 0.1629 | stalled |
| r128_n4_mri4/sample_009 | P-Flow | 470 | 429 | 1.5941 | 0.0773 | stalled |
| r128_n4_mri4/sample_010 | P-Flow | 434 | 395 | 1.9102 | 0.1501 | stalled |
| r128_n4_mri4/sample_011 | P-Flow | 456 | 406 | 1.7674 | 0.2734 | stalled |
| r128_n4_mri4/sample_012 | P-Flow | 90 | 78 | 1.4248 | 0.7270 | stalled |
| r128_n4_mri4/sample_000 | DLR D-Flow | 257 | 247 | 75.9846 | 70.0217 | stalled |
| r128_n4_mri4/sample_001 | DLR D-Flow | 215 | 204 | 40.2861 | 34.8731 | stalled |
| r128_n4_mri4/sample_002 | DLR D-Flow | 1000 | 964 | 55.6732 | 53.8078 | step_limit |
| r128_n4_mri4/sample_003 | DLR D-Flow | 19 | 16 | 106.3951 | 104.9521 | stalled |
| r128_n4_mri4/sample_004 | DLR D-Flow | 182 | 174 | 62.3178 | 57.2612 | stalled |
| r128_n4_mri4/sample_005 | DLR D-Flow | 1000 | 972 | 74.5389 | 70.6738 | step_limit |
| r128_n4_mri4/sample_006 | DLR D-Flow | 483 | 467 | 38.5558 | 35.5361 | stalled |
| r128_n4_mri4/sample_007 | DLR D-Flow | 636 | 609 | 65.7776 | 62.7441 | stalled |
| r128_n4_mri4/sample_008 | DLR D-Flow | 764 | 726 | 6.9187 | 5.2887 | stalled |
| r128_n4_mri4/sample_009 | DLR D-Flow | 198 | 183 | 61.9852 | 58.4767 | stalled |
| r128_n4_mri4/sample_010 | DLR D-Flow | 54 | 49 | 88.4851 | 84.3952 | stalled |
| r128_n4_mri4/sample_011 | DLR D-Flow | 729 | 702 | 11.7689 | 8.6707 | stalled |
| r128_n4_mri4/sample_012 | DLR D-Flow | 425 | 409 | 70.7185 | 67.7185 | stalled |
| r160_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.3323 | 0.0764 | step_limit |
| r160_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.2396 | 0.0494 | step_limit |
| r160_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.4053 | 0.0318 | step_limit |
| r160_n1_ct10/sample_003 | D-Flow | 961 | 959 | 1.0837 | 0.0205 | stalled |
| r160_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.0338 | 0.0182 | step_limit |
| r160_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 1.2007 | 0.0242 | step_limit |
| r160_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.1598 | 0.0187 | step_limit |
| r160_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 0.9638 | 0.0217 | step_limit |
| r160_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 1.6639 | 0.0332 | step_limit |
| r160_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.1687 | 0.1101 | step_limit |
| r160_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.0298 | 0.0172 | step_limit |
| r160_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.0569 | 0.0348 | step_limit |
| r160_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 1.0477 | 0.0336 | step_limit |
| r160_n1_ct10/sample_000 | P-Flow | 112 | 91 | 3.4570 | 0.7567 | stalled |
| r160_n1_ct10/sample_001 | P-Flow | 58 | 50 | 10.8198 | 4.1260 | stalled |
| r160_n1_ct10/sample_002 | P-Flow | 1000 | 814 | 2.6152 | 1.2453 | step_limit |
| r160_n1_ct10/sample_003 | P-Flow | 96 | 82 | 1.3882 | 0.1997 | stalled |
| r160_n1_ct10/sample_004 | P-Flow | 37 | 31 | 7.3620 | 2.2783 | stalled |
| r160_n1_ct10/sample_005 | P-Flow | 59 | 50 | 6.4274 | 2.5491 | stalled |
| r160_n1_ct10/sample_006 | P-Flow | 21 | 17 | 4.6494 | 1.3489 | stalled |
| r160_n1_ct10/sample_007 | P-Flow | 372 | 311 | 1.1220 | 0.1495 | stalled |
| r160_n1_ct10/sample_008 | P-Flow | 648 | 521 | 2.5986 | 0.5371 | stalled |
| r160_n1_ct10/sample_009 | P-Flow | 448 | 362 | 1.3897 | 0.1998 | stalled |
| r160_n1_ct10/sample_010 | P-Flow | 1000 | 783 | 1.3501 | 0.1743 | step_limit |
| r160_n1_ct10/sample_011 | P-Flow | 1000 | 876 | 1.3121 | 0.1827 | step_limit |
| r160_n1_ct10/sample_012 | P-Flow | 79 | 67 | 1.3756 | 0.1742 | stalled |
| r160_n1_ct10/sample_000 | DLR D-Flow | 1000 | 939 | 2.0568 | 0.3014 | step_limit |
| r160_n1_ct10/sample_001 | DLR D-Flow | 118 | 113 | 7.4219 | 5.0947 | stalled |
| r160_n1_ct10/sample_002 | DLR D-Flow | 131 | 124 | 97.1810 | 83.0332 | stalled |
| r160_n1_ct10/sample_003 | DLR D-Flow | 170 | 162 | 1.3691 | 0.2036 | stalled |
| r160_n1_ct10/sample_004 | DLR D-Flow | 390 | 376 | 1.3921 | 0.1829 | stalled |
| r160_n1_ct10/sample_005 | DLR D-Flow | 75 | 72 | 6.3410 | 2.5390 | stalled |
| r160_n1_ct10/sample_006 | DLR D-Flow | 1000 | 992 | 1.5429 | 0.2222 | step_limit |
| r160_n1_ct10/sample_007 | DLR D-Flow | 428 | 402 | 3.9392 | 0.5155 | stalled |
| r160_n1_ct10/sample_008 | DLR D-Flow | 431 | 414 | 2.3189 | 0.4960 | stalled |
| r160_n1_ct10/sample_009 | DLR D-Flow | 606 | 588 | 1.7849 | 0.2161 | stalled |
| r160_n1_ct10/sample_010 | DLR D-Flow | 358 | 342 | 1.2621 | 0.1476 | stalled |
| r160_n1_ct10/sample_011 | DLR D-Flow | 785 | 761 | 1.3263 | 0.1793 | stalled |
| r160_n1_ct10/sample_012 | DLR D-Flow | 253 | 243 | 1.3429 | 0.1752 | stalled |
| r160_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.0109 | 0.0731 | step_limit |
| r160_n2_ct10/sample_001 | D-Flow | 866 | 864 | 0.9562 | 0.0117 | stalled |
| r160_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.4490 | 0.2178 | step_limit |
| r160_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 0.9768 | 0.0175 | step_limit |
| r160_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.0106 | 0.0729 | step_limit |
| r160_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 0.9411 | 0.0170 | step_limit |
| r160_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.0026 | 0.0596 | step_limit |
| r160_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 0.9204 | 0.0200 | step_limit |
| r160_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 1.0604 | 0.0179 | step_limit |
| r160_n2_ct10/sample_009 | D-Flow | 992 | 990 | 0.8899 | 0.0060 | stalled |
| r160_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 0.8674 | 0.0199 | step_limit |
| r160_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.0158 | 0.0940 | step_limit |
| r160_n2_ct10/sample_012 | D-Flow | 264 | 262 | 119.8680 | 100.4067 | stalled |
| r160_n2_ct10/sample_000 | P-Flow | 1000 | 979 | 1.2599 | 0.1308 | step_limit |
| r160_n2_ct10/sample_001 | P-Flow | 15 | 12 | 43.5529 | 18.5815 | stalled |
| r160_n2_ct10/sample_002 | P-Flow | 197 | 162 | 1.4135 | 0.2160 | stalled |
| r160_n2_ct10/sample_003 | P-Flow | 138 | 118 | 1.3178 | 0.1747 | stalled |
| r160_n2_ct10/sample_004 | P-Flow | 99 | 85 | 1.2667 | 0.1762 | stalled |
| r160_n2_ct10/sample_005 | P-Flow | 57 | 46 | 9.2631 | 2.5721 | stalled |
| r160_n2_ct10/sample_006 | P-Flow | 1000 | 807 | 2.1535 | 0.4246 | step_limit |
| r160_n2_ct10/sample_007 | P-Flow | 144 | 128 | 1.2086 | 0.1556 | stalled |
| r160_n2_ct10/sample_008 | P-Flow | 4 | 2 | 98.5154 | 65.0371 | stalled |
| r160_n2_ct10/sample_009 | P-Flow | 1000 | 915 | 1.2293 | 0.1557 | step_limit |
| r160_n2_ct10/sample_010 | P-Flow | 157 | 135 | 1.2415 | 0.1772 | stalled |
| r160_n2_ct10/sample_011 | P-Flow | 291 | 235 | 1.1879 | 0.1473 | stalled |
| r160_n2_ct10/sample_012 | P-Flow | 98 | 82 | 6.7209 | 2.5530 | stalled |
| r160_n2_ct10/sample_000 | DLR D-Flow | 736 | 713 | 1.7310 | 0.3823 | stalled |
| r160_n2_ct10/sample_001 | DLR D-Flow | 539 | 518 | 1.9947 | 0.3664 | stalled |
| r160_n2_ct10/sample_002 | DLR D-Flow | 392 | 382 | 60.3341 | 41.0764 | stalled |
| r160_n2_ct10/sample_003 | DLR D-Flow | 31 | 28 | 76.4030 | 42.1552 | stalled |
| r160_n2_ct10/sample_004 | DLR D-Flow | 381 | 373 | 3.0812 | 1.0358 | stalled |
| r160_n2_ct10/sample_005 | DLR D-Flow | 743 | 721 | 5.8631 | 1.3096 | stalled |
| r160_n2_ct10/sample_006 | DLR D-Flow | 1000 | 970 | 1.9604 | 0.5144 | step_limit |
| r160_n2_ct10/sample_007 | DLR D-Flow | 707 | 683 | 2.1687 | 0.2983 | stalled |
| r160_n2_ct10/sample_008 | DLR D-Flow | 919 | 875 | 3.1244 | 0.7083 | stalled |
| r160_n2_ct10/sample_009 | DLR D-Flow | 103 | 94 | 4.4427 | 1.1630 | stalled |
| r160_n2_ct10/sample_010 | DLR D-Flow | 255 | 241 | 2.0519 | 0.5315 | stalled |
| r160_n2_ct10/sample_011 | DLR D-Flow | 613 | 601 | 45.4137 | 28.1823 | stalled |
| r160_n2_ct10/sample_012 | DLR D-Flow | 220 | 214 | 126.4957 | 107.1895 | stalled |
| r160_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 0.9477 | 0.0122 | step_limit |
| r160_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 0.9853 | 0.0300 | step_limit |
| r160_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 0.9936 | 0.0192 | step_limit |
| r160_n4_ct10/sample_003 | D-Flow | 749 | 747 | 0.9941 | 0.0118 | stalled |
| r160_n4_ct10/sample_004 | D-Flow | 570 | 568 | 0.9056 | 0.0176 | stalled |
| r160_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 0.9589 | 0.0244 | step_limit |
| r160_n4_ct10/sample_006 | D-Flow | 753 | 751 | 1.0722 | 0.0136 | stalled |
| r160_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.0547 | 0.0971 | step_limit |
| r160_n4_ct10/sample_008 | D-Flow | 899 | 897 | 0.9328 | 0.0138 | stalled |
| r160_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 0.8953 | 0.0188 | step_limit |
| r160_n4_ct10/sample_010 | D-Flow | 742 | 740 | 0.9396 | 0.0130 | stalled |
| r160_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 0.9304 | 0.0447 | step_limit |
| r160_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 1.0542 | 0.0264 | step_limit |
| r160_n4_ct10/sample_000 | P-Flow | 81 | 70 | 1.5586 | 0.2319 | stalled |
| r160_n4_ct10/sample_001 | P-Flow | 1000 | 991 | 1.1879 | 0.1371 | step_limit |
| r160_n4_ct10/sample_002 | P-Flow | 1000 | 978 | 1.2102 | 0.1500 | step_limit |
| r160_n4_ct10/sample_003 | P-Flow | 98 | 86 | 1.3438 | 0.1885 | stalled |
| r160_n4_ct10/sample_004 | P-Flow | 117 | 102 | 1.0984 | 0.1266 | stalled |
| r160_n4_ct10/sample_005 | P-Flow | 1000 | 976 | 1.1571 | 0.1380 | step_limit |
| r160_n4_ct10/sample_006 | P-Flow | 1000 | 986 | 1.3070 | 0.1820 | step_limit |
| r160_n4_ct10/sample_007 | P-Flow | 183 | 164 | 1.3063 | 0.1635 | stalled |
| r160_n4_ct10/sample_008 | P-Flow | 1000 | 945 | 1.2006 | 0.1270 | step_limit |
| r160_n4_ct10/sample_009 | P-Flow | 72 | 64 | 1.2726 | 0.1788 | stalled |
| r160_n4_ct10/sample_010 | P-Flow | 4 | 2 | 89.0217 | 47.4085 | stalled |
| r160_n4_ct10/sample_011 | P-Flow | 180 | 158 | 1.3463 | 0.1831 | stalled |
| r160_n4_ct10/sample_012 | P-Flow | 1000 | 974 | 1.2836 | 0.1643 | step_limit |
| r160_n4_ct10/sample_000 | DLR D-Flow | 31 | 29 | 36.0370 | 12.3975 | stalled |
| r160_n4_ct10/sample_001 | DLR D-Flow | 1000 | 967 | 33.7391 | 14.5938 | step_limit |
| r160_n4_ct10/sample_002 | DLR D-Flow | 1000 | 968 | 50.6594 | 23.9533 | step_limit |
| r160_n4_ct10/sample_003 | DLR D-Flow | 1000 | 968 | 23.9567 | 8.0350 | step_limit |
| r160_n4_ct10/sample_004 | DLR D-Flow | 1000 | 954 | 62.5644 | 26.7479 | step_limit |
| r160_n4_ct10/sample_005 | DLR D-Flow | 413 | 390 | 23.9409 | 7.6782 | stalled |
| r160_n4_ct10/sample_006 | DLR D-Flow | 152 | 147 | 15.1303 | 4.5958 | stalled |
| r160_n4_ct10/sample_007 | DLR D-Flow | 130 | 121 | 69.1375 | 30.8524 | stalled |
| r160_n4_ct10/sample_008 | DLR D-Flow | 648 | 629 | 37.3756 | 12.7229 | stalled |
| r160_n4_ct10/sample_009 | DLR D-Flow | 382 | 371 | 40.2078 | 19.8531 | stalled |
| r160_n4_ct10/sample_010 | DLR D-Flow | 515 | 494 | 23.2049 | 6.4383 | stalled |
| r160_n4_ct10/sample_011 | DLR D-Flow | 1000 | 943 | 68.7907 | 29.7338 | step_limit |
| r160_n4_ct10/sample_012 | DLR D-Flow | 234 | 227 | 15.1058 | 4.5236 | stalled |
| r160_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.9807 | 0.1485 | step_limit |
| r160_n1_mri4/sample_001 | D-Flow | 231 | 228 | 1.8728 | 1.0787 | stalled |
| r160_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 53.9125 | 50.9588 | step_limit |
| r160_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.8056 | 0.2460 | step_limit |
| r160_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.8854 | 0.3736 | step_limit |
| r160_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.9207 | 0.1612 | step_limit |
| r160_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.8574 | 0.1187 | step_limit |
| r160_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.7586 | 0.2517 | step_limit |
| r160_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.2391 | 0.3477 | step_limit |
| r160_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.7288 | 0.2394 | step_limit |
| r160_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.6685 | 0.0541 | step_limit |
| r160_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.8142 | 0.2188 | step_limit |
| r160_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 118.9684 | 32.0679 | step_limit |
| r160_n1_mri4/sample_000 | P-Flow | 120 | 106 | 1.5327 | 0.7061 | stalled |
| r160_n1_mri4/sample_001 | P-Flow | 150 | 126 | 1.3156 | 0.6435 | stalled |
| r160_n1_mri4/sample_002 | P-Flow | 125 | 110 | 1.3883 | 0.7176 | stalled |
| r160_n1_mri4/sample_003 | P-Flow | 118 | 100 | 1.2407 | 0.6017 | stalled |
| r160_n1_mri4/sample_004 | P-Flow | 129 | 110 | 1.1639 | 0.6179 | stalled |
| r160_n1_mri4/sample_005 | P-Flow | 224 | 189 | 1.3717 | 0.6307 | stalled |
| r160_n1_mri4/sample_006 | P-Flow | 71 | 61 | 1.3738 | 0.7954 | stalled |
| r160_n1_mri4/sample_007 | P-Flow | 102 | 86 | 1.0322 | 0.5489 | stalled |
| r160_n1_mri4/sample_008 | P-Flow | 200 | 170 | 1.8231 | 1.0062 | stalled |
| r160_n1_mri4/sample_009 | P-Flow | 185 | 153 | 1.1611 | 0.6641 | stalled |
| r160_n1_mri4/sample_010 | P-Flow | 149 | 121 | 1.1099 | 0.5639 | stalled |
| r160_n1_mri4/sample_011 | P-Flow | 58 | 47 | 1.2106 | 0.6323 | stalled |
| r160_n1_mri4/sample_012 | P-Flow | 112 | 98 | 1.1300 | 0.7266 | stalled |
| r160_n1_mri4/sample_000 | DLR D-Flow | 106 | 103 | 99.9486 | 99.1196 | stalled |
| r160_n1_mri4/sample_001 | DLR D-Flow | 286 | 273 | 106.6263 | 106.1697 | stalled |
| r160_n1_mri4/sample_002 | DLR D-Flow | 85 | 81 | 96.4632 | 96.0659 | stalled |
| r160_n1_mri4/sample_003 | DLR D-Flow | 377 | 363 | 1.3704 | 0.7572 | stalled |
| r160_n1_mri4/sample_004 | DLR D-Flow | 482 | 453 | 1.3233 | 0.8478 | stalled |
| r160_n1_mri4/sample_005 | DLR D-Flow | 138 | 135 | 6.8574 | 5.0481 | stalled |
| r160_n1_mri4/sample_006 | DLR D-Flow | 1000 | 902 | 1.8590 | 1.1654 | step_limit |
| r160_n1_mri4/sample_007 | DLR D-Flow | 666 | 649 | 1.1541 | 0.6740 | stalled |
| r160_n1_mri4/sample_008 | DLR D-Flow | 416 | 388 | 2.4871 | 1.5367 | stalled |
| r160_n1_mri4/sample_009 | DLR D-Flow | 275 | 263 | 2.3418 | 1.8998 | stalled |
| r160_n1_mri4/sample_010 | DLR D-Flow | 218 | 212 | 1.2067 | 0.6569 | stalled |
| r160_n1_mri4/sample_011 | DLR D-Flow | 452 | 442 | 1.4397 | 0.8184 | stalled |
| r160_n1_mri4/sample_012 | DLR D-Flow | 325 | 311 | 126.4351 | 53.4396 | stalled |
| r160_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.6470 | 0.0976 | step_limit |
| r160_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.0397 | 0.5755 | step_limit |
| r160_n2_mri4/sample_002 | D-Flow | 717 | 715 | 54.2528 | 52.7654 | stalled |
| r160_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.0598 | 0.5547 | step_limit |
| r160_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.1821 | 0.6680 | step_limit |
| r160_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.8418 | 0.3774 | step_limit |
| r160_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.6805 | 0.1562 | step_limit |
| r160_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.7409 | 0.2512 | step_limit |
| r160_n2_mri4/sample_008 | D-Flow | 561 | 556 | 67.6651 | 67.5634 | stalled |
| r160_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.6318 | 0.1056 | step_limit |
| r160_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.6069 | 0.0740 | step_limit |
| r160_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.8314 | 0.2756 | step_limit |
| r160_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.1285 | 0.3907 | step_limit |
| r160_n2_mri4/sample_000 | P-Flow | 271 | 242 | 1.2317 | 0.4191 | stalled |
| r160_n2_mri4/sample_001 | P-Flow | 744 | 652 | 1.3081 | 0.1321 | stalled |
| r160_n2_mri4/sample_002 | P-Flow | 144 | 121 | 1.2065 | 0.7398 | stalled |
| r160_n2_mri4/sample_003 | P-Flow | 174 | 146 | 1.0995 | 0.5552 | stalled |
| r160_n2_mri4/sample_004 | P-Flow | 920 | 829 | 1.8722 | 0.0599 | stalled |
| r160_n2_mri4/sample_005 | P-Flow | 104 | 89 | 1.0745 | 0.5728 | stalled |
| r160_n2_mri4/sample_006 | P-Flow | 390 | 345 | 1.4721 | 0.1975 | stalled |
| r160_n2_mri4/sample_007 | P-Flow | 138 | 120 | 1.0594 | 0.5519 | stalled |
| r160_n2_mri4/sample_008 | P-Flow | 477 | 421 | 1.6509 | 0.1396 | stalled |
| r160_n2_mri4/sample_009 | P-Flow | 615 | 533 | 1.3452 | 0.2707 | stalled |
| r160_n2_mri4/sample_010 | P-Flow | 282 | 250 | 1.0822 | 0.4376 | stalled |
| r160_n2_mri4/sample_011 | P-Flow | 506 | 453 | 1.4720 | 0.1570 | stalled |
| r160_n2_mri4/sample_012 | P-Flow | 123 | 105 | 1.5659 | 0.7732 | stalled |
| r160_n2_mri4/sample_000 | DLR D-Flow | 353 | 338 | 53.9629 | 51.1336 | stalled |
| r160_n2_mri4/sample_001 | DLR D-Flow | 282 | 273 | 45.2835 | 32.1816 | stalled |
| r160_n2_mri4/sample_002 | DLR D-Flow | 470 | 449 | 54.6724 | 54.0584 | stalled |
| r160_n2_mri4/sample_003 | DLR D-Flow | 1000 | 962 | 56.6757 | 53.4822 | step_limit |
| r160_n2_mri4/sample_004 | DLR D-Flow | 189 | 183 | 61.0452 | 60.4984 | stalled |
| r160_n2_mri4/sample_005 | DLR D-Flow | 460 | 441 | 69.3114 | 61.8325 | stalled |
| r160_n2_mri4/sample_006 | DLR D-Flow | 509 | 493 | 1.6053 | 1.1021 | stalled |
| r160_n2_mri4/sample_007 | DLR D-Flow | 201 | 190 | 2.3545 | 1.6936 | stalled |
| r160_n2_mri4/sample_008 | DLR D-Flow | 74 | 70 | 79.2206 | 78.0238 | stalled |
| r160_n2_mri4/sample_009 | DLR D-Flow | 722 | 695 | 2.5581 | 1.8910 | stalled |
| r160_n2_mri4/sample_010 | DLR D-Flow | 587 | 566 | 1.4691 | 1.1269 | stalled |
| r160_n2_mri4/sample_011 | DLR D-Flow | 541 | 531 | 44.6751 | 43.1597 | stalled |
| r160_n2_mri4/sample_012 | DLR D-Flow | 90 | 86 | 127.2783 | 126.1863 | stalled |
| r160_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.7382 | 0.2482 | step_limit |
| r160_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.8409 | 0.2820 | step_limit |
| r160_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.8010 | 0.1883 | step_limit |
| r160_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.8149 | 0.2481 | step_limit |
| r160_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.7162 | 0.0998 | step_limit |
| r160_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.8997 | 0.3610 | step_limit |
| r160_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.8496 | 0.1632 | step_limit |
| r160_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 58.2063 | 54.7795 | step_limit |
| r160_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.7167 | 0.1673 | step_limit |
| r160_n4_mri4/sample_009 | D-Flow | 281 | 279 | 33.3003 | 30.5646 | stalled |
| r160_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.7422 | 0.1012 | step_limit |
| r160_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.8445 | 0.3669 | step_limit |
| r160_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 0.7726 | 0.2013 | step_limit |
| r160_n4_mri4/sample_000 | P-Flow | 88 | 73 | 1.0404 | 0.5939 | stalled |
| r160_n4_mri4/sample_001 | P-Flow | 457 | 406 | 1.2782 | 0.1687 | stalled |
| r160_n4_mri4/sample_002 | P-Flow | 155 | 133 | 1.0684 | 0.5048 | stalled |
| r160_n4_mri4/sample_003 | P-Flow | 409 | 362 | 1.2899 | 0.2112 | stalled |
| r160_n4_mri4/sample_004 | P-Flow | 254 | 223 | 1.0377 | 0.3834 | stalled |
| r160_n4_mri4/sample_005 | P-Flow | 921 | 803 | 1.2514 | 0.1174 | stalled |
| r160_n4_mri4/sample_006 | P-Flow | 1000 | 888 | 1.4930 | 0.0512 | step_limit |
| r160_n4_mri4/sample_007 | P-Flow | 699 | 612 | 1.2294 | 0.0804 | stalled |
| r160_n4_mri4/sample_008 | P-Flow | 1000 | 883 | 1.2495 | 0.0324 | step_limit |
| r160_n4_mri4/sample_009 | P-Flow | 140 | 121 | 0.9891 | 0.5517 | stalled |
| r160_n4_mri4/sample_010 | P-Flow | 601 | 518 | 1.3113 | 0.1266 | stalled |
| r160_n4_mri4/sample_011 | P-Flow | 526 | 462 | 1.1630 | 0.1748 | stalled |
| r160_n4_mri4/sample_012 | P-Flow | 1000 | 896 | 1.3647 | 0.0261 | step_limit |
| r160_n4_mri4/sample_000 | DLR D-Flow | 333 | 316 | 25.6230 | 22.7102 | stalled |
| r160_n4_mri4/sample_001 | DLR D-Flow | 695 | 666 | 65.8305 | 57.0720 | stalled |
| r160_n4_mri4/sample_002 | DLR D-Flow | 1000 | 972 | 25.0694 | 20.9903 | step_limit |
| r160_n4_mri4/sample_003 | DLR D-Flow | 584 | 561 | 72.6411 | 65.2561 | stalled |
| r160_n4_mri4/sample_004 | DLR D-Flow | 1000 | 968 | 59.1775 | 52.2579 | step_limit |
| r160_n4_mri4/sample_005 | DLR D-Flow | 376 | 362 | 84.7940 | 82.8081 | stalled |
| r160_n4_mri4/sample_006 | DLR D-Flow | 643 | 622 | 43.8674 | 43.1873 | stalled |
| r160_n4_mri4/sample_007 | DLR D-Flow | 882 | 856 | 69.9045 | 66.6101 | stalled |
| r160_n4_mri4/sample_008 | DLR D-Flow | 141 | 136 | 65.4824 | 56.8707 | stalled |
| r160_n4_mri4/sample_009 | DLR D-Flow | 467 | 450 | 46.6402 | 43.0815 | stalled |
| r160_n4_mri4/sample_010 | DLR D-Flow | 1000 | 966 | 52.7253 | 48.6401 | step_limit |
| r160_n4_mri4/sample_011 | DLR D-Flow | 1000 | 975 | 47.3318 | 43.0884 | step_limit |
| r160_n4_mri4/sample_012 | DLR D-Flow | 1000 | 970 | 24.9110 | 22.2267 | step_limit |
| r200_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 0.9236 | 0.0154 | step_limit |
| r200_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.4885 | 0.0814 | step_limit |
| r200_n1_ct10/sample_002 | D-Flow | 588 | 586 | 0.9307 | 0.0184 | stalled |
| r200_n1_ct10/sample_003 | D-Flow | 653 | 651 | 0.9206 | 0.0135 | stalled |
| r200_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.6486 | 0.0382 | step_limit |
| r200_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 1.4029 | 0.0496 | step_limit |
| r200_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.6585 | 0.0970 | step_limit |
| r200_n1_ct10/sample_007 | D-Flow | 64 | 62 | 109.6043 | 95.3884 | stalled |
| r200_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 0.9284 | 0.0176 | step_limit |
| r200_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.4460 | 0.0177 | step_limit |
| r200_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 0.8746 | 0.0109 | step_limit |
| r200_n1_ct10/sample_011 | D-Flow | 517 | 514 | 1.0270 | 0.0195 | stalled |
| r200_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 1.2345 | 0.0597 | step_limit |
| r200_n1_ct10/sample_000 | P-Flow | 41 | 33 | 8.0076 | 2.0387 | stalled |
| r200_n1_ct10/sample_001 | P-Flow | 15 | 13 | 5.2008 | 1.9960 | stalled |
| r200_n1_ct10/sample_002 | P-Flow | 56 | 48 | 1.2083 | 0.1799 | stalled |
| r200_n1_ct10/sample_003 | P-Flow | 267 | 229 | 1.3381 | 0.1729 | stalled |
| r200_n1_ct10/sample_004 | P-Flow | 1000 | 965 | 1.9845 | 0.3288 | step_limit |
| r200_n1_ct10/sample_005 | P-Flow | 220 | 175 | 1.8905 | 0.3384 | stalled |
| r200_n1_ct10/sample_006 | P-Flow | 32 | 27 | 12.4518 | 6.2413 | stalled |
| r200_n1_ct10/sample_007 | P-Flow | 22 | 19 | 110.9080 | 94.9654 | stalled |
| r200_n1_ct10/sample_008 | P-Flow | 201 | 163 | 1.2843 | 0.1760 | stalled |
| r200_n1_ct10/sample_009 | P-Flow | 873 | 848 | 1.7986 | 0.3222 | stalled |
| r200_n1_ct10/sample_010 | P-Flow | 33 | 28 | 7.0479 | 1.9175 | stalled |
| r200_n1_ct10/sample_011 | P-Flow | 76 | 64 | 2.9418 | 0.6049 | stalled |
| r200_n1_ct10/sample_012 | P-Flow | 238 | 197 | 8.7482 | 2.5192 | stalled |
| r200_n1_ct10/sample_000 | DLR D-Flow | 780 | 747 | 1.1694 | 0.1399 | stalled |
| r200_n1_ct10/sample_001 | DLR D-Flow | 687 | 654 | 1.8296 | 0.2579 | stalled |
| r200_n1_ct10/sample_002 | DLR D-Flow | 416 | 395 | 1.1688 | 0.1646 | stalled |
| r200_n1_ct10/sample_003 | DLR D-Flow | 694 | 679 | 1.2353 | 0.1510 | stalled |
| r200_n1_ct10/sample_004 | DLR D-Flow | 624 | 597 | 1.9448 | 0.3237 | stalled |
| r200_n1_ct10/sample_005 | DLR D-Flow | 411 | 391 | 1.6535 | 0.2950 | stalled |
| r200_n1_ct10/sample_006 | DLR D-Flow | 439 | 425 | 7.2667 | 4.9509 | stalled |
| r200_n1_ct10/sample_007 | DLR D-Flow | 11 | 8 | 108.8613 | 96.2243 | stalled |
| r200_n1_ct10/sample_008 | DLR D-Flow | 253 | 244 | 1.3244 | 0.1766 | stalled |
| r200_n1_ct10/sample_009 | DLR D-Flow | 29 | 26 | 57.6213 | 29.2380 | stalled |
| r200_n1_ct10/sample_010 | DLR D-Flow | 416 | 392 | 1.4615 | 0.2338 | stalled |
| r200_n1_ct10/sample_011 | DLR D-Flow | 220 | 212 | 1.2098 | 0.1522 | stalled |
| r200_n1_ct10/sample_012 | DLR D-Flow | 678 | 644 | 1.5038 | 0.2360 | stalled |
| r200_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 0.9608 | 0.0282 | step_limit |
| r200_n2_ct10/sample_001 | D-Flow | 275 | 271 | 55.8958 | 36.5515 | stalled |
| r200_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 0.9718 | 0.0204 | step_limit |
| r200_n2_ct10/sample_003 | D-Flow | 50 | 48 | 91.8360 | 70.9766 | stalled |
| r200_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.3265 | 0.0922 | step_limit |
| r200_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 0.9384 | 0.0159 | step_limit |
| r200_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.4742 | 0.0318 | step_limit |
| r200_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.0588 | 0.0219 | step_limit |
| r200_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 0.9600 | 0.0267 | step_limit |
| r200_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.1710 | 0.0822 | step_limit |
| r200_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.3899 | 0.0160 | step_limit |
| r200_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.1187 | 0.1193 | step_limit |
| r200_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 1.0596 | 0.0312 | step_limit |
| r200_n2_ct10/sample_000 | P-Flow | 41 | 35 | 6.4780 | 2.3785 | stalled |
| r200_n2_ct10/sample_001 | P-Flow | 212 | 179 | 1.1766 | 0.1554 | stalled |
| r200_n2_ct10/sample_002 | P-Flow | 328 | 273 | 1.1689 | 0.1472 | stalled |
| r200_n2_ct10/sample_003 | P-Flow | 216 | 173 | 4.1193 | 1.2232 | stalled |
| r200_n2_ct10/sample_004 | P-Flow | 631 | 516 | 2.1807 | 0.5253 | stalled |
| r200_n2_ct10/sample_005 | P-Flow | 290 | 243 | 1.1567 | 0.1460 | stalled |
| r200_n2_ct10/sample_006 | P-Flow | 353 | 282 | 3.3511 | 0.8964 | stalled |
| r200_n2_ct10/sample_007 | P-Flow | 380 | 304 | 1.1832 | 0.1490 | stalled |
| r200_n2_ct10/sample_008 | P-Flow | 482 | 405 | 1.3138 | 0.1927 | stalled |
| r200_n2_ct10/sample_009 | P-Flow | 666 | 527 | 1.1885 | 0.1440 | stalled |
| r200_n2_ct10/sample_010 | P-Flow | 483 | 396 | 2.5886 | 0.8344 | stalled |
| r200_n2_ct10/sample_011 | P-Flow | 299 | 259 | 1.1125 | 0.1318 | stalled |
| r200_n2_ct10/sample_012 | P-Flow | 370 | 307 | 1.5814 | 0.3489 | stalled |
| r200_n2_ct10/sample_000 | DLR D-Flow | 328 | 319 | 1.5950 | 0.3425 | stalled |
| r200_n2_ct10/sample_001 | DLR D-Flow | 1000 | 970 | 52.3304 | 36.9118 | step_limit |
| r200_n2_ct10/sample_002 | DLR D-Flow | 1000 | 957 | 6.4324 | 1.5746 | step_limit |
| r200_n2_ct10/sample_003 | DLR D-Flow | 210 | 204 | 92.0402 | 68.6537 | stalled |
| r200_n2_ct10/sample_004 | DLR D-Flow | 471 | 454 | 42.7697 | 20.6301 | stalled |
| r200_n2_ct10/sample_005 | DLR D-Flow | 163 | 159 | 10.9861 | 2.9324 | stalled |
| r200_n2_ct10/sample_006 | DLR D-Flow | 792 | 759 | 77.8704 | 55.8116 | stalled |
| r200_n2_ct10/sample_007 | DLR D-Flow | 364 | 349 | 2.3340 | 0.4276 | stalled |
| r200_n2_ct10/sample_008 | DLR D-Flow | 425 | 408 | 4.7779 | 1.7364 | stalled |
| r200_n2_ct10/sample_009 | DLR D-Flow | 618 | 596 | 1.7877 | 0.3036 | stalled |
| r200_n2_ct10/sample_010 | DLR D-Flow | 480 | 456 | 27.0472 | 9.9527 | stalled |
| r200_n2_ct10/sample_011 | DLR D-Flow | 207 | 202 | 2.8994 | 0.6239 | stalled |
| r200_n2_ct10/sample_012 | DLR D-Flow | 654 | 626 | 4.7832 | 1.5505 | stalled |
| r200_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.0340 | 0.0369 | step_limit |
| r200_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.3434 | 0.1153 | step_limit |
| r200_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 0.9689 | 0.0461 | step_limit |
| r200_n4_ct10/sample_003 | D-Flow | 582 | 579 | 81.5545 | 46.6864 | stalled |
| r200_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 0.9624 | 0.0088 | step_limit |
| r200_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 0.9616 | 0.0259 | step_limit |
| r200_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.0720 | 0.0822 | step_limit |
| r200_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 0.9892 | 0.0178 | step_limit |
| r200_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 43.8522 | 18.7899 | step_limit |
| r200_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.2502 | 0.1471 | step_limit |
| r200_n4_ct10/sample_010 | D-Flow | 828 | 825 | 0.9704 | 0.0129 | stalled |
| r200_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 1.2005 | 0.0654 | step_limit |
| r200_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 1.2376 | 0.0939 | step_limit |
| r200_n4_ct10/sample_000 | P-Flow | 1000 | 966 | 1.3273 | 0.0831 | step_limit |
| r200_n4_ct10/sample_001 | P-Flow | 1000 | 959 | 1.4755 | 0.1790 | step_limit |
| r200_n4_ct10/sample_002 | P-Flow | 527 | 439 | 1.4655 | 0.1949 | stalled |
| r200_n4_ct10/sample_003 | P-Flow | 1000 | 981 | 1.3168 | 0.1453 | step_limit |
| r200_n4_ct10/sample_004 | P-Flow | 1000 | 970 | 1.1896 | 0.1316 | step_limit |
| r200_n4_ct10/sample_005 | P-Flow | 128 | 111 | 1.5489 | 0.1909 | stalled |
| r200_n4_ct10/sample_006 | P-Flow | 121 | 109 | 1.3563 | 0.1752 | stalled |
| r200_n4_ct10/sample_007 | P-Flow | 1000 | 979 | 1.3225 | 0.0779 | step_limit |
| r200_n4_ct10/sample_008 | P-Flow | 1000 | 968 | 1.2152 | 0.1268 | step_limit |
| r200_n4_ct10/sample_009 | P-Flow | 1000 | 985 | 1.2661 | 0.1421 | step_limit |
| r200_n4_ct10/sample_010 | P-Flow | 1000 | 963 | 1.4392 | 0.0658 | step_limit |
| r200_n4_ct10/sample_011 | P-Flow | 380 | 307 | 1.4377 | 0.1927 | stalled |
| r200_n4_ct10/sample_012 | P-Flow | 92 | 82 | 1.5706 | 0.2126 | stalled |
| r200_n4_ct10/sample_000 | DLR D-Flow | 1000 | 961 | 13.7307 | 3.2527 | step_limit |
| r200_n4_ct10/sample_001 | DLR D-Flow | 491 | 468 | 84.7484 | 46.1804 | stalled |
| r200_n4_ct10/sample_002 | DLR D-Flow | 234 | 228 | 67.4669 | 33.5918 | stalled |
| r200_n4_ct10/sample_003 | DLR D-Flow | 561 | 538 | 62.7905 | 35.9368 | stalled |
| r200_n4_ct10/sample_004 | DLR D-Flow | 1000 | 959 | 37.3326 | 13.9029 | step_limit |
| r200_n4_ct10/sample_005 | DLR D-Flow | 1000 | 964 | 27.5889 | 8.6396 | step_limit |
| r200_n4_ct10/sample_006 | DLR D-Flow | 1000 | 955 | 17.6590 | 4.7907 | step_limit |
| r200_n4_ct10/sample_007 | DLR D-Flow | 921 | 886 | 65.0383 | 31.5558 | stalled |
| r200_n4_ct10/sample_008 | DLR D-Flow | 1000 | 957 | 30.9766 | 10.9083 | step_limit |
| r200_n4_ct10/sample_009 | DLR D-Flow | 1000 | 967 | 57.2308 | 27.1927 | step_limit |
| r200_n4_ct10/sample_010 | DLR D-Flow | 937 | 888 | 79.2383 | 40.7337 | stalled |
| r200_n4_ct10/sample_011 | DLR D-Flow | 1000 | 961 | 83.1216 | 42.3587 | step_limit |
| r200_n4_ct10/sample_012 | DLR D-Flow | 967 | 940 | 67.5640 | 36.9810 | stalled |
| r200_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.6766 | 0.0902 | step_limit |
| r200_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.9712 | 0.1587 | step_limit |
| r200_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.6513 | 0.1082 | step_limit |
| r200_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.6504 | 0.1291 | step_limit |
| r200_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.0969 | 0.2473 | step_limit |
| r200_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.1171 | 0.3194 | step_limit |
| r200_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.1860 | 0.3101 | step_limit |
| r200_n1_mri4/sample_007 | D-Flow | 26 | 22 | 108.1585 | 107.4076 | stalled |
| r200_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.6377 | 0.0801 | step_limit |
| r200_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.1550 | 0.4355 | step_limit |
| r200_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.6905 | 0.1766 | step_limit |
| r200_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 0.6688 | 0.0889 | step_limit |
| r200_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 0.8735 | 0.2013 | step_limit |
| r200_n1_mri4/sample_000 | P-Flow | 137 | 119 | 1.0685 | 0.5562 | stalled |
| r200_n1_mri4/sample_001 | P-Flow | 199 | 172 | 1.4747 | 0.7593 | stalled |
| r200_n1_mri4/sample_002 | P-Flow | 61 | 51 | 1.0678 | 0.6639 | stalled |
| r200_n1_mri4/sample_003 | P-Flow | 67 | 56 | 1.1366 | 0.5995 | stalled |
| r200_n1_mri4/sample_004 | P-Flow | 64 | 51 | 1.8002 | 1.1076 | stalled |
| r200_n1_mri4/sample_005 | P-Flow | 71 | 60 | 1.5923 | 0.9903 | stalled |
| r200_n1_mri4/sample_006 | P-Flow | 80 | 67 | 1.7193 | 0.9493 | stalled |
| r200_n1_mri4/sample_007 | P-Flow | 525 | 457 | 2.2296 | 0.4805 | stalled |
| r200_n1_mri4/sample_008 | P-Flow | 66 | 55 | 1.0844 | 0.6663 | stalled |
| r200_n1_mri4/sample_009 | P-Flow | 234 | 214 | 1.8187 | 0.7831 | stalled |
| r200_n1_mri4/sample_010 | P-Flow | 75 | 64 | 1.0884 | 0.5959 | stalled |
| r200_n1_mri4/sample_011 | P-Flow | 29 | 25 | 1.7288 | 1.1590 | stalled |
| r200_n1_mri4/sample_012 | P-Flow | 103 | 92 | 1.2707 | 0.7516 | stalled |
| r200_n1_mri4/sample_000 | DLR D-Flow | 439 | 420 | 1.2179 | 0.6535 | stalled |
| r200_n1_mri4/sample_001 | DLR D-Flow | 993 | 942 | 1.5807 | 0.8806 | stalled |
| r200_n1_mri4/sample_002 | DLR D-Flow | 281 | 272 | 1.1596 | 0.7473 | stalled |
| r200_n1_mri4/sample_003 | DLR D-Flow | 433 | 413 | 1.1999 | 0.6571 | stalled |
| r200_n1_mri4/sample_004 | DLR D-Flow | 436 | 415 | 43.8209 | 41.7592 | stalled |
| r200_n1_mri4/sample_005 | DLR D-Flow | 1000 | 962 | 2.1810 | 1.5510 | step_limit |
| r200_n1_mri4/sample_006 | DLR D-Flow | 634 | 611 | 5.5745 | 5.0599 | stalled |
| r200_n1_mri4/sample_007 | DLR D-Flow | 17 | 14 | 104.7155 | 103.0129 | stalled |
| r200_n1_mri4/sample_008 | DLR D-Flow | 640 | 610 | 1.2797 | 0.8208 | stalled |
| r200_n1_mri4/sample_009 | DLR D-Flow | 867 | 832 | 1.7705 | 1.0992 | stalled |
| r200_n1_mri4/sample_010 | DLR D-Flow | 529 | 500 | 1.1795 | 0.6625 | stalled |
| r200_n1_mri4/sample_011 | DLR D-Flow | 274 | 262 | 1.7130 | 1.1495 | stalled |
| r200_n1_mri4/sample_012 | DLR D-Flow | 283 | 275 | 94.7724 | 93.9303 | stalled |
| r200_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.8039 | 0.2904 | step_limit |
| r200_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 50.8506 | 48.3018 | step_limit |
| r200_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.7530 | 0.2327 | step_limit |
| r200_n2_mri4/sample_003 | D-Flow | 417 | 415 | 59.2805 | 58.9595 | stalled |
| r200_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 51.1554 | 49.7276 | step_limit |
| r200_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.7703 | 0.2621 | step_limit |
| r200_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.1048 | 0.4675 | step_limit |
| r200_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.8145 | 0.3477 | step_limit |
| r200_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.8778 | 0.3601 | step_limit |
| r200_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.7815 | 0.3069 | step_limit |
| r200_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.1215 | 0.4391 | step_limit |
| r200_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.0425 | 0.5604 | step_limit |
| r200_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.0574 | 0.5294 | step_limit |
| r200_n2_mri4/sample_000 | P-Flow | 210 | 180 | 1.1343 | 0.5031 | stalled |
| r200_n2_mri4/sample_001 | P-Flow | 118 | 102 | 1.0314 | 0.6132 | stalled |
| r200_n2_mri4/sample_002 | P-Flow | 315 | 274 | 1.2764 | 0.3970 | stalled |
| r200_n2_mri4/sample_003 | P-Flow | 104 | 89 | 1.3022 | 0.6658 | stalled |
| r200_n2_mri4/sample_004 | P-Flow | 271 | 237 | 1.3062 | 0.5321 | stalled |
| r200_n2_mri4/sample_005 | P-Flow | 111 | 93 | 1.0064 | 0.5009 | stalled |
| r200_n2_mri4/sample_006 | P-Flow | 178 | 159 | 1.7746 | 0.6408 | stalled |
| r200_n2_mri4/sample_007 | P-Flow | 128 | 112 | 1.0251 | 0.5691 | stalled |
| r200_n2_mri4/sample_008 | P-Flow | 91 | 76 | 1.0782 | 0.5786 | stalled |
| r200_n2_mri4/sample_009 | P-Flow | 130 | 109 | 1.0071 | 0.5193 | stalled |
| r200_n2_mri4/sample_010 | P-Flow | 265 | 237 | 1.3564 | 0.6464 | stalled |
| r200_n2_mri4/sample_011 | P-Flow | 122 | 103 | 1.0537 | 0.5512 | stalled |
| r200_n2_mri4/sample_012 | P-Flow | 72 | 62 | 1.0652 | 0.5492 | stalled |
| r200_n2_mri4/sample_000 | DLR D-Flow | 311 | 298 | 30.2960 | 25.9211 | stalled |
| r200_n2_mri4/sample_001 | DLR D-Flow | 489 | 468 | 51.6543 | 49.6906 | stalled |
| r200_n2_mri4/sample_002 | DLR D-Flow | 1000 | 970 | 3.1563 | 2.4308 | step_limit |
| r200_n2_mri4/sample_003 | DLR D-Flow | 374 | 362 | 59.6590 | 59.2959 | stalled |
| r200_n2_mri4/sample_004 | DLR D-Flow | 184 | 178 | 61.4158 | 60.3855 | stalled |
| r200_n2_mri4/sample_005 | DLR D-Flow | 50 | 48 | 48.0467 | 44.0043 | stalled |
| r200_n2_mri4/sample_006 | DLR D-Flow | 1000 | 968 | 68.0863 | 66.8382 | step_limit |
| r200_n2_mri4/sample_007 | DLR D-Flow | 120 | 114 | 60.9554 | 59.7251 | stalled |
| r200_n2_mri4/sample_008 | DLR D-Flow | 457 | 445 | 2.5306 | 2.1620 | stalled |
| r200_n2_mri4/sample_009 | DLR D-Flow | 658 | 625 | 3.1011 | 2.2874 | stalled |
| r200_n2_mri4/sample_010 | DLR D-Flow | 63 | 57 | 46.6290 | 43.1037 | stalled |
| r200_n2_mri4/sample_011 | DLR D-Flow | 1000 | 955 | 3.9419 | 3.3565 | step_limit |
| r200_n2_mri4/sample_012 | DLR D-Flow | 207 | 202 | 51.8295 | 51.5943 | stalled |
| r200_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.9146 | 0.3895 | step_limit |
| r200_n4_mri4/sample_001 | D-Flow | 420 | 417 | 48.3633 | 46.2606 | stalled |
| r200_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.9562 | 0.4457 | step_limit |
| r200_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.1534 | 0.5876 | step_limit |
| r200_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.2223 | 0.7257 | step_limit |
| r200_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.8684 | 0.3734 | step_limit |
| r200_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.7801 | 0.1408 | step_limit |
| r200_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.7593 | 0.2176 | step_limit |
| r200_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.0250 | 0.5812 | step_limit |
| r200_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.1094 | 0.5289 | step_limit |
| r200_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.7810 | 0.2717 | step_limit |
| r200_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.0341 | 0.4407 | step_limit |
| r200_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 4.2400 | 4.1800 | step_limit |
| r200_n4_mri4/sample_013 | D-Flow | 821 | 821 | 0.7744 | 0.2483 | paused |
| r200_n4_mri4/sample_000 | P-Flow | 405 | 361 | 1.2232 | 0.2617 | stalled |
| r200_n4_mri4/sample_001 | P-Flow | 523 | 453 | 1.3930 | 0.4746 | stalled |
| r200_n4_mri4/sample_002 | P-Flow | 530 | 462 | 1.1449 | 0.3212 | stalled |
| r200_n4_mri4/sample_003 | P-Flow | 655 | 583 | 1.4526 | 0.1449 | stalled |
| r200_n4_mri4/sample_004 | P-Flow | 163 | 137 | 1.0490 | 0.5072 | stalled |
| r200_n4_mri4/sample_005 | P-Flow | 466 | 408 | 1.1818 | 0.3249 | stalled |
| r200_n4_mri4/sample_006 | P-Flow | 440 | 385 | 1.3674 | 0.3826 | stalled |
| r200_n4_mri4/sample_007 | P-Flow | 341 | 294 | 1.1966 | 0.3825 | stalled |
| r200_n4_mri4/sample_008 | P-Flow | 428 | 375 | 1.1845 | 0.3371 | stalled |
| r200_n4_mri4/sample_009 | P-Flow | 742 | 648 | 1.4705 | 0.1807 | stalled |
| r200_n4_mri4/sample_010 | P-Flow | 490 | 430 | 1.2726 | 0.2405 | stalled |
| r200_n4_mri4/sample_011 | P-Flow | 1000 | 873 | 1.6843 | 0.1852 | step_limit |
| r200_n4_mri4/sample_012 | P-Flow | 183 | 159 | 1.3105 | 0.6613 | stalled |
| r200_n4_mri4/sample_013 | P-Flow | 629 | 541 | 1.3463 | 0.2350 | stalled |
| r200_n4_mri4/sample_000 | DLR D-Flow | 305 | 289 | 18.0703 | 14.2474 | stalled |
| r200_n4_mri4/sample_001 | DLR D-Flow | 1000 | 971 | 33.0978 | 28.2805 | step_limit |
| r200_n4_mri4/sample_002 | DLR D-Flow | 300 | 287 | 57.6747 | 55.9668 | stalled |
| r200_n4_mri4/sample_003 | DLR D-Flow | 1000 | 958 | 22.7052 | 19.9284 | step_limit |
| r200_n4_mri4/sample_004 | DLR D-Flow | 868 | 834 | 66.4000 | 62.5081 | stalled |
| r200_n4_mri4/sample_005 | DLR D-Flow | 865 | 830 | 25.1765 | 21.6593 | stalled |
| r200_n4_mri4/sample_006 | DLR D-Flow | 177 | 168 | 40.9010 | 39.4355 | stalled |
| r200_n4_mri4/sample_007 | DLR D-Flow | 1000 | 963 | 61.4572 | 53.7511 | step_limit |
| r200_n4_mri4/sample_008 | DLR D-Flow | 1000 | 958 | 26.5464 | 23.2037 | step_limit |
| r200_n4_mri4/sample_009 | DLR D-Flow | 57 | 53 | 61.4308 | 59.3700 | stalled |
| r200_n4_mri4/sample_010 | DLR D-Flow | 1000 | 964 | 49.4142 | 46.7239 | step_limit |
| r200_n4_mri4/sample_011 | DLR D-Flow | 1000 | 951 | 67.7167 | 65.4068 | step_limit |
| r200_n4_mri4/sample_012 | DLR D-Flow | 192 | 188 | 63.3892 | 60.9420 | stalled |
| r200_n4_mri4/sample_013 | DLR D-Flow | 1000 | 958 | 55.8402 | 50.4270 | step_limit |
| r250_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.1028 | 0.0238 | step_limit |
| r250_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 0.9935 | 0.0376 | step_limit |
| r250_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.0368 | 0.0297 | step_limit |
| r250_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.1778 | 0.0755 | step_limit |
| r250_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.0157 | 0.0236 | step_limit |
| r250_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 0.9616 | 0.0422 | step_limit |
| r250_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.0242 | 0.0303 | step_limit |
| r250_n1_ct10/sample_007 | D-Flow | 21 | 17 | 99.7809 | 97.3586 | stalled |
| r250_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 1.2065 | 0.0281 | step_limit |
| r250_n1_ct10/sample_009 | D-Flow | 69 | 66 | 1.1987 | 0.1610 | stalled |
| r250_n1_ct10/sample_000 | P-Flow | 322 | 259 | 1.2654 | 0.1588 | stalled |
| r250_n1_ct10/sample_001 | P-Flow | 291 | 234 | 1.7511 | 0.2965 | stalled |
| r250_n1_ct10/sample_002 | P-Flow | 24 | 20 | 3.0155 | 1.5610 | stalled |
| r250_n1_ct10/sample_003 | P-Flow | 186 | 156 | 2.7655 | 0.6173 | stalled |
| r250_n1_ct10/sample_004 | P-Flow | 175 | 142 | 1.1457 | 0.1273 | stalled |
| r250_n1_ct10/sample_005 | P-Flow | 6 | 4 | 112.0409 | 95.7050 | stalled |
| r250_n1_ct10/sample_006 | P-Flow | 83 | 68 | 2.9836 | 0.5710 | stalled |
| r250_n1_ct10/sample_007 | P-Flow | 61 | 48 | 3.4102 | 3.3791 | stalled |
| r250_n1_ct10/sample_008 | P-Flow | 142 | 118 | 1.5963 | 0.2497 | stalled |
| r250_n1_ct10/sample_009 | P-Flow | 179 | 139 | 1.8534 | 0.5649 | stalled |
| r250_n1_ct10/sample_000 | DLR D-Flow | 281 | 272 | 2.2877 | 0.4087 | stalled |
| r250_n1_ct10/sample_001 | DLR D-Flow | 217 | 206 | 2.9440 | 0.3925 | stalled |
| r250_n1_ct10/sample_002 | DLR D-Flow | 455 | 436 | 1.2254 | 0.1430 | stalled |
| r250_n1_ct10/sample_003 | DLR D-Flow | 137 | 128 | 3.7858 | 0.4030 | stalled |
| r250_n1_ct10/sample_004 | DLR D-Flow | 484 | 462 | 1.3006 | 0.1410 | stalled |
| r250_n1_ct10/sample_005 | DLR D-Flow | 403 | 383 | 1.1815 | 0.1510 | stalled |
| r250_n1_ct10/sample_006 | DLR D-Flow | 879 | 854 | 1.3469 | 0.1856 | stalled |
| r250_n1_ct10/sample_007 | DLR D-Flow | 21 | 19 | 99.6341 | 96.9357 | stalled |
| r250_n1_ct10/sample_008 | DLR D-Flow | 420 | 399 | 1.5959 | 0.2346 | stalled |
| r250_n1_ct10/sample_009 | DLR D-Flow | 1000 | 913 | 1.6968 | 0.3637 | step_limit |
| r250_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 0.9282 | 0.0548 | step_limit |
| r250_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 0.8856 | 0.0259 | step_limit |
| r250_n2_ct10/sample_002 | D-Flow | 826 | 824 | 0.9292 | 0.0118 | stalled |
| r250_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 0.9299 | 0.0298 | step_limit |
| r250_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 0.9245 | 0.0707 | step_limit |
| r250_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 0.8617 | 0.0184 | step_limit |
| r250_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.0064 | 0.0773 | step_limit |
| r250_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.5151 | 0.0457 | step_limit |
| r250_n2_ct10/sample_008 | D-Flow | 872 | 870 | 0.8634 | 0.0103 | stalled |
| r250_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 0.8544 | 0.0209 | step_limit |
| r250_n2_ct10/sample_000 | P-Flow | 71 | 62 | 1.3548 | 0.1896 | stalled |
| r250_n2_ct10/sample_001 | P-Flow | 123 | 109 | 1.1947 | 0.1424 | stalled |
| r250_n2_ct10/sample_002 | P-Flow | 219 | 176 | 1.3936 | 0.1638 | stalled |
| r250_n2_ct10/sample_003 | P-Flow | 92 | 80 | 1.3119 | 0.1744 | stalled |
| r250_n2_ct10/sample_004 | P-Flow | 1000 | 801 | 1.7257 | 0.3420 | step_limit |
| r250_n2_ct10/sample_005 | P-Flow | 205 | 181 | 1.0838 | 0.1241 | stalled |
| r250_n2_ct10/sample_006 | P-Flow | 1000 | 888 | 1.2813 | 0.1365 | step_limit |
| r250_n2_ct10/sample_007 | P-Flow | 1000 | 987 | 1.6270 | 0.2772 | step_limit |
| r250_n2_ct10/sample_008 | P-Flow | 107 | 95 | 1.2276 | 0.1584 | stalled |
| r250_n2_ct10/sample_009 | P-Flow | 1000 | 966 | 1.0924 | 0.1126 | step_limit |
| r250_n2_ct10/sample_000 | DLR D-Flow | 114 | 111 | 6.2358 | 2.8287 | stalled |
| r250_n2_ct10/sample_001 | DLR D-Flow | 563 | 541 | 1.6212 | 0.4599 | stalled |
| r250_n2_ct10/sample_002 | DLR D-Flow | 643 | 628 | 2.5011 | 0.7744 | stalled |
| r250_n2_ct10/sample_003 | DLR D-Flow | 952 | 920 | 1.4170 | 0.2339 | stalled |
| r250_n2_ct10/sample_004 | DLR D-Flow | 248 | 240 | 1.9368 | 0.3185 | stalled |
| r250_n2_ct10/sample_005 | DLR D-Flow | 347 | 335 | 7.0755 | 3.4548 | stalled |
| r250_n2_ct10/sample_006 | DLR D-Flow | 485 | 470 | 2.0236 | 0.2978 | stalled |
| r250_n2_ct10/sample_007 | DLR D-Flow | 778 | 747 | 40.8523 | 17.5301 | stalled |
| r250_n2_ct10/sample_008 | DLR D-Flow | 534 | 514 | 3.5071 | 0.7004 | stalled |
| r250_n2_ct10/sample_009 | DLR D-Flow | 544 | 521 | 1.2159 | 0.1597 | stalled |
| r250_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 0.8496 | 0.0198 | step_limit |
| r250_n4_ct10/sample_001 | D-Flow | 797 | 795 | 0.8349 | 0.0175 | stalled |
| r250_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 1.1053 | 0.1001 | step_limit |
| r250_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 0.8951 | 0.0153 | step_limit |
| r250_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.0439 | 0.0879 | step_limit |
| r250_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 0.8553 | 0.0410 | step_limit |
| r250_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 0.8171 | 0.0098 | step_limit |
| r250_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 0.8709 | 0.0201 | step_limit |
| r250_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 0.8855 | 0.0475 | step_limit |
| r250_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 0.9418 | 0.0650 | step_limit |
| r250_n4_ct10/sample_000 | P-Flow | 1000 | 822 | 1.8274 | 0.2847 | step_limit |
| r250_n4_ct10/sample_001 | P-Flow | 881 | 864 | 1.1086 | 0.1230 | stalled |
| r250_n4_ct10/sample_002 | P-Flow | 1000 | 981 | 1.2049 | 0.1275 | step_limit |
| r250_n4_ct10/sample_003 | P-Flow | 1000 | 924 | 1.1072 | 0.0976 | step_limit |
| r250_n4_ct10/sample_004 | P-Flow | 114 | 100 | 1.2987 | 0.1604 | stalled |
| r250_n4_ct10/sample_005 | P-Flow | 1000 | 981 | 1.0491 | 0.0989 | step_limit |
| r250_n4_ct10/sample_006 | P-Flow | 758 | 731 | 1.1068 | 0.1239 | stalled |
| r250_n4_ct10/sample_007 | P-Flow | 143 | 122 | 1.2039 | 0.1440 | stalled |
| r250_n4_ct10/sample_008 | P-Flow | 1000 | 967 | 1.1289 | 0.1138 | step_limit |
| r250_n4_ct10/sample_009 | P-Flow | 1000 | 981 | 1.0985 | 0.0825 | step_limit |
| r250_n4_ct10/sample_000 | DLR D-Flow | 1000 | 963 | 41.2162 | 14.9658 | step_limit |
| r250_n4_ct10/sample_001 | DLR D-Flow | 920 | 888 | 48.8803 | 25.0409 | stalled |
| r250_n4_ct10/sample_002 | DLR D-Flow | 1000 | 963 | 27.6794 | 8.2113 | step_limit |
| r250_n4_ct10/sample_003 | DLR D-Flow | 840 | 793 | 19.8968 | 5.0078 | stalled |
| r250_n4_ct10/sample_004 | DLR D-Flow | 660 | 633 | 55.2649 | 29.1971 | stalled |
| r250_n4_ct10/sample_005 | DLR D-Flow | 345 | 333 | 36.1420 | 14.3720 | stalled |
| r250_n4_ct10/sample_006 | DLR D-Flow | 533 | 512 | 39.5365 | 14.1200 | stalled |
| r250_n4_ct10/sample_007 | DLR D-Flow | 693 | 670 | 24.1299 | 6.7687 | stalled |
| r250_n4_ct10/sample_008 | DLR D-Flow | 194 | 183 | 85.8171 | 42.0052 | stalled |
| r250_n4_ct10/sample_009 | DLR D-Flow | 1000 | 968 | 64.6413 | 29.4779 | step_limit |
| r250_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.7352 | 0.2610 | step_limit |
| r250_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.6785 | 0.1244 | step_limit |
| r250_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.8083 | 0.3191 | step_limit |
| r250_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.7404 | 0.2826 | step_limit |
| r250_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.7440 | 0.2702 | step_limit |
| r250_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.7796 | 0.2956 | step_limit |
| r250_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.7245 | 0.2176 | step_limit |
| r250_n1_mri4/sample_007 | D-Flow | 29 | 26 | 99.6306 | 99.5894 | stalled |
| r250_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.0397 | 0.4234 | step_limit |
| r250_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.7341 | 0.2684 | step_limit |
| r250_n1_mri4/sample_000 | P-Flow | 49 | 41 | 1.0478 | 0.5517 | stalled |
| r250_n1_mri4/sample_001 | P-Flow | 138 | 121 | 0.9978 | 0.4707 | stalled |
| r250_n1_mri4/sample_002 | P-Flow | 28 | 23 | 1.6195 | 1.1540 | stalled |
| r250_n1_mri4/sample_003 | P-Flow | 303 | 251 | 0.9809 | 0.4772 | stalled |
| r250_n1_mri4/sample_004 | P-Flow | 76 | 59 | 1.0406 | 0.6163 | stalled |
| r250_n1_mri4/sample_005 | P-Flow | 49 | 40 | 1.1071 | 0.5918 | stalled |
| r250_n1_mri4/sample_006 | P-Flow | 108 | 90 | 0.9673 | 0.4992 | stalled |
| r250_n1_mri4/sample_007 | P-Flow | 98 | 83 | 1.8966 | 0.9458 | stalled |
| r250_n1_mri4/sample_008 | P-Flow | 32 | 27 | 1.4783 | 0.8029 | stalled |
| r250_n1_mri4/sample_009 | P-Flow | 20 | 17 | 1.5253 | 1.1311 | stalled |
| r250_n1_mri4/sample_000 | DLR D-Flow | 339 | 323 | 1.1340 | 0.6169 | stalled |
| r250_n1_mri4/sample_001 | DLR D-Flow | 294 | 280 | 1.2070 | 0.7061 | stalled |
| r250_n1_mri4/sample_002 | DLR D-Flow | 240 | 231 | 1.2560 | 0.7038 | stalled |
| r250_n1_mri4/sample_003 | DLR D-Flow | 298 | 287 | 1.5829 | 0.9498 | stalled |
| r250_n1_mri4/sample_004 | DLR D-Flow | 242 | 232 | 1.1951 | 0.7172 | stalled |
| r250_n1_mri4/sample_005 | DLR D-Flow | 197 | 186 | 31.6593 | 27.6604 | stalled |
| r250_n1_mri4/sample_006 | DLR D-Flow | 46 | 43 | 1.0740 | 0.5975 | stalled |
| r250_n1_mri4/sample_007 | DLR D-Flow | 19 | 17 | 99.4729 | 99.4185 | stalled |
| r250_n1_mri4/sample_008 | DLR D-Flow | 676 | 661 | 1.8768 | 1.0572 | stalled |
| r250_n1_mri4/sample_009 | DLR D-Flow | 324 | 311 | 1.3578 | 0.8314 | stalled |
| r250_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.6517 | 0.2255 | step_limit |
| r250_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.7840 | 0.3548 | step_limit |
| r250_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 39.1974 | 37.0715 | step_limit |
| r250_n2_mri4/sample_003 | D-Flow | 904 | 902 | 33.4972 | 32.2600 | stalled |
| r250_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 35.7826 | 34.6006 | step_limit |
| r250_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.6609 | 0.2552 | step_limit |
| r250_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.7267 | 0.2821 | step_limit |
| r250_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.0695 | 0.3547 | step_limit |
| r250_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.6307 | 0.2267 | step_limit |
| r250_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.7436 | 0.3337 | step_limit |
| r250_n2_mri4/sample_000 | P-Flow | 338 | 298 | 1.0246 | 0.4168 | stalled |
| r250_n2_mri4/sample_001 | P-Flow | 152 | 132 | 0.9471 | 0.4518 | stalled |
| r250_n2_mri4/sample_002 | P-Flow | 344 | 298 | 1.1075 | 0.4482 | stalled |
| r250_n2_mri4/sample_003 | P-Flow | 159 | 141 | 1.0302 | 0.4558 | stalled |
| r250_n2_mri4/sample_004 | P-Flow | 147 | 126 | 0.9027 | 0.4456 | stalled |
| r250_n2_mri4/sample_005 | P-Flow | 204 | 179 | 0.9364 | 0.4951 | stalled |
| r250_n2_mri4/sample_006 | P-Flow | 129 | 109 | 0.9577 | 0.4790 | stalled |
| r250_n2_mri4/sample_007 | P-Flow | 110 | 95 | 1.5302 | 0.7342 | stalled |
| r250_n2_mri4/sample_008 | P-Flow | 77 | 65 | 0.9598 | 0.5958 | stalled |
| r250_n2_mri4/sample_009 | P-Flow | 221 | 193 | 0.9281 | 0.4370 | stalled |
| r250_n2_mri4/sample_000 | DLR D-Flow | 1000 | 979 | 7.2410 | 4.8563 | step_limit |
| r250_n2_mri4/sample_001 | DLR D-Flow | 477 | 457 | 1.4985 | 0.9421 | stalled |
| r250_n2_mri4/sample_002 | DLR D-Flow | 302 | 292 | 44.8905 | 43.6465 | stalled |
| r250_n2_mri4/sample_003 | DLR D-Flow | 373 | 359 | 48.3885 | 48.0501 | stalled |
| r250_n2_mri4/sample_004 | DLR D-Flow | 269 | 262 | 3.1408 | 2.2239 | stalled |
| r250_n2_mri4/sample_005 | DLR D-Flow | 217 | 213 | 16.5504 | 15.7696 | stalled |
| r250_n2_mri4/sample_006 | DLR D-Flow | 121 | 116 | 8.1506 | 7.5605 | stalled |
| r250_n2_mri4/sample_007 | DLR D-Flow | 54 | 50 | 74.4663 | 71.1116 | stalled |
| r250_n2_mri4/sample_008 | DLR D-Flow | 234 | 226 | 3.1269 | 2.6771 | stalled |
| r250_n2_mri4/sample_009 | DLR D-Flow | 219 | 212 | 1.5835 | 1.1800 | stalled |
| r250_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 0.7993 | 0.3978 | step_limit |
| r250_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 0.6963 | 0.2646 | step_limit |
| r250_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 0.8978 | 0.4273 | step_limit |
| r250_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 0.6446 | 0.1423 | step_limit |
| r250_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.9328 | 0.4550 | step_limit |
| r250_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 0.7728 | 0.3461 | step_limit |
| r250_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 0.6363 | 0.1468 | step_limit |
| r250_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 0.7893 | 0.3257 | step_limit |
| r250_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 0.8034 | 0.3621 | step_limit |
| r250_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 0.7100 | 0.2262 | step_limit |
| r250_n4_mri4/sample_000 | P-Flow | 143 | 130 | 0.9359 | 0.4674 | stalled |
| r250_n4_mri4/sample_001 | P-Flow | 284 | 251 | 0.9774 | 0.3691 | stalled |
| r250_n4_mri4/sample_002 | P-Flow | 256 | 218 | 1.1356 | 0.4831 | stalled |
| r250_n4_mri4/sample_003 | P-Flow | 148 | 129 | 1.0196 | 0.4867 | stalled |
| r250_n4_mri4/sample_004 | P-Flow | 242 | 215 | 1.1929 | 0.4153 | stalled |
| r250_n4_mri4/sample_005 | P-Flow | 121 | 107 | 0.9657 | 0.4626 | stalled |
| r250_n4_mri4/sample_006 | P-Flow | 183 | 164 | 0.9581 | 0.3847 | stalled |
| r250_n4_mri4/sample_007 | P-Flow | 283 | 245 | 1.0382 | 0.4310 | stalled |
| r250_n4_mri4/sample_008 | P-Flow | 156 | 140 | 1.0356 | 0.3873 | stalled |
| r250_n4_mri4/sample_009 | P-Flow | 115 | 100 | 0.9841 | 0.4663 | stalled |
| r250_n4_mri4/sample_000 | DLR D-Flow | 1000 | 963 | 55.1567 | 51.3245 | step_limit |
| r250_n4_mri4/sample_001 | DLR D-Flow | 1000 | 965 | 19.3784 | 14.8411 | step_limit |
| r250_n4_mri4/sample_002 | DLR D-Flow | 1000 | 962 | 50.4465 | 48.5728 | step_limit |
| r250_n4_mri4/sample_003 | DLR D-Flow | 313 | 299 | 44.6435 | 43.3714 | stalled |
| r250_n4_mri4/sample_004 | DLR D-Flow | 383 | 361 | 67.4916 | 64.7791 | stalled |
| r250_n4_mri4/sample_005 | DLR D-Flow | 294 | 283 | 56.9564 | 51.2844 | stalled |
| r250_n4_mri4/sample_006 | DLR D-Flow | 1000 | 970 | 54.0020 | 49.9033 | step_limit |
| r250_n4_mri4/sample_007 | DLR D-Flow | 296 | 288 | 20.5883 | 18.0406 | stalled |
| r250_n4_mri4/sample_008 | DLR D-Flow | 1000 | 968 | 73.9701 | 71.8736 | step_limit |
| r250_n4_mri4/sample_009 | DLR D-Flow | 793 | 765 | 68.6062 | 66.9540 | stalled |

Terminal method runs: 1310/3240. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.
