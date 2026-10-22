# Five-step Euler pilot

10% relative measurement noise; CT10 and Cartesian MRI4; 30 test samples per acquisition at each resolution; circle count in [1, 2, 4]. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 8, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_n1_ct10/sample_000 | D-Flow | 935 | 933 | 67.8776 | 0.5256 | stalled |
| r64_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 47.1661 | 1.3445 | step_limit |
| r64_n1_ct10/sample_002 | D-Flow | 518 | 515 | 51.3289 | 1.2281 | stalled |
| r64_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 56.7354 | 0.3963 | step_limit |
| r64_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 15.1087 | 8.6157 | step_limit |
| r64_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 61.6858 | 0.8977 | step_limit |
| r64_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 61.7759 | 1.0488 | step_limit |
| r64_n1_ct10/sample_007 | D-Flow | 742 | 740 | 80.8274 | 0.8876 | stalled |
| r64_n1_ct10/sample_008 | D-Flow | 726 | 723 | 66.6809 | 0.3077 | stalled |
| r64_n1_ct10/sample_009 | D-Flow | 864 | 862 | 54.3943 | 1.3846 | stalled |
| r64_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 62.7250 | 0.5921 | step_limit |
| r64_n1_ct10/sample_011 | D-Flow | 518 | 515 | 8.8465 | 7.9795 | stalled |
| r64_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 64.1128 | 0.6432 | step_limit |
| r64_n1_ct10/sample_013 | D-Flow | 587 | 585 | 52.5462 | 0.6184 | stalled |
| r64_n1_ct10/sample_014 | D-Flow | 820 | 818 | 10.2624 | 7.3489 | stalled |
| r64_n1_ct10/sample_015 | D-Flow | 1000 | 1000 | 5.1222 | 10.5555 | step_limit |
| r64_n1_ct10/sample_016 | D-Flow | 873 | 870 | 58.1600 | 0.8667 | stalled |
| r64_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 54.7718 | 0.7368 | step_limit |
| r64_n1_ct10/sample_018 | D-Flow | 871 | 869 | 63.2889 | 0.9714 | stalled |
| r64_n1_ct10/sample_019 | D-Flow | 737 | 734 | 87.0405 | 0.6162 | stalled |
| r64_n1_ct10/sample_020 | D-Flow | 456 | 454 | 8.2053 | 7.7200 | stalled |
| r64_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 35.7533 | 1.9142 | step_limit |
| r64_n1_ct10/sample_022 | D-Flow | 803 | 801 | 101.4111 | 0.4606 | stalled |
| r64_n1_ct10/sample_023 | D-Flow | 691 | 689 | 85.1051 | 0.6250 | stalled |
| r64_n1_ct10/sample_024 | D-Flow | 624 | 621 | 88.5360 | 0.8921 | stalled |
| r64_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 56.2045 | 1.1480 | step_limit |
| r64_n1_ct10/sample_026 | D-Flow | 466 | 464 | 76.2843 | 1.1218 | stalled |
| r64_n1_ct10/sample_027 | D-Flow | 722 | 720 | 7.8564 | 7.8996 | stalled |
| r64_n1_ct10/sample_028 | D-Flow | 784 | 782 | 55.5198 | 0.8524 | stalled |
| r64_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 65.5562 | 0.6344 | step_limit |
| r64_n1_ct10/sample_000 | P-Flow | 352 | 287 | 22.6033 | 13.7597 | stalled |
| r64_n1_ct10/sample_001 | P-Flow | 95 | 81 | 4.0465 | 9.9889 | stalled |
| r64_n1_ct10/sample_002 | P-Flow | 285 | 235 | 3.0714 | 9.8304 | stalled |
| r64_n1_ct10/sample_003 | P-Flow | 143 | 129 | 4.3152 | 9.7211 | stalled |
| r64_n1_ct10/sample_004 | P-Flow | 1000 | 981 | 52.6713 | 0.6127 | step_limit |
| r64_n1_ct10/sample_005 | P-Flow | 55 | 48 | 5.5214 | 9.9911 | stalled |
| r64_n1_ct10/sample_006 | P-Flow | 469 | 458 | 61.2124 | 1.0439 | stalled |
| r64_n1_ct10/sample_007 | P-Flow | 1000 | 928 | 72.0046 | 0.8876 | step_limit |
| r64_n1_ct10/sample_008 | P-Flow | 52 | 44 | 4.0708 | 10.0399 | stalled |
| r64_n1_ct10/sample_009 | P-Flow | 1000 | 911 | 45.0652 | 1.4418 | step_limit |
| r64_n1_ct10/sample_010 | P-Flow | 78 | 67 | 4.2154 | 10.2089 | stalled |
| r64_n1_ct10/sample_011 | P-Flow | 490 | 477 | 76.1363 | 0.7650 | stalled |
| r64_n1_ct10/sample_012 | P-Flow | 30 | 25 | 3.7278 | 10.0143 | stalled |
| r64_n1_ct10/sample_013 | P-Flow | 1000 | 858 | 38.2139 | 1.0607 | step_limit |
| r64_n1_ct10/sample_014 | P-Flow | 62 | 50 | 7.1297 | 10.3539 | stalled |
| r64_n1_ct10/sample_015 | P-Flow | 1000 | 958 | 42.7306 | 0.8061 | step_limit |
| r64_n1_ct10/sample_016 | P-Flow | 36 | 30 | 37.0200 | 21.4566 | stalled |
| r64_n1_ct10/sample_017 | P-Flow | 198 | 166 | 2.9064 | 9.9231 | stalled |
| r64_n1_ct10/sample_018 | P-Flow | 69 | 58 | 3.1433 | 10.0966 | stalled |
| r64_n1_ct10/sample_019 | P-Flow | 59 | 52 | 3.6843 | 10.0959 | stalled |
| r64_n1_ct10/sample_020 | P-Flow | 30 | 26 | 4.2390 | 9.9433 | stalled |
| r64_n1_ct10/sample_021 | P-Flow | 94 | 80 | 4.0646 | 9.9929 | stalled |
| r64_n1_ct10/sample_022 | P-Flow | 323 | 261 | 9.1279 | 10.7353 | stalled |
| r64_n1_ct10/sample_023 | P-Flow | 583 | 567 | 76.2779 | 0.6250 | stalled |
| r64_n1_ct10/sample_024 | P-Flow | 587 | 568 | 71.6870 | 0.8921 | stalled |
| r64_n1_ct10/sample_025 | P-Flow | 39 | 34 | 2.7892 | 9.8967 | stalled |
| r64_n1_ct10/sample_026 | P-Flow | 1000 | 897 | 47.4264 | 1.2001 | step_limit |
| r64_n1_ct10/sample_027 | P-Flow | 32 | 27 | 4.2485 | 9.9272 | stalled |
| r64_n1_ct10/sample_028 | P-Flow | 16 | 14 | 7.5576 | 10.8977 | stalled |
| r64_n1_ct10/sample_029 | P-Flow | 35 | 30 | 3.0113 | 10.0849 | stalled |
| r64_n1_ct10/sample_000 | DLR D-Flow | 686 | 658 | 8.4102 | 10.8516 | stalled |
| r64_n1_ct10/sample_001 | DLR D-Flow | 167 | 152 | 2.8332 | 9.9240 | stalled |
| r64_n1_ct10/sample_002 | DLR D-Flow | 150 | 148 | 5.7758 | 9.9338 | stalled |
| r64_n1_ct10/sample_003 | DLR D-Flow | 675 | 641 | 3.3428 | 9.9286 | stalled |
| r64_n1_ct10/sample_004 | DLR D-Flow | 1000 | 927 | 101.5167 | 83.7463 | step_limit |
| r64_n1_ct10/sample_005 | DLR D-Flow | 201 | 189 | 4.0957 | 10.0149 | stalled |
| r64_n1_ct10/sample_006 | DLR D-Flow | 156 | 150 | 3.0995 | 10.0003 | stalled |
| r64_n1_ct10/sample_007 | DLR D-Flow | 424 | 402 | 5.0269 | 9.8903 | stalled |
| r64_n1_ct10/sample_008 | DLR D-Flow | 351 | 334 | 2.4768 | 9.9583 | stalled |
| r64_n1_ct10/sample_009 | DLR D-Flow | 635 | 607 | 7.4796 | 9.8437 | stalled |
| r64_n1_ct10/sample_010 | DLR D-Flow | 199 | 187 | 2.8693 | 9.9858 | stalled |
| r64_n1_ct10/sample_011 | DLR D-Flow | 103 | 99 | 8.2476 | 9.8925 | stalled |
| r64_n1_ct10/sample_012 | DLR D-Flow | 122 | 120 | 3.8653 | 9.8624 | stalled |
| r64_n1_ct10/sample_013 | DLR D-Flow | 97 | 92 | 5.5195 | 9.7782 | stalled |
| r64_n1_ct10/sample_014 | DLR D-Flow | 717 | 689 | 4.6566 | 10.1037 | stalled |
| r64_n1_ct10/sample_015 | DLR D-Flow | 126 | 120 | 122.0087 | 99.5890 | stalled |
| r64_n1_ct10/sample_016 | DLR D-Flow | 130 | 127 | 5.4582 | 9.8809 | stalled |
| r64_n1_ct10/sample_017 | DLR D-Flow | 324 | 302 | 2.9200 | 9.9470 | stalled |
| r64_n1_ct10/sample_018 | DLR D-Flow | 162 | 159 | 3.5462 | 9.9734 | stalled |
| r64_n1_ct10/sample_019 | DLR D-Flow | 189 | 182 | 7.2004 | 9.8524 | stalled |
| r64_n1_ct10/sample_020 | DLR D-Flow | 193 | 184 | 2.2367 | 9.8772 | stalled |
| r64_n1_ct10/sample_021 | DLR D-Flow | 20 | 18 | 113.8509 | 94.3110 | stalled |
| r64_n1_ct10/sample_022 | DLR D-Flow | 228 | 217 | 2.9728 | 9.8351 | stalled |
| r64_n1_ct10/sample_023 | DLR D-Flow | 173 | 167 | 6.3722 | 9.7543 | stalled |
| r64_n1_ct10/sample_024 | DLR D-Flow | 383 | 367 | 3.9849 | 9.8983 | stalled |
| r64_n1_ct10/sample_025 | DLR D-Flow | 110 | 104 | 9.6562 | 9.8504 | stalled |
| r64_n1_ct10/sample_026 | DLR D-Flow | 220 | 208 | 2.6770 | 10.0021 | stalled |
| r64_n1_ct10/sample_027 | DLR D-Flow | 342 | 330 | 2.5150 | 9.8868 | stalled |
| r64_n1_ct10/sample_028 | DLR D-Flow | 242 | 237 | 5.3383 | 9.9464 | stalled |
| r64_n1_ct10/sample_029 | DLR D-Flow | 129 | 123 | 2.1644 | 9.9620 | stalled |
| r64_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 60.0310 | 0.8550 | step_limit |
| r64_n2_ct10/sample_001 | D-Flow | 818 | 815 | 80.5834 | 0.7193 | stalled |
| r64_n2_ct10/sample_002 | D-Flow | 909 | 907 | 55.1079 | 0.6359 | stalled |
| r64_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 65.7096 | 0.7716 | step_limit |
| r64_n2_ct10/sample_004 | D-Flow | 320 | 316 | 71.0412 | 47.5212 | stalled |
| r64_n2_ct10/sample_005 | D-Flow | 611 | 609 | 82.8531 | 0.8310 | stalled |
| r64_n2_ct10/sample_006 | D-Flow | 740 | 738 | 61.6670 | 1.0068 | stalled |
| r64_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 48.7090 | 1.7666 | step_limit |
| r64_n2_ct10/sample_008 | D-Flow | 609 | 607 | 76.6395 | 0.5883 | stalled |
| r64_n2_ct10/sample_009 | D-Flow | 815 | 813 | 72.2102 | 0.4545 | stalled |
| r64_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 66.5357 | 1.1731 | step_limit |
| r64_n2_ct10/sample_000 | P-Flow | 66 | 58 | 3.6640 | 9.9160 | stalled |
| r64_n2_ct10/sample_001 | P-Flow | 678 | 665 | 70.3905 | 0.7193 | stalled |
| r64_n2_ct10/sample_002 | P-Flow | 1000 | 871 | 46.1719 | 0.7415 | step_limit |
| r64_n2_ct10/sample_003 | P-Flow | 10 | 8 | 112.1927 | 70.2852 | stalled |
| r64_n2_ct10/sample_004 | P-Flow | 26 | 21 | 52.9653 | 27.8266 | stalled |
| r64_n2_ct10/sample_005 | P-Flow | 23 | 19 | 72.7053 | 41.4626 | stalled |
| r64_n2_ct10/sample_006 | P-Flow | 51 | 43 | 5.1947 | 9.8757 | stalled |
| r64_n2_ct10/sample_007 | P-Flow | 57 | 48 | 4.9588 | 9.9976 | stalled |
| r64_n2_ct10/sample_008 | P-Flow | 956 | 868 | 62.6970 | 0.5883 | stalled |
| r64_n2_ct10/sample_009 | P-Flow | 742 | 709 | 70.7834 | 0.4545 | stalled |
| r64_n2_ct10/sample_010 | P-Flow | 1000 | 913 | 62.9871 | 1.1729 | step_limit |
| r64_n2_ct10/sample_000 | DLR D-Flow | 930 | 893 | 43.4409 | 27.7717 | stalled |
| r64_n2_ct10/sample_001 | DLR D-Flow | 230 | 226 | 6.9118 | 10.0153 | stalled |
| r64_n2_ct10/sample_002 | DLR D-Flow | 885 | 831 | 3.4020 | 9.8756 | stalled |
| r64_n2_ct10/sample_003 | DLR D-Flow | 1000 | 974 | 65.8511 | 36.0484 | step_limit |
| r64_n2_ct10/sample_004 | DLR D-Flow | 356 | 340 | 85.5306 | 55.4510 | stalled |
| r64_n2_ct10/sample_005 | DLR D-Flow | 195 | 192 | 5.7921 | 10.0256 | stalled |
| r64_n2_ct10/sample_006 | DLR D-Flow | 258 | 247 | 73.0164 | 57.0093 | stalled |
| r64_n2_ct10/sample_007 | DLR D-Flow | 342 | 330 | 77.5208 | 64.0208 | stalled |
| r64_n2_ct10/sample_008 | DLR D-Flow | 307 | 297 | 75.2697 | 51.0394 | stalled |
| r64_n2_ct10/sample_009 | DLR D-Flow | 230 | 223 | 3.1543 | 9.9594 | stalled |
| r64_n2_ct10/sample_010 | DLR D-Flow | 437 | 422 | 4.1960 | 9.8834 | stalled |
| r64_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 80.3252 | 1.1404 | step_limit |
| r64_n4_ct10/sample_001 | D-Flow | 894 | 892 | 72.1318 | 0.8624 | stalled |
| r64_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 35.4723 | 1.2173 | step_limit |
| r64_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 56.3961 | 0.9497 | step_limit |
| r64_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 72.9061 | 1.0037 | step_limit |
| r64_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 64.7237 | 0.9208 | step_limit |
| r64_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 44.7659 | 2.0157 | step_limit |
| r64_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 55.2365 | 1.2861 | step_limit |
| r64_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 49.5051 | 1.4655 | step_limit |
| r64_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 66.2507 | 1.2173 | step_limit |
| r64_n4_ct10/sample_010 | D-Flow | 630 | 628 | 82.2042 | 1.3940 | stalled |
| r64_n4_ct10/sample_000 | P-Flow | 1000 | 910 | 46.7849 | 1.6187 | step_limit |
| r64_n4_ct10/sample_001 | P-Flow | 1000 | 898 | 56.7985 | 0.9673 | step_limit |
| r64_n4_ct10/sample_002 | P-Flow | 132 | 120 | 8.8024 | 9.0063 | stalled |
| r64_n4_ct10/sample_003 | P-Flow | 14 | 12 | 58.8461 | 30.7787 | stalled |
| r64_n4_ct10/sample_004 | P-Flow | 26 | 22 | 42.0528 | 20.4597 | stalled |
| r64_n4_ct10/sample_005 | P-Flow | 1000 | 876 | 37.6874 | 1.5248 | step_limit |
| r64_n4_ct10/sample_006 | P-Flow | 230 | 189 | 7.5275 | 9.8266 | stalled |
| r64_n4_ct10/sample_007 | P-Flow | 576 | 554 | 57.8663 | 1.2244 | stalled |
| r64_n4_ct10/sample_008 | P-Flow | 1000 | 894 | 42.6144 | 1.4478 | step_limit |
| r64_n4_ct10/sample_009 | P-Flow | 16 | 13 | 57.4375 | 29.7388 | stalled |
| r64_n4_ct10/sample_010 | P-Flow | 826 | 782 | 60.4939 | 1.3940 | stalled |
| r64_n4_ct10/sample_000 | DLR D-Flow | 163 | 158 | 24.6930 | 13.4385 | stalled |
| r64_n4_ct10/sample_001 | DLR D-Flow | 157 | 151 | 27.8130 | 15.0605 | stalled |
| r64_n4_ct10/sample_002 | DLR D-Flow | 266 | 255 | 84.9214 | 48.2885 | stalled |
| r64_n4_ct10/sample_003 | DLR D-Flow | 118 | 115 | 21.2037 | 12.8406 | stalled |
| r64_n4_ct10/sample_004 | DLR D-Flow | 765 | 739 | 13.4795 | 11.0078 | stalled |
| r64_n4_ct10/sample_005 | DLR D-Flow | 248 | 236 | 76.6562 | 46.1257 | stalled |
| r64_n4_ct10/sample_006 | DLR D-Flow | 1000 | 971 | 57.9171 | 29.0692 | step_limit |
| r64_n4_ct10/sample_007 | DLR D-Flow | 286 | 277 | 89.3501 | 51.3189 | stalled |
| r64_n4_ct10/sample_008 | DLR D-Flow | 925 | 901 | 80.2526 | 43.1309 | stalled |
| r64_n4_ct10/sample_009 | DLR D-Flow | 1000 | 964 | 34.7664 | 17.7512 | step_limit |
| r64_n4_ct10/sample_010 | DLR D-Flow | 242 | 233 | 14.4432 | 11.0760 | stalled |
| r64_n1_mri4/sample_000 | D-Flow | 684 | 682 | 6.6062 | 6.3291 | stalled |
| r64_n1_mri4/sample_001 | D-Flow | 519 | 517 | 10.1254 | 5.3927 | stalled |
| r64_n1_mri4/sample_002 | D-Flow | 347 | 345 | 14.8614 | 4.1486 | stalled |
| r64_n1_mri4/sample_003 | D-Flow | 638 | 636 | 6.4270 | 6.3238 | stalled |
| r64_n1_mri4/sample_004 | D-Flow | 249 | 247 | 100.4763 | 95.1889 | stalled |
| r64_n1_mri4/sample_005 | D-Flow | 903 | 900 | 7.2333 | 6.2405 | stalled |
| r64_n1_mri4/sample_006 | D-Flow | 611 | 609 | 7.0604 | 6.5794 | stalled |
| r64_n1_mri4/sample_007 | D-Flow | 667 | 665 | 7.1909 | 6.5346 | stalled |
| r64_n1_mri4/sample_008 | D-Flow | 377 | 375 | 13.9061 | 4.5125 | stalled |
| r64_n1_mri4/sample_009 | D-Flow | 994 | 992 | 6.2008 | 6.4986 | stalled |
| r64_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 7.5734 | 6.3475 | step_limit |
| r64_n1_mri4/sample_011 | D-Flow | 490 | 487 | 7.0131 | 6.6040 | stalled |
| r64_n1_mri4/sample_012 | D-Flow | 448 | 446 | 13.2529 | 5.3036 | stalled |
| r64_n1_mri4/sample_013 | D-Flow | 568 | 566 | 6.8571 | 6.5748 | stalled |
| r64_n1_mri4/sample_014 | D-Flow | 467 | 464 | 12.5409 | 6.3147 | stalled |
| r64_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 7.5942 | 6.1264 | step_limit |
| r64_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 7.0934 | 5.9571 | step_limit |
| r64_n1_mri4/sample_017 | D-Flow | 685 | 682 | 6.2949 | 6.9072 | stalled |
| r64_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 7.2430 | 6.9996 | step_limit |
| r64_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 7.0463 | 7.0165 | step_limit |
| r64_n1_mri4/sample_020 | D-Flow | 474 | 471 | 12.7488 | 5.3614 | stalled |
| r64_n1_mri4/sample_021 | D-Flow | 491 | 489 | 6.5707 | 6.1409 | stalled |
| r64_n1_mri4/sample_022 | D-Flow | 366 | 363 | 15.0547 | 5.5512 | stalled |
| r64_n1_mri4/sample_023 | D-Flow | 517 | 515 | 7.0389 | 6.9985 | stalled |
| r64_n1_mri4/sample_024 | D-Flow | 447 | 444 | 13.7222 | 4.7247 | stalled |
| r64_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 6.8853 | 6.0807 | step_limit |
| r64_n1_mri4/sample_026 | D-Flow | 744 | 742 | 6.5144 | 6.2762 | stalled |
| r64_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 7.0919 | 6.7493 | step_limit |
| r64_n1_mri4/sample_028 | D-Flow | 869 | 866 | 6.4528 | 7.0771 | stalled |
| r64_n1_mri4/sample_029 | D-Flow | 380 | 378 | 13.7577 | 5.1238 | stalled |
| r64_n1_mri4/sample_000 | P-Flow | 193 | 181 | 13.3457 | 4.5391 | stalled |
| r64_n1_mri4/sample_001 | P-Flow | 197 | 185 | 11.9221 | 5.3927 | stalled |
| r64_n1_mri4/sample_002 | P-Flow | 181 | 175 | 13.3258 | 4.1486 | stalled |
| r64_n1_mri4/sample_003 | P-Flow | 58 | 51 | 2.4492 | 10.1696 | stalled |
| r64_n1_mri4/sample_004 | P-Flow | 191 | 181 | 12.1923 | 5.6664 | stalled |
| r64_n1_mri4/sample_005 | P-Flow | 189 | 185 | 12.9680 | 3.9301 | stalled |
| r64_n1_mri4/sample_006 | P-Flow | 196 | 191 | 13.6588 | 5.0198 | stalled |
| r64_n1_mri4/sample_007 | P-Flow | 202 | 196 | 14.4640 | 4.5021 | stalled |
| r64_n1_mri4/sample_008 | P-Flow | 246 | 227 | 14.5153 | 4.5125 | stalled |
| r64_n1_mri4/sample_009 | P-Flow | 189 | 181 | 11.8429 | 5.2751 | stalled |
| r64_n1_mri4/sample_010 | P-Flow | 180 | 174 | 12.9450 | 4.5800 | stalled |
| r64_n1_mri4/sample_011 | P-Flow | 176 | 173 | 15.1534 | 3.6756 | stalled |
| r64_n1_mri4/sample_012 | P-Flow | 215 | 210 | 13.5314 | 5.3036 | stalled |
| r64_n1_mri4/sample_013 | P-Flow | 233 | 203 | 13.1291 | 5.9815 | stalled |
| r64_n1_mri4/sample_014 | P-Flow | 66 | 55 | 2.8564 | 10.1248 | stalled |
| r64_n1_mri4/sample_015 | P-Flow | 215 | 192 | 12.6739 | 5.7562 | stalled |
| r64_n1_mri4/sample_016 | P-Flow | 213 | 202 | 12.2044 | 5.4399 | stalled |
| r64_n1_mri4/sample_017 | P-Flow | 47 | 39 | 1.9721 | 9.9851 | stalled |
| r64_n1_mri4/sample_018 | P-Flow | 216 | 202 | 14.1227 | 5.1032 | stalled |
| r64_n1_mri4/sample_019 | P-Flow | 37 | 30 | 2.1386 | 9.9872 | stalled |
| r64_n1_mri4/sample_020 | P-Flow | 24 | 20 | 2.3757 | 10.0494 | stalled |
| r64_n1_mri4/sample_021 | P-Flow | 173 | 166 | 14.6927 | 3.9962 | stalled |
| r64_n1_mri4/sample_022 | P-Flow | 36 | 29 | 2.3221 | 10.0201 | stalled |
| r64_n1_mri4/sample_023 | P-Flow | 172 | 167 | 12.7618 | 5.7249 | stalled |
| r64_n1_mri4/sample_024 | P-Flow | 183 | 180 | 12.6628 | 4.7247 | stalled |
| r64_n1_mri4/sample_025 | P-Flow | 182 | 176 | 13.2709 | 3.7666 | stalled |
| r64_n1_mri4/sample_026 | P-Flow | 177 | 170 | 13.8603 | 4.6353 | stalled |
| r64_n1_mri4/sample_027 | P-Flow | 182 | 175 | 14.0227 | 4.7345 | stalled |
| r64_n1_mri4/sample_028 | P-Flow | 177 | 170 | 12.9492 | 6.0073 | stalled |
| r64_n1_mri4/sample_029 | P-Flow | 160 | 155 | 13.5525 | 5.1238 | stalled |
| r64_n1_mri4/sample_000 | DLR D-Flow | 66 | 61 | 76.8242 | 76.3349 | stalled |
| r64_n1_mri4/sample_001 | DLR D-Flow | 53 | 47 | 3.3751 | 10.1928 | stalled |
| r64_n1_mri4/sample_002 | DLR D-Flow | 132 | 129 | 2.0269 | 9.9614 | stalled |
| r64_n1_mri4/sample_003 | DLR D-Flow | 268 | 256 | 2.2620 | 10.0066 | stalled |
| r64_n1_mri4/sample_004 | DLR D-Flow | 153 | 142 | 100.4444 | 95.6929 | stalled |
| r64_n1_mri4/sample_005 | DLR D-Flow | 493 | 469 | 2.5327 | 10.1584 | stalled |
| r64_n1_mri4/sample_006 | DLR D-Flow | 169 | 163 | 1.9088 | 10.0184 | stalled |
| r64_n1_mri4/sample_007 | DLR D-Flow | 244 | 232 | 1.6606 | 9.9734 | stalled |
| r64_n1_mri4/sample_008 | DLR D-Flow | 60 | 55 | 41.0832 | 36.4388 | stalled |
| r64_n1_mri4/sample_009 | DLR D-Flow | 127 | 119 | 3.4858 | 10.3768 | stalled |
| r64_n1_mri4/sample_010 | DLR D-Flow | 140 | 132 | 31.5027 | 27.9559 | stalled |
| r64_n1_mri4/sample_011 | DLR D-Flow | 187 | 181 | 2.0246 | 9.9637 | stalled |
| r64_n1_mri4/sample_012 | DLR D-Flow | 256 | 245 | 2.7520 | 10.1334 | stalled |
| r64_n1_mri4/sample_013 | DLR D-Flow | 118 | 114 | 2.3102 | 9.9615 | stalled |
| r64_n1_mri4/sample_014 | DLR D-Flow | 208 | 195 | 13.4221 | 13.5455 | stalled |
| r64_n1_mri4/sample_015 | DLR D-Flow | 637 | 620 | 2.3656 | 10.0031 | stalled |
| r64_n1_mri4/sample_016 | DLR D-Flow | 247 | 235 | 4.2919 | 10.4966 | stalled |
| r64_n1_mri4/sample_017 | DLR D-Flow | 1000 | 946 | 2.1152 | 9.9461 | step_limit |
| r64_n1_mri4/sample_018 | DLR D-Flow | 434 | 423 | 1.9902 | 9.9483 | stalled |
| r64_n1_mri4/sample_019 | DLR D-Flow | 82 | 78 | 1.9928 | 9.9708 | stalled |
| r64_n1_mri4/sample_020 | DLR D-Flow | 418 | 399 | 1.8490 | 10.0040 | stalled |
| r64_n1_mri4/sample_021 | DLR D-Flow | 291 | 276 | 2.3100 | 9.9581 | stalled |
| r64_n1_mri4/sample_022 | DLR D-Flow | 62 | 57 | 2.2956 | 10.0054 | stalled |
| r64_n1_mri4/sample_023 | DLR D-Flow | 135 | 127 | 2.0569 | 9.9508 | stalled |
| r64_n1_mri4/sample_024 | DLR D-Flow | 118 | 110 | 2.0009 | 10.0173 | stalled |
| r64_n1_mri4/sample_025 | DLR D-Flow | 209 | 197 | 1.8965 | 9.9888 | stalled |
| r64_n1_mri4/sample_026 | DLR D-Flow | 1000 | 948 | 2.5482 | 10.1395 | step_limit |
| r64_n1_mri4/sample_027 | DLR D-Flow | 129 | 124 | 1.4635 | 10.0025 | stalled |
| r64_n1_mri4/sample_028 | DLR D-Flow | 278 | 271 | 5.2855 | 10.6981 | stalled |
| r64_n1_mri4/sample_029 | DLR D-Flow | 164 | 158 | 2.2343 | 9.9775 | stalled |
| r64_n2_mri4/sample_000 | D-Flow | 552 | 550 | 12.3818 | 5.2607 | stalled |
| r64_n2_mri4/sample_001 | D-Flow | 702 | 699 | 10.8566 | 5.3905 | stalled |
| r64_n2_mri4/sample_002 | D-Flow | 488 | 486 | 13.3715 | 5.1647 | stalled |
| r64_n2_mri4/sample_003 | D-Flow | 664 | 661 | 11.5499 | 4.7160 | stalled |
| r64_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 4.4153 | 9.0953 | step_limit |
| r64_n2_mri4/sample_005 | D-Flow | 719 | 716 | 14.4148 | 3.6460 | stalled |
| r64_n2_mri4/sample_006 | D-Flow | 961 | 959 | 11.9344 | 4.8509 | stalled |
| r64_n2_mri4/sample_007 | D-Flow | 550 | 548 | 13.1577 | 4.3693 | stalled |
| r64_n2_mri4/sample_008 | D-Flow | 616 | 613 | 13.8354 | 3.9430 | stalled |
| r64_n2_mri4/sample_009 | D-Flow | 523 | 521 | 14.5174 | 4.5583 | stalled |
| r64_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 4.5467 | 9.4585 | step_limit |
| r64_n2_mri4/sample_000 | P-Flow | 268 | 237 | 14.3961 | 5.2607 | stalled |
| r64_n2_mri4/sample_001 | P-Flow | 264 | 242 | 12.3445 | 5.3905 | stalled |
| r64_n2_mri4/sample_002 | P-Flow | 170 | 166 | 14.5942 | 5.1647 | stalled |
| r64_n2_mri4/sample_003 | P-Flow | 9 | 7 | 108.7445 | 108.7075 | stalled |
| r64_n2_mri4/sample_004 | P-Flow | 235 | 217 | 14.8777 | 4.6438 | stalled |
| r64_n2_mri4/sample_005 | P-Flow | 193 | 180 | 15.2931 | 3.6460 | stalled |
| r64_n2_mri4/sample_006 | P-Flow | 170 | 163 | 13.7433 | 4.8509 | stalled |
| r64_n2_mri4/sample_007 | P-Flow | 185 | 176 | 15.2588 | 4.3693 | stalled |
| r64_n2_mri4/sample_008 | P-Flow | 182 | 174 | 14.9222 | 3.9429 | stalled |
| r64_n2_mri4/sample_009 | P-Flow | 49 | 42 | 2.5578 | 9.9843 | stalled |
| r64_n2_mri4/sample_010 | P-Flow | 6 | 4 | 112.4339 | 108.3126 | stalled |
| r64_n2_mri4/sample_000 | DLR D-Flow | 1000 | 979 | 41.7772 | 39.7428 | step_limit |
| r64_n2_mri4/sample_001 | DLR D-Flow | 192 | 184 | 2.1654 | 9.9982 | stalled |
| r64_n2_mri4/sample_002 | DLR D-Flow | 425 | 413 | 2.2956 | 10.0228 | stalled |
| r64_n2_mri4/sample_003 | DLR D-Flow | 506 | 484 | 58.5948 | 58.9346 | stalled |
| r64_n2_mri4/sample_004 | DLR D-Flow | 227 | 220 | 77.8066 | 74.8851 | stalled |
| r64_n2_mri4/sample_005 | DLR D-Flow | 273 | 260 | 69.5036 | 63.0180 | stalled |
| r64_n2_mri4/sample_006 | DLR D-Flow | 804 | 778 | 2.7413 | 10.0525 | stalled |
| r64_n2_mri4/sample_007 | DLR D-Flow | 552 | 526 | 71.7906 | 72.2275 | stalled |
| r64_n2_mri4/sample_008 | DLR D-Flow | 440 | 426 | 1.8042 | 9.9546 | stalled |
| r64_n2_mri4/sample_009 | DLR D-Flow | 176 | 171 | 2.7383 | 10.0064 | stalled |
| r64_n2_mri4/sample_010 | DLR D-Flow | 1000 | 976 | 104.4647 | 100.0185 | step_limit |
| r64_n4_mri4/sample_000 | D-Flow | 799 | 797 | 13.0651 | 5.6703 | stalled |
| r64_n4_mri4/sample_001 | D-Flow | 595 | 592 | 11.5318 | 5.2992 | stalled |
| r64_n4_mri4/sample_002 | D-Flow | 473 | 471 | 11.1386 | 5.2207 | stalled |
| r64_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 4.2325 | 7.8157 | step_limit |
| r64_n4_mri4/sample_004 | D-Flow | 884 | 882 | 12.6519 | 5.1761 | stalled |
| r64_n4_mri4/sample_005 | D-Flow | 717 | 714 | 5.1397 | 7.4103 | stalled |
| r64_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 12.6543 | 5.7428 | step_limit |
| r64_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 13.9448 | 3.9970 | step_limit |
| r64_n4_mri4/sample_008 | D-Flow | 723 | 721 | 12.0678 | 5.8027 | stalled |
| r64_n4_mri4/sample_009 | D-Flow | 868 | 866 | 10.8606 | 5.7611 | stalled |
| r64_n4_mri4/sample_010 | D-Flow | 567 | 565 | 13.3737 | 4.1295 | stalled |
| r64_n4_mri4/sample_000 | P-Flow | 208 | 194 | 13.3324 | 5.6703 | stalled |
| r64_n4_mri4/sample_001 | P-Flow | 219 | 206 | 13.2620 | 5.2992 | stalled |
| r64_n4_mri4/sample_002 | P-Flow | 170 | 164 | 12.7478 | 5.2207 | stalled |
| r64_n4_mri4/sample_003 | P-Flow | 11 | 8 | 62.3836 | 58.4233 | stalled |
| r64_n4_mri4/sample_004 | P-Flow | 184 | 181 | 14.1960 | 5.1761 | stalled |
| r64_n4_mri4/sample_005 | P-Flow | 245 | 223 | 13.2171 | 5.9508 | stalled |
| r64_n4_mri4/sample_006 | P-Flow | 207 | 195 | 13.6156 | 5.7427 | stalled |
| r64_n4_mri4/sample_007 | P-Flow | 197 | 193 | 15.3708 | 3.9390 | stalled |
| r64_n4_mri4/sample_008 | P-Flow | 244 | 225 | 13.1138 | 5.8027 | stalled |
| r64_n4_mri4/sample_009 | P-Flow | 33 | 25 | 44.1855 | 40.4399 | stalled |
| r64_n4_mri4/sample_010 | P-Flow | 176 | 174 | 13.8369 | 4.1295 | stalled |
| r64_n4_mri4/sample_000 | DLR D-Flow | 1000 | 959 | 78.9593 | 73.0121 | step_limit |
| r64_n4_mri4/sample_001 | DLR D-Flow | 120 | 112 | 19.5691 | 17.9050 | stalled |
| r64_n4_mri4/sample_002 | DLR D-Flow | 1000 | 978 | 15.8200 | 14.7482 | step_limit |
| r64_n4_mri4/sample_003 | DLR D-Flow | 475 | 456 | 19.7379 | 18.9087 | stalled |
| r64_n4_mri4/sample_004 | DLR D-Flow | 151 | 147 | 17.2587 | 15.6006 | stalled |
| r64_n4_mri4/sample_005 | DLR D-Flow | 1000 | 969 | 16.5913 | 15.7434 | step_limit |
| r64_n4_mri4/sample_006 | DLR D-Flow | 306 | 298 | 66.9934 | 61.9720 | stalled |
| r64_n4_mri4/sample_007 | DLR D-Flow | 1000 | 968 | 18.3490 | 16.5970 | step_limit |
| r64_n4_mri4/sample_008 | DLR D-Flow | 486 | 470 | 23.5885 | 19.8564 | stalled |
| r64_n4_mri4/sample_009 | DLR D-Flow | 1000 | 969 | 61.5037 | 53.3287 | step_limit |
| r64_n4_mri4/sample_010 | DLR D-Flow | 94 | 88 | 16.8946 | 16.4666 | stalled |
| r96_n1_ct10/sample_000 | D-Flow | 916 | 914 | 44.9185 | 0.3542 | stalled |
| r96_n1_ct10/sample_001 | D-Flow | 721 | 719 | 56.8219 | 0.7870 | stalled |
| r96_n1_ct10/sample_002 | D-Flow | 710 | 708 | 53.6763 | 0.8492 | stalled |
| r96_n1_ct10/sample_003 | D-Flow | 610 | 608 | 56.1627 | 0.6358 | stalled |
| r96_n1_ct10/sample_004 | D-Flow | 61 | 59 | 105.3711 | 108.3831 | stalled |
| r96_n1_ct10/sample_005 | D-Flow | 729 | 727 | 64.6357 | 0.5517 | stalled |
| r96_n1_ct10/sample_006 | D-Flow | 803 | 801 | 62.4113 | 0.4179 | stalled |
| r96_n1_ct10/sample_007 | D-Flow | 594 | 592 | 88.9574 | 0.4267 | stalled |
| r96_n1_ct10/sample_008 | D-Flow | 772 | 770 | 48.9491 | 0.8056 | stalled |
| r96_n1_ct10/sample_009 | D-Flow | 46 | 44 | 114.9632 | 103.6353 | stalled |
| r96_n1_ct10/sample_010 | D-Flow | 723 | 721 | 65.1950 | 0.7696 | stalled |
| r96_n1_ct10/sample_011 | D-Flow | 535 | 533 | 86.3326 | 0.5431 | stalled |
| r96_n1_ct10/sample_012 | D-Flow | 670 | 668 | 8.5434 | 7.8809 | stalled |
| r96_n1_ct10/sample_013 | D-Flow | 249 | 245 | 105.4650 | 105.6555 | stalled |
| r96_n1_ct10/sample_014 | D-Flow | 183 | 174 | 116.1078 | 106.1508 | stalled |
| r96_n1_ct10/sample_015 | D-Flow | 889 | 886 | 47.1045 | 0.7301 | stalled |
| r96_n1_ct10/sample_016 | D-Flow | 957 | 954 | 76.8796 | 0.5899 | stalled |
| r96_n1_ct10/sample_017 | D-Flow | 563 | 561 | 67.6292 | 0.6191 | stalled |
| r96_n1_ct10/sample_018 | D-Flow | 969 | 967 | 9.3961 | 6.9516 | stalled |
| r96_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 66.0899 | 0.7136 | step_limit |
| r96_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 53.0496 | 0.4629 | step_limit |
| r96_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 54.5152 | 0.6290 | step_limit |
| r96_n1_ct10/sample_022 | D-Flow | 697 | 695 | 53.5492 | 0.9243 | stalled |
| r96_n1_ct10/sample_023 | D-Flow | 620 | 618 | 53.4425 | 0.7347 | stalled |
| r96_n1_ct10/sample_024 | D-Flow | 627 | 625 | 51.6644 | 0.9298 | stalled |
| r96_n1_ct10/sample_025 | D-Flow | 644 | 642 | 55.9157 | 0.6700 | stalled |
| r96_n1_ct10/sample_026 | D-Flow | 780 | 778 | 43.5791 | 0.6702 | stalled |
| r96_n1_ct10/sample_027 | D-Flow | 481 | 479 | 130.1100 | 0.3283 | stalled |
| r96_n1_ct10/sample_028 | D-Flow | 501 | 499 | 9.6392 | 5.8015 | stalled |
| r96_n1_ct10/sample_029 | D-Flow | 665 | 663 | 53.5276 | 0.4432 | stalled |
| r96_n1_ct10/sample_000 | P-Flow | 40 | 35 | 4.3645 | 9.9083 | stalled |
| r96_n1_ct10/sample_001 | P-Flow | 43 | 39 | 2.5449 | 9.9998 | stalled |
| r96_n1_ct10/sample_002 | P-Flow | 42 | 36 | 3.7725 | 9.9144 | stalled |
| r96_n1_ct10/sample_003 | P-Flow | 39 | 32 | 2.5154 | 9.9956 | stalled |
| r96_n1_ct10/sample_004 | P-Flow | 46 | 38 | 3.9234 | 9.9588 | stalled |
| r96_n1_ct10/sample_005 | P-Flow | 128 | 106 | 2.4013 | 9.9807 | stalled |
| r96_n1_ct10/sample_006 | P-Flow | 1000 | 845 | 35.7746 | 1.1452 | step_limit |
| r96_n1_ct10/sample_007 | P-Flow | 34 | 27 | 4.8294 | 9.9810 | stalled |
| r96_n1_ct10/sample_008 | P-Flow | 1000 | 971 | 36.1323 | 0.9628 | step_limit |
| r96_n1_ct10/sample_009 | P-Flow | 914 | 722 | 2.8262 | 9.8601 | stalled |
| r96_n1_ct10/sample_010 | P-Flow | 1000 | 870 | 43.6822 | 0.9048 | step_limit |
| r96_n1_ct10/sample_011 | P-Flow | 54 | 45 | 3.3799 | 9.9691 | stalled |
| r96_n1_ct10/sample_012 | P-Flow | 55 | 45 | 4.7154 | 9.9337 | stalled |
| r96_n1_ct10/sample_013 | P-Flow | 826 | 781 | 47.7846 | 0.7417 | stalled |
| r96_n1_ct10/sample_014 | P-Flow | 1000 | 943 | 31.7943 | 1.2591 | step_limit |
| r96_n1_ct10/sample_015 | P-Flow | 1000 | 813 | 3.4660 | 10.2018 | step_limit |
| r96_n1_ct10/sample_016 | P-Flow | 42 | 36 | 2.3910 | 10.0138 | stalled |
| r96_n1_ct10/sample_017 | P-Flow | 36 | 30 | 2.7969 | 9.9509 | stalled |
| r96_n1_ct10/sample_018 | P-Flow | 900 | 852 | 53.9761 | 0.7824 | stalled |
| r96_n1_ct10/sample_019 | P-Flow | 61 | 53 | 3.0612 | 9.9998 | stalled |
| r96_n1_ct10/sample_020 | P-Flow | 525 | 506 | 46.2961 | 0.4605 | stalled |
| r96_n1_ct10/sample_021 | P-Flow | 1000 | 856 | 33.5399 | 1.2624 | step_limit |
| r96_n1_ct10/sample_022 | P-Flow | 1000 | 871 | 43.8040 | 0.9747 | step_limit |
| r96_n1_ct10/sample_023 | P-Flow | 941 | 822 | 51.9480 | 0.7347 | stalled |
| r96_n1_ct10/sample_024 | P-Flow | 30 | 25 | 27.2232 | 15.4792 | stalled |
| r96_n1_ct10/sample_025 | P-Flow | 43 | 38 | 3.5300 | 9.9401 | stalled |
| r96_n1_ct10/sample_026 | P-Flow | 141 | 114 | 4.1478 | 9.9746 | stalled |
| r96_n1_ct10/sample_027 | P-Flow | 33 | 28 | 4.1563 | 9.9237 | stalled |
| r96_n1_ct10/sample_028 | P-Flow | 295 | 247 | 2.5330 | 9.9193 | stalled |
| r96_n1_ct10/sample_029 | P-Flow | 49 | 42 | 2.5125 | 9.9632 | stalled |
| r96_n1_ct10/sample_000 | DLR D-Flow | 295 | 286 | 2.8563 | 9.8728 | stalled |
| r96_n1_ct10/sample_001 | DLR D-Flow | 37 | 33 | 113.2756 | 101.0677 | stalled |
| r96_n1_ct10/sample_002 | DLR D-Flow | 632 | 600 | 2.7469 | 9.8617 | stalled |
| r96_n1_ct10/sample_003 | DLR D-Flow | 119 | 113 | 2.7477 | 9.9797 | stalled |
| r96_n1_ct10/sample_004 | DLR D-Flow | 209 | 200 | 4.4798 | 10.0018 | stalled |
| r96_n1_ct10/sample_005 | DLR D-Flow | 279 | 265 | 2.1811 | 9.9388 | stalled |
| r96_n1_ct10/sample_006 | DLR D-Flow | 208 | 197 | 3.0720 | 9.8936 | stalled |
| r96_n1_ct10/sample_007 | DLR D-Flow | 179 | 171 | 2.7673 | 9.8614 | stalled |
| r96_n1_ct10/sample_008 | DLR D-Flow | 87 | 82 | 114.5719 | 102.6563 | stalled |
| r96_n1_ct10/sample_009 | DLR D-Flow | 883 | 849 | 3.7685 | 9.9259 | stalled |
| r96_n1_ct10/sample_010 | DLR D-Flow | 206 | 201 | 5.7914 | 9.9496 | stalled |
| r96_n1_ct10/sample_011 | DLR D-Flow | 198 | 188 | 3.0984 | 9.9305 | stalled |
| r96_n1_ct10/sample_012 | DLR D-Flow | 299 | 285 | 3.5207 | 9.8792 | stalled |
| r96_n1_ct10/sample_013 | DLR D-Flow | 24 | 21 | 124.1965 | 112.5329 | stalled |
| r96_n1_ct10/sample_014 | DLR D-Flow | 32 | 28 | 116.3447 | 107.2080 | stalled |
| r96_n1_ct10/sample_015 | DLR D-Flow | 208 | 201 | 5.1294 | 10.9139 | stalled |
| r96_n1_ct10/sample_016 | DLR D-Flow | 193 | 188 | 7.1181 | 10.0690 | stalled |
| r96_n1_ct10/sample_017 | DLR D-Flow | 438 | 420 | 4.8318 | 9.9483 | stalled |
| r96_n1_ct10/sample_018 | DLR D-Flow | 165 | 157 | 3.5857 | 9.8445 | stalled |
| r96_n1_ct10/sample_019 | DLR D-Flow | 298 | 289 | 31.7811 | 16.8749 | stalled |
| r96_n1_ct10/sample_020 | DLR D-Flow | 214 | 209 | 2.7939 | 9.9753 | stalled |
| r96_n1_ct10/sample_021 | DLR D-Flow | 154 | 146 | 2.5990 | 9.9227 | stalled |
| r96_n1_ct10/sample_022 | DLR D-Flow | 151 | 141 | 3.9044 | 9.9358 | stalled |
| r96_n1_ct10/sample_023 | DLR D-Flow | 747 | 721 | 3.2357 | 9.8500 | stalled |
| r96_n1_ct10/sample_024 | DLR D-Flow | 804 | 760 | 2.9288 | 9.9616 | stalled |
| r96_n1_ct10/sample_025 | DLR D-Flow | 1000 | 956 | 4.2012 | 9.9686 | step_limit |
| r96_n1_ct10/sample_026 | DLR D-Flow | 185 | 176 | 122.3940 | 101.0086 | stalled |
| r96_n1_ct10/sample_027 | DLR D-Flow | 337 | 322 | 5.3336 | 9.8068 | stalled |
| r96_n1_ct10/sample_028 | DLR D-Flow | 115 | 113 | 120.5156 | 99.7650 | stalled |
| r96_n1_ct10/sample_029 | DLR D-Flow | 163 | 157 | 2.7851 | 9.9487 | stalled |
| r96_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 62.5484 | 0.7662 | step_limit |
| r96_n2_ct10/sample_001 | D-Flow | 976 | 974 | 65.4482 | 0.7801 | stalled |
| r96_n2_ct10/sample_002 | D-Flow | 449 | 447 | 76.6798 | 0.5876 | stalled |
| r96_n2_ct10/sample_003 | D-Flow | 877 | 874 | 65.9237 | 0.4079 | stalled |
| r96_n2_ct10/sample_004 | D-Flow | 807 | 805 | 81.4348 | 0.5660 | stalled |
| r96_n2_ct10/sample_005 | D-Flow | 771 | 769 | 58.2122 | 0.6745 | stalled |
| r96_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 31.1797 | 3.0742 | step_limit |
| r96_n2_ct10/sample_007 | D-Flow | 277 | 273 | 62.9947 | 36.4739 | stalled |
| r96_n2_ct10/sample_008 | D-Flow | 908 | 906 | 7.2166 | 7.2242 | stalled |
| r96_n2_ct10/sample_009 | D-Flow | 31 | 28 | 83.4622 | 64.2911 | stalled |
| r96_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 44.2217 | 1.1777 | step_limit |
| r96_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 78.6932 | 0.3179 | step_limit |
| r96_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 39.2419 | 3.5223 | step_limit |
| r96_n2_ct10/sample_000 | P-Flow | 1000 | 890 | 37.0820 | 1.2133 | step_limit |
| r96_n2_ct10/sample_001 | P-Flow | 12 | 10 | 13.1464 | 11.4651 | stalled |
| r96_n2_ct10/sample_002 | P-Flow | 786 | 733 | 56.7516 | 0.5877 | stalled |
| r96_n2_ct10/sample_003 | P-Flow | 108 | 91 | 6.9280 | 10.1842 | stalled |
| r96_n2_ct10/sample_004 | P-Flow | 81 | 72 | 4.1851 | 9.9569 | stalled |
| r96_n2_ct10/sample_005 | P-Flow | 831 | 795 | 55.2393 | 0.6746 | stalled |
| r96_n2_ct10/sample_006 | P-Flow | 146 | 120 | 5.1189 | 10.1011 | stalled |
| r96_n2_ct10/sample_007 | P-Flow | 32 | 27 | 4.9237 | 10.0709 | stalled |
| r96_n2_ct10/sample_008 | P-Flow | 1000 | 866 | 42.0270 | 0.8756 | step_limit |
| r96_n2_ct10/sample_009 | P-Flow | 1000 | 880 | 42.1532 | 1.0922 | step_limit |
| r96_n2_ct10/sample_010 | P-Flow | 37 | 29 | 5.7518 | 10.1210 | stalled |
| r96_n2_ct10/sample_011 | P-Flow | 695 | 669 | 69.5346 | 0.3162 | stalled |
| r96_n2_ct10/sample_012 | P-Flow | 791 | 722 | 61.0992 | 0.7997 | stalled |
| r96_n2_ct10/sample_000 | DLR D-Flow | 842 | 818 | 61.6004 | 44.6793 | stalled |
| r96_n2_ct10/sample_001 | DLR D-Flow | 1000 | 972 | 32.6508 | 16.5781 | step_limit |
| r96_n2_ct10/sample_002 | DLR D-Flow | 1000 | 953 | 3.3201 | 9.9190 | step_limit |
| r96_n2_ct10/sample_003 | DLR D-Flow | 412 | 397 | 4.7798 | 9.9349 | stalled |
| r96_n2_ct10/sample_004 | DLR D-Flow | 426 | 413 | 7.5927 | 10.0959 | stalled |
| r96_n2_ct10/sample_005 | DLR D-Flow | 99 | 96 | 3.2292 | 9.9228 | stalled |
| r96_n2_ct10/sample_006 | DLR D-Flow | 77 | 69 | 103.1362 | 72.7757 | stalled |
| r96_n2_ct10/sample_007 | DLR D-Flow | 452 | 437 | 63.1312 | 36.9989 | stalled |
| r96_n2_ct10/sample_008 | DLR D-Flow | 1000 | 957 | 3.4831 | 10.0551 | step_limit |
| r96_n2_ct10/sample_009 | DLR D-Flow | 23 | 20 | 79.4559 | 65.0150 | stalled |
| r96_n2_ct10/sample_010 | DLR D-Flow | 148 | 142 | 5.2457 | 10.0842 | stalled |
| r96_n2_ct10/sample_011 | DLR D-Flow | 138 | 128 | 51.0022 | 26.7188 | stalled |
| r96_n2_ct10/sample_012 | DLR D-Flow | 739 | 707 | 80.1093 | 52.4907 | stalled |
| r96_n4_ct10/sample_000 | D-Flow | 629 | 627 | 97.8514 | 0.8261 | stalled |
| r96_n4_ct10/sample_001 | D-Flow | 726 | 724 | 76.4168 | 0.7274 | stalled |
| r96_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 70.6460 | 0.4522 | step_limit |
| r96_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 59.3247 | 0.4015 | step_limit |
| r96_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 54.7781 | 29.1177 | step_limit |
| r96_n4_ct10/sample_005 | D-Flow | 865 | 863 | 61.4596 | 0.6696 | stalled |
| r96_n4_ct10/sample_006 | D-Flow | 905 | 903 | 57.6881 | 0.8237 | stalled |
| r96_n4_ct10/sample_007 | D-Flow | 682 | 679 | 97.2083 | 0.6472 | stalled |
| r96_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 62.7722 | 0.4349 | step_limit |
| r96_n4_ct10/sample_009 | D-Flow | 843 | 841 | 75.1125 | 0.5122 | stalled |
| r96_n4_ct10/sample_010 | D-Flow | 964 | 962 | 63.2291 | 0.2373 | stalled |
| r96_n4_ct10/sample_011 | D-Flow | 800 | 798 | 69.0265 | 0.5664 | stalled |
| r96_n4_ct10/sample_012 | D-Flow | 699 | 697 | 76.8315 | 1.2695 | stalled |
| r96_n4_ct10/sample_000 | P-Flow | 81 | 71 | 6.3063 | 9.8514 | stalled |
| r96_n4_ct10/sample_001 | P-Flow | 20 | 16 | 36.3405 | 16.8701 | stalled |
| r96_n4_ct10/sample_002 | P-Flow | 104 | 88 | 4.6332 | 9.9236 | stalled |
| r96_n4_ct10/sample_003 | P-Flow | 16 | 13 | 53.1163 | 25.4926 | stalled |
| r96_n4_ct10/sample_004 | P-Flow | 1000 | 946 | 54.4560 | 1.1781 | step_limit |
| r96_n4_ct10/sample_005 | P-Flow | 47 | 39 | 5.3864 | 9.9168 | stalled |
| r96_n4_ct10/sample_006 | P-Flow | 44 | 35 | 6.6484 | 9.9661 | stalled |
| r96_n4_ct10/sample_007 | P-Flow | 1000 | 932 | 41.0195 | 1.4446 | step_limit |
| r96_n4_ct10/sample_008 | P-Flow | 1000 | 912 | 54.7487 | 0.4334 | step_limit |
| r96_n4_ct10/sample_009 | P-Flow | 1000 | 904 | 42.6270 | 0.7570 | step_limit |
| r96_n4_ct10/sample_010 | P-Flow | 1000 | 789 | 5.2931 | 9.6723 | step_limit |
| r96_n4_ct10/sample_011 | P-Flow | 1000 | 907 | 39.3288 | 1.1332 | step_limit |
| r96_n4_ct10/sample_012 | P-Flow | 669 | 641 | 60.3236 | 1.2695 | stalled |
| r96_n4_ct10/sample_000 | DLR D-Flow | 198 | 189 | 12.0636 | 10.3089 | stalled |
| r96_n4_ct10/sample_001 | DLR D-Flow | 1000 | 963 | 39.6783 | 18.4104 | step_limit |
| r96_n4_ct10/sample_002 | DLR D-Flow | 811 | 772 | 17.2002 | 11.4088 | stalled |
| r96_n4_ct10/sample_003 | DLR D-Flow | 825 | 793 | 30.7112 | 15.0663 | stalled |
| r96_n4_ct10/sample_004 | DLR D-Flow | 251 | 241 | 89.4602 | 50.3629 | stalled |
| r96_n4_ct10/sample_005 | DLR D-Flow | 831 | 782 | 8.6784 | 10.3238 | stalled |
| r96_n4_ct10/sample_006 | DLR D-Flow | 929 | 899 | 55.2458 | 26.6105 | stalled |
| r96_n4_ct10/sample_007 | DLR D-Flow | 1000 | 977 | 60.3878 | 26.7519 | step_limit |
| r96_n4_ct10/sample_008 | DLR D-Flow | 209 | 202 | 21.3826 | 11.7324 | stalled |
| r96_n4_ct10/sample_009 | DLR D-Flow | 1000 | 967 | 48.5871 | 27.7121 | step_limit |
| r96_n4_ct10/sample_010 | DLR D-Flow | 1000 | 962 | 87.4544 | 48.0201 | step_limit |
| r96_n4_ct10/sample_011 | DLR D-Flow | 76 | 69 | 81.3034 | 39.9323 | stalled |
| r96_n4_ct10/sample_012 | DLR D-Flow | 1000 | 966 | 70.5331 | 33.2322 | step_limit |
| r96_n1_mri4/sample_000 | D-Flow | 646 | 644 | 12.7135 | 4.7521 | stalled |
| r96_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 14.1150 | 4.7013 | step_limit |
| r96_n1_mri4/sample_002 | D-Flow | 840 | 838 | 12.3646 | 5.1422 | stalled |
| r96_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 6.2737 | 6.7229 | step_limit |
| r96_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 9.9739 | 5.1685 | step_limit |
| r96_n1_mri4/sample_005 | D-Flow | 540 | 538 | 11.8104 | 5.1471 | stalled |
| r96_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 7.2755 | 5.7810 | step_limit |
| r96_n1_mri4/sample_007 | D-Flow | 702 | 700 | 11.9348 | 4.5301 | stalled |
| r96_n1_mri4/sample_008 | D-Flow | 989 | 987 | 12.7348 | 5.6499 | stalled |
| r96_n1_mri4/sample_009 | D-Flow | 175 | 171 | 103.6711 | 102.0049 | stalled |
| r96_n1_mri4/sample_010 | D-Flow | 489 | 486 | 14.2650 | 5.1468 | stalled |
| r96_n1_mri4/sample_011 | D-Flow | 606 | 602 | 14.1624 | 5.2010 | stalled |
| r96_n1_mri4/sample_012 | D-Flow | 678 | 676 | 14.0814 | 4.2979 | stalled |
| r96_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 14.2238 | 4.3197 | step_limit |
| r96_n1_mri4/sample_014 | D-Flow | 367 | 364 | 103.3544 | 101.5372 | stalled |
| r96_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 6.1156 | 7.9927 | step_limit |
| r96_n1_mri4/sample_016 | D-Flow | 503 | 501 | 13.5081 | 4.6820 | stalled |
| r96_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 4.4789 | 7.2516 | step_limit |
| r96_n1_mri4/sample_018 | D-Flow | 563 | 560 | 12.8900 | 4.3965 | stalled |
| r96_n1_mri4/sample_019 | D-Flow | 989 | 987 | 11.6757 | 4.9198 | stalled |
| r96_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 6.9144 | 6.7641 | step_limit |
| r96_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 6.5760 | 7.3340 | step_limit |
| r96_n1_mri4/sample_022 | D-Flow | 441 | 439 | 12.2320 | 5.9690 | stalled |
| r96_n1_mri4/sample_023 | D-Flow | 803 | 801 | 11.9106 | 4.3046 | stalled |
| r96_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 7.6744 | 6.2402 | step_limit |
| r96_n1_mri4/sample_025 | D-Flow | 657 | 654 | 13.2094 | 5.0777 | stalled |
| r96_n1_mri4/sample_026 | D-Flow | 94 | 92 | 103.6243 | 103.1319 | stalled |
| r96_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 12.2305 | 5.2861 | step_limit |
| r96_n1_mri4/sample_028 | D-Flow | 643 | 640 | 98.6819 | 94.3273 | stalled |
| r96_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 5.8655 | 6.4197 | step_limit |
| r96_n1_mri4/sample_000 | P-Flow | 266 | 236 | 14.4133 | 4.7521 | stalled |
| r96_n1_mri4/sample_001 | P-Flow | 22 | 17 | 117.0688 | 115.6723 | stalled |
| r96_n1_mri4/sample_002 | P-Flow | 41 | 32 | 1.7738 | 9.9740 | stalled |
| r96_n1_mri4/sample_003 | P-Flow | 179 | 171 | 14.1351 | 5.5184 | stalled |
| r96_n1_mri4/sample_004 | P-Flow | 41 | 34 | 4.1235 | 10.2640 | stalled |
| r96_n1_mri4/sample_005 | P-Flow | 178 | 170 | 12.8265 | 5.1471 | stalled |
| r96_n1_mri4/sample_006 | P-Flow | 169 | 161 | 13.8212 | 4.7198 | stalled |
| r96_n1_mri4/sample_007 | P-Flow | 23 | 18 | 1.9102 | 10.0345 | stalled |
| r96_n1_mri4/sample_008 | P-Flow | 164 | 160 | 13.4997 | 5.6499 | stalled |
| r96_n1_mri4/sample_009 | P-Flow | 18 | 14 | 120.4986 | 111.7442 | stalled |
| r96_n1_mri4/sample_010 | P-Flow | 163 | 159 | 13.9888 | 5.1467 | stalled |
| r96_n1_mri4/sample_011 | P-Flow | 168 | 165 | 14.1622 | 5.2010 | stalled |
| r96_n1_mri4/sample_012 | P-Flow | 29 | 25 | 1.8057 | 9.9730 | stalled |
| r96_n1_mri4/sample_013 | P-Flow | 178 | 170 | 15.3752 | 4.3167 | stalled |
| r96_n1_mri4/sample_014 | P-Flow | 11 | 9 | 134.6998 | 127.0807 | stalled |
| r96_n1_mri4/sample_015 | P-Flow | 10 | 7 | 112.9006 | 112.6722 | stalled |
| r96_n1_mri4/sample_016 | P-Flow | 48 | 37 | 2.0652 | 9.9932 | stalled |
| r96_n1_mri4/sample_017 | P-Flow | 204 | 191 | 12.7514 | 4.8238 | stalled |
| r96_n1_mri4/sample_018 | P-Flow | 207 | 190 | 13.7384 | 4.3965 | stalled |
| r96_n1_mri4/sample_019 | P-Flow | 28 | 23 | 1.7898 | 10.0110 | stalled |
| r96_n1_mri4/sample_020 | P-Flow | 179 | 170 | 14.7173 | 5.0683 | stalled |
| r96_n1_mri4/sample_021 | P-Flow | 22 | 17 | 2.1715 | 10.0362 | stalled |
| r96_n1_mri4/sample_022 | P-Flow | 22 | 17 | 2.0640 | 10.0588 | stalled |
| r96_n1_mri4/sample_023 | P-Flow | 210 | 198 | 13.1520 | 4.3047 | stalled |
| r96_n1_mri4/sample_024 | P-Flow | 90 | 78 | 3.2572 | 10.1923 | stalled |
| r96_n1_mri4/sample_025 | P-Flow | 179 | 171 | 14.7316 | 5.0777 | stalled |
| r96_n1_mri4/sample_026 | P-Flow | 78 | 64 | 2.6136 | 10.0480 | stalled |
| r96_n1_mri4/sample_027 | P-Flow | 26 | 21 | 1.4582 | 9.9880 | stalled |
| r96_n1_mri4/sample_028 | P-Flow | 36 | 30 | 2.0619 | 10.0220 | stalled |
| r96_n1_mri4/sample_029 | P-Flow | 22 | 18 | 1.8160 | 10.0018 | stalled |
| r96_n1_mri4/sample_000 | DLR D-Flow | 182 | 177 | 1.7930 | 10.0109 | stalled |
| r96_n1_mri4/sample_001 | DLR D-Flow | 50 | 47 | 102.6920 | 102.4440 | stalled |
| r96_n1_mri4/sample_002 | DLR D-Flow | 683 | 645 | 1.7258 | 9.9727 | stalled |
| r96_n1_mri4/sample_003 | DLR D-Flow | 283 | 269 | 1.7514 | 10.0005 | stalled |
| r96_n1_mri4/sample_004 | DLR D-Flow | 73 | 69 | 19.1671 | 17.4574 | stalled |
| r96_n1_mri4/sample_005 | DLR D-Flow | 534 | 500 | 1.5238 | 9.9713 | stalled |
| r96_n1_mri4/sample_006 | DLR D-Flow | 391 | 367 | 1.7856 | 10.0027 | stalled |
| r96_n1_mri4/sample_007 | DLR D-Flow | 388 | 367 | 1.8108 | 10.0476 | stalled |
| r96_n1_mri4/sample_008 | DLR D-Flow | 88 | 79 | 102.9123 | 102.8612 | stalled |
| r96_n1_mri4/sample_009 | DLR D-Flow | 192 | 186 | 103.7688 | 101.9054 | stalled |
| r96_n1_mri4/sample_010 | DLR D-Flow | 326 | 311 | 1.5071 | 9.9886 | stalled |
| r96_n1_mri4/sample_011 | DLR D-Flow | 389 | 369 | 1.5260 | 9.9680 | stalled |
| r96_n1_mri4/sample_012 | DLR D-Flow | 334 | 319 | 1.6961 | 9.9751 | stalled |
| r96_n1_mri4/sample_013 | DLR D-Flow | 30 | 27 | 123.8580 | 122.6570 | stalled |
| r96_n1_mri4/sample_014 | DLR D-Flow | 35 | 31 | 103.2368 | 102.4071 | stalled |
| r96_n1_mri4/sample_015 | DLR D-Flow | 1000 | 968 | 2.4171 | 10.0959 | step_limit |
| r96_n1_mri4/sample_016 | DLR D-Flow | 245 | 235 | 2.0424 | 9.9830 | stalled |
| r96_n1_mri4/sample_017 | DLR D-Flow | 152 | 146 | 1.4734 | 9.9608 | stalled |
| r96_n1_mri4/sample_018 | DLR D-Flow | 195 | 178 | 1.9129 | 10.0113 | stalled |
| r96_n1_mri4/sample_019 | DLR D-Flow | 97 | 91 | 102.4168 | 102.7275 | stalled |
| r96_n1_mri4/sample_020 | DLR D-Flow | 19 | 14 | 115.7460 | 115.3215 | stalled |
| r96_n1_mri4/sample_021 | DLR D-Flow | 429 | 417 | 1.9919 | 10.0194 | stalled |
| r96_n1_mri4/sample_022 | DLR D-Flow | 414 | 398 | 1.7675 | 10.0185 | stalled |
| r96_n1_mri4/sample_023 | DLR D-Flow | 308 | 293 | 6.7280 | 11.3328 | stalled |
| r96_n1_mri4/sample_024 | DLR D-Flow | 440 | 427 | 4.1078 | 10.4286 | stalled |
| r96_n1_mri4/sample_025 | DLR D-Flow | 173 | 162 | 3.0343 | 10.2003 | stalled |
| r96_n1_mri4/sample_026 | DLR D-Flow | 53 | 50 | 103.6627 | 103.1882 | stalled |
| r96_n1_mri4/sample_027 | DLR D-Flow | 168 | 158 | 1.3319 | 9.9629 | stalled |
| r96_n1_mri4/sample_028 | DLR D-Flow | 49 | 44 | 121.0568 | 118.1537 | stalled |
| r96_n1_mri4/sample_029 | DLR D-Flow | 464 | 443 | 1.6936 | 9.9626 | stalled |
| r96_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 4.4992 | 7.3008 | step_limit |
| r96_n2_mri4/sample_001 | D-Flow | 718 | 714 | 11.6109 | 5.2024 | stalled |
| r96_n2_mri4/sample_002 | D-Flow | 847 | 845 | 12.6355 | 3.8527 | stalled |
| r96_n2_mri4/sample_003 | D-Flow | 867 | 864 | 13.3538 | 5.0864 | stalled |
| r96_n2_mri4/sample_004 | D-Flow | 576 | 574 | 13.1671 | 5.1961 | stalled |
| r96_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 4.3106 | 7.2870 | step_limit |
| r96_n2_mri4/sample_006 | D-Flow | 1000 | 997 | 12.3908 | 4.8371 | step_limit |
| r96_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 3.7075 | 9.5691 | step_limit |
| r96_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 11.1964 | 5.8611 | step_limit |
| r96_n2_mri4/sample_009 | D-Flow | 964 | 960 | 67.4667 | 67.0258 | stalled |
| r96_n2_mri4/sample_010 | D-Flow | 631 | 629 | 11.4393 | 5.5973 | stalled |
| r96_n2_mri4/sample_011 | D-Flow | 384 | 381 | 41.7745 | 41.1713 | stalled |
| r96_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 62.0729 | 61.1719 | step_limit |
| r96_n2_mri4/sample_000 | P-Flow | 209 | 197 | 12.0323 | 5.1793 | stalled |
| r96_n2_mri4/sample_001 | P-Flow | 216 | 195 | 14.0959 | 5.2024 | stalled |
| r96_n2_mri4/sample_002 | P-Flow | 223 | 209 | 14.8045 | 3.8527 | stalled |
| r96_n2_mri4/sample_003 | P-Flow | 190 | 179 | 14.4931 | 5.0864 | stalled |
| r96_n2_mri4/sample_004 | P-Flow | 187 | 178 | 13.9646 | 5.1961 | stalled |
| r96_n2_mri4/sample_005 | P-Flow | 197 | 188 | 13.0250 | 4.6893 | stalled |
| r96_n2_mri4/sample_006 | P-Flow | 57 | 50 | 6.1640 | 10.7713 | stalled |
| r96_n2_mri4/sample_007 | P-Flow | 239 | 219 | 14.8596 | 4.3131 | stalled |
| r96_n2_mri4/sample_008 | P-Flow | 41 | 33 | 2.8048 | 10.1181 | stalled |
| r96_n2_mri4/sample_009 | P-Flow | 259 | 235 | 13.2984 | 5.9000 | stalled |
| r96_n2_mri4/sample_010 | P-Flow | 228 | 213 | 13.1088 | 5.5973 | stalled |
| r96_n2_mri4/sample_011 | P-Flow | 173 | 166 | 12.8043 | 4.7888 | stalled |
| r96_n2_mri4/sample_012 | P-Flow | 182 | 178 | 13.8410 | 3.7512 | stalled |
| r96_n2_mri4/sample_000 | DLR D-Flow | 570 | 550 | 57.2650 | 57.2292 | stalled |
| r96_n2_mri4/sample_001 | DLR D-Flow | 229 | 221 | 2.4199 | 10.0340 | stalled |
| r96_n2_mri4/sample_002 | DLR D-Flow | 824 | 792 | 2.2670 | 9.9804 | stalled |
| r96_n2_mri4/sample_003 | DLR D-Flow | 502 | 489 | 51.4548 | 49.4468 | stalled |
| r96_n2_mri4/sample_004 | DLR D-Flow | 1000 | 973 | 3.8522 | 10.2973 | step_limit |
| r96_n2_mri4/sample_005 | DLR D-Flow | 772 | 731 | 1.8357 | 9.9612 | stalled |
| r96_n2_mri4/sample_006 | DLR D-Flow | 943 | 911 | 77.9044 | 74.8364 | stalled |
| r96_n2_mri4/sample_007 | DLR D-Flow | 237 | 227 | 63.6295 | 59.1939 | stalled |
| r96_n2_mri4/sample_008 | DLR D-Flow | 179 | 171 | 66.1069 | 66.1593 | stalled |
| r96_n2_mri4/sample_009 | DLR D-Flow | 127 | 119 | 74.5634 | 73.0161 | stalled |
| r96_n2_mri4/sample_010 | DLR D-Flow | 508 | 484 | 1.7631 | 10.0193 | stalled |
| r96_n2_mri4/sample_011 | DLR D-Flow | 669 | 647 | 43.1092 | 43.7028 | stalled |
| r96_n2_mri4/sample_012 | DLR D-Flow | 332 | 320 | 102.8714 | 101.2928 | stalled |
| r96_n4_mri4/sample_000 | D-Flow | 891 | 889 | 13.6664 | 3.7455 | stalled |
| r96_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 11.4623 | 7.2505 | step_limit |
| r96_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 3.9421 | 9.0877 | step_limit |
| r96_n4_mri4/sample_003 | D-Flow | 666 | 664 | 13.2625 | 5.1005 | stalled |
| r96_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 13.7091 | 4.8606 | step_limit |
| r96_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 11.3778 | 6.6334 | step_limit |
| r96_n4_mri4/sample_006 | D-Flow | 871 | 868 | 13.3851 | 3.8579 | stalled |
| r96_n4_mri4/sample_007 | D-Flow | 614 | 612 | 12.9905 | 5.6464 | stalled |
| r96_n4_mri4/sample_008 | D-Flow | 497 | 495 | 11.2278 | 6.6207 | stalled |
| r96_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 5.5270 | 7.9004 | step_limit |
| r96_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 2.1184 | 9.8916 | step_limit |
| r96_n4_mri4/sample_011 | D-Flow | 561 | 558 | 12.6498 | 5.4694 | stalled |
| r96_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 12.9663 | 4.7590 | step_limit |
| r96_n4_mri4/sample_000 | P-Flow | 259 | 245 | 13.3088 | 3.7455 | stalled |
| r96_n4_mri4/sample_001 | P-Flow | 279 | 255 | 14.1324 | 5.1485 | stalled |
| r96_n4_mri4/sample_002 | P-Flow | 182 | 178 | 13.9048 | 4.8179 | stalled |
| r96_n4_mri4/sample_003 | P-Flow | 216 | 205 | 13.6842 | 5.1005 | stalled |
| r96_n4_mri4/sample_004 | P-Flow | 211 | 201 | 13.6656 | 4.8606 | stalled |
| r96_n4_mri4/sample_005 | P-Flow | 4 | 2 | 105.7076 | 89.8617 | stalled |
| r96_n4_mri4/sample_006 | P-Flow | 202 | 195 | 13.6490 | 3.8579 | stalled |
| r96_n4_mri4/sample_007 | P-Flow | 185 | 181 | 12.9798 | 5.6464 | stalled |
| r96_n4_mri4/sample_008 | P-Flow | 217 | 200 | 11.0068 | 6.6207 | stalled |
| r96_n4_mri4/sample_009 | P-Flow | 206 | 196 | 13.8738 | 4.2364 | stalled |
| r96_n4_mri4/sample_010 | P-Flow | 177 | 173 | 13.1999 | 5.4889 | stalled |
| r96_n4_mri4/sample_011 | P-Flow | 212 | 202 | 12.8085 | 5.4694 | stalled |
| r96_n4_mri4/sample_012 | P-Flow | 191 | 188 | 12.8796 | 4.7590 | stalled |
| r96_n4_mri4/sample_000 | DLR D-Flow | 165 | 157 | 58.2142 | 54.0627 | stalled |
| r96_n4_mri4/sample_001 | DLR D-Flow | 1000 | 961 | 46.4256 | 41.0650 | step_limit |
| r96_n4_mri4/sample_002 | DLR D-Flow | 485 | 466 | 11.9697 | 12.9665 | stalled |
| r96_n4_mri4/sample_003 | DLR D-Flow | 1000 | 968 | 55.4434 | 53.4956 | step_limit |
| r96_n4_mri4/sample_004 | DLR D-Flow | 1000 | 971 | 85.5837 | 75.6438 | step_limit |
| r96_n4_mri4/sample_005 | DLR D-Flow | 484 | 468 | 35.0158 | 27.4378 | stalled |
| r96_n4_mri4/sample_006 | DLR D-Flow | 465 | 442 | 63.9066 | 58.4637 | stalled |
| r96_n4_mri4/sample_007 | DLR D-Flow | 1000 | 971 | 64.5578 | 62.1904 | step_limit |
| r96_n4_mri4/sample_008 | DLR D-Flow | 252 | 244 | 62.9329 | 59.3335 | stalled |
| r96_n4_mri4/sample_009 | DLR D-Flow | 117 | 111 | 39.9537 | 40.1245 | stalled |
| r96_n4_mri4/sample_010 | DLR D-Flow | 1000 | 966 | 47.6856 | 47.7792 | step_limit |
| r96_n4_mri4/sample_011 | DLR D-Flow | 268 | 260 | 22.8220 | 20.6931 | stalled |
| r96_n4_mri4/sample_012 | DLR D-Flow | 1000 | 962 | 68.5376 | 66.4235 | step_limit |
| r128_n1_ct10/sample_000 | D-Flow | 491 | 489 | 38.4868 | 0.6403 | stalled |
| r128_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 9.5007 | 7.3161 | step_limit |
| r128_n1_ct10/sample_002 | D-Flow | 574 | 571 | 50.6973 | 0.8538 | stalled |
| r128_n1_ct10/sample_003 | D-Flow | 571 | 569 | 62.9652 | 0.6846 | stalled |
| r128_n1_ct10/sample_004 | D-Flow | 670 | 668 | 46.3148 | 0.5022 | stalled |
| r128_n1_ct10/sample_005 | D-Flow | 717 | 715 | 51.0587 | 0.7322 | stalled |
| r128_n1_ct10/sample_006 | D-Flow | 532 | 529 | 58.3395 | 0.6596 | stalled |
| r128_n1_ct10/sample_007 | D-Flow | 653 | 650 | 58.7500 | 0.5508 | stalled |
| r128_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 8.8555 | 6.4086 | step_limit |
| r128_n1_ct10/sample_009 | D-Flow | 713 | 711 | 44.6867 | 0.6070 | stalled |
| r128_n1_ct10/sample_010 | D-Flow | 697 | 692 | 113.2981 | 99.1963 | stalled |
| r128_n1_ct10/sample_011 | D-Flow | 534 | 532 | 64.0519 | 0.6957 | stalled |
| r128_n1_ct10/sample_012 | D-Flow | 414 | 411 | 75.2792 | 0.6116 | stalled |
| r128_n1_ct10/sample_013 | D-Flow | 1000 | 1000 | 43.0001 | 0.5626 | step_limit |
| r128_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 13.6319 | 3.0021 | step_limit |
| r128_n1_ct10/sample_015 | D-Flow | 670 | 668 | 51.4202 | 0.4514 | stalled |
| r128_n1_ct10/sample_016 | D-Flow | 852 | 850 | 54.5272 | 0.4238 | stalled |
| r128_n1_ct10/sample_017 | D-Flow | 953 | 950 | 64.7236 | 0.3296 | stalled |
| r128_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 6.8010 | 7.8282 | step_limit |
| r128_n1_ct10/sample_019 | D-Flow | 490 | 488 | 65.1761 | 0.3528 | stalled |
| r128_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 7.7424 | 7.2900 | step_limit |
| r128_n1_ct10/sample_021 | D-Flow | 378 | 376 | 74.8066 | 0.7901 | stalled |
| r128_n1_ct10/sample_022 | D-Flow | 733 | 731 | 64.7037 | 0.6231 | stalled |
| r128_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 7.2298 | 7.3674 | step_limit |
| r128_n1_ct10/sample_024 | D-Flow | 623 | 620 | 53.1758 | 0.2741 | stalled |
| r128_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 43.0847 | 0.6179 | step_limit |
| r128_n1_ct10/sample_026 | D-Flow | 894 | 892 | 48.8972 | 0.7684 | stalled |
| r128_n1_ct10/sample_027 | D-Flow | 629 | 627 | 73.8929 | 0.3492 | stalled |
| r128_n1_ct10/sample_028 | D-Flow | 1000 | 1000 | 116.4669 | 0.3988 | step_limit |
| r128_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 7.7310 | 7.3146 | step_limit |
| r128_n1_ct10/sample_000 | P-Flow | 567 | 535 | 44.1634 | 0.6403 | stalled |
| r128_n1_ct10/sample_001 | P-Flow | 546 | 532 | 51.3250 | 0.2120 | stalled |
| r128_n1_ct10/sample_002 | P-Flow | 69 | 61 | 4.2520 | 9.9761 | stalled |
| r128_n1_ct10/sample_003 | P-Flow | 1000 | 797 | 4.2800 | 9.9032 | step_limit |
| r128_n1_ct10/sample_004 | P-Flow | 955 | 907 | 45.3222 | 0.5098 | stalled |
| r128_n1_ct10/sample_005 | P-Flow | 620 | 581 | 53.4582 | 0.7325 | stalled |
| r128_n1_ct10/sample_006 | P-Flow | 642 | 614 | 54.8447 | 0.6596 | stalled |
| r128_n1_ct10/sample_007 | P-Flow | 44 | 37 | 3.7841 | 9.9661 | stalled |
| r128_n1_ct10/sample_008 | P-Flow | 145 | 118 | 4.5209 | 9.9753 | stalled |
| r128_n1_ct10/sample_009 | P-Flow | 40 | 33 | 3.3338 | 10.0496 | stalled |
| r128_n1_ct10/sample_010 | P-Flow | 44 | 36 | 3.9029 | 10.0288 | stalled |
| r128_n1_ct10/sample_011 | P-Flow | 67 | 57 | 4.0806 | 10.1387 | stalled |
| r128_n1_ct10/sample_012 | P-Flow | 47 | 39 | 4.1152 | 10.1580 | stalled |
| r128_n1_ct10/sample_013 | P-Flow | 37 | 31 | 5.8820 | 10.2638 | stalled |
| r128_n1_ct10/sample_014 | P-Flow | 18 | 15 | 25.0088 | 13.6797 | stalled |
| r128_n1_ct10/sample_015 | P-Flow | 30 | 26 | 5.6051 | 10.0725 | stalled |
| r128_n1_ct10/sample_016 | P-Flow | 1000 | 813 | 2.6939 | 9.9522 | step_limit |
| r128_n1_ct10/sample_017 | P-Flow | 394 | 384 | 58.4866 | 0.3296 | stalled |
| r128_n1_ct10/sample_018 | P-Flow | 29 | 25 | 2.7282 | 10.0141 | stalled |
| r128_n1_ct10/sample_019 | P-Flow | 80 | 71 | 2.8963 | 9.9709 | stalled |
| r128_n1_ct10/sample_020 | P-Flow | 40 | 34 | 7.0050 | 10.0081 | stalled |
| r128_n1_ct10/sample_021 | P-Flow | 732 | 590 | 2.3870 | 9.9077 | stalled |
| r128_n1_ct10/sample_022 | P-Flow | 1000 | 968 | 34.6793 | 1.0759 | step_limit |
| r128_n1_ct10/sample_023 | P-Flow | 39 | 32 | 12.4122 | 10.4643 | stalled |
| r128_n1_ct10/sample_024 | P-Flow | 26 | 21 | 3.9568 | 10.0744 | stalled |
| r128_n1_ct10/sample_025 | P-Flow | 36 | 30 | 3.9514 | 10.0227 | stalled |
| r128_n1_ct10/sample_026 | P-Flow | 73 | 63 | 3.2181 | 10.0098 | stalled |
| r128_n1_ct10/sample_027 | P-Flow | 54 | 47 | 2.8087 | 9.9271 | stalled |
| r128_n1_ct10/sample_028 | P-Flow | 491 | 478 | 57.9713 | 0.3984 | stalled |
| r128_n1_ct10/sample_029 | P-Flow | 651 | 620 | 59.3158 | 0.5780 | stalled |
| r128_n1_ct10/sample_000 | DLR D-Flow | 710 | 678 | 6.6434 | 10.1112 | stalled |
| r128_n1_ct10/sample_001 | DLR D-Flow | 151 | 144 | 3.4785 | 9.9502 | stalled |
| r128_n1_ct10/sample_002 | DLR D-Flow | 800 | 777 | 2.7068 | 9.9329 | stalled |
| r128_n1_ct10/sample_003 | DLR D-Flow | 293 | 281 | 4.9993 | 9.9557 | stalled |
| r128_n1_ct10/sample_004 | DLR D-Flow | 509 | 487 | 3.5150 | 9.9756 | stalled |
| r128_n1_ct10/sample_005 | DLR D-Flow | 574 | 552 | 3.7712 | 9.9210 | stalled |
| r128_n1_ct10/sample_006 | DLR D-Flow | 188 | 178 | 2.3479 | 9.9264 | stalled |
| r128_n1_ct10/sample_007 | DLR D-Flow | 416 | 400 | 6.8430 | 9.9131 | stalled |
| r128_n1_ct10/sample_008 | DLR D-Flow | 34 | 30 | 109.9352 | 98.8419 | stalled |
| r128_n1_ct10/sample_009 | DLR D-Flow | 377 | 355 | 3.2042 | 9.9381 | stalled |
| r128_n1_ct10/sample_010 | DLR D-Flow | 428 | 412 | 117.9597 | 111.1194 | stalled |
| r128_n1_ct10/sample_011 | DLR D-Flow | 232 | 229 | 3.0467 | 9.9593 | stalled |
| r128_n1_ct10/sample_012 | DLR D-Flow | 474 | 462 | 13.4761 | 10.4519 | stalled |
| r128_n1_ct10/sample_013 | DLR D-Flow | 89 | 85 | 5.5224 | 10.0493 | stalled |
| r128_n1_ct10/sample_014 | DLR D-Flow | 394 | 374 | 10.3152 | 10.4391 | stalled |
| r128_n1_ct10/sample_015 | DLR D-Flow | 1000 | 927 | 2.1818 | 9.9267 | step_limit |
| r128_n1_ct10/sample_016 | DLR D-Flow | 1000 | 966 | 7.2850 | 9.9050 | step_limit |
| r128_n1_ct10/sample_017 | DLR D-Flow | 138 | 133 | 4.5684 | 9.9594 | stalled |
| r128_n1_ct10/sample_018 | DLR D-Flow | 124 | 117 | 4.0470 | 9.9991 | stalled |
| r128_n1_ct10/sample_019 | DLR D-Flow | 178 | 170 | 11.5907 | 10.5857 | stalled |
| r128_n1_ct10/sample_020 | DLR D-Flow | 861 | 825 | 3.8475 | 9.8733 | stalled |
| r128_n1_ct10/sample_021 | DLR D-Flow | 1000 | 882 | 3.0023 | 9.9340 | step_limit |
| r128_n1_ct10/sample_022 | DLR D-Flow | 84 | 82 | 4.2587 | 9.8556 | stalled |
| r128_n1_ct10/sample_023 | DLR D-Flow | 1000 | 942 | 3.0296 | 9.9202 | step_limit |
| r128_n1_ct10/sample_024 | DLR D-Flow | 154 | 149 | 9.7344 | 10.3443 | stalled |
| r128_n1_ct10/sample_025 | DLR D-Flow | 754 | 714 | 4.4925 | 9.9414 | stalled |
| r128_n1_ct10/sample_026 | DLR D-Flow | 530 | 515 | 4.8947 | 10.2457 | stalled |
| r128_n1_ct10/sample_027 | DLR D-Flow | 866 | 842 | 3.7004 | 9.8897 | stalled |
| r128_n1_ct10/sample_028 | DLR D-Flow | 227 | 221 | 7.9882 | 9.8784 | stalled |
| r128_n1_ct10/sample_029 | DLR D-Flow | 486 | 465 | 2.8852 | 9.9463 | stalled |
| r128_n2_ct10/sample_000 | D-Flow | 723 | 721 | 90.1832 | 73.6499 | stalled |
| r128_n2_ct10/sample_001 | D-Flow | 549 | 547 | 68.3378 | 0.2764 | stalled |
| r128_n2_ct10/sample_002 | D-Flow | 622 | 620 | 60.0288 | 0.6322 | stalled |
| r128_n2_ct10/sample_003 | D-Flow | 958 | 955 | 60.7059 | 0.9898 | stalled |
| r128_n2_ct10/sample_004 | D-Flow | 936 | 934 | 43.1798 | 0.6223 | stalled |
| r128_n2_ct10/sample_005 | D-Flow | 651 | 649 | 64.0942 | 0.6732 | stalled |
| r128_n2_ct10/sample_006 | D-Flow | 563 | 561 | 72.2298 | 0.8736 | stalled |
| r128_n2_ct10/sample_007 | D-Flow | 571 | 568 | 61.4653 | 0.7437 | stalled |
| r128_n2_ct10/sample_008 | D-Flow | 579 | 576 | 78.2945 | 0.4737 | stalled |
| r128_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 47.1944 | 0.6640 | step_limit |
| r128_n2_ct10/sample_010 | D-Flow | 582 | 579 | 60.8236 | 1.1627 | stalled |
| r128_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 63.7697 | 0.2164 | step_limit |
| r128_n2_ct10/sample_012 | D-Flow | 467 | 465 | 82.1597 | 0.5957 | stalled |
| r128_n2_ct10/sample_000 | P-Flow | 61 | 51 | 3.3585 | 9.8985 | stalled |
| r128_n2_ct10/sample_001 | P-Flow | 53 | 45 | 4.9185 | 9.9471 | stalled |
| r128_n2_ct10/sample_002 | P-Flow | 45 | 37 | 7.9346 | 10.1868 | stalled |
| r128_n2_ct10/sample_003 | P-Flow | 73 | 63 | 3.0115 | 10.0144 | stalled |
| r128_n2_ct10/sample_004 | P-Flow | 1000 | 940 | 35.3633 | 0.8440 | step_limit |
| r128_n2_ct10/sample_005 | P-Flow | 655 | 540 | 2.7855 | 10.0252 | stalled |
| r128_n2_ct10/sample_006 | P-Flow | 808 | 743 | 48.3917 | 0.8975 | stalled |
| r128_n2_ct10/sample_007 | P-Flow | 483 | 467 | 48.8129 | 0.7438 | stalled |
| r128_n2_ct10/sample_008 | P-Flow | 16 | 13 | 47.5428 | 21.3199 | stalled |
| r128_n2_ct10/sample_009 | P-Flow | 42 | 34 | 8.2830 | 10.1840 | stalled |
| r128_n2_ct10/sample_010 | P-Flow | 480 | 465 | 50.2257 | 1.1627 | stalled |
| r128_n2_ct10/sample_011 | P-Flow | 129 | 108 | 3.7445 | 9.9065 | stalled |
| r128_n2_ct10/sample_012 | P-Flow | 578 | 554 | 53.8939 | 0.5957 | stalled |
| r128_n2_ct10/sample_000 | DLR D-Flow | 48 | 45 | 114.2627 | 89.2665 | stalled |
| r128_n2_ct10/sample_001 | DLR D-Flow | 1000 | 982 | 14.3053 | 10.6921 | step_limit |
| r128_n2_ct10/sample_002 | DLR D-Flow | 698 | 673 | 3.0943 | 9.9007 | stalled |
| r128_n2_ct10/sample_003 | DLR D-Flow | 84 | 80 | 119.6619 | 87.1555 | stalled |
| r128_n2_ct10/sample_004 | DLR D-Flow | 1000 | 969 | 29.0400 | 15.9731 | step_limit |
| r128_n2_ct10/sample_005 | DLR D-Flow | 233 | 218 | 50.9130 | 27.7597 | stalled |
| r128_n2_ct10/sample_006 | DLR D-Flow | 214 | 206 | 4.1600 | 9.9357 | stalled |
| r128_n2_ct10/sample_007 | DLR D-Flow | 597 | 585 | 56.3040 | 29.4527 | stalled |
| r128_n2_ct10/sample_008 | DLR D-Flow | 508 | 478 | 4.3223 | 9.9214 | stalled |
| r128_n2_ct10/sample_009 | DLR D-Flow | 118 | 112 | 103.5819 | 73.6985 | stalled |
| r128_n2_ct10/sample_010 | DLR D-Flow | 193 | 185 | 3.5453 | 10.0065 | stalled |
| r128_n2_ct10/sample_011 | DLR D-Flow | 138 | 130 | 82.6023 | 55.1758 | stalled |
| r128_n2_ct10/sample_012 | DLR D-Flow | 302 | 295 | 12.7894 | 10.4232 | stalled |
| r128_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 47.9792 | 26.4120 | step_limit |
| r128_n4_ct10/sample_001 | D-Flow | 989 | 986 | 60.6293 | 0.4258 | stalled |
| r128_n4_ct10/sample_002 | D-Flow | 996 | 994 | 43.4542 | 24.0177 | stalled |
| r128_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 33.6405 | 6.6672 | step_limit |
| r128_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 61.2260 | 0.3306 | step_limit |
| r128_n4_ct10/sample_005 | D-Flow | 557 | 555 | 69.1116 | 34.7994 | stalled |
| r128_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 49.0406 | 1.0700 | step_limit |
| r128_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 32.7011 | 6.6152 | step_limit |
| r128_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 77.4926 | 0.4641 | step_limit |
| r128_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 47.0673 | 0.9208 | step_limit |
| r128_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 61.9856 | 0.7101 | step_limit |
| r128_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 50.4130 | 29.4867 | step_limit |
| r128_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 10.5726 | 9.4148 | step_limit |
| r128_n4_ct10/sample_000 | P-Flow | 94 | 87 | 6.1443 | 9.7903 | stalled |
| r128_n4_ct10/sample_001 | P-Flow | 1000 | 881 | 39.4631 | 0.7045 | step_limit |
| r128_n4_ct10/sample_002 | P-Flow | 1000 | 910 | 40.5779 | 0.8429 | step_limit |
| r128_n4_ct10/sample_003 | P-Flow | 58 | 48 | 8.0760 | 10.0712 | stalled |
| r128_n4_ct10/sample_004 | P-Flow | 67 | 56 | 6.2044 | 10.0142 | stalled |
| r128_n4_ct10/sample_005 | P-Flow | 1000 | 898 | 36.2682 | 1.3688 | step_limit |
| r128_n4_ct10/sample_006 | P-Flow | 408 | 341 | 7.0308 | 9.2981 | stalled |
| r128_n4_ct10/sample_007 | P-Flow | 67 | 60 | 5.6063 | 9.8267 | stalled |
| r128_n4_ct10/sample_008 | P-Flow | 128 | 115 | 7.9361 | 9.1507 | stalled |
| r128_n4_ct10/sample_009 | P-Flow | 59 | 48 | 6.0976 | 10.0055 | stalled |
| r128_n4_ct10/sample_010 | P-Flow | 1000 | 875 | 34.9800 | 1.3567 | step_limit |
| r128_n4_ct10/sample_011 | P-Flow | 1000 | 893 | 37.6080 | 1.0026 | step_limit |
| r128_n4_ct10/sample_012 | P-Flow | 1000 | 909 | 36.0435 | 1.3826 | step_limit |
| r128_n4_ct10/sample_000 | DLR D-Flow | 1000 | 977 | 65.0534 | 32.8466 | step_limit |
| r128_n4_ct10/sample_001 | DLR D-Flow | 225 | 216 | 39.7404 | 20.4926 | stalled |
| r128_n4_ct10/sample_002 | DLR D-Flow | 78 | 70 | 94.3229 | 50.7492 | stalled |
| r128_n4_ct10/sample_003 | DLR D-Flow | 341 | 327 | 70.1762 | 36.9822 | stalled |
| r128_n4_ct10/sample_004 | DLR D-Flow | 421 | 406 | 79.4599 | 39.4473 | stalled |
| r128_n4_ct10/sample_005 | DLR D-Flow | 509 | 493 | 86.8608 | 46.9746 | stalled |
| r128_n4_ct10/sample_006 | DLR D-Flow | 1000 | 964 | 53.7053 | 25.3625 | step_limit |
| r128_n4_ct10/sample_007 | DLR D-Flow | 1000 | 965 | 55.5655 | 29.3033 | step_limit |
| r128_n4_ct10/sample_008 | DLR D-Flow | 252 | 240 | 41.9294 | 25.1873 | stalled |
| r128_n4_ct10/sample_009 | DLR D-Flow | 210 | 205 | 66.8054 | 34.0529 | stalled |
| r128_n4_ct10/sample_010 | DLR D-Flow | 875 | 843 | 78.9518 | 44.8172 | stalled |
| r128_n4_ct10/sample_011 | DLR D-Flow | 764 | 738 | 94.9855 | 53.9901 | stalled |
| r128_n4_ct10/sample_012 | DLR D-Flow | 319 | 310 | 44.5079 | 24.1527 | stalled |
| r128_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 10.5019 | 4.9105 | step_limit |
| r128_n1_mri4/sample_001 | D-Flow | 643 | 640 | 12.0251 | 5.3138 | stalled |
| r128_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 8.7044 | 5.6768 | step_limit |
| r128_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 7.1785 | 7.6371 | step_limit |
| r128_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 8.3404 | 6.2294 | step_limit |
| r128_n1_mri4/sample_005 | D-Flow | 503 | 500 | 12.3897 | 5.4904 | stalled |
| r128_n1_mri4/sample_006 | D-Flow | 578 | 576 | 12.8458 | 4.7458 | stalled |
| r128_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 6.7763 | 6.7565 | step_limit |
| r128_n1_mri4/sample_008 | D-Flow | 358 | 356 | 104.6343 | 101.6797 | stalled |
| r128_n1_mri4/sample_009 | D-Flow | 571 | 568 | 13.0375 | 5.1585 | stalled |
| r128_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 11.0147 | 5.1592 | step_limit |
| r128_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 5.5987 | 8.1084 | step_limit |
| r128_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 7.3812 | 6.2528 | step_limit |
| r128_n1_mri4/sample_013 | D-Flow | 813 | 810 | 12.2146 | 4.0378 | stalled |
| r128_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 10.1563 | 6.3694 | step_limit |
| r128_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 8.2965 | 5.8362 | step_limit |
| r128_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 9.6584 | 5.5367 | step_limit |
| r128_n1_mri4/sample_017 | D-Flow | 622 | 619 | 12.9762 | 4.7667 | stalled |
| r128_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 6.2628 | 7.2156 | step_limit |
| r128_n1_mri4/sample_019 | D-Flow | 426 | 424 | 11.8089 | 4.7331 | stalled |
| r128_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 6.5430 | 7.6714 | step_limit |
| r128_n1_mri4/sample_021 | D-Flow | 512 | 510 | 11.3668 | 5.7351 | stalled |
| r128_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 8.1741 | 6.6346 | step_limit |
| r128_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 6.4382 | 7.4909 | step_limit |
| r128_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 6.4084 | 7.9697 | step_limit |
| r128_n1_mri4/sample_025 | D-Flow | 667 | 664 | 12.1312 | 5.2149 | stalled |
| r128_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 12.0326 | 4.7908 | step_limit |
| r128_n1_mri4/sample_027 | D-Flow | 474 | 472 | 12.6140 | 5.7559 | stalled |
| r128_n1_mri4/sample_028 | D-Flow | 488 | 486 | 12.7819 | 5.0065 | stalled |
| r128_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 6.5936 | 7.6881 | step_limit |
| r128_n1_mri4/sample_000 | P-Flow | 152 | 149 | 15.1210 | 4.4486 | stalled |
| r128_n1_mri4/sample_001 | P-Flow | 31 | 24 | 1.7073 | 10.0300 | stalled |
| r128_n1_mri4/sample_002 | P-Flow | 30 | 24 | 2.1742 | 10.0425 | stalled |
| r128_n1_mri4/sample_003 | P-Flow | 182 | 173 | 14.5589 | 5.3924 | stalled |
| r128_n1_mri4/sample_004 | P-Flow | 178 | 166 | 13.2315 | 5.4355 | stalled |
| r128_n1_mri4/sample_005 | P-Flow | 28 | 22 | 1.9253 | 10.0125 | stalled |
| r128_n1_mri4/sample_006 | P-Flow | 25 | 21 | 1.7048 | 10.0080 | stalled |
| r128_n1_mri4/sample_007 | P-Flow | 189 | 175 | 15.5725 | 4.9071 | stalled |
| r128_n1_mri4/sample_008 | P-Flow | 176 | 165 | 13.4361 | 5.5060 | stalled |
| r128_n1_mri4/sample_009 | P-Flow | 34 | 27 | 1.5829 | 9.9684 | stalled |
| r128_n1_mri4/sample_010 | P-Flow | 158 | 155 | 14.9110 | 4.9678 | stalled |
| r128_n1_mri4/sample_011 | P-Flow | 172 | 162 | 13.0567 | 5.5086 | stalled |
| r128_n1_mri4/sample_012 | P-Flow | 181 | 172 | 15.2614 | 4.7804 | stalled |
| r128_n1_mri4/sample_013 | P-Flow | 37 | 32 | 65.6223 | 62.8525 | stalled |
| r128_n1_mri4/sample_014 | P-Flow | 210 | 195 | 15.2743 | 4.7930 | stalled |
| r128_n1_mri4/sample_015 | P-Flow | 159 | 155 | 14.2672 | 4.8199 | stalled |
| r128_n1_mri4/sample_016 | P-Flow | 46 | 36 | 1.9803 | 10.0346 | stalled |
| r128_n1_mri4/sample_017 | P-Flow | 170 | 162 | 15.1618 | 4.7667 | stalled |
| r128_n1_mri4/sample_018 | P-Flow | 27 | 22 | 1.8001 | 9.9993 | stalled |
| r128_n1_mri4/sample_019 | P-Flow | 171 | 162 | 13.4425 | 4.7331 | stalled |
| r128_n1_mri4/sample_020 | P-Flow | 227 | 208 | 14.3275 | 5.7211 | stalled |
| r128_n1_mri4/sample_021 | P-Flow | 49 | 40 | 1.7962 | 9.9868 | stalled |
| r128_n1_mri4/sample_022 | P-Flow | 180 | 170 | 12.6846 | 5.1403 | stalled |
| r128_n1_mri4/sample_023 | P-Flow | 50 | 38 | 1.7190 | 10.0053 | stalled |
| r128_n1_mri4/sample_024 | P-Flow | 39 | 32 | 2.0365 | 9.9343 | stalled |
| r128_n1_mri4/sample_025 | P-Flow | 231 | 210 | 14.7083 | 5.2149 | stalled |
| r128_n1_mri4/sample_026 | P-Flow | 40 | 33 | 2.8461 | 10.1155 | stalled |
| r128_n1_mri4/sample_027 | P-Flow | 203 | 186 | 14.4271 | 5.7559 | stalled |
| r128_n1_mri4/sample_028 | P-Flow | 172 | 164 | 15.3932 | 5.0065 | stalled |
| r128_n1_mri4/sample_029 | P-Flow | 43 | 37 | 1.3606 | 9.9956 | stalled |
| r128_n1_mri4/sample_000 | DLR D-Flow | 191 | 184 | 37.5791 | 33.3708 | stalled |
| r128_n1_mri4/sample_001 | DLR D-Flow | 805 | 766 | 1.4087 | 9.9907 | stalled |
| r128_n1_mri4/sample_002 | DLR D-Flow | 1000 | 983 | 5.0571 | 10.5601 | step_limit |
| r128_n1_mri4/sample_003 | DLR D-Flow | 205 | 195 | 103.2360 | 102.8360 | stalled |
| r128_n1_mri4/sample_004 | DLR D-Flow | 416 | 396 | 2.1818 | 10.0589 | stalled |
| r128_n1_mri4/sample_005 | DLR D-Flow | 315 | 303 | 1.9324 | 9.9989 | stalled |
| r128_n1_mri4/sample_006 | DLR D-Flow | 415 | 394 | 2.1236 | 10.0642 | stalled |
| r128_n1_mri4/sample_007 | DLR D-Flow | 648 | 618 | 1.4494 | 9.9787 | stalled |
| r128_n1_mri4/sample_008 | DLR D-Flow | 18 | 16 | 104.4726 | 104.3591 | stalled |
| r128_n1_mri4/sample_009 | DLR D-Flow | 1000 | 953 | 1.3650 | 9.9431 | step_limit |
| r128_n1_mri4/sample_010 | DLR D-Flow | 375 | 359 | 106.1801 | 104.7287 | stalled |
| r128_n1_mri4/sample_011 | DLR D-Flow | 403 | 385 | 1.7102 | 10.0150 | stalled |
| r128_n1_mri4/sample_012 | DLR D-Flow | 23 | 21 | 109.8521 | 105.7126 | stalled |
| r128_n1_mri4/sample_013 | DLR D-Flow | 198 | 185 | 2.0806 | 10.0534 | stalled |
| r128_n1_mri4/sample_014 | DLR D-Flow | 1000 | 958 | 2.9903 | 10.1666 | step_limit |
| r128_n1_mri4/sample_015 | DLR D-Flow | 229 | 217 | 1.6714 | 10.0488 | stalled |
| r128_n1_mri4/sample_016 | DLR D-Flow | 482 | 455 | 2.2835 | 10.0548 | stalled |
| r128_n1_mri4/sample_017 | DLR D-Flow | 245 | 237 | 1.5756 | 9.9888 | stalled |
| r128_n1_mri4/sample_018 | DLR D-Flow | 275 | 267 | 1.2853 | 9.9462 | stalled |
| r128_n1_mri4/sample_019 | DLR D-Flow | 351 | 335 | 2.3610 | 10.1026 | stalled |
| r128_n1_mri4/sample_020 | DLR D-Flow | 605 | 589 | 1.4112 | 9.9779 | stalled |
| r128_n1_mri4/sample_021 | DLR D-Flow | 643 | 622 | 1.5003 | 9.9740 | stalled |
| r128_n1_mri4/sample_022 | DLR D-Flow | 119 | 114 | 2.1477 | 10.0159 | stalled |
| r128_n1_mri4/sample_023 | DLR D-Flow | 131 | 123 | 2.4629 | 10.1156 | stalled |
| r128_n1_mri4/sample_024 | DLR D-Flow | 82 | 79 | 4.2795 | 10.5929 | stalled |
| r128_n1_mri4/sample_025 | DLR D-Flow | 394 | 378 | 49.0054 | 45.3481 | stalled |
| r128_n1_mri4/sample_026 | DLR D-Flow | 44 | 40 | 7.9303 | 11.4641 | stalled |
| r128_n1_mri4/sample_027 | DLR D-Flow | 680 | 654 | 6.2038 | 11.4225 | stalled |
| r128_n1_mri4/sample_028 | DLR D-Flow | 1000 | 959 | 1.9043 | 10.0109 | step_limit |
| r128_n1_mri4/sample_029 | DLR D-Flow | 1000 | 946 | 1.5180 | 9.9942 | step_limit |
| r128_n2_mri4/sample_000 | D-Flow | 742 | 740 | 75.8322 | 75.7972 | stalled |
| r128_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 2.9696 | 9.4410 | step_limit |
| r128_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 3.4091 | 9.4729 | step_limit |
| r128_n2_mri4/sample_003 | D-Flow | 742 | 740 | 71.5244 | 71.0961 | stalled |
| r128_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 2.8646 | 9.7468 | step_limit |
| r128_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 12.3461 | 4.8807 | step_limit |
| r128_n2_mri4/sample_006 | D-Flow | 597 | 595 | 12.0238 | 4.8453 | stalled |
| r128_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 10.3978 | 8.0626 | step_limit |
| r128_n2_mri4/sample_008 | D-Flow | 704 | 702 | 11.9965 | 4.8805 | stalled |
| r128_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.7762 | 9.9730 | step_limit |
| r128_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 3.8398 | 9.2291 | step_limit |
| r128_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 3.0601 | 10.2269 | step_limit |
| r128_n2_mri4/sample_012 | D-Flow | 894 | 892 | 11.2726 | 5.1126 | stalled |
| r128_n2_mri4/sample_000 | P-Flow | 185 | 175 | 15.1650 | 4.9022 | stalled |
| r128_n2_mri4/sample_001 | P-Flow | 213 | 202 | 12.9718 | 4.4337 | stalled |
| r128_n2_mri4/sample_002 | P-Flow | 180 | 170 | 13.6383 | 5.5815 | stalled |
| r128_n2_mri4/sample_003 | P-Flow | 179 | 171 | 14.7571 | 4.8595 | stalled |
| r128_n2_mri4/sample_004 | P-Flow | 167 | 160 | 14.7871 | 5.4791 | stalled |
| r128_n2_mri4/sample_005 | P-Flow | 173 | 166 | 14.8771 | 4.8697 | stalled |
| r128_n2_mri4/sample_006 | P-Flow | 183 | 175 | 14.3882 | 4.8453 | stalled |
| r128_n2_mri4/sample_007 | P-Flow | 179 | 170 | 14.7443 | 4.8027 | stalled |
| r128_n2_mri4/sample_008 | P-Flow | 154 | 151 | 14.2560 | 4.8805 | stalled |
| r128_n2_mri4/sample_009 | P-Flow | 176 | 169 | 13.5210 | 5.5409 | stalled |
| r128_n2_mri4/sample_010 | P-Flow | 202 | 186 | 14.2936 | 4.8975 | stalled |
| r128_n2_mri4/sample_011 | P-Flow | 175 | 166 | 14.5691 | 5.0312 | stalled |
| r128_n2_mri4/sample_012 | P-Flow | 30 | 25 | 95.9683 | 63.4310 | stalled |
| r128_n2_mri4/sample_000 | DLR D-Flow | 1000 | 954 | 41.2081 | 37.6435 | step_limit |
| r128_n2_mri4/sample_001 | DLR D-Flow | 1000 | 953 | 1.9989 | 10.0626 | step_limit |
| r128_n2_mri4/sample_002 | DLR D-Flow | 1000 | 956 | 52.4581 | 52.8542 | step_limit |
| r128_n2_mri4/sample_003 | DLR D-Flow | 285 | 270 | 118.2166 | 117.6011 | stalled |
| r128_n2_mri4/sample_004 | DLR D-Flow | 1000 | 966 | 2.8359 | 10.0933 | step_limit |
| r128_n2_mri4/sample_005 | DLR D-Flow | 176 | 169 | 68.5615 | 64.8803 | stalled |
| r128_n2_mri4/sample_006 | DLR D-Flow | 728 | 697 | 1.7384 | 10.0015 | stalled |
| r128_n2_mri4/sample_007 | DLR D-Flow | 1000 | 966 | 3.9438 | 10.2703 | step_limit |
| r128_n2_mri4/sample_008 | DLR D-Flow | 1000 | 961 | 1.5425 | 9.9907 | step_limit |
| r128_n2_mri4/sample_009 | DLR D-Flow | 156 | 152 | 116.4311 | 111.1455 | stalled |
| r128_n2_mri4/sample_010 | DLR D-Flow | 1000 | 972 | 1.6830 | 9.9929 | step_limit |
| r128_n2_mri4/sample_011 | DLR D-Flow | 1000 | 977 | 55.8662 | 53.9782 | step_limit |
| r128_n2_mri4/sample_012 | DLR D-Flow | 301 | 291 | 93.8935 | 75.9575 | stalled |
| r128_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 3.4099 | 9.6218 | step_limit |
| r128_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 5.0770 | 9.1326 | step_limit |
| r128_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 3.7520 | 10.3374 | step_limit |
| r128_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 3.8330 | 9.5703 | step_limit |
| r128_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 5.0318 | 9.2078 | step_limit |
| r128_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 2.6091 | 9.7573 | step_limit |
| r128_n4_mri4/sample_006 | D-Flow | 634 | 632 | 12.5566 | 4.1231 | stalled |
| r128_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 2.2523 | 9.8560 | step_limit |
| r128_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 2.4311 | 9.7474 | step_limit |
| r128_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 35.2114 | 34.4216 | step_limit |
| r128_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 2.4937 | 9.6426 | step_limit |
| r128_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 3.1750 | 9.6395 | step_limit |
| r128_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 7.8181 | 8.6271 | step_limit |
| r128_n4_mri4/sample_000 | P-Flow | 163 | 160 | 14.0213 | 4.0917 | stalled |
| r128_n4_mri4/sample_001 | P-Flow | 232 | 216 | 13.6304 | 5.1817 | stalled |
| r128_n4_mri4/sample_002 | P-Flow | 175 | 172 | 13.6390 | 5.1361 | stalled |
| r128_n4_mri4/sample_003 | P-Flow | 185 | 178 | 12.9034 | 5.7008 | stalled |
| r128_n4_mri4/sample_004 | P-Flow | 195 | 189 | 13.8668 | 4.8286 | stalled |
| r128_n4_mri4/sample_005 | P-Flow | 227 | 213 | 12.9254 | 5.6523 | stalled |
| r128_n4_mri4/sample_006 | P-Flow | 179 | 177 | 13.0847 | 4.1231 | stalled |
| r128_n4_mri4/sample_007 | P-Flow | 178 | 175 | 12.2090 | 5.4890 | stalled |
| r128_n4_mri4/sample_008 | P-Flow | 183 | 178 | 12.2716 | 5.5041 | stalled |
| r128_n4_mri4/sample_009 | P-Flow | 190 | 183 | 13.3886 | 5.1335 | stalled |
| r128_n4_mri4/sample_010 | P-Flow | 211 | 203 | 12.9783 | 4.6808 | stalled |
| r128_n4_mri4/sample_011 | P-Flow | 259 | 244 | 13.3415 | 4.5196 | stalled |
| r128_n4_mri4/sample_012 | P-Flow | 245 | 231 | 13.1625 | 5.4379 | stalled |
| r128_n4_mri4/sample_000 | DLR D-Flow | 163 | 158 | 68.8214 | 60.4686 | stalled |
| r128_n4_mri4/sample_001 | DLR D-Flow | 477 | 458 | 52.8097 | 47.9316 | stalled |
| r128_n4_mri4/sample_002 | DLR D-Flow | 74 | 70 | 91.2308 | 89.3644 | stalled |
| r128_n4_mri4/sample_003 | DLR D-Flow | 25 | 21 | 106.3667 | 104.8604 | stalled |
| r128_n4_mri4/sample_004 | DLR D-Flow | 456 | 443 | 41.6541 | 37.8333 | stalled |
| r128_n4_mri4/sample_005 | DLR D-Flow | 117 | 113 | 88.7380 | 86.4177 | stalled |
| r128_n4_mri4/sample_006 | DLR D-Flow | 1000 | 964 | 38.3081 | 38.1481 | step_limit |
| r128_n4_mri4/sample_007 | DLR D-Flow | 445 | 432 | 43.2389 | 42.4866 | stalled |
| r128_n4_mri4/sample_008 | DLR D-Flow | 847 | 807 | 51.4077 | 48.6320 | stalled |
| r128_n4_mri4/sample_009 | DLR D-Flow | 1000 | 965 | 53.8751 | 48.1859 | step_limit |
| r128_n4_mri4/sample_010 | DLR D-Flow | 865 | 838 | 79.8438 | 75.3566 | stalled |
| r128_n4_mri4/sample_011 | DLR D-Flow | 626 | 602 | 71.7722 | 66.4782 | stalled |
| r128_n4_mri4/sample_012 | DLR D-Flow | 443 | 426 | 62.5448 | 59.7983 | stalled |
| r160_n1_ct10/sample_000 | D-Flow | 769 | 767 | 54.5845 | 0.3026 | stalled |
| r160_n1_ct10/sample_001 | D-Flow | 560 | 558 | 59.4053 | 0.5805 | stalled |
| r160_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 17.1570 | 4.1298 | step_limit |
| r160_n1_ct10/sample_003 | D-Flow | 602 | 600 | 52.6586 | 0.5673 | stalled |
| r160_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 13.0074 | 7.1987 | step_limit |
| r160_n1_ct10/sample_005 | D-Flow | 615 | 612 | 82.1501 | 0.2897 | stalled |
| r160_n1_ct10/sample_006 | D-Flow | 483 | 481 | 110.3156 | 0.3649 | stalled |
| r160_n1_ct10/sample_007 | D-Flow | 495 | 493 | 80.4788 | 0.6269 | stalled |
| r160_n1_ct10/sample_008 | D-Flow | 594 | 592 | 49.8210 | 0.3610 | stalled |
| r160_n1_ct10/sample_009 | D-Flow | 662 | 660 | 58.4761 | 0.6992 | stalled |
| r160_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 13.3663 | 7.3376 | step_limit |
| r160_n1_ct10/sample_011 | D-Flow | 776 | 773 | 60.0069 | 0.3846 | stalled |
| r160_n1_ct10/sample_012 | D-Flow | 654 | 652 | 67.2472 | 0.5609 | stalled |
| r160_n1_ct10/sample_013 | D-Flow | 579 | 577 | 20.4390 | 1.7647 | stalled |
| r160_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 17.7945 | 3.3688 | step_limit |
| r160_n1_ct10/sample_015 | D-Flow | 301 | 299 | 100.1647 | 86.2136 | stalled |
| r160_n1_ct10/sample_016 | D-Flow | 589 | 587 | 59.4918 | 0.4907 | stalled |
| r160_n1_ct10/sample_017 | D-Flow | 819 | 817 | 47.0915 | 0.6204 | stalled |
| r160_n1_ct10/sample_018 | D-Flow | 600 | 598 | 19.3439 | 1.5856 | stalled |
| r160_n1_ct10/sample_019 | D-Flow | 411 | 409 | 83.9555 | 0.7427 | stalled |
| r160_n1_ct10/sample_020 | D-Flow | 37 | 34 | 103.1443 | 91.8081 | stalled |
| r160_n1_ct10/sample_021 | D-Flow | 428 | 426 | 107.3201 | 0.8240 | stalled |
| r160_n1_ct10/sample_022 | D-Flow | 124 | 121 | 93.6358 | 79.8603 | stalled |
| r160_n1_ct10/sample_023 | D-Flow | 659 | 657 | 62.7526 | 0.6544 | stalled |
| r160_n1_ct10/sample_024 | D-Flow | 698 | 696 | 61.2298 | 0.4583 | stalled |
| r160_n1_ct10/sample_025 | D-Flow | 518 | 516 | 80.1462 | 0.2889 | stalled |
| r160_n1_ct10/sample_026 | D-Flow | 727 | 725 | 63.9210 | 0.3692 | stalled |
| r160_n1_ct10/sample_027 | D-Flow | 804 | 802 | 70.7258 | 0.6404 | stalled |
| r160_n1_ct10/sample_028 | D-Flow | 139 | 136 | 94.4542 | 79.1530 | stalled |
| r160_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 15.3991 | 7.0757 | step_limit |
| r160_n1_ct10/sample_000 | P-Flow | 520 | 433 | 8.2976 | 9.1620 | stalled |
| r160_n1_ct10/sample_001 | P-Flow | 37 | 31 | 5.8297 | 10.5527 | stalled |
| r160_n1_ct10/sample_002 | P-Flow | 153 | 127 | 2.1538 | 9.9839 | stalled |
| r160_n1_ct10/sample_003 | P-Flow | 914 | 887 | 42.5759 | 0.6046 | stalled |
| r160_n1_ct10/sample_004 | P-Flow | 36 | 31 | 7.0409 | 10.0842 | stalled |
| r160_n1_ct10/sample_005 | P-Flow | 35 | 28 | 5.6858 | 10.1263 | stalled |
| r160_n1_ct10/sample_006 | P-Flow | 52 | 43 | 4.0534 | 9.9592 | stalled |
| r160_n1_ct10/sample_007 | P-Flow | 862 | 805 | 58.0783 | 0.6390 | stalled |
| r160_n1_ct10/sample_008 | P-Flow | 1000 | 964 | 38.0118 | 0.7436 | step_limit |
| r160_n1_ct10/sample_009 | P-Flow | 677 | 627 | 51.7605 | 0.6993 | stalled |
| r160_n1_ct10/sample_010 | P-Flow | 515 | 483 | 51.2536 | 1.0501 | stalled |
| r160_n1_ct10/sample_011 | P-Flow | 129 | 115 | 8.5962 | 9.0919 | stalled |
| r160_n1_ct10/sample_012 | P-Flow | 746 | 693 | 48.3876 | 0.5758 | stalled |
| r160_n1_ct10/sample_013 | P-Flow | 929 | 853 | 47.5383 | 0.3848 | stalled |
| r160_n1_ct10/sample_014 | P-Flow | 412 | 341 | 3.0383 | 9.9903 | stalled |
| r160_n1_ct10/sample_015 | P-Flow | 1000 | 956 | 37.5554 | 0.8043 | step_limit |
| r160_n1_ct10/sample_016 | P-Flow | 166 | 143 | 12.1207 | 8.0910 | stalled |
| r160_n1_ct10/sample_017 | P-Flow | 1000 | 825 | 27.4762 | 3.5432 | step_limit |
| r160_n1_ct10/sample_018 | P-Flow | 42 | 36 | 3.6180 | 9.9155 | stalled |
| r160_n1_ct10/sample_019 | P-Flow | 49 | 42 | 5.3781 | 10.0629 | stalled |
| r160_n1_ct10/sample_020 | P-Flow | 1000 | 954 | 52.4649 | 0.1693 | step_limit |
| r160_n1_ct10/sample_021 | P-Flow | 33 | 26 | 4.6768 | 10.1540 | stalled |
| r160_n1_ct10/sample_022 | P-Flow | 25 | 21 | 3.8719 | 9.9436 | stalled |
| r160_n1_ct10/sample_023 | P-Flow | 41 | 35 | 4.6636 | 9.9692 | stalled |
| r160_n1_ct10/sample_024 | P-Flow | 578 | 556 | 58.7286 | 0.4599 | stalled |
| r160_n1_ct10/sample_025 | P-Flow | 18 | 15 | 3.3395 | 10.1246 | stalled |
| r160_n1_ct10/sample_026 | P-Flow | 90 | 76 | 2.3983 | 9.9339 | stalled |
| r160_n1_ct10/sample_027 | P-Flow | 301 | 254 | 5.4050 | 9.8967 | stalled |
| r160_n1_ct10/sample_028 | P-Flow | 883 | 806 | 42.7432 | 0.7150 | stalled |
| r160_n1_ct10/sample_029 | P-Flow | 750 | 666 | 49.6873 | 0.6751 | stalled |
| r160_n1_ct10/sample_000 | DLR D-Flow | 26 | 24 | 89.9310 | 75.3792 | stalled |
| r160_n1_ct10/sample_001 | DLR D-Flow | 309 | 297 | 4.5594 | 9.9946 | stalled |
| r160_n1_ct10/sample_002 | DLR D-Flow | 42 | 39 | 97.7102 | 83.4034 | stalled |
| r160_n1_ct10/sample_003 | DLR D-Flow | 284 | 275 | 2.6519 | 9.9667 | stalled |
| r160_n1_ct10/sample_004 | DLR D-Flow | 330 | 319 | 4.3253 | 9.9131 | stalled |
| r160_n1_ct10/sample_005 | DLR D-Flow | 799 | 777 | 2.3387 | 9.9067 | stalled |
| r160_n1_ct10/sample_006 | DLR D-Flow | 569 | 549 | 4.1594 | 9.9477 | stalled |
| r160_n1_ct10/sample_007 | DLR D-Flow | 593 | 568 | 5.3271 | 9.9396 | stalled |
| r160_n1_ct10/sample_008 | DLR D-Flow | 264 | 252 | 5.4164 | 9.8863 | stalled |
| r160_n1_ct10/sample_009 | DLR D-Flow | 253 | 245 | 5.2106 | 10.0024 | stalled |
| r160_n1_ct10/sample_010 | DLR D-Flow | 81 | 78 | 2.1507 | 9.9382 | stalled |
| r160_n1_ct10/sample_011 | DLR D-Flow | 489 | 471 | 2.5378 | 9.9471 | stalled |
| r160_n1_ct10/sample_012 | DLR D-Flow | 880 | 851 | 6.7642 | 9.8874 | stalled |
| r160_n1_ct10/sample_013 | DLR D-Flow | 219 | 210 | 4.1224 | 9.9464 | stalled |
| r160_n1_ct10/sample_014 | DLR D-Flow | 1000 | 966 | 3.4280 | 10.0148 | step_limit |
| r160_n1_ct10/sample_015 | DLR D-Flow | 16 | 13 | 107.0703 | 91.6292 | stalled |
| r160_n1_ct10/sample_016 | DLR D-Flow | 289 | 274 | 3.0742 | 9.9193 | stalled |
| r160_n1_ct10/sample_017 | DLR D-Flow | 322 | 305 | 3.9721 | 9.9534 | stalled |
| r160_n1_ct10/sample_018 | DLR D-Flow | 439 | 422 | 32.2090 | 18.3691 | stalled |
| r160_n1_ct10/sample_019 | DLR D-Flow | 336 | 320 | 2.0346 | 9.9402 | stalled |
| r160_n1_ct10/sample_020 | DLR D-Flow | 78 | 73 | 114.5619 | 98.8438 | stalled |
| r160_n1_ct10/sample_021 | DLR D-Flow | 263 | 255 | 2.5835 | 9.9514 | stalled |
| r160_n1_ct10/sample_022 | DLR D-Flow | 26 | 23 | 94.0949 | 81.3876 | stalled |
| r160_n1_ct10/sample_023 | DLR D-Flow | 25 | 22 | 96.4965 | 81.8250 | stalled |
| r160_n1_ct10/sample_024 | DLR D-Flow | 169 | 161 | 3.6489 | 9.9415 | stalled |
| r160_n1_ct10/sample_025 | DLR D-Flow | 205 | 193 | 2.2675 | 9.9342 | stalled |
| r160_n1_ct10/sample_026 | DLR D-Flow | 398 | 382 | 9.3240 | 9.9491 | stalled |
| r160_n1_ct10/sample_027 | DLR D-Flow | 65 | 62 | 5.3797 | 9.9846 | stalled |
| r160_n1_ct10/sample_028 | DLR D-Flow | 23 | 19 | 95.4470 | 81.5448 | stalled |
| r160_n1_ct10/sample_029 | DLR D-Flow | 254 | 243 | 2.5205 | 9.9513 | stalled |
| r160_n2_ct10/sample_000 | D-Flow | 649 | 646 | 95.0165 | 0.4790 | stalled |
| r160_n2_ct10/sample_001 | D-Flow | 731 | 729 | 93.0941 | 0.3504 | stalled |
| r160_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 46.2861 | 0.4000 | step_limit |
| r160_n2_ct10/sample_003 | D-Flow | 439 | 437 | 79.9654 | 0.8752 | stalled |
| r160_n2_ct10/sample_004 | D-Flow | 536 | 533 | 102.8320 | 0.1443 | stalled |
| r160_n2_ct10/sample_005 | D-Flow | 418 | 415 | 94.0860 | 0.2170 | stalled |
| r160_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 47.0375 | 0.2480 | step_limit |
| r160_n2_ct10/sample_007 | D-Flow | 542 | 540 | 72.5318 | 0.7163 | stalled |
| r160_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 40.3786 | 4.9031 | step_limit |
| r160_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 7.4023 | 8.4706 | step_limit |
| r160_n2_ct10/sample_010 | D-Flow | 558 | 556 | 75.6813 | 0.3056 | stalled |
| r160_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 21.3299 | 8.8735 | step_limit |
| r160_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 40.2269 | 2.0054 | step_limit |
| r160_n2_ct10/sample_000 | P-Flow | 618 | 602 | 58.5214 | 0.4790 | stalled |
| r160_n2_ct10/sample_001 | P-Flow | 23 | 20 | 43.2188 | 20.8859 | stalled |
| r160_n2_ct10/sample_002 | P-Flow | 103 | 86 | 3.9193 | 9.9286 | stalled |
| r160_n2_ct10/sample_003 | P-Flow | 586 | 564 | 51.9771 | 0.8752 | stalled |
| r160_n2_ct10/sample_004 | P-Flow | 1000 | 925 | 53.8619 | 0.1444 | step_limit |
| r160_n2_ct10/sample_005 | P-Flow | 636 | 599 | 53.0307 | 0.2327 | stalled |
| r160_n2_ct10/sample_006 | P-Flow | 666 | 616 | 43.9157 | 0.2606 | stalled |
| r160_n2_ct10/sample_007 | P-Flow | 1000 | 929 | 52.8455 | 0.7164 | step_limit |
| r160_n2_ct10/sample_008 | P-Flow | 4 | 2 | 98.5154 | 65.3697 | stalled |
| r160_n2_ct10/sample_009 | P-Flow | 417 | 407 | 49.1496 | 0.2440 | stalled |
| r160_n2_ct10/sample_010 | P-Flow | 619 | 583 | 45.9350 | 0.3190 | stalled |
| r160_n2_ct10/sample_011 | P-Flow | 1000 | 809 | 19.6962 | 5.6628 | step_limit |
| r160_n2_ct10/sample_012 | P-Flow | 68 | 57 | 8.3064 | 10.2635 | stalled |
| r160_n2_ct10/sample_000 | DLR D-Flow | 484 | 470 | 4.8661 | 9.9464 | stalled |
| r160_n2_ct10/sample_001 | DLR D-Flow | 212 | 201 | 3.6106 | 9.9456 | stalled |
| r160_n2_ct10/sample_002 | DLR D-Flow | 396 | 380 | 60.2986 | 42.1840 | stalled |
| r160_n2_ct10/sample_003 | DLR D-Flow | 36 | 33 | 76.3490 | 43.8159 | stalled |
| r160_n2_ct10/sample_004 | DLR D-Flow | 336 | 321 | 72.1629 | 48.8113 | stalled |
| r160_n2_ct10/sample_005 | DLR D-Flow | 513 | 500 | 9.1797 | 10.1628 | stalled |
| r160_n2_ct10/sample_006 | DLR D-Flow | 882 | 849 | 16.9033 | 10.8369 | stalled |
| r160_n2_ct10/sample_007 | DLR D-Flow | 274 | 266 | 21.6250 | 11.3124 | stalled |
| r160_n2_ct10/sample_008 | DLR D-Flow | 980 | 940 | 4.0127 | 10.0387 | stalled |
| r160_n2_ct10/sample_009 | DLR D-Flow | 95 | 90 | 59.8234 | 30.7730 | stalled |
| r160_n2_ct10/sample_010 | DLR D-Flow | 325 | 311 | 2.8896 | 9.9696 | stalled |
| r160_n2_ct10/sample_011 | DLR D-Flow | 119 | 112 | 49.2607 | 30.4287 | stalled |
| r160_n2_ct10/sample_012 | DLR D-Flow | 230 | 222 | 123.1992 | 104.1602 | stalled |
| r160_n4_ct10/sample_000 | D-Flow | 761 | 759 | 78.0564 | 0.1739 | stalled |
| r160_n4_ct10/sample_001 | D-Flow | 601 | 599 | 66.0929 | 0.4176 | stalled |
| r160_n4_ct10/sample_002 | D-Flow | 619 | 617 | 58.0328 | 0.4965 | stalled |
| r160_n4_ct10/sample_003 | D-Flow | 720 | 718 | 95.4825 | 0.5074 | stalled |
| r160_n4_ct10/sample_004 | D-Flow | 547 | 545 | 92.7442 | 0.7854 | stalled |
| r160_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 41.5091 | 3.1255 | step_limit |
| r160_n4_ct10/sample_006 | D-Flow | 613 | 611 | 64.6118 | 0.1571 | stalled |
| r160_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 24.5395 | 8.9978 | step_limit |
| r160_n4_ct10/sample_008 | D-Flow | 774 | 772 | 69.4083 | 0.9033 | stalled |
| r160_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 63.6943 | 0.9082 | step_limit |
| r160_n4_ct10/sample_010 | D-Flow | 604 | 602 | 66.3722 | 0.4096 | stalled |
| r160_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 54.1210 | 0.5060 | step_limit |
| r160_n4_ct10/sample_012 | D-Flow | 741 | 739 | 61.9118 | 0.6929 | stalled |
| r160_n4_ct10/sample_000 | P-Flow | 1000 | 912 | 47.3563 | 0.1892 | step_limit |
| r160_n4_ct10/sample_001 | P-Flow | 877 | 839 | 58.4585 | 0.4176 | stalled |
| r160_n4_ct10/sample_002 | P-Flow | 1000 | 905 | 37.4842 | 0.8577 | step_limit |
| r160_n4_ct10/sample_003 | P-Flow | 721 | 692 | 49.9962 | 0.5075 | stalled |
| r160_n4_ct10/sample_004 | P-Flow | 453 | 447 | 48.5632 | 0.7855 | stalled |
| r160_n4_ct10/sample_005 | P-Flow | 736 | 692 | 51.2674 | 0.7566 | stalled |
| r160_n4_ct10/sample_006 | P-Flow | 717 | 702 | 52.5788 | 0.1760 | stalled |
| r160_n4_ct10/sample_007 | P-Flow | 588 | 559 | 53.0262 | 0.3276 | stalled |
| r160_n4_ct10/sample_008 | P-Flow | 1000 | 929 | 55.3644 | 0.9036 | step_limit |
| r160_n4_ct10/sample_009 | P-Flow | 765 | 732 | 56.8378 | 0.9080 | stalled |
| r160_n4_ct10/sample_010 | P-Flow | 4 | 2 | 89.0216 | 48.4824 | stalled |
| r160_n4_ct10/sample_011 | P-Flow | 1000 | 894 | 40.4187 | 0.7568 | step_limit |
| r160_n4_ct10/sample_012 | P-Flow | 7 | 5 | 93.6559 | 54.4174 | stalled |
| r160_n4_ct10/sample_000 | DLR D-Flow | 1000 | 957 | 15.5816 | 10.5666 | step_limit |
| r160_n4_ct10/sample_001 | DLR D-Flow | 372 | 354 | 10.5421 | 10.2085 | stalled |
| r160_n4_ct10/sample_002 | DLR D-Flow | 359 | 348 | 21.5723 | 11.3196 | stalled |
| r160_n4_ct10/sample_003 | DLR D-Flow | 398 | 384 | 18.1091 | 10.7931 | stalled |
| r160_n4_ct10/sample_004 | DLR D-Flow | 134 | 129 | 23.3826 | 11.8419 | stalled |
| r160_n4_ct10/sample_005 | DLR D-Flow | 557 | 538 | 26.2033 | 12.6256 | stalled |
| r160_n4_ct10/sample_006 | DLR D-Flow | 927 | 892 | 70.8171 | 37.2455 | stalled |
| r160_n4_ct10/sample_007 | DLR D-Flow | 905 | 873 | 70.1346 | 33.5981 | stalled |
| r160_n4_ct10/sample_008 | DLR D-Flow | 211 | 206 | 26.4909 | 12.9500 | stalled |
| r160_n4_ct10/sample_009 | DLR D-Flow | 248 | 239 | 42.7535 | 22.7601 | stalled |
| r160_n4_ct10/sample_010 | DLR D-Flow | 1000 | 954 | 29.5939 | 13.3836 | step_limit |
| r160_n4_ct10/sample_011 | DLR D-Flow | 975 | 943 | 43.8176 | 19.4493 | stalled |
| r160_n4_ct10/sample_012 | DLR D-Flow | 139 | 134 | 32.7988 | 16.6491 | stalled |
| r160_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 8.8571 | 7.6478 | step_limit |
| r160_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.6157 | 9.9353 | step_limit |
| r160_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 63.4171 | 62.3200 | step_limit |
| r160_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 13.8987 | 4.3217 | step_limit |
| r160_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 6.6970 | 6.4579 | step_limit |
| r160_n1_mri4/sample_005 | D-Flow | 130 | 128 | 95.7957 | 95.5288 | stalled |
| r160_n1_mri4/sample_006 | D-Flow | 540 | 538 | 11.2135 | 5.1208 | stalled |
| r160_n1_mri4/sample_007 | D-Flow | 939 | 936 | 11.3276 | 5.5717 | stalled |
| r160_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 11.2445 | 4.6550 | step_limit |
| r160_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 4.8855 | 9.0317 | step_limit |
| r160_n1_mri4/sample_010 | D-Flow | 910 | 908 | 12.6454 | 5.2783 | stalled |
| r160_n1_mri4/sample_011 | D-Flow | 689 | 687 | 12.9325 | 5.8591 | stalled |
| r160_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 121.5349 | 15.7404 | step_limit |
| r160_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 9.6780 | 6.1042 | step_limit |
| r160_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 9.4951 | 6.1320 | step_limit |
| r160_n1_mri4/sample_015 | D-Flow | 69 | 67 | 100.6053 | 100.1054 | stalled |
| r160_n1_mri4/sample_016 | D-Flow | 790 | 788 | 12.6787 | 4.9806 | stalled |
| r160_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 9.8707 | 6.9145 | step_limit |
| r160_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 10.3721 | 5.1998 | step_limit |
| r160_n1_mri4/sample_019 | D-Flow | 919 | 916 | 11.2078 | 5.5530 | stalled |
| r160_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 10.1367 | 5.6651 | step_limit |
| r160_n1_mri4/sample_021 | D-Flow | 810 | 808 | 12.7464 | 5.1339 | stalled |
| r160_n1_mri4/sample_022 | D-Flow | 821 | 819 | 13.1888 | 5.0274 | stalled |
| r160_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 7.2577 | 6.7161 | step_limit |
| r160_n1_mri4/sample_024 | D-Flow | 798 | 796 | 11.2480 | 4.8208 | stalled |
| r160_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 6.9509 | 8.0702 | step_limit |
| r160_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 7.5456 | 5.9508 | step_limit |
| r160_n1_mri4/sample_027 | D-Flow | 354 | 352 | 1.5282 | 9.9384 | stalled |
| r160_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 7.8183 | 7.4354 | step_limit |
| r160_n1_mri4/sample_029 | D-Flow | 890 | 887 | 12.0903 | 5.4139 | stalled |
| r160_n1_mri4/sample_000 | P-Flow | 235 | 213 | 15.3579 | 5.0185 | stalled |
| r160_n1_mri4/sample_001 | P-Flow | 200 | 191 | 15.3180 | 4.7891 | stalled |
| r160_n1_mri4/sample_002 | P-Flow | 155 | 150 | 15.0637 | 5.0973 | stalled |
| r160_n1_mri4/sample_003 | P-Flow | 176 | 164 | 15.5639 | 4.3207 | stalled |
| r160_n1_mri4/sample_004 | P-Flow | 161 | 158 | 12.3172 | 5.6012 | stalled |
| r160_n1_mri4/sample_005 | P-Flow | 56 | 45 | 1.8397 | 9.9747 | stalled |
| r160_n1_mri4/sample_006 | P-Flow | 208 | 192 | 13.1007 | 5.1208 | stalled |
| r160_n1_mri4/sample_007 | P-Flow | 214 | 202 | 13.7503 | 5.5716 | stalled |
| r160_n1_mri4/sample_008 | P-Flow | 197 | 183 | 14.7430 | 4.3411 | stalled |
| r160_n1_mri4/sample_009 | P-Flow | 210 | 198 | 12.9089 | 4.9593 | stalled |
| r160_n1_mri4/sample_010 | P-Flow | 203 | 194 | 14.1163 | 5.2783 | stalled |
| r160_n1_mri4/sample_011 | P-Flow | 197 | 185 | 13.7744 | 5.8591 | stalled |
| r160_n1_mri4/sample_012 | P-Flow | 166 | 163 | 11.6495 | 4.8133 | stalled |
| r160_n1_mri4/sample_013 | P-Flow | 161 | 154 | 13.8887 | 5.4483 | stalled |
| r160_n1_mri4/sample_014 | P-Flow | 33 | 26 | 3.1498 | 10.2617 | stalled |
| r160_n1_mri4/sample_015 | P-Flow | 271 | 243 | 14.2755 | 5.3026 | stalled |
| r160_n1_mri4/sample_016 | P-Flow | 183 | 171 | 14.7502 | 4.9806 | stalled |
| r160_n1_mri4/sample_017 | P-Flow | 182 | 175 | 15.7629 | 4.6410 | stalled |
| r160_n1_mri4/sample_018 | P-Flow | 259 | 235 | 14.7572 | 4.4219 | stalled |
| r160_n1_mri4/sample_019 | P-Flow | 180 | 174 | 12.3463 | 5.5530 | stalled |
| r160_n1_mri4/sample_020 | P-Flow | 183 | 172 | 13.7166 | 5.4899 | stalled |
| r160_n1_mri4/sample_021 | P-Flow | 13 | 10 | 81.7482 | 80.1820 | stalled |
| r160_n1_mri4/sample_022 | P-Flow | 169 | 163 | 14.9950 | 5.0274 | stalled |
| r160_n1_mri4/sample_023 | P-Flow | 172 | 161 | 11.2806 | 5.5413 | stalled |
| r160_n1_mri4/sample_024 | P-Flow | 204 | 197 | 13.2011 | 4.8208 | stalled |
| r160_n1_mri4/sample_025 | P-Flow | 173 | 169 | 15.4791 | 4.2530 | stalled |
| r160_n1_mri4/sample_026 | P-Flow | 179 | 169 | 13.1631 | 4.9909 | stalled |
| r160_n1_mri4/sample_027 | P-Flow | 218 | 202 | 12.6417 | 5.0372 | stalled |
| r160_n1_mri4/sample_028 | P-Flow | 163 | 156 | 14.7756 | 5.1594 | stalled |
| r160_n1_mri4/sample_029 | P-Flow | 177 | 170 | 14.2819 | 5.4139 | stalled |
| r160_n1_mri4/sample_000 | DLR D-Flow | 30 | 27 | 88.1670 | 87.7952 | stalled |
| r160_n1_mri4/sample_001 | DLR D-Flow | 234 | 224 | 106.2638 | 105.7742 | stalled |
| r160_n1_mri4/sample_002 | DLR D-Flow | 81 | 77 | 96.6061 | 96.2522 | stalled |
| r160_n1_mri4/sample_003 | DLR D-Flow | 146 | 139 | 1.4035 | 9.9619 | stalled |
| r160_n1_mri4/sample_004 | DLR D-Flow | 673 | 652 | 1.3752 | 9.9778 | stalled |
| r160_n1_mri4/sample_005 | DLR D-Flow | 85 | 81 | 7.7062 | 11.3298 | stalled |
| r160_n1_mri4/sample_006 | DLR D-Flow | 290 | 276 | 1.6256 | 10.0093 | stalled |
| r160_n1_mri4/sample_007 | DLR D-Flow | 197 | 187 | 1.3446 | 9.9750 | stalled |
| r160_n1_mri4/sample_008 | DLR D-Flow | 419 | 403 | 2.1868 | 10.0322 | stalled |
| r160_n1_mri4/sample_009 | DLR D-Flow | 323 | 314 | 1.4441 | 9.9642 | stalled |
| r160_n1_mri4/sample_010 | DLR D-Flow | 247 | 234 | 1.3944 | 9.9808 | stalled |
| r160_n1_mri4/sample_011 | DLR D-Flow | 1000 | 966 | 1.5474 | 9.9859 | step_limit |
| r160_n1_mri4/sample_012 | DLR D-Flow | 164 | 159 | 119.1405 | 51.2474 | stalled |
| r160_n1_mri4/sample_013 | DLR D-Flow | 1000 | 966 | 2.4064 | 10.0524 | step_limit |
| r160_n1_mri4/sample_014 | DLR D-Flow | 129 | 118 | 6.1366 | 11.1676 | stalled |
| r160_n1_mri4/sample_015 | DLR D-Flow | 107 | 101 | 98.6383 | 98.1372 | stalled |
| r160_n1_mri4/sample_016 | DLR D-Flow | 806 | 761 | 1.7353 | 9.9965 | stalled |
| r160_n1_mri4/sample_017 | DLR D-Flow | 28 | 26 | 100.1648 | 99.9005 | stalled |
| r160_n1_mri4/sample_018 | DLR D-Flow | 62 | 58 | 90.3739 | 85.7775 | stalled |
| r160_n1_mri4/sample_019 | DLR D-Flow | 422 | 408 | 1.3610 | 9.9880 | stalled |
| r160_n1_mri4/sample_020 | DLR D-Flow | 27 | 22 | 111.5731 | 110.3933 | stalled |
| r160_n1_mri4/sample_021 | DLR D-Flow | 348 | 338 | 2.3358 | 10.1146 | stalled |
| r160_n1_mri4/sample_022 | DLR D-Flow | 279 | 267 | 93.9533 | 93.7459 | stalled |
| r160_n1_mri4/sample_023 | DLR D-Flow | 20 | 17 | 103.1607 | 103.8763 | stalled |
| r160_n1_mri4/sample_024 | DLR D-Flow | 680 | 652 | 1.3608 | 9.9764 | stalled |
| r160_n1_mri4/sample_025 | DLR D-Flow | 233 | 216 | 1.4581 | 9.9755 | stalled |
| r160_n1_mri4/sample_026 | DLR D-Flow | 361 | 345 | 1.3795 | 9.9702 | stalled |
| r160_n1_mri4/sample_027 | DLR D-Flow | 142 | 135 | 7.6109 | 11.8306 | stalled |
| r160_n1_mri4/sample_028 | DLR D-Flow | 36 | 32 | 94.8867 | 94.5800 | stalled |
| r160_n1_mri4/sample_029 | DLR D-Flow | 461 | 446 | 1.2771 | 9.9626 | stalled |
| r160_n2_mri4/sample_000 | D-Flow | 793 | 791 | 11.7750 | 5.1759 | stalled |
| r160_n2_mri4/sample_001 | D-Flow | 835 | 832 | 11.9793 | 5.1556 | stalled |
| r160_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.4937 | 9.9635 | step_limit |
| r160_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 2.2218 | 9.8333 | step_limit |
| r160_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.9110 | 9.7617 | step_limit |
| r160_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 2.2149 | 9.6127 | step_limit |
| r160_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 7.9990 | 8.6906 | step_limit |
| r160_n2_mri4/sample_007 | D-Flow | 834 | 832 | 12.7961 | 4.5445 | stalled |
| r160_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.5909 | 9.9575 | step_limit |
| r160_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 4.0788 | 8.9093 | step_limit |
| r160_n2_mri4/sample_010 | D-Flow | 534 | 531 | 12.0770 | 4.8515 | stalled |
| r160_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 2.5384 | 9.8425 | step_limit |
| r160_n2_mri4/sample_012 | D-Flow | 193 | 188 | 68.1346 | 65.6174 | stalled |
| r160_n2_mri4/sample_000 | P-Flow | 166 | 159 | 14.5799 | 5.1759 | stalled |
| r160_n2_mri4/sample_001 | P-Flow | 219 | 203 | 12.9540 | 5.1556 | stalled |
| r160_n2_mri4/sample_002 | P-Flow | 174 | 168 | 13.6175 | 3.9906 | stalled |
| r160_n2_mri4/sample_003 | P-Flow | 189 | 182 | 14.8710 | 4.4980 | stalled |
| r160_n2_mri4/sample_004 | P-Flow | 178 | 171 | 14.0253 | 5.0935 | stalled |
| r160_n2_mri4/sample_005 | P-Flow | 184 | 175 | 13.6743 | 4.8857 | stalled |
| r160_n2_mri4/sample_006 | P-Flow | 164 | 161 | 12.5966 | 5.3077 | stalled |
| r160_n2_mri4/sample_007 | P-Flow | 176 | 166 | 14.7000 | 4.5445 | stalled |
| r160_n2_mri4/sample_008 | P-Flow | 195 | 180 | 14.8699 | 5.0874 | stalled |
| r160_n2_mri4/sample_009 | P-Flow | 210 | 199 | 14.5312 | 5.1316 | stalled |
| r160_n2_mri4/sample_010 | P-Flow | 220 | 209 | 13.3649 | 4.8515 | stalled |
| r160_n2_mri4/sample_011 | P-Flow | 185 | 176 | 13.2968 | 5.8187 | stalled |
| r160_n2_mri4/sample_012 | P-Flow | 191 | 179 | 15.4865 | 4.6816 | stalled |
| r160_n2_mri4/sample_000 | DLR D-Flow | 276 | 266 | 51.9209 | 50.5308 | stalled |
| r160_n2_mri4/sample_001 | DLR D-Flow | 233 | 226 | 44.9318 | 33.7705 | stalled |
| r160_n2_mri4/sample_002 | DLR D-Flow | 158 | 153 | 54.4943 | 54.4782 | stalled |
| r160_n2_mri4/sample_003 | DLR D-Flow | 111 | 106 | 57.4235 | 53.8499 | stalled |
| r160_n2_mri4/sample_004 | DLR D-Flow | 303 | 293 | 62.2310 | 61.5934 | stalled |
| r160_n2_mri4/sample_005 | DLR D-Flow | 1000 | 981 | 2.1971 | 10.0716 | step_limit |
| r160_n2_mri4/sample_006 | DLR D-Flow | 548 | 524 | 3.5203 | 10.3462 | stalled |
| r160_n2_mri4/sample_007 | DLR D-Flow | 285 | 272 | 2.7836 | 10.1202 | stalled |
| r160_n2_mri4/sample_008 | DLR D-Flow | 393 | 376 | 82.2598 | 81.3637 | stalled |
| r160_n2_mri4/sample_009 | DLR D-Flow | 852 | 812 | 2.9823 | 10.1503 | stalled |
| r160_n2_mri4/sample_010 | DLR D-Flow | 281 | 266 | 1.4205 | 9.9967 | stalled |
| r160_n2_mri4/sample_011 | DLR D-Flow | 694 | 678 | 44.8277 | 44.1004 | stalled |
| r160_n2_mri4/sample_012 | DLR D-Flow | 212 | 202 | 124.7032 | 123.1668 | stalled |
| r160_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 8.3450 | 8.5458 | step_limit |
| r160_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 2.4509 | 9.5868 | step_limit |
| r160_n4_mri4/sample_002 | D-Flow | 767 | 765 | 12.8445 | 5.0888 | stalled |
| r160_n4_mri4/sample_003 | D-Flow | 761 | 758 | 11.1961 | 5.4746 | stalled |
| r160_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 11.2451 | 5.8588 | step_limit |
| r160_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 2.1639 | 9.6406 | step_limit |
| r160_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 2.2795 | 9.6708 | step_limit |
| r160_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 56.5036 | 53.6133 | step_limit |
| r160_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.7937 | 9.8744 | step_limit |
| r160_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 4.0696 | 9.5273 | step_limit |
| r160_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 7.7512 | 8.7899 | step_limit |
| r160_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.7667 | 9.9210 | step_limit |
| r160_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 7.6845 | 8.3590 | step_limit |
| r160_n4_mri4/sample_000 | P-Flow | 180 | 174 | 13.2005 | 4.4122 | stalled |
| r160_n4_mri4/sample_001 | P-Flow | 182 | 176 | 11.6399 | 5.4139 | stalled |
| r160_n4_mri4/sample_002 | P-Flow | 210 | 198 | 13.8738 | 5.0888 | stalled |
| r160_n4_mri4/sample_003 | P-Flow | 172 | 166 | 11.8169 | 5.4746 | stalled |
| r160_n4_mri4/sample_004 | P-Flow | 183 | 179 | 12.8776 | 5.3260 | stalled |
| r160_n4_mri4/sample_005 | P-Flow | 195 | 186 | 13.7073 | 5.2758 | stalled |
| r160_n4_mri4/sample_006 | P-Flow | 211 | 198 | 13.5367 | 5.4970 | stalled |
| r160_n4_mri4/sample_007 | P-Flow | 176 | 170 | 13.0562 | 5.0679 | stalled |
| r160_n4_mri4/sample_008 | P-Flow | 193 | 187 | 12.4288 | 5.3333 | stalled |
| r160_n4_mri4/sample_009 | P-Flow | 197 | 184 | 12.4115 | 5.3553 | stalled |
| r160_n4_mri4/sample_010 | P-Flow | 197 | 188 | 12.6237 | 5.8470 | stalled |
| r160_n4_mri4/sample_011 | P-Flow | 218 | 208 | 12.6542 | 5.5261 | stalled |
| r160_n4_mri4/sample_012 | P-Flow | 167 | 164 | 12.2902 | 5.0365 | stalled |
| r160_n4_mri4/sample_000 | DLR D-Flow | 1000 | 972 | 27.9890 | 26.6485 | step_limit |
| r160_n4_mri4/sample_001 | DLR D-Flow | 1000 | 959 | 42.1808 | 38.2749 | step_limit |
| r160_n4_mri4/sample_002 | DLR D-Flow | 999 | 951 | 21.8764 | 20.6288 | stalled |
| r160_n4_mri4/sample_003 | DLR D-Flow | 428 | 415 | 60.3426 | 57.1197 | stalled |
| r160_n4_mri4/sample_004 | DLR D-Flow | 1000 | 961 | 11.9253 | 13.6066 | step_limit |
| r160_n4_mri4/sample_005 | DLR D-Flow | 659 | 630 | 15.9016 | 15.8504 | stalled |
| r160_n4_mri4/sample_006 | DLR D-Flow | 930 | 894 | 15.7604 | 16.0288 | stalled |
| r160_n4_mri4/sample_007 | DLR D-Flow | 1000 | 969 | 48.6156 | 46.3096 | step_limit |
| r160_n4_mri4/sample_008 | DLR D-Flow | 136 | 127 | 64.6899 | 56.7659 | stalled |
| r160_n4_mri4/sample_009 | DLR D-Flow | 821 | 788 | 34.1835 | 32.8756 | stalled |
| r160_n4_mri4/sample_010 | DLR D-Flow | 1000 | 968 | 56.7587 | 52.2748 | step_limit |
| r160_n4_mri4/sample_011 | DLR D-Flow | 1000 | 961 | 39.3392 | 37.8531 | step_limit |
| r160_n4_mri4/sample_012 | DLR D-Flow | 1000 | 968 | 75.9519 | 62.4248 | step_limit |
| r200_n1_ct10/sample_000 | D-Flow | 581 | 579 | 102.6718 | 0.3407 | stalled |
| r200_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 10.4289 | 5.6112 | step_limit |
| r200_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 8.0701 | 7.4372 | step_limit |
| r200_n1_ct10/sample_003 | D-Flow | 488 | 486 | 82.8175 | 0.6947 | stalled |
| r200_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 11.1028 | 4.6535 | step_limit |
| r200_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 10.6485 | 5.2158 | step_limit |
| r200_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 36.9312 | 0.5193 | step_limit |
| r200_n1_ct10/sample_007 | D-Flow | 69 | 66 | 109.6502 | 95.6134 | stalled |
| r200_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 8.2764 | 7.8155 | step_limit |
| r200_n1_ct10/sample_009 | D-Flow | 518 | 516 | 49.8129 | 0.4190 | stalled |
| r200_n1_ct10/sample_010 | D-Flow | 738 | 736 | 73.7802 | 0.4061 | stalled |
| r200_n1_ct10/sample_011 | D-Flow | 378 | 376 | 88.4609 | 0.5418 | stalled |
| r200_n1_ct10/sample_012 | D-Flow | 512 | 510 | 95.9857 | 0.4058 | stalled |
| r200_n1_ct10/sample_013 | D-Flow | 462 | 460 | 74.2942 | 0.4808 | stalled |
| r200_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 54.5625 | 0.4883 | step_limit |
| r200_n1_ct10/sample_015 | D-Flow | 497 | 495 | 59.3896 | 0.2744 | stalled |
| r200_n1_ct10/sample_016 | D-Flow | 420 | 418 | 93.9740 | 0.4780 | stalled |
| r200_n1_ct10/sample_017 | D-Flow | 432 | 430 | 120.1125 | 0.6630 | stalled |
| r200_n1_ct10/sample_018 | D-Flow | 417 | 414 | 72.6427 | 0.5408 | stalled |
| r200_n1_ct10/sample_019 | D-Flow | 524 | 522 | 65.3807 | 0.4982 | stalled |
| r200_n1_ct10/sample_020 | D-Flow | 462 | 460 | 98.0553 | 0.5208 | stalled |
| r200_n1_ct10/sample_021 | D-Flow | 86 | 84 | 103.4652 | 97.2328 | stalled |
| r200_n1_ct10/sample_022 | D-Flow | 485 | 483 | 55.0536 | 0.5836 | stalled |
| r200_n1_ct10/sample_023 | D-Flow | 793 | 791 | 56.9596 | 0.4433 | stalled |
| r200_n1_ct10/sample_024 | D-Flow | 574 | 572 | 63.6524 | 0.3917 | stalled |
| r200_n1_ct10/sample_025 | D-Flow | 576 | 573 | 95.6370 | 0.5906 | stalled |
| r200_n1_ct10/sample_026 | D-Flow | 421 | 419 | 101.1046 | 0.6238 | stalled |
| r200_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 84.8434 | 0.6771 | step_limit |
| r200_n1_ct10/sample_028 | D-Flow | 728 | 726 | 50.5113 | 0.3153 | stalled |
| r200_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 53.4449 | 0.2250 | step_limit |
| r200_n1_ct10/sample_000 | P-Flow | 40 | 33 | 4.9427 | 10.0280 | stalled |
| r200_n1_ct10/sample_001 | P-Flow | 1000 | 789 | 4.3774 | 9.8266 | step_limit |
| r200_n1_ct10/sample_002 | P-Flow | 1000 | 914 | 41.3494 | 0.6319 | step_limit |
| r200_n1_ct10/sample_003 | P-Flow | 160 | 129 | 5.1994 | 9.9708 | stalled |
| r200_n1_ct10/sample_004 | P-Flow | 1000 | 921 | 34.2164 | 0.9300 | step_limit |
| r200_n1_ct10/sample_005 | P-Flow | 250 | 202 | 5.2612 | 10.0190 | stalled |
| r200_n1_ct10/sample_006 | P-Flow | 56 | 49 | 11.5405 | 11.5168 | stalled |
| r200_n1_ct10/sample_007 | P-Flow | 17 | 14 | 110.4990 | 95.0144 | stalled |
| r200_n1_ct10/sample_008 | P-Flow | 279 | 217 | 3.6024 | 10.0412 | stalled |
| r200_n1_ct10/sample_009 | P-Flow | 466 | 437 | 42.7269 | 0.4192 | stalled |
| r200_n1_ct10/sample_010 | P-Flow | 49 | 42 | 2.7371 | 9.9583 | stalled |
| r200_n1_ct10/sample_011 | P-Flow | 414 | 403 | 47.1499 | 0.5420 | stalled |
| r200_n1_ct10/sample_012 | P-Flow | 69 | 59 | 7.1422 | 10.1565 | stalled |
| r200_n1_ct10/sample_013 | P-Flow | 48 | 40 | 2.9263 | 10.0021 | stalled |
| r200_n1_ct10/sample_014 | P-Flow | 30 | 25 | 5.1863 | 10.4944 | stalled |
| r200_n1_ct10/sample_015 | P-Flow | 445 | 416 | 43.9304 | 0.2754 | stalled |
| r200_n1_ct10/sample_016 | P-Flow | 4 | 2 | 104.7073 | 97.1622 | stalled |
| r200_n1_ct10/sample_017 | P-Flow | 40 | 35 | 3.9983 | 9.8822 | stalled |
| r200_n1_ct10/sample_018 | P-Flow | 103 | 85 | 5.9016 | 9.9512 | stalled |
| r200_n1_ct10/sample_019 | P-Flow | 28 | 24 | 18.6918 | 11.0676 | stalled |
| r200_n1_ct10/sample_020 | P-Flow | 19 | 16 | 8.0797 | 10.2854 | stalled |
| r200_n1_ct10/sample_021 | P-Flow | 22 | 18 | 22.8155 | 12.5836 | stalled |
| r200_n1_ct10/sample_022 | P-Flow | 1000 | 955 | 41.5589 | 0.6214 | step_limit |
| r200_n1_ct10/sample_023 | P-Flow | 130 | 107 | 12.2230 | 11.0943 | stalled |
| r200_n1_ct10/sample_024 | P-Flow | 5 | 3 | 108.7669 | 94.0855 | stalled |
| r200_n1_ct10/sample_025 | P-Flow | 610 | 581 | 45.7163 | 0.5906 | stalled |
| r200_n1_ct10/sample_026 | P-Flow | 970 | 880 | 46.1360 | 0.6406 | stalled |
| r200_n1_ct10/sample_027 | P-Flow | 159 | 129 | 5.2116 | 10.0439 | stalled |
| r200_n1_ct10/sample_028 | P-Flow | 1000 | 949 | 38.7004 | 0.4441 | step_limit |
| r200_n1_ct10/sample_029 | P-Flow | 56 | 46 | 7.6206 | 10.5060 | stalled |
| r200_n1_ct10/sample_000 | DLR D-Flow | 349 | 336 | 2.8583 | 9.8945 | stalled |
| r200_n1_ct10/sample_001 | DLR D-Flow | 437 | 418 | 2.7310 | 9.9530 | stalled |
| r200_n1_ct10/sample_002 | DLR D-Flow | 213 | 199 | 2.0378 | 9.8942 | stalled |
| r200_n1_ct10/sample_003 | DLR D-Flow | 143 | 137 | 2.5939 | 9.9408 | stalled |
| r200_n1_ct10/sample_004 | DLR D-Flow | 875 | 846 | 40.7049 | 21.2195 | stalled |
| r200_n1_ct10/sample_005 | DLR D-Flow | 421 | 405 | 2.7034 | 9.9311 | stalled |
| r200_n1_ct10/sample_006 | DLR D-Flow | 953 | 916 | 3.1048 | 9.9413 | stalled |
| r200_n1_ct10/sample_007 | DLR D-Flow | 174 | 164 | 109.1571 | 96.0186 | stalled |
| r200_n1_ct10/sample_008 | DLR D-Flow | 245 | 234 | 2.0596 | 9.9585 | stalled |
| r200_n1_ct10/sample_009 | DLR D-Flow | 55 | 48 | 96.2517 | 61.9577 | stalled |
| r200_n1_ct10/sample_010 | DLR D-Flow | 176 | 169 | 2.2330 | 9.9265 | stalled |
| r200_n1_ct10/sample_011 | DLR D-Flow | 376 | 363 | 2.6436 | 9.9305 | stalled |
| r200_n1_ct10/sample_012 | DLR D-Flow | 599 | 565 | 6.3241 | 10.0613 | stalled |
| r200_n1_ct10/sample_013 | DLR D-Flow | 267 | 259 | 2.6427 | 9.9565 | stalled |
| r200_n1_ct10/sample_014 | DLR D-Flow | 32 | 27 | 111.8351 | 95.7725 | stalled |
| r200_n1_ct10/sample_015 | DLR D-Flow | 143 | 136 | 2.8646 | 9.9700 | stalled |
| r200_n1_ct10/sample_016 | DLR D-Flow | 324 | 309 | 2.6856 | 9.9634 | stalled |
| r200_n1_ct10/sample_017 | DLR D-Flow | 258 | 246 | 3.7359 | 9.9174 | stalled |
| r200_n1_ct10/sample_018 | DLR D-Flow | 361 | 342 | 3.4305 | 9.8918 | stalled |
| r200_n1_ct10/sample_019 | DLR D-Flow | 426 | 408 | 2.6680 | 9.9300 | stalled |
| r200_n1_ct10/sample_020 | DLR D-Flow | 725 | 692 | 3.2246 | 9.9536 | stalled |
| r200_n1_ct10/sample_021 | DLR D-Flow | 100 | 96 | 105.2078 | 97.8918 | stalled |
| r200_n1_ct10/sample_022 | DLR D-Flow | 314 | 305 | 2.7393 | 9.9949 | stalled |
| r200_n1_ct10/sample_023 | DLR D-Flow | 283 | 273 | 9.3843 | 11.0462 | stalled |
| r200_n1_ct10/sample_024 | DLR D-Flow | 376 | 361 | 2.7604 | 9.9424 | stalled |
| r200_n1_ct10/sample_025 | DLR D-Flow | 219 | 210 | 1.6319 | 9.9204 | stalled |
| r200_n1_ct10/sample_026 | DLR D-Flow | 267 | 258 | 2.2392 | 9.9184 | stalled |
| r200_n1_ct10/sample_027 | DLR D-Flow | 543 | 509 | 2.2329 | 9.9816 | stalled |
| r200_n1_ct10/sample_028 | DLR D-Flow | 424 | 411 | 3.0319 | 9.9388 | stalled |
| r200_n1_ct10/sample_029 | DLR D-Flow | 206 | 196 | 63.4225 | 43.0509 | stalled |
| r200_n2_ct10/sample_000 | D-Flow | 923 | 921 | 60.3155 | 0.3068 | stalled |
| r200_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 32.2149 | 6.6655 | step_limit |
| r200_n2_ct10/sample_002 | D-Flow | 631 | 629 | 65.4135 | 0.6152 | stalled |
| r200_n2_ct10/sample_003 | D-Flow | 65 | 62 | 91.5617 | 71.0609 | stalled |
| r200_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 42.2844 | 3.3616 | step_limit |
| r200_n2_ct10/sample_005 | D-Flow | 525 | 522 | 99.9920 | 0.4843 | stalled |
| r200_n2_ct10/sample_006 | D-Flow | 687 | 685 | 50.1037 | 0.6994 | stalled |
| r200_n2_ct10/sample_007 | D-Flow | 800 | 797 | 70.9753 | 0.4507 | stalled |
| r200_n2_ct10/sample_008 | D-Flow | 574 | 572 | 82.6986 | 0.3628 | stalled |
| r200_n2_ct10/sample_009 | D-Flow | 838 | 835 | 66.5057 | 0.1938 | stalled |
| r200_n2_ct10/sample_010 | D-Flow | 687 | 684 | 54.3690 | 0.5577 | stalled |
| r200_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 46.9741 | 0.8708 | step_limit |
| r200_n2_ct10/sample_012 | D-Flow | 749 | 746 | 58.7113 | 0.3750 | stalled |
| r200_n2_ct10/sample_000 | P-Flow | 35 | 28 | 9.4392 | 10.2866 | stalled |
| r200_n2_ct10/sample_001 | P-Flow | 1000 | 837 | 41.8469 | 0.7368 | step_limit |
| r200_n2_ct10/sample_002 | P-Flow | 68 | 57 | 7.5927 | 10.2396 | stalled |
| r200_n2_ct10/sample_003 | P-Flow | 42 | 36 | 3.2781 | 9.9567 | stalled |
| r200_n2_ct10/sample_004 | P-Flow | 531 | 428 | 5.0417 | 9.9422 | stalled |
| r200_n2_ct10/sample_005 | P-Flow | 1000 | 830 | 22.6252 | 4.5901 | step_limit |
| r200_n2_ct10/sample_006 | P-Flow | 249 | 199 | 3.9014 | 9.9813 | stalled |
| r200_n2_ct10/sample_007 | P-Flow | 1000 | 810 | 2.8742 | 9.9203 | step_limit |
| r200_n2_ct10/sample_008 | P-Flow | 33 | 28 | 5.6192 | 10.0235 | stalled |
| r200_n2_ct10/sample_009 | P-Flow | 872 | 824 | 52.9551 | 0.1939 | stalled |
| r200_n2_ct10/sample_010 | P-Flow | 1000 | 802 | 3.8269 | 9.9436 | step_limit |
| r200_n2_ct10/sample_011 | P-Flow | 1000 | 905 | 38.1239 | 0.8302 | step_limit |
| r200_n2_ct10/sample_012 | P-Flow | 1000 | 832 | 4.2117 | 9.9510 | step_limit |
| r200_n2_ct10/sample_000 | DLR D-Flow | 359 | 348 | 3.2421 | 9.9173 | stalled |
| r200_n2_ct10/sample_001 | DLR D-Flow | 327 | 313 | 53.0670 | 37.8132 | stalled |
| r200_n2_ct10/sample_002 | DLR D-Flow | 987 | 952 | 9.4634 | 10.2143 | stalled |
| r200_n2_ct10/sample_003 | DLR D-Flow | 82 | 76 | 91.5105 | 69.8414 | stalled |
| r200_n2_ct10/sample_004 | DLR D-Flow | 1000 | 966 | 2.7281 | 9.9527 | step_limit |
| r200_n2_ct10/sample_005 | DLR D-Flow | 979 | 946 | 3.0183 | 9.9304 | stalled |
| r200_n2_ct10/sample_006 | DLR D-Flow | 270 | 254 | 83.0683 | 58.4447 | stalled |
| r200_n2_ct10/sample_007 | DLR D-Flow | 171 | 165 | 4.6876 | 10.0172 | stalled |
| r200_n2_ct10/sample_008 | DLR D-Flow | 500 | 474 | 3.5389 | 9.9623 | stalled |
| r200_n2_ct10/sample_009 | DLR D-Flow | 55 | 52 | 59.3946 | 38.4505 | stalled |
| r200_n2_ct10/sample_010 | DLR D-Flow | 1000 | 968 | 3.9724 | 9.9786 | step_limit |
| r200_n2_ct10/sample_011 | DLR D-Flow | 318 | 306 | 6.0971 | 10.1079 | stalled |
| r200_n2_ct10/sample_012 | DLR D-Flow | 322 | 313 | 4.0093 | 10.0539 | stalled |
| r200_n4_ct10/sample_000 | D-Flow | 915 | 913 | 63.3732 | 0.5661 | stalled |
| r200_n4_ct10/sample_001 | D-Flow | 517 | 514 | 53.4865 | 35.9281 | stalled |
| r200_n4_ct10/sample_002 | D-Flow | 832 | 829 | 59.0099 | 0.1268 | stalled |
| r200_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 61.0381 | 33.6380 | step_limit |
| r200_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 13.2629 | 10.6321 | step_limit |
| r200_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 62.4366 | 0.3182 | step_limit |
| r200_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 72.7067 | 0.8569 | step_limit |
| r200_n4_ct10/sample_007 | D-Flow | 869 | 866 | 65.0772 | 0.2311 | stalled |
| r200_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 3.9232 | 9.9136 | step_limit |
| r200_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 45.7356 | 28.4746 | step_limit |
| r200_n4_ct10/sample_010 | D-Flow | 552 | 550 | 65.1513 | 0.5578 | stalled |
| r200_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 65.3804 | 0.6072 | step_limit |
| r200_n4_ct10/sample_012 | D-Flow | 711 | 709 | 64.4243 | 0.5556 | stalled |
| r200_n4_ct10/sample_000 | P-Flow | 1000 | 882 | 37.6647 | 0.7874 | step_limit |
| r200_n4_ct10/sample_001 | P-Flow | 1000 | 879 | 35.5291 | 0.9555 | step_limit |
| r200_n4_ct10/sample_002 | P-Flow | 1000 | 901 | 40.2269 | 0.3955 | step_limit |
| r200_n4_ct10/sample_003 | P-Flow | 1000 | 866 | 38.8944 | 0.4747 | step_limit |
| r200_n4_ct10/sample_004 | P-Flow | 1000 | 887 | 38.1168 | 0.9214 | step_limit |
| r200_n4_ct10/sample_005 | P-Flow | 581 | 564 | 46.8453 | 0.3182 | stalled |
| r200_n4_ct10/sample_006 | P-Flow | 716 | 663 | 51.5652 | 0.8398 | stalled |
| r200_n4_ct10/sample_007 | P-Flow | 1000 | 874 | 39.1932 | 0.7278 | step_limit |
| r200_n4_ct10/sample_008 | P-Flow | 55 | 45 | 7.4474 | 10.0191 | stalled |
| r200_n4_ct10/sample_009 | P-Flow | 1000 | 889 | 36.8101 | 0.9892 | step_limit |
| r200_n4_ct10/sample_010 | P-Flow | 613 | 595 | 45.9208 | 0.5582 | stalled |
| r200_n4_ct10/sample_011 | P-Flow | 1000 | 883 | 35.5205 | 1.2391 | step_limit |
| r200_n4_ct10/sample_012 | P-Flow | 50 | 42 | 4.7004 | 9.9277 | stalled |
| r200_n4_ct10/sample_000 | DLR D-Flow | 1000 | 965 | 62.4133 | 30.9389 | step_limit |
| r200_n4_ct10/sample_001 | DLR D-Flow | 382 | 362 | 88.9485 | 49.4491 | stalled |
| r200_n4_ct10/sample_002 | DLR D-Flow | 311 | 298 | 75.7618 | 37.6172 | stalled |
| r200_n4_ct10/sample_003 | DLR D-Flow | 400 | 391 | 70.9298 | 41.0972 | stalled |
| r200_n4_ct10/sample_004 | DLR D-Flow | 331 | 320 | 74.1242 | 37.7967 | stalled |
| r200_n4_ct10/sample_005 | DLR D-Flow | 339 | 323 | 26.7965 | 12.7380 | stalled |
| r200_n4_ct10/sample_006 | DLR D-Flow | 257 | 245 | 49.2957 | 23.5016 | stalled |
| r200_n4_ct10/sample_007 | DLR D-Flow | 213 | 204 | 78.8353 | 37.9903 | stalled |
| r200_n4_ct10/sample_008 | DLR D-Flow | 1000 | 959 | 10.6387 | 10.2137 | step_limit |
| r200_n4_ct10/sample_009 | DLR D-Flow | 1000 | 958 | 17.5921 | 11.0767 | step_limit |
| r200_n4_ct10/sample_010 | DLR D-Flow | 138 | 130 | 91.6898 | 49.6663 | stalled |
| r200_n4_ct10/sample_011 | DLR D-Flow | 1000 | 961 | 62.9092 | 34.6774 | step_limit |
| r200_n4_ct10/sample_012 | DLR D-Flow | 703 | 673 | 54.6605 | 29.8245 | stalled |
| r200_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 5.5161 | 7.1426 | step_limit |
| r200_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 8.9912 | 5.7462 | step_limit |
| r200_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 5.1350 | 8.6017 | step_limit |
| r200_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 5.5781 | 7.4052 | step_limit |
| r200_n1_mri4/sample_004 | D-Flow | 804 | 801 | 11.6598 | 4.7664 | stalled |
| r200_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 9.2063 | 5.4588 | step_limit |
| r200_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 8.1827 | 7.8560 | step_limit |
| r200_n1_mri4/sample_007 | D-Flow | 57 | 55 | 102.6488 | 100.8886 | stalled |
| r200_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 5.7934 | 6.4789 | step_limit |
| r200_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 11.9971 | 5.4930 | step_limit |
| r200_n1_mri4/sample_010 | D-Flow | 941 | 939 | 13.7421 | 4.9736 | stalled |
| r200_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 2.7584 | 9.6694 | step_limit |
| r200_n1_mri4/sample_012 | D-Flow | 869 | 867 | 11.0201 | 5.1440 | stalled |
| r200_n1_mri4/sample_013 | D-Flow | 710 | 708 | 11.6520 | 5.1264 | stalled |
| r200_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 13.2021 | 5.4175 | step_limit |
| r200_n1_mri4/sample_015 | D-Flow | 663 | 660 | 11.7804 | 4.8094 | stalled |
| r200_n1_mri4/sample_016 | D-Flow | 904 | 902 | 13.4934 | 4.9953 | stalled |
| r200_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 6.9548 | 6.3892 | step_limit |
| r200_n1_mri4/sample_018 | D-Flow | 551 | 549 | 13.2392 | 4.7550 | stalled |
| r200_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 6.4917 | 5.4604 | step_limit |
| r200_n1_mri4/sample_020 | D-Flow | 562 | 559 | 12.4160 | 4.8463 | stalled |
| r200_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 14.2179 | 4.5129 | step_limit |
| r200_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 13.5887 | 5.2405 | step_limit |
| r200_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 4.2687 | 8.7638 | step_limit |
| r200_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 6.3416 | 6.3307 | step_limit |
| r200_n1_mri4/sample_025 | D-Flow | 616 | 614 | 13.8951 | 4.5585 | stalled |
| r200_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 12.1495 | 4.7757 | step_limit |
| r200_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 8.3895 | 5.9888 | step_limit |
| r200_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 7.1250 | 6.0548 | step_limit |
| r200_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 7.1953 | 7.5778 | step_limit |
| r200_n1_mri4/sample_000 | P-Flow | 174 | 170 | 14.1551 | 5.3159 | stalled |
| r200_n1_mri4/sample_001 | P-Flow | 194 | 188 | 14.6263 | 4.7365 | stalled |
| r200_n1_mri4/sample_002 | P-Flow | 198 | 187 | 13.0829 | 4.9407 | stalled |
| r200_n1_mri4/sample_003 | P-Flow | 227 | 211 | 14.4673 | 5.1681 | stalled |
| r200_n1_mri4/sample_004 | P-Flow | 205 | 192 | 13.5961 | 4.7664 | stalled |
| r200_n1_mri4/sample_005 | P-Flow | 455 | 396 | 12.8086 | 5.1323 | stalled |
| r200_n1_mri4/sample_006 | P-Flow | 224 | 218 | 14.9455 | 4.7939 | stalled |
| r200_n1_mri4/sample_007 | P-Flow | 17 | 14 | 102.3027 | 100.5775 | stalled |
| r200_n1_mri4/sample_008 | P-Flow | 46 | 39 | 1.4820 | 10.0303 | stalled |
| r200_n1_mri4/sample_009 | P-Flow | 173 | 164 | 13.9140 | 5.2203 | stalled |
| r200_n1_mri4/sample_010 | P-Flow | 181 | 177 | 14.5862 | 4.9736 | stalled |
| r200_n1_mri4/sample_011 | P-Flow | 38 | 31 | 1.3199 | 9.9777 | stalled |
| r200_n1_mri4/sample_012 | P-Flow | 177 | 170 | 13.1964 | 5.1440 | stalled |
| r200_n1_mri4/sample_013 | P-Flow | 183 | 173 | 12.9984 | 5.1264 | stalled |
| r200_n1_mri4/sample_014 | P-Flow | 205 | 189 | 15.0161 | 5.4150 | stalled |
| r200_n1_mri4/sample_015 | P-Flow | 224 | 210 | 13.1668 | 4.8094 | stalled |
| r200_n1_mri4/sample_016 | P-Flow | 185 | 180 | 14.4764 | 4.9953 | stalled |
| r200_n1_mri4/sample_017 | P-Flow | 179 | 169 | 14.2667 | 5.4173 | stalled |
| r200_n1_mri4/sample_018 | P-Flow | 165 | 160 | 14.8586 | 4.7550 | stalled |
| r200_n1_mri4/sample_019 | P-Flow | 194 | 179 | 11.0490 | 4.6110 | stalled |
| r200_n1_mri4/sample_020 | P-Flow | 25 | 20 | 3.0478 | 10.2602 | stalled |
| r200_n1_mri4/sample_021 | P-Flow | 182 | 173 | 15.5360 | 4.5128 | stalled |
| r200_n1_mri4/sample_022 | P-Flow | 188 | 175 | 15.2058 | 4.9989 | stalled |
| r200_n1_mri4/sample_023 | P-Flow | 58 | 47 | 3.4698 | 10.4140 | stalled |
| r200_n1_mri4/sample_024 | P-Flow | 162 | 156 | 13.3915 | 4.7991 | stalled |
| r200_n1_mri4/sample_025 | P-Flow | 199 | 186 | 14.9130 | 4.5585 | stalled |
| r200_n1_mri4/sample_026 | P-Flow | 187 | 182 | 13.5493 | 4.7755 | stalled |
| r200_n1_mri4/sample_027 | P-Flow | 256 | 239 | 14.4004 | 5.4147 | stalled |
| r200_n1_mri4/sample_028 | P-Flow | 170 | 161 | 12.7472 | 5.1051 | stalled |
| r200_n1_mri4/sample_029 | P-Flow | 162 | 156 | 13.2145 | 5.3931 | stalled |
| r200_n1_mri4/sample_000 | DLR D-Flow | 293 | 280 | 1.1927 | 9.9714 | stalled |
| r200_n1_mri4/sample_001 | DLR D-Flow | 776 | 747 | 1.6421 | 9.9975 | stalled |
| r200_n1_mri4/sample_002 | DLR D-Flow | 215 | 207 | 1.2345 | 9.9663 | stalled |
| r200_n1_mri4/sample_003 | DLR D-Flow | 406 | 386 | 1.2165 | 9.9637 | stalled |
| r200_n1_mri4/sample_004 | DLR D-Flow | 79 | 75 | 50.4961 | 50.2296 | stalled |
| r200_n1_mri4/sample_005 | DLR D-Flow | 282 | 268 | 40.9901 | 40.4023 | stalled |
| r200_n1_mri4/sample_006 | DLR D-Flow | 552 | 533 | 5.7012 | 11.2082 | stalled |
| r200_n1_mri4/sample_007 | DLR D-Flow | 17 | 15 | 104.7306 | 103.0052 | stalled |
| r200_n1_mri4/sample_008 | DLR D-Flow | 528 | 515 | 1.7008 | 10.0097 | stalled |
| r200_n1_mri4/sample_009 | DLR D-Flow | 197 | 188 | 77.1619 | 76.4970 | stalled |
| r200_n1_mri4/sample_010 | DLR D-Flow | 307 | 297 | 1.5327 | 9.9968 | stalled |
| r200_n1_mri4/sample_011 | DLR D-Flow | 240 | 227 | 1.2108 | 9.9603 | stalled |
| r200_n1_mri4/sample_012 | DLR D-Flow | 332 | 303 | 99.7405 | 98.7565 | stalled |
| r200_n1_mri4/sample_013 | DLR D-Flow | 89 | 86 | 80.6186 | 81.3858 | stalled |
| r200_n1_mri4/sample_014 | DLR D-Flow | 68 | 63 | 96.2644 | 94.1704 | stalled |
| r200_n1_mri4/sample_015 | DLR D-Flow | 201 | 193 | 1.2050 | 9.9804 | stalled |
| r200_n1_mri4/sample_016 | DLR D-Flow | 482 | 466 | 2.3132 | 10.0505 | stalled |
| r200_n1_mri4/sample_017 | DLR D-Flow | 289 | 276 | 1.7637 | 9.9953 | stalled |
| r200_n1_mri4/sample_018 | DLR D-Flow | 228 | 222 | 1.3022 | 9.9779 | stalled |
| r200_n1_mri4/sample_019 | DLR D-Flow | 143 | 138 | 23.1037 | 25.7530 | stalled |
| r200_n1_mri4/sample_020 | DLR D-Flow | 689 | 662 | 1.2503 | 9.9748 | stalled |
| r200_n1_mri4/sample_021 | DLR D-Flow | 314 | 302 | 99.9463 | 98.9407 | stalled |
| r200_n1_mri4/sample_022 | DLR D-Flow | 35 | 32 | 101.0301 | 99.9776 | stalled |
| r200_n1_mri4/sample_023 | DLR D-Flow | 261 | 250 | 6.0993 | 11.3932 | stalled |
| r200_n1_mri4/sample_024 | DLR D-Flow | 301 | 284 | 1.2890 | 9.9854 | stalled |
| r200_n1_mri4/sample_025 | DLR D-Flow | 166 | 159 | 1.2190 | 9.9752 | stalled |
| r200_n1_mri4/sample_026 | DLR D-Flow | 315 | 302 | 1.1958 | 9.9887 | stalled |
| r200_n1_mri4/sample_027 | DLR D-Flow | 1000 | 967 | 3.8116 | 10.1478 | step_limit |
| r200_n1_mri4/sample_028 | DLR D-Flow | 336 | 323 | 2.1708 | 10.0804 | stalled |
| r200_n1_mri4/sample_029 | DLR D-Flow | 329 | 321 | 2.7418 | 10.2375 | stalled |
| r200_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 13.0470 | 5.2043 | step_limit |
| r200_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 45.8916 | 44.5407 | step_limit |
| r200_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 13.1521 | 5.3694 | step_limit |
| r200_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.8021 | 9.8470 | step_limit |
| r200_n2_mri4/sample_004 | D-Flow | 448 | 445 | 60.9568 | 59.9866 | stalled |
| r200_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 2.1371 | 9.8500 | step_limit |
| r200_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 3.0203 | 9.4051 | step_limit |
| r200_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 11.8809 | 4.7906 | step_limit |
| r200_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.6230 | 9.9936 | step_limit |
| r200_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 2.3067 | 9.7735 | step_limit |
| r200_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 40.6081 | 39.1630 | step_limit |
| r200_n2_mri4/sample_000 | P-Flow | 134 | 113 | 1.7314 | 9.7896 | stalled |
| r200_n2_mri4/sample_001 | P-Flow | 179 | 175 | 12.2751 | 5.1975 | stalled |
| r200_n2_mri4/sample_002 | P-Flow | 191 | 182 | 14.1655 | 5.0088 | stalled |
| r200_n2_mri4/sample_003 | P-Flow | 192 | 187 | 14.2378 | 5.3990 | stalled |
| r200_n2_mri4/sample_004 | P-Flow | 180 | 177 | 13.5257 | 5.3856 | stalled |
| r200_n2_mri4/sample_005 | P-Flow | 181 | 177 | 14.3129 | 5.0199 | stalled |
| r200_n2_mri4/sample_006 | P-Flow | 218 | 209 | 14.3977 | 4.7335 | stalled |
| r200_n2_mri4/sample_007 | P-Flow | 171 | 165 | 13.1570 | 4.7874 | stalled |
| r200_n2_mri4/sample_008 | P-Flow | 216 | 196 | 13.1947 | 5.5766 | stalled |
| r200_n2_mri4/sample_009 | P-Flow | 187 | 181 | 14.2092 | 4.8465 | stalled |
| r200_n2_mri4/sample_010 | P-Flow | 193 | 183 | 14.7121 | 4.8811 | stalled |
| r200_n2_mri4/sample_000 | DLR D-Flow | 1000 | 975 | 2.1774 | 10.0889 | step_limit |
| r200_n2_mri4/sample_001 | DLR D-Flow | 1000 | 971 | 1.6924 | 10.0673 | step_limit |
| r200_n2_mri4/sample_002 | DLR D-Flow | 313 | 301 | 1.5676 | 10.0024 | stalled |
| r200_n2_mri4/sample_003 | DLR D-Flow | 508 | 493 | 66.7988 | 65.0829 | stalled |
| r200_n2_mri4/sample_004 | DLR D-Flow | 1000 | 970 | 2.4212 | 10.1326 | step_limit |
| r200_n2_mri4/sample_005 | DLR D-Flow | 734 | 702 | 1.8896 | 10.0444 | stalled |
| r200_n2_mri4/sample_006 | DLR D-Flow | 686 | 669 | 77.7152 | 74.5516 | stalled |
| r200_n2_mri4/sample_007 | DLR D-Flow | 324 | 312 | 1.4361 | 9.9815 | stalled |
| r200_n2_mri4/sample_008 | DLR D-Flow | 1000 | 984 | 14.8210 | 14.9230 | step_limit |
| r200_n2_mri4/sample_009 | DLR D-Flow | 219 | 208 | 2.1782 | 10.0672 | stalled |
| r200_n2_mri4/sample_010 | DLR D-Flow | 598 | 574 | 2.6793 | 10.1025 | stalled |
| r200_n2_mri4/sample_011 | DLR D-Flow | 110 | 109 | 72.3314 | 72.9938 | paused |
| r200_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 5.9098 | 9.2293 | step_limit |
| r200_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.7011 | 9.9933 | step_limit |
| r200_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 2.5794 | 9.7834 | step_limit |
| r200_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 2.6895 | 9.7463 | step_limit |
| r200_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 26.1522 | 23.9134 | step_limit |
| r200_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 2.4611 | 9.7656 | step_limit |
| r200_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 2.5615 | 9.7866 | step_limit |
| r200_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 12.9671 | 4.9688 | step_limit |
| r200_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 3.8494 | 9.6018 | step_limit |
| r200_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 30.3303 | 29.4078 | step_limit |
| r200_n4_mri4/sample_000 | P-Flow | 224 | 212 | 14.0741 | 4.9354 | stalled |
| r200_n4_mri4/sample_001 | P-Flow | 208 | 203 | 13.3052 | 4.8323 | stalled |
| r200_n4_mri4/sample_002 | P-Flow | 286 | 252 | 13.9344 | 5.0585 | stalled |
| r200_n4_mri4/sample_003 | P-Flow | 180 | 175 | 13.0068 | 5.8816 | stalled |
| r200_n4_mri4/sample_004 | P-Flow | 192 | 182 | 13.8501 | 4.2947 | stalled |
| r200_n4_mri4/sample_005 | P-Flow | 190 | 186 | 14.0086 | 4.5384 | stalled |
| r200_n4_mri4/sample_006 | P-Flow | 190 | 182 | 12.4290 | 6.0671 | stalled |
| r200_n4_mri4/sample_007 | P-Flow | 173 | 169 | 13.7174 | 4.9688 | stalled |
| r200_n4_mri4/sample_008 | P-Flow | 212 | 196 | 13.1792 | 4.5568 | stalled |
| r200_n4_mri4/sample_009 | P-Flow | 174 | 169 | 13.0530 | 5.8653 | stalled |
| r200_n4_mri4/sample_010 | P-Flow | 189 | 185 | 14.1284 | 4.9891 | paused |
| r200_n4_mri4/sample_000 | DLR D-Flow | 1000 | 972 | 18.9310 | 17.9838 | step_limit |
| r200_n4_mri4/sample_001 | DLR D-Flow | 273 | 265 | 56.3424 | 53.0938 | stalled |
| r200_n4_mri4/sample_002 | DLR D-Flow | 318 | 305 | 51.0670 | 48.6163 | stalled |
| r200_n4_mri4/sample_003 | DLR D-Flow | 417 | 400 | 54.5667 | 53.0705 | stalled |
| r200_n4_mri4/sample_004 | DLR D-Flow | 1000 | 966 | 25.5348 | 22.9993 | step_limit |
| r200_n4_mri4/sample_005 | DLR D-Flow | 169 | 165 | 40.2060 | 38.6497 | stalled |
| r200_n4_mri4/sample_006 | DLR D-Flow | 992 | 953 | 33.1909 | 30.8831 | stalled |
| r200_n4_mri4/sample_007 | DLR D-Flow | 271 | 261 | 44.5046 | 43.3364 | stalled |
| r200_n4_mri4/sample_008 | DLR D-Flow | 1000 | 957 | 23.5098 | 22.9348 | step_limit |
| r200_n4_mri4/sample_009 | DLR D-Flow | 1000 | 973 | 48.8956 | 48.6040 | step_limit |
| r250_n1_ct10/sample_000 | D-Flow | 481 | 479 | 118.7209 | 0.3975 | stalled |
| r250_n1_ct10/sample_001 | D-Flow | 544 | 542 | 61.9640 | 0.3082 | stalled |
| r250_n1_ct10/sample_002 | D-Flow | 505 | 503 | 65.3811 | 0.4624 | stalled |
| r250_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 86.0459 | 0.4405 | step_limit |
| r250_n1_ct10/sample_004 | D-Flow | 442 | 440 | 67.9036 | 0.4599 | stalled |
| r250_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 6.9820 | 7.6482 | step_limit |
| r250_n1_ct10/sample_006 | D-Flow | 483 | 481 | 87.8173 | 0.1681 | stalled |
| r250_n1_ct10/sample_007 | D-Flow | 36 | 33 | 99.7915 | 97.3570 | stalled |
| r250_n1_ct10/sample_008 | D-Flow | 465 | 463 | 102.1023 | 0.5253 | stalled |
| r250_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 6.5768 | 7.5852 | step_limit |
| r250_n1_ct10/sample_010 | D-Flow | 426 | 424 | 83.8854 | 0.5572 | stalled |
| r250_n1_ct10/sample_011 | D-Flow | 563 | 561 | 56.2140 | 0.4573 | stalled |
| r250_n1_ct10/sample_012 | D-Flow | 14 | 10 | 99.7328 | 97.7217 | stalled |
| r250_n1_ct10/sample_013 | D-Flow | 618 | 616 | 50.5464 | 0.5339 | stalled |
| r250_n1_ct10/sample_014 | D-Flow | 608 | 606 | 82.3890 | 0.4239 | stalled |
| r250_n1_ct10/sample_015 | D-Flow | 414 | 412 | 86.9429 | 0.4385 | stalled |
| r250_n1_ct10/sample_016 | D-Flow | 504 | 501 | 63.0792 | 0.6367 | stalled |
| r250_n1_ct10/sample_017 | D-Flow | 399 | 396 | 71.9641 | 0.4362 | stalled |
| r250_n1_ct10/sample_018 | D-Flow | 149 | 146 | 107.3937 | 92.0144 | stalled |
| r250_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 6.2875 | 7.5635 | step_limit |
| r250_n1_ct10/sample_020 | D-Flow | 642 | 639 | 70.2072 | 0.5242 | stalled |
| r250_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 10.2525 | 4.9881 | step_limit |
| r250_n1_ct10/sample_022 | D-Flow | 526 | 524 | 66.9889 | 0.3248 | stalled |
| r250_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 9.3926 | 5.8489 | step_limit |
| r250_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 6.2577 | 7.8041 | step_limit |
| r250_n1_ct10/sample_025 | D-Flow | 596 | 594 | 67.6612 | 0.5018 | stalled |
| r250_n1_ct10/sample_026 | D-Flow | 508 | 506 | 58.9326 | 0.5786 | stalled |
| r250_n1_ct10/sample_027 | D-Flow | 742 | 740 | 50.5237 | 0.3908 | stalled |
| r250_n1_ct10/sample_028 | D-Flow | 819 | 817 | 49.5091 | 0.4586 | stalled |
| r250_n1_ct10/sample_029 | D-Flow | 527 | 525 | 75.7975 | 0.3798 | stalled |
| r250_n1_ct10/sample_000 | P-Flow | 653 | 622 | 48.2829 | 0.3978 | stalled |
| r250_n1_ct10/sample_001 | P-Flow | 1000 | 826 | 41.6721 | 0.4976 | step_limit |
| r250_n1_ct10/sample_002 | P-Flow | 146 | 124 | 4.0828 | 9.9860 | stalled |
| r250_n1_ct10/sample_003 | P-Flow | 759 | 621 | 2.4803 | 9.9240 | stalled |
| r250_n1_ct10/sample_004 | P-Flow | 1000 | 897 | 43.6987 | 0.4837 | step_limit |
| r250_n1_ct10/sample_005 | P-Flow | 7 | 5 | 112.0706 | 95.6297 | stalled |
| r250_n1_ct10/sample_006 | P-Flow | 957 | 852 | 53.7367 | 0.1684 | stalled |
| r250_n1_ct10/sample_007 | P-Flow | 95 | 79 | 2.7086 | 9.9826 | stalled |
| r250_n1_ct10/sample_008 | P-Flow | 166 | 134 | 2.5904 | 9.9462 | stalled |
| r250_n1_ct10/sample_009 | P-Flow | 1000 | 788 | 2.8159 | 9.9685 | step_limit |
| r250_n1_ct10/sample_010 | P-Flow | 869 | 846 | 42.6466 | 0.5602 | stalled |
| r250_n1_ct10/sample_011 | P-Flow | 41 | 34 | 3.7958 | 10.3448 | stalled |
| r250_n1_ct10/sample_012 | P-Flow | 39 | 31 | 2.3769 | 9.9658 | stalled |
| r250_n1_ct10/sample_013 | P-Flow | 31 | 26 | 2.0659 | 9.9592 | stalled |
| r250_n1_ct10/sample_014 | P-Flow | 33 | 26 | 4.3368 | 9.9589 | stalled |
| r250_n1_ct10/sample_015 | P-Flow | 122 | 98 | 3.6837 | 9.9411 | stalled |
| r250_n1_ct10/sample_016 | P-Flow | 543 | 530 | 49.8691 | 0.6367 | stalled |
| r250_n1_ct10/sample_017 | P-Flow | 384 | 302 | 2.1107 | 9.9407 | stalled |
| r250_n1_ct10/sample_018 | P-Flow | 913 | 816 | 37.6295 | 0.5626 | stalled |
| r250_n1_ct10/sample_019 | P-Flow | 25 | 21 | 6.6773 | 10.1631 | stalled |
| r250_n1_ct10/sample_020 | P-Flow | 35 | 30 | 2.7797 | 10.0018 | stalled |
| r250_n1_ct10/sample_021 | P-Flow | 32 | 26 | 3.5610 | 9.9484 | stalled |
| r250_n1_ct10/sample_022 | P-Flow | 529 | 511 | 40.2701 | 0.3248 | stalled |
| r250_n1_ct10/sample_023 | P-Flow | 1000 | 972 | 34.4249 | 0.4757 | step_limit |
| r250_n1_ct10/sample_024 | P-Flow | 528 | 428 | 1.7003 | 9.9399 | stalled |
| r250_n1_ct10/sample_025 | P-Flow | 269 | 211 | 3.1989 | 9.9857 | stalled |
| r250_n1_ct10/sample_026 | P-Flow | 29 | 24 | 3.5936 | 10.1016 | stalled |
| r250_n1_ct10/sample_027 | P-Flow | 115 | 98 | 3.4192 | 9.9679 | stalled |
| r250_n1_ct10/sample_028 | P-Flow | 50 | 42 | 8.6972 | 10.2948 | stalled |
| r250_n1_ct10/sample_029 | P-Flow | 75 | 66 | 3.2220 | 9.8736 | stalled |
| r250_n1_ct10/sample_000 | DLR D-Flow | 70 | 67 | 12.0152 | 10.1600 | stalled |
| r250_n1_ct10/sample_001 | DLR D-Flow | 95 | 92 | 14.3143 | 10.2720 | stalled |
| r250_n1_ct10/sample_002 | DLR D-Flow | 290 | 280 | 2.7105 | 9.9695 | stalled |
| r250_n1_ct10/sample_003 | DLR D-Flow | 482 | 461 | 1.9968 | 9.9507 | stalled |
| r250_n1_ct10/sample_004 | DLR D-Flow | 325 | 309 | 2.8895 | 9.9392 | stalled |
| r250_n1_ct10/sample_005 | DLR D-Flow | 409 | 386 | 2.0476 | 9.9588 | stalled |
| r250_n1_ct10/sample_006 | DLR D-Flow | 179 | 170 | 2.3390 | 9.9490 | stalled |
| r250_n1_ct10/sample_007 | DLR D-Flow | 22 | 19 | 99.6357 | 96.9046 | stalled |
| r250_n1_ct10/sample_008 | DLR D-Flow | 233 | 225 | 2.3880 | 9.9244 | stalled |
| r250_n1_ct10/sample_009 | DLR D-Flow | 483 | 464 | 3.0996 | 9.9513 | stalled |
| r250_n1_ct10/sample_010 | DLR D-Flow | 502 | 491 | 2.4537 | 9.9323 | stalled |
| r250_n1_ct10/sample_011 | DLR D-Flow | 428 | 414 | 2.4915 | 9.9106 | stalled |
| r250_n1_ct10/sample_012 | DLR D-Flow | 10 | 8 | 99.7274 | 97.7209 | stalled |
| r250_n1_ct10/sample_013 | DLR D-Flow | 598 | 571 | 6.9142 | 9.9444 | stalled |
| r250_n1_ct10/sample_014 | DLR D-Flow | 26 | 23 | 103.2562 | 97.6850 | stalled |
| r250_n1_ct10/sample_015 | DLR D-Flow | 138 | 133 | 3.2713 | 9.9474 | stalled |
| r250_n1_ct10/sample_016 | DLR D-Flow | 488 | 470 | 4.6692 | 9.9425 | stalled |
| r250_n1_ct10/sample_017 | DLR D-Flow | 384 | 372 | 3.9175 | 9.9373 | stalled |
| r250_n1_ct10/sample_018 | DLR D-Flow | 97 | 92 | 108.3513 | 93.7064 | stalled |
| r250_n1_ct10/sample_019 | DLR D-Flow | 261 | 250 | 1.4302 | 9.9429 | stalled |
| r250_n1_ct10/sample_020 | DLR D-Flow | 334 | 322 | 2.7634 | 9.9495 | stalled |
| r250_n1_ct10/sample_021 | DLR D-Flow | 151 | 144 | 2.6741 | 9.9324 | stalled |
| r250_n1_ct10/sample_022 | DLR D-Flow | 330 | 318 | 3.0128 | 9.8994 | stalled |
| r250_n1_ct10/sample_023 | DLR D-Flow | 26 | 23 | 111.5311 | 91.7259 | stalled |
| r250_n1_ct10/sample_024 | DLR D-Flow | 192 | 183 | 1.8912 | 9.9245 | stalled |
| r250_n1_ct10/sample_025 | DLR D-Flow | 59 | 55 | 2.3167 | 9.9719 | stalled |
| r250_n1_ct10/sample_026 | DLR D-Flow | 214 | 206 | 2.2780 | 9.9388 | stalled |
| r250_n1_ct10/sample_027 | DLR D-Flow | 257 | 243 | 16.0912 | 12.1463 | stalled |
| r250_n1_ct10/sample_028 | DLR D-Flow | 780 | 756 | 2.7699 | 9.9439 | stalled |
| r250_n1_ct10/sample_029 | DLR D-Flow | 425 | 401 | 3.7957 | 9.9194 | stalled |
| r250_n2_ct10/sample_000 | D-Flow | 512 | 510 | 89.2698 | 0.2735 | stalled |
| r250_n2_ct10/sample_001 | D-Flow | 532 | 529 | 89.8927 | 0.4666 | stalled |
| r250_n2_ct10/sample_002 | D-Flow | 424 | 422 | 129.4139 | 0.4833 | stalled |
| r250_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 12.0820 | 7.8064 | step_limit |
| r250_n2_ct10/sample_004 | D-Flow | 840 | 838 | 63.3979 | 0.3066 | stalled |
| r250_n2_ct10/sample_005 | D-Flow | 591 | 589 | 86.2372 | 0.5798 | stalled |
| r250_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 10.1092 | 8.1708 | step_limit |
| r250_n2_ct10/sample_007 | D-Flow | 835 | 833 | 24.4685 | 4.0407 | stalled |
| r250_n2_ct10/sample_008 | D-Flow | 505 | 503 | 62.3087 | 0.4483 | stalled |
| r250_n2_ct10/sample_009 | D-Flow | 415 | 413 | 74.6209 | 0.5153 | stalled |
| r250_n2_ct10/sample_000 | P-Flow | 477 | 464 | 48.5016 | 0.2756 | stalled |
| r250_n2_ct10/sample_001 | P-Flow | 1000 | 831 | 9.0115 | 8.3859 | step_limit |
| r250_n2_ct10/sample_002 | P-Flow | 943 | 838 | 44.1559 | 0.4884 | stalled |
| r250_n2_ct10/sample_003 | P-Flow | 656 | 620 | 43.7219 | 0.4343 | stalled |
| r250_n2_ct10/sample_004 | P-Flow | 1000 | 862 | 42.6227 | 0.3676 | step_limit |
| r250_n2_ct10/sample_005 | P-Flow | 1000 | 893 | 42.9039 | 0.7651 | step_limit |
| r250_n2_ct10/sample_006 | P-Flow | 171 | 146 | 4.2285 | 9.8951 | stalled |
| r250_n2_ct10/sample_007 | P-Flow | 517 | 417 | 4.4302 | 9.9007 | stalled |
| r250_n2_ct10/sample_008 | P-Flow | 70 | 63 | 3.3193 | 9.9201 | stalled |
| r250_n2_ct10/sample_009 | P-Flow | 477 | 438 | 45.1642 | 0.5183 | stalled |
| r250_n2_ct10/sample_000 | DLR D-Flow | 148 | 139 | 4.8644 | 10.0446 | stalled |
| r250_n2_ct10/sample_001 | DLR D-Flow | 99 | 95 | 3.9375 | 9.9494 | stalled |
| r250_n2_ct10/sample_002 | DLR D-Flow | 192 | 184 | 4.7200 | 9.8887 | stalled |
| r250_n2_ct10/sample_003 | DLR D-Flow | 588 | 567 | 3.5743 | 9.9483 | stalled |
| r250_n2_ct10/sample_004 | DLR D-Flow | 267 | 261 | 3.1849 | 9.9489 | stalled |
| r250_n2_ct10/sample_005 | DLR D-Flow | 441 | 428 | 8.2003 | 10.1231 | stalled |
| r250_n2_ct10/sample_006 | DLR D-Flow | 259 | 250 | 6.4373 | 10.0295 | stalled |
| r250_n2_ct10/sample_007 | DLR D-Flow | 192 | 188 | 76.5373 | 47.2648 | stalled |
| r250_n2_ct10/sample_008 | DLR D-Flow | 394 | 384 | 3.4590 | 10.0220 | stalled |
| r250_n2_ct10/sample_009 | DLR D-Flow | 361 | 348 | 2.3847 | 9.9421 | stalled |
| r250_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 9.6112 | 9.7731 | step_limit |
| r250_n4_ct10/sample_001 | D-Flow | 487 | 484 | 97.3583 | 0.3972 | stalled |
| r250_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 11.5125 | 10.3776 | step_limit |
| r250_n4_ct10/sample_003 | D-Flow | 524 | 522 | 79.2712 | 0.4257 | stalled |
| r250_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 5.1424 | 9.9089 | step_limit |
| r250_n4_ct10/sample_005 | D-Flow | 461 | 459 | 81.0926 | 0.1168 | stalled |
| r250_n4_ct10/sample_006 | D-Flow | 615 | 613 | 66.8078 | 0.3129 | stalled |
| r250_n4_ct10/sample_007 | D-Flow | 724 | 722 | 67.0817 | 0.3582 | stalled |
| r250_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 14.8566 | 9.4998 | step_limit |
| r250_n4_ct10/sample_009 | D-Flow | 148 | 146 | 53.9765 | 26.0742 | stalled |
| r250_n4_ct10/sample_000 | P-Flow | 66 | 57 | 7.6844 | 9.8923 | stalled |
| r250_n4_ct10/sample_001 | P-Flow | 1000 | 926 | 45.6336 | 0.4551 | step_limit |
| r250_n4_ct10/sample_002 | P-Flow | 112 | 101 | 5.6156 | 9.8744 | stalled |
| r250_n4_ct10/sample_003 | P-Flow | 1000 | 915 | 47.4038 | 0.4751 | step_limit |
| r250_n4_ct10/sample_004 | P-Flow | 1000 | 904 | 38.8244 | 0.6044 | step_limit |
| r250_n4_ct10/sample_005 | P-Flow | 1000 | 894 | 39.8187 | 0.4180 | step_limit |
| r250_n4_ct10/sample_006 | P-Flow | 849 | 816 | 48.4034 | 0.3173 | stalled |
| r250_n4_ct10/sample_007 | P-Flow | 215 | 187 | 6.3884 | 9.4760 | stalled |
| r250_n4_ct10/sample_008 | P-Flow | 161 | 143 | 6.0782 | 9.5646 | stalled |
| r250_n4_ct10/sample_009 | P-Flow | 641 | 519 | 4.8870 | 9.8531 | stalled |
| r250_n4_ct10/sample_000 | DLR D-Flow | 922 | 880 | 40.1922 | 17.8928 | stalled |
| r250_n4_ct10/sample_001 | DLR D-Flow | 203 | 195 | 43.8345 | 26.1192 | stalled |
| r250_n4_ct10/sample_002 | DLR D-Flow | 841 | 805 | 52.9608 | 24.1451 | stalled |
| r250_n4_ct10/sample_003 | DLR D-Flow | 828 | 793 | 18.1866 | 10.7296 | stalled |
| r250_n4_ct10/sample_004 | DLR D-Flow | 81 | 77 | 66.1329 | 31.8591 | stalled |
| r250_n4_ct10/sample_005 | DLR D-Flow | 1000 | 953 | 50.7975 | 25.9570 | step_limit |
| r250_n4_ct10/sample_006 | DLR D-Flow | 1000 | 951 | 55.6257 | 26.9076 | step_limit |
| r250_n4_ct10/sample_007 | DLR D-Flow | 128 | 120 | 74.9531 | 36.5864 | stalled |
| r250_n4_ct10/sample_008 | DLR D-Flow | 516 | 494 | 82.6813 | 41.4229 | stalled |
| r250_n4_ct10/sample_009 | DLR D-Flow | 1000 | 969 | 48.7131 | 22.1121 | step_limit |
| r250_n1_mri4/sample_000 | D-Flow | 677 | 675 | 12.7591 | 4.9975 | stalled |
| r250_n1_mri4/sample_001 | D-Flow | 898 | 896 | 12.4292 | 5.5475 | stalled |
| r250_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 12.7436 | 5.5529 | step_limit |
| r250_n1_mri4/sample_003 | D-Flow | 809 | 806 | 13.0245 | 4.9961 | stalled |
| r250_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 12.1438 | 4.6311 | step_limit |
| r250_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 12.9536 | 5.2885 | step_limit |
| r250_n1_mri4/sample_006 | D-Flow | 655 | 652 | 12.9018 | 5.4284 | stalled |
| r250_n1_mri4/sample_007 | D-Flow | 23 | 21 | 99.6311 | 99.5946 | stalled |
| r250_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 7.7302 | 5.8604 | step_limit |
| r250_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 14.0415 | 4.2986 | step_limit |
| r250_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 6.8315 | 7.3021 | step_limit |
| r250_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 11.3517 | 6.7091 | step_limit |
| r250_n1_mri4/sample_012 | D-Flow | 11 | 9 | 99.5923 | 99.5413 | stalled |
| r250_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 5.5976 | 7.1050 | step_limit |
| r250_n1_mri4/sample_014 | D-Flow | 75 | 73 | 99.2284 | 98.9014 | stalled |
| r250_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 12.5089 | 5.4260 | step_limit |
| r250_n1_mri4/sample_016 | D-Flow | 901 | 898 | 12.5594 | 5.2866 | stalled |
| r250_n1_mri4/sample_017 | D-Flow | 23 | 21 | 99.5400 | 99.5097 | stalled |
| r250_n1_mri4/sample_018 | D-Flow | 149 | 145 | 99.4530 | 99.4085 | stalled |
| r250_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 12.0638 | 4.9442 | step_limit |
| r250_n1_mri4/sample_020 | D-Flow | 825 | 822 | 13.7968 | 4.6216 | stalled |
| r250_n1_mri4/sample_021 | D-Flow | 19 | 15 | 99.5006 | 99.4722 | stalled |
| r250_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 10.8041 | 7.3903 | step_limit |
| r250_n1_mri4/sample_023 | D-Flow | 269 | 265 | 99.4590 | 99.4323 | stalled |
| r250_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 12.6121 | 5.3444 | step_limit |
| r250_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 7.3085 | 6.6326 | step_limit |
| r250_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 5.7461 | 8.2925 | step_limit |
| r250_n1_mri4/sample_027 | D-Flow | 103 | 100 | 99.6714 | 99.6475 | stalled |
| r250_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 3.8884 | 9.2701 | step_limit |
| r250_n1_mri4/sample_029 | D-Flow | 18 | 14 | 99.5092 | 99.3482 | stalled |
| r250_n1_mri4/sample_000 | P-Flow | 224 | 208 | 14.2536 | 4.9975 | stalled |
| r250_n1_mri4/sample_001 | P-Flow | 187 | 178 | 13.6030 | 5.5475 | stalled |
| r250_n1_mri4/sample_002 | P-Flow | 26 | 21 | 1.5330 | 10.0137 | stalled |
| r250_n1_mri4/sample_003 | P-Flow | 206 | 200 | 14.3319 | 4.9961 | stalled |
| r250_n1_mri4/sample_004 | P-Flow | 218 | 204 | 13.2467 | 4.6311 | stalled |
| r250_n1_mri4/sample_005 | P-Flow | 18 | 14 | 1.9405 | 10.0540 | stalled |
| r250_n1_mri4/sample_006 | P-Flow | 229 | 211 | 13.8270 | 5.4284 | stalled |
| r250_n1_mri4/sample_007 | P-Flow | 246 | 226 | 15.5981 | 4.8407 | stalled |
| r250_n1_mri4/sample_008 | P-Flow | 76 | 65 | 1.5288 | 9.9482 | stalled |
| r250_n1_mri4/sample_009 | P-Flow | 19 | 15 | 1.9711 | 10.0667 | stalled |
| r250_n1_mri4/sample_010 | P-Flow | 185 | 176 | 12.7788 | 5.4816 | stalled |
| r250_n1_mri4/sample_011 | P-Flow | 187 | 179 | 13.7567 | 5.4568 | stalled |
| r250_n1_mri4/sample_012 | P-Flow | 203 | 187 | 14.9201 | 5.1816 | stalled |
| r250_n1_mri4/sample_013 | P-Flow | 187 | 175 | 12.6499 | 5.3892 | stalled |
| r250_n1_mri4/sample_014 | P-Flow | 190 | 177 | 14.6552 | 4.9546 | stalled |
| r250_n1_mri4/sample_015 | P-Flow | 260 | 236 | 13.8984 | 5.4241 | stalled |
| r250_n1_mri4/sample_016 | P-Flow | 227 | 208 | 13.8851 | 5.2866 | stalled |
| r250_n1_mri4/sample_017 | P-Flow | 243 | 222 | 14.8945 | 4.7513 | stalled |
| r250_n1_mri4/sample_018 | P-Flow | 195 | 185 | 15.5560 | 4.4751 | stalled |
| r250_n1_mri4/sample_019 | P-Flow | 161 | 157 | 13.1241 | 4.9442 | stalled |
| r250_n1_mri4/sample_020 | P-Flow | 64 | 55 | 1.4905 | 9.9946 | stalled |
| r250_n1_mri4/sample_021 | P-Flow | 72 | 63 | 1.7428 | 9.9659 | stalled |
| r250_n1_mri4/sample_022 | P-Flow | 191 | 183 | 14.2853 | 5.4146 | stalled |
| r250_n1_mri4/sample_023 | P-Flow | 246 | 225 | 15.4713 | 4.6733 | stalled |
| r250_n1_mri4/sample_024 | P-Flow | 211 | 202 | 13.7672 | 5.3156 | stalled |
| r250_n1_mri4/sample_025 | P-Flow | 19 | 15 | 1.9769 | 10.0381 | stalled |
| r250_n1_mri4/sample_026 | P-Flow | 167 | 162 | 14.5808 | 5.1220 | stalled |
| r250_n1_mri4/sample_027 | P-Flow | 214 | 206 | 15.2895 | 5.3297 | stalled |
| r250_n1_mri4/sample_028 | P-Flow | 177 | 175 | 14.6626 | 4.9941 | stalled |
| r250_n1_mri4/sample_029 | P-Flow | 200 | 187 | 13.0525 | 5.0485 | stalled |
| r250_n1_mri4/sample_000 | DLR D-Flow | 253 | 246 | 1.2767 | 9.9563 | stalled |
| r250_n1_mri4/sample_001 | DLR D-Flow | 204 | 195 | 1.4533 | 9.9929 | stalled |
| r250_n1_mri4/sample_002 | DLR D-Flow | 290 | 280 | 1.2762 | 9.9759 | stalled |
| r250_n1_mri4/sample_003 | DLR D-Flow | 329 | 317 | 1.4609 | 9.9818 | stalled |
| r250_n1_mri4/sample_004 | DLR D-Flow | 189 | 183 | 1.1710 | 9.9707 | stalled |
| r250_n1_mri4/sample_005 | DLR D-Flow | 370 | 361 | 1.5752 | 9.9909 | stalled |
| r250_n1_mri4/sample_006 | DLR D-Flow | 354 | 342 | 1.1552 | 9.9757 | stalled |
| r250_n1_mri4/sample_007 | DLR D-Flow | 20 | 17 | 99.4747 | 99.4269 | stalled |
| r250_n1_mri4/sample_008 | DLR D-Flow | 470 | 452 | 1.6782 | 9.9883 | stalled |
| r250_n1_mri4/sample_009 | DLR D-Flow | 355 | 340 | 1.2984 | 9.9683 | stalled |
| r250_n1_mri4/sample_010 | DLR D-Flow | 819 | 798 | 1.6618 | 10.0045 | stalled |
| r250_n1_mri4/sample_011 | DLR D-Flow | 600 | 584 | 1.2757 | 9.9857 | stalled |
| r250_n1_mri4/sample_012 | DLR D-Flow | 19 | 17 | 99.5921 | 99.5411 | stalled |
| r250_n1_mri4/sample_013 | DLR D-Flow | 161 | 153 | 1.2960 | 9.9693 | stalled |
| r250_n1_mri4/sample_014 | DLR D-Flow | 44 | 40 | 107.9178 | 105.0677 | stalled |
| r250_n1_mri4/sample_015 | DLR D-Flow | 236 | 228 | 1.1335 | 9.9657 | stalled |
| r250_n1_mri4/sample_016 | DLR D-Flow | 377 | 362 | 1.1852 | 9.9734 | stalled |
| r250_n1_mri4/sample_017 | DLR D-Flow | 19 | 16 | 99.5225 | 99.4931 | stalled |
| r250_n1_mri4/sample_018 | DLR D-Flow | 30 | 26 | 99.6677 | 99.6390 | stalled |
| r250_n1_mri4/sample_019 | DLR D-Flow | 404 | 388 | 1.4154 | 10.0007 | stalled |
| r250_n1_mri4/sample_020 | DLR D-Flow | 573 | 558 | 1.2930 | 9.9731 | stalled |
| r250_n1_mri4/sample_021 | DLR D-Flow | 15 | 13 | 97.8101 | 97.6426 | stalled |
| r250_n1_mri4/sample_022 | DLR D-Flow | 508 | 493 | 1.2540 | 9.9712 | stalled |
| r250_n1_mri4/sample_023 | DLR D-Flow | 21 | 19 | 99.2970 | 99.2650 | stalled |
| r250_n1_mri4/sample_024 | DLR D-Flow | 92 | 88 | 1.2265 | 9.9707 | stalled |
| r250_n1_mri4/sample_025 | DLR D-Flow | 381 | 367 | 1.4315 | 9.9824 | stalled |
| r250_n1_mri4/sample_026 | DLR D-Flow | 29 | 27 | 1.5439 | 10.0073 | stalled |
| r250_n1_mri4/sample_027 | DLR D-Flow | 21 | 17 | 99.6171 | 99.5930 | stalled |
| r250_n1_mri4/sample_028 | DLR D-Flow | 58 | 55 | 2.0316 | 10.0759 | stalled |
| r250_n1_mri4/sample_029 | DLR D-Flow | 16 | 12 | 99.4976 | 99.3331 | stalled |
| r250_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 2.4948 | 9.8318 | step_limit |
| r250_n2_mri4/sample_001 | D-Flow | 938 | 936 | 12.2251 | 4.9661 | stalled |
| r250_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 32.5115 | 31.5044 | step_limit |
| r250_n2_mri4/sample_003 | D-Flow | 828 | 826 | 33.7550 | 33.3412 | stalled |
| r250_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 2.0248 | 9.8513 | step_limit |
| r250_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 9.9601 | 5.8888 | step_limit |
| r250_n2_mri4/sample_006 | D-Flow | 540 | 538 | 12.3973 | 5.1426 | stalled |
| r250_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 7.0237 | 8.5453 | step_limit |
| r250_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 11.1563 | 4.9271 | step_limit |
| r250_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 8.1182 | 8.8094 | step_limit |
| r250_n2_mri4/sample_000 | P-Flow | 168 | 164 | 13.4941 | 5.1684 | stalled |
| r250_n2_mri4/sample_001 | P-Flow | 156 | 151 | 13.9805 | 4.9661 | stalled |
| r250_n2_mri4/sample_002 | P-Flow | 198 | 183 | 14.3456 | 5.0201 | stalled |
| r250_n2_mri4/sample_003 | P-Flow | 196 | 183 | 14.1016 | 5.2516 | stalled |
| r250_n2_mri4/sample_004 | P-Flow | 168 | 164 | 13.8858 | 5.2841 | stalled |
| r250_n2_mri4/sample_005 | P-Flow | 177 | 175 | 12.4189 | 5.1007 | stalled |
| r250_n2_mri4/sample_006 | P-Flow | 172 | 166 | 14.2783 | 5.1426 | stalled |
| r250_n2_mri4/sample_007 | P-Flow | 206 | 185 | 13.6957 | 5.4008 | stalled |
| r250_n2_mri4/sample_008 | P-Flow | 174 | 165 | 12.8050 | 4.4312 | stalled |
| r250_n2_mri4/sample_009 | P-Flow | 221 | 209 | 14.6865 | 4.6728 | stalled |
| r250_n2_mri4/sample_000 | DLR D-Flow | 343 | 332 | 5.6819 | 10.4884 | stalled |
| r250_n2_mri4/sample_001 | DLR D-Flow | 601 | 583 | 1.6545 | 10.0123 | stalled |
| r250_n2_mri4/sample_002 | DLR D-Flow | 591 | 576 | 44.6671 | 44.2971 | stalled |
| r250_n2_mri4/sample_003 | DLR D-Flow | 1000 | 961 | 1.6722 | 10.0121 | step_limit |
| r250_n2_mri4/sample_004 | DLR D-Flow | 405 | 388 | 3.7138 | 10.3344 | stalled |
| r250_n2_mri4/sample_005 | DLR D-Flow | 252 | 247 | 53.0677 | 51.7410 | stalled |
| r250_n2_mri4/sample_006 | DLR D-Flow | 905 | 883 | 1.7025 | 10.0262 | stalled |
| r250_n2_mri4/sample_007 | DLR D-Flow | 276 | 266 | 68.5170 | 68.8177 | stalled |
| r250_n2_mri4/sample_008 | DLR D-Flow | 231 | 221 | 3.8473 | 10.5296 | stalled |
| r250_n2_mri4/sample_009 | DLR D-Flow | 509 | 489 | 3.1955 | 10.1424 | stalled |
| r250_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.6524 | 9.8086 | step_limit |
| r250_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 2.3219 | 9.6288 | step_limit |
| r250_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 2.0319 | 9.8087 | step_limit |
| r250_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 3.4664 | 9.6195 | step_limit |
| r250_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.5911 | 9.9048 | step_limit |
| r250_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 6.8683 | 9.0303 | step_limit |
| r250_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 13.0346 | 5.3981 | step_limit |
| r250_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 2.7221 | 9.7386 | step_limit |
| r250_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 52.4926 | 51.8876 | step_limit |
| r250_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 9.9592 | 8.0416 | step_limit |
| r250_n4_mri4/sample_000 | P-Flow | 248 | 232 | 13.0003 | 5.1558 | stalled |
| r250_n4_mri4/sample_001 | P-Flow | 214 | 204 | 13.9069 | 5.3059 | stalled |
| r250_n4_mri4/sample_002 | P-Flow | 193 | 186 | 13.5560 | 5.6085 | stalled |
| r250_n4_mri4/sample_003 | P-Flow | 189 | 180 | 13.0869 | 5.3045 | stalled |
| r250_n4_mri4/sample_004 | P-Flow | 177 | 174 | 14.1021 | 5.2860 | stalled |
| r250_n4_mri4/sample_005 | P-Flow | 202 | 198 | 12.3851 | 5.7490 | stalled |
| r250_n4_mri4/sample_006 | P-Flow | 197 | 194 | 13.8396 | 5.2622 | stalled |
| r250_n4_mri4/sample_007 | P-Flow | 189 | 184 | 14.3172 | 5.0845 | stalled |
| r250_n4_mri4/sample_008 | P-Flow | 289 | 263 | 13.9602 | 5.4331 | stalled |
| r250_n4_mri4/sample_009 | P-Flow | 196 | 190 | 14.0444 | 5.4249 | stalled |
| r250_n4_mri4/sample_000 | DLR D-Flow | 1000 | 957 | 46.8294 | 44.0174 | step_limit |
| r250_n4_mri4/sample_001 | DLR D-Flow | 1000 | 949 | 29.5258 | 27.5698 | step_limit |
| r250_n4_mri4/sample_002 | DLR D-Flow | 185 | 177 | 51.4126 | 50.1355 | stalled |
| r250_n4_mri4/sample_003 | DLR D-Flow | 723 | 694 | 40.8529 | 41.1016 | stalled |
| r250_n4_mri4/sample_004 | DLR D-Flow | 653 | 627 | 46.4856 | 44.2521 | stalled |
| r250_n4_mri4/sample_005 | DLR D-Flow | 182 | 175 | 44.8068 | 43.4340 | stalled |
| r250_n4_mri4/sample_006 | DLR D-Flow | 1000 | 952 | 52.8323 | 49.3914 | step_limit |
| r250_n4_mri4/sample_007 | DLR D-Flow | 1000 | 963 | 64.1230 | 61.1629 | step_limit |
| r250_n4_mri4/sample_008 | DLR D-Flow | 868 | 832 | 73.8078 | 71.9650 | stalled |
| r250_n4_mri4/sample_009 | DLR D-Flow | 672 | 649 | 68.3019 | 66.8728 | stalled |

Terminal method runs: 1941/3240. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.
