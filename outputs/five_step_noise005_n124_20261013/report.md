# Five-step Euler pilot

5% relative measurement noise; CT10 and Cartesian MRI4; 30 test samples per acquisition at each resolution; circle count in [1, 2, 4]. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 8, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 38.5657 | 0.2643 | step_limit |
| r64_n1_ct10/sample_001 | D-Flow | 711 | 709 | 36.5453 | 0.5032 | stalled |
| r64_n1_ct10/sample_002 | D-Flow | 978 | 976 | 23.5474 | 0.6166 | stalled |
| r64_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 31.0509 | 0.2378 | step_limit |
| r64_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 72.0870 | 57.5237 | step_limit |
| r64_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 33.3906 | 0.4516 | step_limit |
| r64_n1_ct10/sample_006 | D-Flow | 925 | 922 | 32.4474 | 0.5225 | stalled |
| r64_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 38.9876 | 0.4463 | step_limit |
| r64_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 31.4876 | 0.1773 | step_limit |
| r64_n1_ct10/sample_009 | D-Flow | 687 | 685 | 8.5915 | 2.8993 | stalled |
| r64_n1_ct10/sample_010 | D-Flow | 789 | 787 | 34.2192 | 0.2956 | stalled |
| r64_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 41.4859 | 0.3843 | step_limit |
| r64_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 30.6603 | 0.3360 | step_limit |
| r64_n1_ct10/sample_013 | D-Flow | 684 | 682 | 31.0325 | 0.3114 | stalled |
| r64_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 9.6572 | 2.7309 | step_limit |
| r64_n1_ct10/sample_015 | D-Flow | 369 | 367 | 104.0610 | 89.4143 | stalled |
| r64_n1_ct10/sample_016 | D-Flow | 1000 | 1000 | 29.0249 | 0.4366 | step_limit |
| r64_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 29.0898 | 0.3726 | step_limit |
| r64_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 33.0830 | 0.4869 | step_limit |
| r64_n1_ct10/sample_019 | D-Flow | 796 | 793 | 37.2885 | 0.3098 | stalled |
| r64_n1_ct10/sample_020 | D-Flow | 704 | 702 | 7.1614 | 3.2848 | stalled |
| r64_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 18.4932 | 1.1599 | step_limit |
| r64_n1_ct10/sample_022 | D-Flow | 1000 | 1000 | 36.8214 | 0.2341 | step_limit |
| r64_n1_ct10/sample_023 | D-Flow | 454 | 452 | 55.4737 | 0.3144 | stalled |
| r64_n1_ct10/sample_024 | D-Flow | 819 | 817 | 45.2518 | 0.4479 | stalled |
| r64_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 30.9886 | 0.3352 | step_limit |
| r64_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 37.4295 | 0.5623 | step_limit |
| r64_n1_ct10/sample_027 | D-Flow | 679 | 677 | 8.0082 | 3.6321 | stalled |
| r64_n1_ct10/sample_028 | D-Flow | 1000 | 1000 | 26.4031 | 0.4387 | step_limit |
| r64_n1_ct10/sample_029 | D-Flow | 993 | 991 | 36.9452 | 0.3181 | stalled |
| r64_n1_ct10/sample_000 | P-Flow | 17 | 13 | 39.0710 | 19.3895 | stalled |
| r64_n1_ct10/sample_001 | P-Flow | 64 | 51 | 3.1562 | 5.1009 | stalled |
| r64_n1_ct10/sample_002 | P-Flow | 66 | 59 | 2.9509 | 4.8850 | stalled |
| r64_n1_ct10/sample_003 | P-Flow | 184 | 160 | 3.2804 | 4.9242 | stalled |
| r64_n1_ct10/sample_004 | P-Flow | 769 | 706 | 25.5317 | 0.3660 | stalled |
| r64_n1_ct10/sample_005 | P-Flow | 41 | 33 | 5.3807 | 5.2542 | stalled |
| r64_n1_ct10/sample_006 | P-Flow | 1000 | 888 | 32.4511 | 0.5235 | step_limit |
| r64_n1_ct10/sample_007 | P-Flow | 775 | 739 | 38.6114 | 0.4463 | stalled |
| r64_n1_ct10/sample_008 | P-Flow | 57 | 49 | 3.9036 | 5.1570 | stalled |
| r64_n1_ct10/sample_009 | P-Flow | 1000 | 899 | 19.7324 | 0.8595 | step_limit |
| r64_n1_ct10/sample_010 | P-Flow | 68 | 58 | 5.9934 | 5.8288 | stalled |
| r64_n1_ct10/sample_011 | P-Flow | 555 | 531 | 40.7370 | 0.3839 | stalled |
| r64_n1_ct10/sample_012 | P-Flow | 29 | 24 | 2.9834 | 5.0848 | stalled |
| r64_n1_ct10/sample_013 | P-Flow | 1000 | 785 | 2.9228 | 4.9092 | step_limit |
| r64_n1_ct10/sample_014 | P-Flow | 58 | 52 | 4.3537 | 5.3637 | stalled |
| r64_n1_ct10/sample_015 | P-Flow | 69 | 61 | 2.4241 | 5.0752 | stalled |
| r64_n1_ct10/sample_016 | P-Flow | 59 | 50 | 11.4038 | 8.1698 | stalled |
| r64_n1_ct10/sample_017 | P-Flow | 116 | 100 | 2.3720 | 4.9789 | stalled |
| r64_n1_ct10/sample_018 | P-Flow | 61 | 51 | 3.9242 | 5.3448 | stalled |
| r64_n1_ct10/sample_019 | P-Flow | 42 | 36 | 2.9020 | 5.2911 | stalled |
| r64_n1_ct10/sample_020 | P-Flow | 28 | 23 | 3.9434 | 5.1710 | stalled |
| r64_n1_ct10/sample_021 | P-Flow | 135 | 115 | 3.6653 | 5.0187 | stalled |
| r64_n1_ct10/sample_022 | P-Flow | 32 | 27 | 4.0317 | 5.4539 | stalled |
| r64_n1_ct10/sample_023 | P-Flow | 728 | 701 | 40.9325 | 0.3145 | stalled |
| r64_n1_ct10/sample_024 | P-Flow | 1000 | 892 | 31.3613 | 0.5115 | step_limit |
| r64_n1_ct10/sample_025 | P-Flow | 116 | 104 | 4.2379 | 4.6298 | stalled |
| r64_n1_ct10/sample_026 | P-Flow | 1000 | 954 | 20.1052 | 0.7206 | step_limit |
| r64_n1_ct10/sample_027 | P-Flow | 36 | 30 | 3.3864 | 5.0799 | stalled |
| r64_n1_ct10/sample_028 | P-Flow | 22 | 19 | 8.1460 | 6.7750 | stalled |
| r64_n1_ct10/sample_029 | P-Flow | 48 | 39 | 2.3063 | 5.0397 | stalled |
| r64_n1_ct10/sample_000 | DLR D-Flow | 402 | 385 | 24.9686 | 11.8819 | stalled |
| r64_n1_ct10/sample_001 | DLR D-Flow | 334 | 309 | 2.9242 | 5.0294 | stalled |
| r64_n1_ct10/sample_002 | DLR D-Flow | 230 | 214 | 2.7025 | 4.9951 | stalled |
| r64_n1_ct10/sample_003 | DLR D-Flow | 428 | 415 | 3.1261 | 4.9952 | stalled |
| r64_n1_ct10/sample_004 | DLR D-Flow | 1000 | 955 | 100.9990 | 81.7843 | step_limit |
| r64_n1_ct10/sample_005 | DLR D-Flow | 419 | 398 | 3.0833 | 5.1069 | stalled |
| r64_n1_ct10/sample_006 | DLR D-Flow | 109 | 106 | 2.4720 | 5.0150 | stalled |
| r64_n1_ct10/sample_007 | DLR D-Flow | 207 | 197 | 2.5672 | 4.9827 | stalled |
| r64_n1_ct10/sample_008 | DLR D-Flow | 1000 | 908 | 1.8706 | 4.9983 | step_limit |
| r64_n1_ct10/sample_009 | DLR D-Flow | 92 | 86 | 4.0533 | 5.4713 | stalled |
| r64_n1_ct10/sample_010 | DLR D-Flow | 186 | 177 | 2.2179 | 4.9925 | stalled |
| r64_n1_ct10/sample_011 | DLR D-Flow | 104 | 101 | 4.5086 | 4.9762 | stalled |
| r64_n1_ct10/sample_012 | DLR D-Flow | 190 | 184 | 5.6973 | 5.1482 | stalled |
| r64_n1_ct10/sample_013 | DLR D-Flow | 97 | 92 | 2.4168 | 4.9305 | stalled |
| r64_n1_ct10/sample_014 | DLR D-Flow | 341 | 326 | 5.0424 | 5.9823 | stalled |
| r64_n1_ct10/sample_015 | DLR D-Flow | 462 | 442 | 5.0531 | 5.5201 | stalled |
| r64_n1_ct10/sample_016 | DLR D-Flow | 85 | 79 | 5.3802 | 5.1662 | stalled |
| r64_n1_ct10/sample_017 | DLR D-Flow | 438 | 415 | 2.3319 | 4.9867 | stalled |
| r64_n1_ct10/sample_018 | DLR D-Flow | 220 | 216 | 2.2857 | 5.0203 | stalled |
| r64_n1_ct10/sample_019 | DLR D-Flow | 211 | 198 | 1.9574 | 4.9682 | stalled |
| r64_n1_ct10/sample_020 | DLR D-Flow | 239 | 226 | 1.9193 | 4.9781 | stalled |
| r64_n1_ct10/sample_021 | DLR D-Flow | 39 | 35 | 100.4580 | 89.2554 | stalled |
| r64_n1_ct10/sample_022 | DLR D-Flow | 177 | 167 | 2.2699 | 4.9650 | stalled |
| r64_n1_ct10/sample_023 | DLR D-Flow | 66 | 64 | 3.0052 | 4.9925 | stalled |
| r64_n1_ct10/sample_024 | DLR D-Flow | 496 | 467 | 2.1711 | 4.9783 | stalled |
| r64_n1_ct10/sample_025 | DLR D-Flow | 198 | 189 | 5.3276 | 4.9733 | stalled |
| r64_n1_ct10/sample_026 | DLR D-Flow | 1000 | 957 | 2.5773 | 5.0525 | step_limit |
| r64_n1_ct10/sample_027 | DLR D-Flow | 394 | 379 | 1.8859 | 4.9834 | stalled |
| r64_n1_ct10/sample_028 | DLR D-Flow | 328 | 315 | 3.8785 | 5.0211 | stalled |
| r64_n1_ct10/sample_029 | DLR D-Flow | 84 | 81 | 1.9810 | 5.0060 | stalled |
| r64_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 31.7172 | 0.4297 | step_limit |
| r64_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 31.3321 | 0.4860 | step_limit |
| r64_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 28.4731 | 0.3200 | step_limit |
| r64_n2_ct10/sample_003 | D-Flow | 843 | 841 | 34.4152 | 0.3865 | stalled |
| r64_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 9.9725 | 6.0129 | step_limit |
| r64_n2_ct10/sample_005 | D-Flow | 626 | 624 | 40.5401 | 0.4176 | stalled |
| r64_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 7.8312 | 2.3529 | step_limit |
| r64_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 6.1276 | 3.5615 | step_limit |
| r64_n2_ct10/sample_008 | D-Flow | 761 | 759 | 32.3054 | 0.2947 | stalled |
| r64_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 4.0401 | 3.8779 | step_limit |
| r64_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 30.6108 | 0.6520 | step_limit |
| r64_n2_ct10/sample_000 | P-Flow | 1000 | 893 | 23.5127 | 0.6187 | step_limit |
| r64_n2_ct10/sample_001 | P-Flow | 948 | 885 | 37.6917 | 0.3622 | stalled |
| r64_n2_ct10/sample_002 | P-Flow | 137 | 114 | 3.4929 | 4.9839 | stalled |
| r64_n2_ct10/sample_003 | P-Flow | 10 | 8 | 113.3400 | 70.3850 | stalled |
| r64_n2_ct10/sample_004 | P-Flow | 14 | 11 | 71.1015 | 39.5942 | stalled |
| r64_n2_ct10/sample_005 | P-Flow | 17 | 12 | 71.4122 | 39.8984 | stalled |
| r64_n2_ct10/sample_006 | P-Flow | 32 | 27 | 3.9217 | 5.0435 | stalled |
| r64_n2_ct10/sample_007 | P-Flow | 114 | 98 | 4.5108 | 5.1343 | stalled |
| r64_n2_ct10/sample_008 | P-Flow | 1000 | 897 | 23.3073 | 0.6066 | step_limit |
| r64_n2_ct10/sample_009 | P-Flow | 914 | 877 | 38.2574 | 0.2279 | stalled |
| r64_n2_ct10/sample_010 | P-Flow | 1000 | 895 | 25.1686 | 0.7421 | step_limit |
| r64_n2_ct10/sample_000 | DLR D-Flow | 1000 | 975 | 46.3296 | 27.3275 | step_limit |
| r64_n2_ct10/sample_001 | DLR D-Flow | 414 | 406 | 3.1993 | 5.0177 | stalled |
| r64_n2_ct10/sample_002 | DLR D-Flow | 307 | 294 | 3.3107 | 5.0624 | stalled |
| r64_n2_ct10/sample_003 | DLR D-Flow | 704 | 686 | 64.4659 | 33.6142 | stalled |
| r64_n2_ct10/sample_004 | DLR D-Flow | 373 | 360 | 75.8473 | 51.7826 | stalled |
| r64_n2_ct10/sample_005 | DLR D-Flow | 330 | 320 | 3.4288 | 5.0190 | stalled |
| r64_n2_ct10/sample_006 | DLR D-Flow | 215 | 205 | 74.0988 | 56.4035 | stalled |
| r64_n2_ct10/sample_007 | DLR D-Flow | 909 | 876 | 3.9047 | 5.1876 | stalled |
| r64_n2_ct10/sample_008 | DLR D-Flow | 101 | 94 | 75.4857 | 48.4246 | stalled |
| r64_n2_ct10/sample_009 | DLR D-Flow | 315 | 299 | 3.4714 | 5.0558 | stalled |
| r64_n2_ct10/sample_010 | DLR D-Flow | 431 | 416 | 2.2201 | 4.9866 | stalled |
| r64_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 41.8289 | 0.5696 | step_limit |
| r64_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 34.4679 | 0.4330 | step_limit |
| r64_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 15.9154 | 1.6771 | step_limit |
| r64_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 21.6132 | 0.6141 | step_limit |
| r64_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 37.4816 | 0.5024 | step_limit |
| r64_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 32.8914 | 0.4706 | step_limit |
| r64_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 22.0244 | 1.0776 | step_limit |
| r64_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 26.8859 | 0.6738 | step_limit |
| r64_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 20.7627 | 0.9435 | step_limit |
| r64_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 23.7081 | 0.7975 | step_limit |
| r64_n4_ct10/sample_010 | D-Flow | 752 | 750 | 36.0326 | 0.6979 | stalled |
| r64_n4_ct10/sample_000 | P-Flow | 1000 | 888 | 29.1387 | 0.7125 | step_limit |
| r64_n4_ct10/sample_001 | P-Flow | 1000 | 896 | 24.9470 | 0.5997 | step_limit |
| r64_n4_ct10/sample_002 | P-Flow | 170 | 146 | 5.2569 | 4.5049 | stalled |
| r64_n4_ct10/sample_003 | P-Flow | 16 | 13 | 59.3797 | 30.1217 | stalled |
| r64_n4_ct10/sample_004 | P-Flow | 18 | 15 | 33.4173 | 14.4429 | stalled |
| r64_n4_ct10/sample_005 | P-Flow | 1000 | 883 | 20.5578 | 0.7672 | step_limit |
| r64_n4_ct10/sample_006 | P-Flow | 33 | 28 | 15.3719 | 6.7583 | stalled |
| r64_n4_ct10/sample_007 | P-Flow | 686 | 635 | 30.3330 | 0.6208 | stalled |
| r64_n4_ct10/sample_008 | P-Flow | 1000 | 874 | 24.2696 | 0.7060 | step_limit |
| r64_n4_ct10/sample_009 | P-Flow | 22 | 17 | 56.0275 | 27.2829 | stalled |
| r64_n4_ct10/sample_010 | P-Flow | 451 | 436 | 31.6085 | 0.7022 | stalled |
| r64_n4_ct10/sample_000 | DLR D-Flow | 599 | 574 | 19.9840 | 8.4131 | stalled |
| r64_n4_ct10/sample_001 | DLR D-Flow | 601 | 572 | 18.0369 | 8.3845 | stalled |
| r64_n4_ct10/sample_002 | DLR D-Flow | 1000 | 975 | 64.3963 | 33.3131 | step_limit |
| r64_n4_ct10/sample_003 | DLR D-Flow | 197 | 186 | 14.2433 | 6.6901 | stalled |
| r64_n4_ct10/sample_004 | DLR D-Flow | 769 | 734 | 14.2958 | 7.2948 | stalled |
| r64_n4_ct10/sample_005 | DLR D-Flow | 106 | 98 | 79.0686 | 46.2925 | stalled |
| r64_n4_ct10/sample_006 | DLR D-Flow | 1000 | 966 | 82.1095 | 41.0760 | step_limit |
| r64_n4_ct10/sample_007 | DLR D-Flow | 993 | 957 | 102.8664 | 60.3147 | stalled |
| r64_n4_ct10/sample_008 | DLR D-Flow | 495 | 479 | 79.8341 | 42.7863 | stalled |
| r64_n4_ct10/sample_009 | DLR D-Flow | 850 | 823 | 30.0222 | 13.4175 | stalled |
| r64_n4_ct10/sample_010 | DLR D-Flow | 202 | 190 | 15.7932 | 6.9346 | stalled |
| r64_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 3.8267 | 3.1909 | step_limit |
| r64_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 6.1345 | 2.7066 | step_limit |
| r64_n1_mri4/sample_002 | D-Flow | 645 | 643 | 6.6785 | 2.0830 | stalled |
| r64_n1_mri4/sample_003 | D-Flow | 903 | 901 | 6.7675 | 2.4118 | stalled |
| r64_n1_mri4/sample_004 | D-Flow | 703 | 701 | 99.7547 | 94.4304 | stalled |
| r64_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 3.6225 | 2.8247 | step_limit |
| r64_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 4.4071 | 3.0183 | step_limit |
| r64_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 4.1501 | 2.9565 | step_limit |
| r64_n1_mri4/sample_008 | D-Flow | 644 | 642 | 7.0010 | 2.2650 | stalled |
| r64_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 3.7681 | 3.6170 | step_limit |
| r64_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 4.0954 | 2.7511 | step_limit |
| r64_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 4.7866 | 2.9115 | step_limit |
| r64_n1_mri4/sample_012 | D-Flow | 516 | 514 | 7.0413 | 2.6581 | stalled |
| r64_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 4.9371 | 3.0528 | step_limit |
| r64_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 6.4942 | 3.1721 | step_limit |
| r64_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 4.5079 | 3.3191 | step_limit |
| r64_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 5.1801 | 2.7386 | step_limit |
| r64_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 4.4123 | 3.0417 | step_limit |
| r64_n1_mri4/sample_018 | D-Flow | 266 | 264 | 1.8838 | 4.9081 | stalled |
| r64_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 3.8626 | 3.2642 | step_limit |
| r64_n1_mri4/sample_020 | D-Flow | 582 | 580 | 6.9133 | 2.6888 | stalled |
| r64_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 4.7844 | 2.2911 | step_limit |
| r64_n1_mri4/sample_022 | D-Flow | 661 | 659 | 6.7128 | 2.7864 | stalled |
| r64_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 4.0948 | 3.1761 | step_limit |
| r64_n1_mri4/sample_024 | D-Flow | 593 | 589 | 6.6671 | 2.3695 | stalled |
| r64_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 4.0374 | 2.4331 | step_limit |
| r64_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 4.7121 | 2.7653 | step_limit |
| r64_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 3.5273 | 3.6046 | step_limit |
| r64_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 4.7247 | 3.1136 | step_limit |
| r64_n1_mri4/sample_029 | D-Flow | 593 | 590 | 6.6427 | 2.5723 | stalled |
| r64_n1_mri4/sample_000 | P-Flow | 242 | 228 | 7.2940 | 2.2762 | stalled |
| r64_n1_mri4/sample_001 | P-Flow | 280 | 254 | 6.6169 | 2.7049 | stalled |
| r64_n1_mri4/sample_002 | P-Flow | 229 | 214 | 7.2439 | 2.0830 | stalled |
| r64_n1_mri4/sample_003 | P-Flow | 232 | 217 | 7.0792 | 2.4118 | stalled |
| r64_n1_mri4/sample_004 | P-Flow | 279 | 257 | 7.1058 | 2.8439 | stalled |
| r64_n1_mri4/sample_005 | P-Flow | 306 | 285 | 7.0273 | 1.9722 | stalled |
| r64_n1_mri4/sample_006 | P-Flow | 339 | 310 | 7.5508 | 2.5180 | stalled |
| r64_n1_mri4/sample_007 | P-Flow | 189 | 185 | 7.9203 | 2.2566 | stalled |
| r64_n1_mri4/sample_008 | P-Flow | 235 | 219 | 7.9056 | 2.2650 | stalled |
| r64_n1_mri4/sample_009 | P-Flow | 287 | 258 | 6.6423 | 2.6459 | stalled |
| r64_n1_mri4/sample_010 | P-Flow | 388 | 358 | 6.9814 | 2.3010 | stalled |
| r64_n1_mri4/sample_011 | P-Flow | 279 | 261 | 8.1488 | 1.8455 | stalled |
| r64_n1_mri4/sample_012 | P-Flow | 255 | 238 | 7.4675 | 2.6581 | stalled |
| r64_n1_mri4/sample_013 | P-Flow | 196 | 181 | 7.3572 | 3.0044 | stalled |
| r64_n1_mri4/sample_014 | P-Flow | 45 | 37 | 2.8794 | 5.3893 | stalled |
| r64_n1_mri4/sample_015 | P-Flow | 198 | 182 | 7.0177 | 2.8883 | stalled |
| r64_n1_mri4/sample_016 | P-Flow | 197 | 182 | 6.8249 | 2.7295 | stalled |
| r64_n1_mri4/sample_017 | P-Flow | 37 | 30 | 1.9672 | 5.1103 | stalled |
| r64_n1_mri4/sample_018 | P-Flow | 29 | 24 | 1.6132 | 5.0868 | stalled |
| r64_n1_mri4/sample_019 | P-Flow | 59 | 49 | 1.8989 | 5.1094 | stalled |
| r64_n1_mri4/sample_020 | P-Flow | 285 | 260 | 7.3241 | 2.6946 | stalled |
| r64_n1_mri4/sample_021 | P-Flow | 206 | 195 | 7.9431 | 2.0095 | stalled |
| r64_n1_mri4/sample_022 | P-Flow | 73 | 60 | 2.1120 | 5.1083 | stalled |
| r64_n1_mri4/sample_023 | P-Flow | 186 | 181 | 7.1232 | 2.8786 | stalled |
| r64_n1_mri4/sample_024 | P-Flow | 268 | 245 | 6.9220 | 2.3695 | stalled |
| r64_n1_mri4/sample_025 | P-Flow | 251 | 237 | 7.3840 | 1.8891 | stalled |
| r64_n1_mri4/sample_026 | P-Flow | 204 | 191 | 7.4661 | 2.3253 | stalled |
| r64_n1_mri4/sample_027 | P-Flow | 48 | 39 | 1.7114 | 5.1351 | stalled |
| r64_n1_mri4/sample_028 | P-Flow | 377 | 332 | 7.3186 | 3.0159 | stalled |
| r64_n1_mri4/sample_029 | P-Flow | 201 | 185 | 7.4382 | 2.5723 | stalled |
| r64_n1_mri4/sample_000 | DLR D-Flow | 69 | 63 | 77.0606 | 76.4350 | stalled |
| r64_n1_mri4/sample_001 | DLR D-Flow | 127 | 114 | 2.6050 | 5.2356 | stalled |
| r64_n1_mri4/sample_002 | DLR D-Flow | 297 | 289 | 2.0480 | 5.1263 | stalled |
| r64_n1_mri4/sample_003 | DLR D-Flow | 375 | 353 | 2.0000 | 5.1624 | stalled |
| r64_n1_mri4/sample_004 | DLR D-Flow | 115 | 110 | 100.4092 | 95.7003 | stalled |
| r64_n1_mri4/sample_005 | DLR D-Flow | 1000 | 957 | 2.0944 | 5.2346 | step_limit |
| r64_n1_mri4/sample_006 | DLR D-Flow | 164 | 157 | 1.6842 | 5.0991 | stalled |
| r64_n1_mri4/sample_007 | DLR D-Flow | 332 | 311 | 1.6140 | 5.0430 | stalled |
| r64_n1_mri4/sample_008 | DLR D-Flow | 427 | 404 | 1.6099 | 5.0466 | stalled |
| r64_n1_mri4/sample_009 | DLR D-Flow | 1000 | 962 | 1.8294 | 5.1340 | step_limit |
| r64_n1_mri4/sample_010 | DLR D-Flow | 88 | 85 | 14.4326 | 13.0151 | stalled |
| r64_n1_mri4/sample_011 | DLR D-Flow | 181 | 176 | 1.8985 | 5.1179 | stalled |
| r64_n1_mri4/sample_012 | DLR D-Flow | 161 | 155 | 2.9643 | 5.3543 | stalled |
| r64_n1_mri4/sample_013 | DLR D-Flow | 149 | 143 | 2.1287 | 5.1158 | stalled |
| r64_n1_mri4/sample_014 | DLR D-Flow | 220 | 211 | 3.2051 | 5.4706 | stalled |
| r64_n1_mri4/sample_015 | DLR D-Flow | 847 | 821 | 2.1488 | 5.1239 | stalled |
| r64_n1_mri4/sample_016 | DLR D-Flow | 332 | 315 | 5.0274 | 6.5897 | stalled |
| r64_n1_mri4/sample_017 | DLR D-Flow | 1000 | 958 | 2.0338 | 5.0889 | step_limit |
| r64_n1_mri4/sample_018 | DLR D-Flow | 221 | 214 | 1.6159 | 5.0618 | stalled |
| r64_n1_mri4/sample_019 | DLR D-Flow | 162 | 152 | 1.7782 | 5.0717 | stalled |
| r64_n1_mri4/sample_020 | DLR D-Flow | 216 | 202 | 1.8034 | 5.0917 | stalled |
| r64_n1_mri4/sample_021 | DLR D-Flow | 179 | 169 | 2.1629 | 5.1420 | stalled |
| r64_n1_mri4/sample_022 | DLR D-Flow | 197 | 182 | 2.0742 | 5.1017 | stalled |
| r64_n1_mri4/sample_023 | DLR D-Flow | 109 | 104 | 1.9188 | 5.1039 | stalled |
| r64_n1_mri4/sample_024 | DLR D-Flow | 210 | 195 | 1.7431 | 5.0945 | stalled |
| r64_n1_mri4/sample_025 | DLR D-Flow | 244 | 232 | 1.8288 | 5.0930 | stalled |
| r64_n1_mri4/sample_026 | DLR D-Flow | 1000 | 959 | 3.0028 | 5.5695 | step_limit |
| r64_n1_mri4/sample_027 | DLR D-Flow | 116 | 110 | 1.4693 | 5.0646 | stalled |
| r64_n1_mri4/sample_028 | DLR D-Flow | 450 | 437 | 3.4557 | 5.3561 | stalled |
| r64_n1_mri4/sample_029 | DLR D-Flow | 297 | 286 | 1.8355 | 5.0759 | stalled |
| r64_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 5.8057 | 3.1975 | step_limit |
| r64_n2_mri4/sample_001 | D-Flow | 819 | 817 | 5.9380 | 2.7060 | stalled |
| r64_n2_mri4/sample_002 | D-Flow | 858 | 856 | 6.8232 | 2.5884 | stalled |
| r64_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 2.4979 | 4.3357 | step_limit |
| r64_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 4.0954 | 3.6084 | step_limit |
| r64_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 7.3282 | 1.8298 | step_limit |
| r64_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 7.0872 | 2.4329 | step_limit |
| r64_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 5.2529 | 3.1763 | step_limit |
| r64_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 6.9360 | 1.9824 | step_limit |
| r64_n2_mri4/sample_009 | D-Flow | 732 | 729 | 7.2729 | 2.2905 | stalled |
| r64_n2_mri4/sample_010 | D-Flow | 972 | 969 | 6.2583 | 2.2669 | stalled |
| r64_n2_mri4/sample_000 | P-Flow | 259 | 240 | 7.9626 | 2.6390 | stalled |
| r64_n2_mri4/sample_001 | P-Flow | 366 | 327 | 7.0340 | 2.7060 | stalled |
| r64_n2_mri4/sample_002 | P-Flow | 34 | 30 | 1.8423 | 5.0779 | stalled |
| r64_n2_mri4/sample_003 | P-Flow | 9 | 7 | 108.7707 | 108.6981 | stalled |
| r64_n2_mri4/sample_004 | P-Flow | 393 | 351 | 7.9811 | 2.3312 | stalled |
| r64_n2_mri4/sample_005 | P-Flow | 203 | 192 | 8.1909 | 1.8261 | stalled |
| r64_n2_mri4/sample_006 | P-Flow | 35 | 29 | 2.8681 | 5.2157 | stalled |
| r64_n2_mri4/sample_007 | P-Flow | 245 | 228 | 8.3350 | 2.1957 | stalled |
| r64_n2_mri4/sample_008 | P-Flow | 292 | 262 | 8.0884 | 1.9822 | stalled |
| r64_n2_mri4/sample_009 | P-Flow | 58 | 48 | 2.3359 | 5.1642 | stalled |
| r64_n2_mri4/sample_010 | P-Flow | 4 | 2 | 112.4306 | 108.4202 | stalled |
| r64_n2_mri4/sample_000 | DLR D-Flow | 817 | 789 | 41.7403 | 39.1296 | stalled |
| r64_n2_mri4/sample_001 | DLR D-Flow | 368 | 357 | 6.2899 | 7.5197 | stalled |
| r64_n2_mri4/sample_002 | DLR D-Flow | 511 | 480 | 2.5083 | 5.2393 | stalled |
| r64_n2_mri4/sample_003 | DLR D-Flow | 627 | 601 | 66.7091 | 64.2435 | stalled |
| r64_n2_mri4/sample_004 | DLR D-Flow | 1000 | 968 | 10.9310 | 8.8239 | step_limit |
| r64_n2_mri4/sample_005 | DLR D-Flow | 237 | 230 | 76.9230 | 71.9364 | stalled |
| r64_n2_mri4/sample_006 | DLR D-Flow | 1000 | 963 | 67.5621 | 67.2499 | step_limit |
| r64_n2_mri4/sample_007 | DLR D-Flow | 607 | 581 | 2.3427 | 5.1035 | stalled |
| r64_n2_mri4/sample_008 | DLR D-Flow | 514 | 505 | 53.7729 | 51.8195 | stalled |
| r64_n2_mri4/sample_009 | DLR D-Flow | 350 | 331 | 2.2430 | 5.1198 | stalled |
| r64_n2_mri4/sample_010 | DLR D-Flow | 286 | 277 | 105.0005 | 100.2975 | stalled |
| r64_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 3.0753 | 3.5655 | step_limit |
| r64_n4_mri4/sample_001 | D-Flow | 798 | 796 | 6.5779 | 2.6593 | stalled |
| r64_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 6.2187 | 2.6232 | step_limit |
| r64_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 3.3299 | 3.3799 | step_limit |
| r64_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 7.0325 | 2.6054 | step_limit |
| r64_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 4.0859 | 3.1445 | step_limit |
| r64_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 7.0704 | 2.9739 | step_limit |
| r64_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 2.7352 | 4.6268 | step_limit |
| r64_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 6.9778 | 2.9136 | step_limit |
| r64_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 6.2478 | 2.8944 | step_limit |
| r64_n4_mri4/sample_010 | D-Flow | 571 | 569 | 7.4778 | 2.0725 | stalled |
| r64_n4_mri4/sample_000 | P-Flow | 243 | 227 | 7.5390 | 2.8403 | stalled |
| r64_n4_mri4/sample_001 | P-Flow | 265 | 240 | 7.4666 | 2.6593 | stalled |
| r64_n4_mri4/sample_002 | P-Flow | 200 | 192 | 7.0318 | 2.6225 | stalled |
| r64_n4_mri4/sample_003 | P-Flow | 13 | 10 | 63.4548 | 59.0057 | stalled |
| r64_n4_mri4/sample_004 | P-Flow | 255 | 236 | 7.8748 | 2.6048 | stalled |
| r64_n4_mri4/sample_005 | P-Flow | 311 | 281 | 7.5682 | 2.9887 | stalled |
| r64_n4_mri4/sample_006 | P-Flow | 282 | 262 | 7.6879 | 2.8831 | stalled |
| r64_n4_mri4/sample_007 | P-Flow | 259 | 241 | 8.4209 | 1.9786 | stalled |
| r64_n4_mri4/sample_008 | P-Flow | 315 | 290 | 7.2551 | 2.9132 | stalled |
| r64_n4_mri4/sample_009 | P-Flow | 229 | 214 | 6.9301 | 2.8935 | stalled |
| r64_n4_mri4/sample_010 | P-Flow | 205 | 197 | 7.6173 | 2.0725 | stalled |
| r64_n4_mri4/sample_000 | DLR D-Flow | 303 | 295 | 67.6890 | 66.0417 | stalled |
| r64_n4_mri4/sample_001 | DLR D-Flow | 172 | 165 | 28.2970 | 22.9377 | stalled |
| r64_n4_mri4/sample_002 | DLR D-Flow | 1000 | 965 | 6.5782 | 7.0474 | step_limit |
| r64_n4_mri4/sample_003 | DLR D-Flow | 194 | 185 | 30.6189 | 27.6306 | stalled |
| r64_n4_mri4/sample_004 | DLR D-Flow | 1000 | 968 | 48.0154 | 44.9706 | step_limit |
| r64_n4_mri4/sample_005 | DLR D-Flow | 1000 | 965 | 59.0170 | 55.6169 | step_limit |
| r64_n4_mri4/sample_006 | DLR D-Flow | 1000 | 958 | 84.3863 | 78.2968 | step_limit |
| r64_n4_mri4/sample_007 | DLR D-Flow | 1000 | 972 | 36.0454 | 29.9512 | step_limit |
| r64_n4_mri4/sample_008 | DLR D-Flow | 505 | 491 | 26.0542 | 20.4618 | stalled |
| r64_n4_mri4/sample_009 | DLR D-Flow | 1000 | 969 | 62.2451 | 54.9580 | step_limit |
| r64_n4_mri4/sample_010 | DLR D-Flow | 101 | 98 | 18.3251 | 14.5023 | stalled |
| r96_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 16.4010 | 0.5821 | step_limit |
| r96_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 19.7389 | 0.6626 | step_limit |
| r96_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 6.2027 | 3.1401 | step_limit |
| r96_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 32.1760 | 0.3189 | step_limit |
| r96_n1_ct10/sample_004 | D-Flow | 95 | 93 | 105.4968 | 108.6108 | stalled |
| r96_n1_ct10/sample_005 | D-Flow | 1000 | 998 | 29.0811 | 0.2767 | stalled |
| r96_n1_ct10/sample_006 | D-Flow | 778 | 776 | 33.5787 | 0.2099 | stalled |
| r96_n1_ct10/sample_007 | D-Flow | 451 | 449 | 44.6011 | 0.2146 | stalled |
| r96_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 23.7905 | 0.4211 | step_limit |
| r96_n1_ct10/sample_009 | D-Flow | 386 | 384 | 112.4158 | 98.0302 | stalled |
| r96_n1_ct10/sample_010 | D-Flow | 532 | 530 | 38.4674 | 0.3860 | stalled |
| r96_n1_ct10/sample_011 | D-Flow | 582 | 580 | 41.8336 | 0.2727 | stalled |
| r96_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 30.8099 | 0.3673 | step_limit |
| r96_n1_ct10/sample_013 | D-Flow | 192 | 189 | 116.0198 | 108.3664 | stalled |
| r96_n1_ct10/sample_014 | D-Flow | 198 | 195 | 115.7646 | 105.9107 | stalled |
| r96_n1_ct10/sample_015 | D-Flow | 1000 | 1000 | 6.8475 | 2.5025 | step_limit |
| r96_n1_ct10/sample_016 | D-Flow | 1000 | 1000 | 24.3563 | 0.3588 | step_limit |
| r96_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 31.9142 | 0.3120 | step_limit |
| r96_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 7.7950 | 2.5877 | step_limit |
| r96_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 6.3598 | 3.1564 | step_limit |
| r96_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 26.4581 | 0.2734 | step_limit |
| r96_n1_ct10/sample_021 | D-Flow | 760 | 758 | 26.9661 | 0.3096 | stalled |
| r96_n1_ct10/sample_022 | D-Flow | 853 | 851 | 28.2947 | 0.4635 | stalled |
| r96_n1_ct10/sample_023 | D-Flow | 944 | 942 | 32.7163 | 0.3701 | stalled |
| r96_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 26.4890 | 0.4997 | step_limit |
| r96_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 7.2786 | 2.5035 | step_limit |
| r96_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 8.4836 | 1.4657 | step_limit |
| r96_n1_ct10/sample_027 | D-Flow | 780 | 777 | 51.4721 | 0.1651 | stalled |
| r96_n1_ct10/sample_028 | D-Flow | 1000 | 1000 | 6.8062 | 3.5925 | step_limit |
| r96_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 28.3046 | 0.2317 | step_limit |
| r96_n1_ct10/sample_000 | P-Flow | 50 | 42 | 2.3840 | 4.9850 | stalled |
| r96_n1_ct10/sample_001 | P-Flow | 1000 | 901 | 18.0795 | 0.6418 | step_limit |
| r96_n1_ct10/sample_002 | P-Flow | 26 | 21 | 4.0368 | 5.2242 | stalled |
| r96_n1_ct10/sample_003 | P-Flow | 62 | 52 | 1.7485 | 4.9883 | stalled |
| r96_n1_ct10/sample_004 | P-Flow | 55 | 44 | 3.6918 | 5.0661 | stalled |
| r96_n1_ct10/sample_005 | P-Flow | 1000 | 807 | 2.3419 | 5.0861 | step_limit |
| r96_n1_ct10/sample_006 | P-Flow | 1000 | 953 | 19.0104 | 0.5752 | step_limit |
| r96_n1_ct10/sample_007 | P-Flow | 43 | 36 | 3.7478 | 5.1183 | stalled |
| r96_n1_ct10/sample_008 | P-Flow | 1000 | 970 | 18.5149 | 0.6049 | step_limit |
| r96_n1_ct10/sample_009 | P-Flow | 632 | 499 | 2.4683 | 4.9806 | stalled |
| r96_n1_ct10/sample_010 | P-Flow | 1000 | 906 | 25.7459 | 0.4409 | step_limit |
| r96_n1_ct10/sample_011 | P-Flow | 198 | 159 | 2.7548 | 5.0240 | stalled |
| r96_n1_ct10/sample_012 | P-Flow | 21 | 17 | 4.8762 | 5.1299 | stalled |
| r96_n1_ct10/sample_013 | P-Flow | 1000 | 961 | 18.7074 | 0.6634 | step_limit |
| r96_n1_ct10/sample_014 | P-Flow | 151 | 120 | 3.3222 | 5.0788 | stalled |
| r96_n1_ct10/sample_015 | P-Flow | 1000 | 809 | 3.5910 | 5.5346 | step_limit |
| r96_n1_ct10/sample_016 | P-Flow | 30 | 25 | 3.1954 | 5.2436 | stalled |
| r96_n1_ct10/sample_017 | P-Flow | 104 | 89 | 2.0766 | 4.9708 | stalled |
| r96_n1_ct10/sample_018 | P-Flow | 34 | 29 | 3.9169 | 5.0967 | stalled |
| r96_n1_ct10/sample_019 | P-Flow | 130 | 108 | 2.1534 | 5.0092 | stalled |
| r96_n1_ct10/sample_020 | P-Flow | 550 | 514 | 24.1691 | 0.2456 | stalled |
| r96_n1_ct10/sample_021 | P-Flow | 1000 | 896 | 18.5841 | 0.6431 | step_limit |
| r96_n1_ct10/sample_022 | P-Flow | 24 | 19 | 4.2585 | 5.1682 | stalled |
| r96_n1_ct10/sample_023 | P-Flow | 29 | 25 | 3.8120 | 5.2849 | stalled |
| r96_n1_ct10/sample_024 | P-Flow | 53 | 46 | 12.4817 | 7.4836 | stalled |
| r96_n1_ct10/sample_025 | P-Flow | 399 | 324 | 2.8769 | 5.0072 | stalled |
| r96_n1_ct10/sample_026 | P-Flow | 237 | 197 | 3.3448 | 5.0590 | stalled |
| r96_n1_ct10/sample_027 | P-Flow | 22 | 18 | 2.8262 | 5.0534 | stalled |
| r96_n1_ct10/sample_028 | P-Flow | 930 | 758 | 2.1787 | 5.0176 | stalled |
| r96_n1_ct10/sample_029 | P-Flow | 51 | 41 | 2.4215 | 5.0075 | stalled |
| r96_n1_ct10/sample_000 | DLR D-Flow | 71 | 66 | 2.0437 | 4.9975 | stalled |
| r96_n1_ct10/sample_001 | DLR D-Flow | 23 | 19 | 112.8620 | 101.0725 | stalled |
| r96_n1_ct10/sample_002 | DLR D-Flow | 845 | 815 | 2.0862 | 4.9641 | stalled |
| r96_n1_ct10/sample_003 | DLR D-Flow | 170 | 162 | 2.2342 | 5.0549 | stalled |
| r96_n1_ct10/sample_004 | DLR D-Flow | 89 | 84 | 4.2681 | 5.2601 | stalled |
| r96_n1_ct10/sample_005 | DLR D-Flow | 343 | 322 | 1.7990 | 4.9860 | stalled |
| r96_n1_ct10/sample_006 | DLR D-Flow | 174 | 167 | 2.1775 | 4.9710 | stalled |
| r96_n1_ct10/sample_007 | DLR D-Flow | 219 | 210 | 1.9742 | 4.9568 | stalled |
| r96_n1_ct10/sample_008 | DLR D-Flow | 68 | 65 | 114.6102 | 102.6279 | stalled |
| r96_n1_ct10/sample_009 | DLR D-Flow | 242 | 229 | 119.6748 | 104.9035 | stalled |
| r96_n1_ct10/sample_010 | DLR D-Flow | 422 | 405 | 4.4632 | 5.0857 | stalled |
| r96_n1_ct10/sample_011 | DLR D-Flow | 163 | 155 | 2.0788 | 4.9981 | stalled |
| r96_n1_ct10/sample_012 | DLR D-Flow | 206 | 198 | 2.3231 | 4.9661 | stalled |
| r96_n1_ct10/sample_013 | DLR D-Flow | 56 | 53 | 124.0300 | 112.3812 | stalled |
| r96_n1_ct10/sample_014 | DLR D-Flow | 116 | 108 | 123.2668 | 108.1000 | stalled |
| r96_n1_ct10/sample_015 | DLR D-Flow | 391 | 370 | 3.6541 | 5.0361 | stalled |
| r96_n1_ct10/sample_016 | DLR D-Flow | 152 | 146 | 3.5131 | 5.1221 | stalled |
| r96_n1_ct10/sample_017 | DLR D-Flow | 592 | 563 | 3.4216 | 4.9864 | stalled |
| r96_n1_ct10/sample_018 | DLR D-Flow | 423 | 405 | 2.5699 | 4.9644 | stalled |
| r96_n1_ct10/sample_019 | DLR D-Flow | 145 | 138 | 94.3708 | 66.4434 | stalled |
| r96_n1_ct10/sample_020 | DLR D-Flow | 134 | 128 | 112.6348 | 91.3801 | stalled |
| r96_n1_ct10/sample_021 | DLR D-Flow | 776 | 738 | 2.9154 | 5.0100 | stalled |
| r96_n1_ct10/sample_022 | DLR D-Flow | 152 | 146 | 2.6075 | 4.9854 | stalled |
| r96_n1_ct10/sample_023 | DLR D-Flow | 51 | 49 | 4.4537 | 5.3596 | stalled |
| r96_n1_ct10/sample_024 | DLR D-Flow | 120 | 114 | 9.1381 | 6.8174 | stalled |
| r96_n1_ct10/sample_025 | DLR D-Flow | 282 | 269 | 4.0804 | 5.1634 | stalled |
| r96_n1_ct10/sample_026 | DLR D-Flow | 168 | 161 | 122.9570 | 101.2191 | stalled |
| r96_n1_ct10/sample_027 | DLR D-Flow | 166 | 159 | 3.0291 | 4.9364 | stalled |
| r96_n1_ct10/sample_028 | DLR D-Flow | 64 | 60 | 119.6535 | 101.7117 | stalled |
| r96_n1_ct10/sample_029 | DLR D-Flow | 279 | 269 | 2.0190 | 4.9797 | stalled |
| r96_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 2.9404 | 4.8908 | step_limit |
| r96_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 19.9974 | 1.9362 | step_limit |
| r96_n2_ct10/sample_002 | D-Flow | 567 | 565 | 34.0581 | 0.2949 | stalled |
| r96_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 33.4910 | 0.2424 | step_limit |
| r96_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 41.1023 | 0.2850 | step_limit |
| r96_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 30.3093 | 0.3387 | step_limit |
| r96_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 20.6144 | 0.6854 | step_limit |
| r96_n2_ct10/sample_007 | D-Flow | 190 | 188 | 62.3624 | 35.3875 | stalled |
| r96_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 5.8919 | 2.7451 | step_limit |
| r96_n2_ct10/sample_009 | D-Flow | 50 | 46 | 83.1152 | 63.8585 | stalled |
| r96_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 30.6548 | 0.4407 | step_limit |
| r96_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 38.9010 | 0.1618 | step_limit |
| r96_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 22.1550 | 0.8660 | step_limit |
| r96_n2_ct10/sample_000 | P-Flow | 1000 | 878 | 20.2123 | 0.6426 | step_limit |
| r96_n2_ct10/sample_001 | P-Flow | 224 | 194 | 3.0315 | 5.1125 | stalled |
| r96_n2_ct10/sample_002 | P-Flow | 378 | 308 | 3.8179 | 5.0035 | stalled |
| r96_n2_ct10/sample_003 | P-Flow | 31 | 25 | 72.7215 | 42.5751 | stalled |
| r96_n2_ct10/sample_004 | P-Flow | 86 | 74 | 2.4630 | 5.0427 | stalled |
| r96_n2_ct10/sample_005 | P-Flow | 51 | 44 | 2.7616 | 4.9977 | stalled |
| r96_n2_ct10/sample_006 | P-Flow | 570 | 455 | 4.3327 | 5.3622 | stalled |
| r96_n2_ct10/sample_007 | P-Flow | 44 | 38 | 4.2713 | 5.1951 | stalled |
| r96_n2_ct10/sample_008 | P-Flow | 624 | 508 | 3.2880 | 5.1047 | stalled |
| r96_n2_ct10/sample_009 | P-Flow | 919 | 739 | 2.4264 | 5.0190 | stalled |
| r96_n2_ct10/sample_010 | P-Flow | 33 | 28 | 6.3982 | 5.4301 | stalled |
| r96_n2_ct10/sample_011 | P-Flow | 1000 | 908 | 34.2426 | 0.2005 | step_limit |
| r96_n2_ct10/sample_012 | P-Flow | 751 | 707 | 34.6202 | 0.3985 | stalled |
| r96_n2_ct10/sample_000 | DLR D-Flow | 703 | 681 | 59.5456 | 44.5677 | stalled |
| r96_n2_ct10/sample_001 | DLR D-Flow | 234 | 228 | 6.7715 | 5.3185 | stalled |
| r96_n2_ct10/sample_002 | DLR D-Flow | 449 | 434 | 2.9055 | 4.9989 | stalled |
| r96_n2_ct10/sample_003 | DLR D-Flow | 667 | 652 | 28.6531 | 11.5534 | stalled |
| r96_n2_ct10/sample_004 | DLR D-Flow | 757 | 728 | 2.4180 | 4.9793 | stalled |
| r96_n2_ct10/sample_005 | DLR D-Flow | 139 | 136 | 5.5134 | 5.2087 | stalled |
| r96_n2_ct10/sample_006 | DLR D-Flow | 89 | 84 | 103.0925 | 72.3582 | stalled |
| r96_n2_ct10/sample_007 | DLR D-Flow | 857 | 827 | 61.8337 | 36.2956 | stalled |
| r96_n2_ct10/sample_008 | DLR D-Flow | 1000 | 965 | 3.3813 | 5.2439 | step_limit |
| r96_n2_ct10/sample_009 | DLR D-Flow | 86 | 83 | 78.2632 | 64.3508 | stalled |
| r96_n2_ct10/sample_010 | DLR D-Flow | 547 | 516 | 2.8340 | 5.0168 | stalled |
| r96_n2_ct10/sample_011 | DLR D-Flow | 960 | 915 | 7.1864 | 5.3257 | stalled |
| r96_n2_ct10/sample_012 | DLR D-Flow | 1000 | 964 | 80.9024 | 52.2875 | step_limit |
| r96_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 34.2022 | 0.4158 | step_limit |
| r96_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 32.5099 | 0.6510 | step_limit |
| r96_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 11.1097 | 3.8486 | step_limit |
| r96_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 5.7047 | 4.7837 | step_limit |
| r96_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 22.1866 | 0.8951 | step_limit |
| r96_n4_ct10/sample_005 | D-Flow | 885 | 883 | 34.8639 | 0.3353 | stalled |
| r96_n4_ct10/sample_006 | D-Flow | 791 | 789 | 31.9081 | 0.4133 | stalled |
| r96_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 47.3002 | 0.3270 | step_limit |
| r96_n4_ct10/sample_008 | D-Flow | 810 | 808 | 34.1298 | 0.2133 | stalled |
| r96_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 33.0662 | 0.2569 | step_limit |
| r96_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 23.2801 | 0.4881 | step_limit |
| r96_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 25.1691 | 0.7332 | step_limit |
| r96_n4_ct10/sample_012 | D-Flow | 896 | 894 | 38.7862 | 0.6361 | stalled |
| r96_n4_ct10/sample_000 | P-Flow | 66 | 57 | 3.5271 | 4.9703 | stalled |
| r96_n4_ct10/sample_001 | P-Flow | 1000 | 898 | 24.3257 | 0.5407 | step_limit |
| r96_n4_ct10/sample_002 | P-Flow | 180 | 154 | 2.7650 | 4.8930 | stalled |
| r96_n4_ct10/sample_003 | P-Flow | 25 | 20 | 51.9829 | 23.6189 | stalled |
| r96_n4_ct10/sample_004 | P-Flow | 58 | 50 | 3.5038 | 4.9996 | stalled |
| r96_n4_ct10/sample_005 | P-Flow | 1000 | 909 | 20.0286 | 0.5911 | step_limit |
| r96_n4_ct10/sample_006 | P-Flow | 71 | 60 | 3.9722 | 5.0243 | stalled |
| r96_n4_ct10/sample_007 | P-Flow | 1000 | 834 | 7.0529 | 4.0552 | step_limit |
| r96_n4_ct10/sample_008 | P-Flow | 1000 | 762 | 3.7482 | 4.9151 | step_limit |
| r96_n4_ct10/sample_009 | P-Flow | 1000 | 882 | 21.5960 | 0.4619 | step_limit |
| r96_n4_ct10/sample_010 | P-Flow | 1000 | 780 | 3.1385 | 4.8729 | step_limit |
| r96_n4_ct10/sample_011 | P-Flow | 66 | 57 | 3.4059 | 4.9418 | stalled |
| r96_n4_ct10/sample_012 | P-Flow | 679 | 653 | 32.3402 | 0.6362 | stalled |
| r96_n4_ct10/sample_000 | DLR D-Flow | 431 | 409 | 17.9160 | 7.1376 | stalled |
| r96_n4_ct10/sample_001 | DLR D-Flow | 656 | 632 | 46.0170 | 21.1108 | stalled |
| r96_n4_ct10/sample_002 | DLR D-Flow | 1000 | 960 | 9.3272 | 5.6935 | step_limit |
| r96_n4_ct10/sample_003 | DLR D-Flow | 251 | 244 | 61.0706 | 33.0942 | stalled |
| r96_n4_ct10/sample_004 | DLR D-Flow | 190 | 182 | 90.6830 | 50.6729 | stalled |
| r96_n4_ct10/sample_005 | DLR D-Flow | 1000 | 965 | 10.7511 | 5.8535 | step_limit |
| r96_n4_ct10/sample_006 | DLR D-Flow | 1000 | 960 | 67.3865 | 32.0868 | step_limit |
| r96_n4_ct10/sample_007 | DLR D-Flow | 760 | 741 | 8.3613 | 5.4091 | stalled |
| r96_n4_ct10/sample_008 | DLR D-Flow | 210 | 204 | 27.5174 | 10.2780 | stalled |
| r96_n4_ct10/sample_009 | DLR D-Flow | 1000 | 970 | 51.1339 | 22.3770 | step_limit |
| r96_n4_ct10/sample_010 | DLR D-Flow | 806 | 776 | 82.7083 | 43.2853 | stalled |
| r96_n4_ct10/sample_011 | DLR D-Flow | 1000 | 961 | 72.7689 | 35.6530 | step_limit |
| r96_n4_ct10/sample_012 | DLR D-Flow | 1000 | 971 | 79.9762 | 39.1922 | step_limit |
| r96_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 2.5995 | 4.2091 | step_limit |
| r96_n1_mri4/sample_001 | D-Flow | 113 | 111 | 101.6449 | 101.2932 | stalled |
| r96_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 7.1734 | 2.5880 | step_limit |
| r96_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 4.8167 | 3.5580 | step_limit |
| r96_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 6.2946 | 2.6272 | step_limit |
| r96_n1_mri4/sample_005 | D-Flow | 724 | 722 | 6.5330 | 2.5840 | stalled |
| r96_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 4.9460 | 3.0420 | step_limit |
| r96_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 3.7920 | 3.1724 | step_limit |
| r96_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 4.6912 | 3.3199 | step_limit |
| r96_n1_mri4/sample_009 | D-Flow | 49 | 44 | 103.5418 | 102.2047 | stalled |
| r96_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 2.8692 | 4.1927 | step_limit |
| r96_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 3.1663 | 3.5377 | step_limit |
| r96_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 7.8006 | 2.1709 | step_limit |
| r96_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 102.0486 | 101.7537 | step_limit |
| r96_n1_mri4/sample_014 | D-Flow | 230 | 228 | 103.6283 | 102.1359 | stalled |
| r96_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 2.5866 | 4.0788 | step_limit |
| r96_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 5.6165 | 3.6309 | step_limit |
| r96_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 3.4716 | 3.5791 | step_limit |
| r96_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 6.1409 | 3.0934 | step_limit |
| r96_n1_mri4/sample_019 | D-Flow | 25 | 23 | 102.4489 | 102.7789 | stalled |
| r96_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 5.3305 | 2.8567 | step_limit |
| r96_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 4.6860 | 3.5551 | step_limit |
| r96_n1_mri4/sample_022 | D-Flow | 578 | 576 | 6.6657 | 2.9917 | stalled |
| r96_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 4.1927 | 3.3762 | step_limit |
| r96_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 4.6935 | 3.5255 | step_limit |
| r96_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 4.5724 | 3.8203 | step_limit |
| r96_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 4.3538 | 6.1513 | step_limit |
| r96_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 2.7324 | 4.1656 | step_limit |
| r96_n1_mri4/sample_028 | D-Flow | 118 | 113 | 116.8932 | 114.4288 | stalled |
| r96_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 3.8290 | 3.5347 | step_limit |
| r96_n1_mri4/sample_000 | P-Flow | 41 | 33 | 1.7124 | 5.1116 | stalled |
| r96_n1_mri4/sample_001 | P-Flow | 10 | 7 | 117.0998 | 115.7854 | stalled |
| r96_n1_mri4/sample_002 | P-Flow | 57 | 49 | 1.7522 | 5.0777 | stalled |
| r96_n1_mri4/sample_003 | P-Flow | 271 | 248 | 7.6584 | 2.7682 | stalled |
| r96_n1_mri4/sample_004 | P-Flow | 45 | 38 | 3.3235 | 5.4025 | stalled |
| r96_n1_mri4/sample_005 | P-Flow | 360 | 334 | 7.1051 | 2.5840 | stalled |
| r96_n1_mri4/sample_006 | P-Flow | 226 | 210 | 7.4923 | 2.3681 | stalled |
| r96_n1_mri4/sample_007 | P-Flow | 52 | 44 | 1.5707 | 5.0992 | stalled |
| r96_n1_mri4/sample_008 | P-Flow | 179 | 172 | 7.4031 | 2.8341 | stalled |
| r96_n1_mri4/sample_009 | P-Flow | 15 | 12 | 120.4648 | 111.7476 | stalled |
| r96_n1_mri4/sample_010 | P-Flow | 173 | 166 | 7.5256 | 2.5816 | stalled |
| r96_n1_mri4/sample_011 | P-Flow | 36 | 28 | 1.4598 | 5.0476 | stalled |
| r96_n1_mri4/sample_012 | P-Flow | 74 | 62 | 1.6995 | 5.0630 | stalled |
| r96_n1_mri4/sample_013 | P-Flow | 238 | 217 | 8.3655 | 2.1647 | stalled |
| r96_n1_mri4/sample_014 | P-Flow | 12 | 9 | 134.7223 | 127.2168 | stalled |
| r96_n1_mri4/sample_015 | P-Flow | 13 | 11 | 112.8870 | 112.7172 | stalled |
| r96_n1_mri4/sample_016 | P-Flow | 271 | 249 | 7.9355 | 2.3491 | stalled |
| r96_n1_mri4/sample_017 | P-Flow | 204 | 194 | 6.8770 | 2.4215 | stalled |
| r96_n1_mri4/sample_018 | P-Flow | 197 | 184 | 7.3934 | 2.2062 | stalled |
| r96_n1_mri4/sample_019 | P-Flow | 26 | 22 | 1.7702 | 5.1228 | stalled |
| r96_n1_mri4/sample_020 | P-Flow | 176 | 167 | 7.9511 | 2.5438 | stalled |
| r96_n1_mri4/sample_021 | P-Flow | 39 | 32 | 2.0441 | 5.1336 | stalled |
| r96_n1_mri4/sample_022 | P-Flow | 22 | 17 | 2.1375 | 5.1921 | stalled |
| r96_n1_mri4/sample_023 | P-Flow | 264 | 240 | 7.0774 | 2.1592 | stalled |
| r96_n1_mri4/sample_024 | P-Flow | 34 | 28 | 3.6917 | 5.6647 | stalled |
| r96_n1_mri4/sample_025 | P-Flow | 31 | 26 | 2.0431 | 5.1370 | stalled |
| r96_n1_mri4/sample_026 | P-Flow | 92 | 76 | 2.3282 | 5.1324 | stalled |
| r96_n1_mri4/sample_027 | P-Flow | 52 | 43 | 1.3339 | 5.0601 | stalled |
| r96_n1_mri4/sample_028 | P-Flow | 62 | 51 | 1.9764 | 5.1313 | stalled |
| r96_n1_mri4/sample_029 | P-Flow | 31 | 25 | 1.7049 | 5.1317 | stalled |
| r96_n1_mri4/sample_000 | DLR D-Flow | 741 | 692 | 1.7114 | 5.0934 | stalled |
| r96_n1_mri4/sample_001 | DLR D-Flow | 68 | 66 | 102.6121 | 102.3694 | stalled |
| r96_n1_mri4/sample_002 | DLR D-Flow | 129 | 123 | 2.9231 | 5.4224 | stalled |
| r96_n1_mri4/sample_003 | DLR D-Flow | 90 | 85 | 1.7955 | 5.1057 | stalled |
| r96_n1_mri4/sample_004 | DLR D-Flow | 228 | 213 | 3.9054 | 5.7162 | stalled |
| r96_n1_mri4/sample_005 | DLR D-Flow | 562 | 535 | 1.5122 | 5.0592 | stalled |
| r96_n1_mri4/sample_006 | DLR D-Flow | 205 | 195 | 1.8345 | 5.1227 | stalled |
| r96_n1_mri4/sample_007 | DLR D-Flow | 603 | 581 | 1.4474 | 5.0689 | stalled |
| r96_n1_mri4/sample_008 | DLR D-Flow | 135 | 132 | 102.9024 | 102.8750 | stalled |
| r96_n1_mri4/sample_009 | DLR D-Flow | 188 | 181 | 103.6300 | 102.0442 | stalled |
| r96_n1_mri4/sample_010 | DLR D-Flow | 249 | 245 | 8.2711 | 8.2135 | stalled |
| r96_n1_mri4/sample_011 | DLR D-Flow | 169 | 160 | 1.4811 | 5.0473 | stalled |
| r96_n1_mri4/sample_012 | DLR D-Flow | 423 | 401 | 1.4358 | 5.0389 | stalled |
| r96_n1_mri4/sample_013 | DLR D-Flow | 21 | 19 | 125.3744 | 124.3975 | stalled |
| r96_n1_mri4/sample_014 | DLR D-Flow | 52 | 47 | 103.2028 | 102.3392 | stalled |
| r96_n1_mri4/sample_015 | DLR D-Flow | 349 | 327 | 3.0463 | 5.5888 | stalled |
| r96_n1_mri4/sample_016 | DLR D-Flow | 167 | 162 | 1.9109 | 5.0861 | stalled |
| r96_n1_mri4/sample_017 | DLR D-Flow | 124 | 121 | 1.9955 | 5.1772 | stalled |
| r96_n1_mri4/sample_018 | DLR D-Flow | 367 | 357 | 1.6758 | 5.0673 | stalled |
| r96_n1_mri4/sample_019 | DLR D-Flow | 53 | 49 | 102.4274 | 102.7468 | stalled |
| r96_n1_mri4/sample_020 | DLR D-Flow | 15 | 13 | 115.7415 | 115.4450 | stalled |
| r96_n1_mri4/sample_021 | DLR D-Flow | 439 | 422 | 1.9719 | 5.1090 | stalled |
| r96_n1_mri4/sample_022 | DLR D-Flow | 283 | 272 | 1.6115 | 5.0760 | stalled |
| r96_n1_mri4/sample_023 | DLR D-Flow | 881 | 844 | 1.4683 | 5.0841 | stalled |
| r96_n1_mri4/sample_024 | DLR D-Flow | 308 | 292 | 5.0016 | 6.3527 | stalled |
| r96_n1_mri4/sample_025 | DLR D-Flow | 1000 | 949 | 2.1616 | 5.1459 | step_limit |
| r96_n1_mri4/sample_026 | DLR D-Flow | 470 | 452 | 103.6284 | 103.0995 | stalled |
| r96_n1_mri4/sample_027 | DLR D-Flow | 172 | 163 | 1.2522 | 5.0309 | stalled |
| r96_n1_mri4/sample_028 | DLR D-Flow | 118 | 111 | 121.3809 | 118.4643 | stalled |
| r96_n1_mri4/sample_029 | DLR D-Flow | 156 | 150 | 1.6021 | 5.0861 | stalled |
| r96_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 3.0989 | 4.0853 | step_limit |
| r96_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 2.6031 | 4.4380 | step_limit |
| r96_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 6.8681 | 2.5194 | step_limit |
| r96_n2_mri4/sample_003 | D-Flow | 855 | 852 | 6.8440 | 2.5539 | stalled |
| r96_n2_mri4/sample_004 | D-Flow | 899 | 897 | 6.2371 | 2.6086 | stalled |
| r96_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 3.6489 | 3.6043 | step_limit |
| r96_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 5.5089 | 3.6440 | step_limit |
| r96_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 7.3351 | 2.1666 | step_limit |
| r96_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 2.6261 | 4.1663 | step_limit |
| r96_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.9561 | 4.7866 | step_limit |
| r96_n2_mri4/sample_010 | D-Flow | 957 | 955 | 5.9716 | 2.8046 | stalled |
| r96_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 2.6315 | 4.6284 | step_limit |
| r96_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.6103 | 4.9682 | step_limit |
| r96_n2_mri4/sample_000 | P-Flow | 217 | 208 | 6.5901 | 2.5990 | stalled |
| r96_n2_mri4/sample_001 | P-Flow | 215 | 206 | 7.6247 | 2.6122 | stalled |
| r96_n2_mri4/sample_002 | P-Flow | 210 | 203 | 7.9359 | 1.9363 | stalled |
| r96_n2_mri4/sample_003 | P-Flow | 192 | 185 | 7.8907 | 2.5539 | stalled |
| r96_n2_mri4/sample_004 | P-Flow | 208 | 188 | 7.5638 | 2.6086 | stalled |
| r96_n2_mri4/sample_005 | P-Flow | 198 | 189 | 7.0677 | 2.3549 | stalled |
| r96_n2_mri4/sample_006 | P-Flow | 45 | 37 | 6.8527 | 6.9425 | stalled |
| r96_n2_mri4/sample_007 | P-Flow | 238 | 215 | 7.9610 | 2.1647 | stalled |
| r96_n2_mri4/sample_008 | P-Flow | 69 | 59 | 2.7092 | 5.2830 | stalled |
| r96_n2_mri4/sample_009 | P-Flow | 304 | 278 | 7.2889 | 2.9594 | stalled |
| r96_n2_mri4/sample_010 | P-Flow | 222 | 208 | 7.1721 | 2.8046 | stalled |
| r96_n2_mri4/sample_011 | P-Flow | 171 | 163 | 6.8957 | 2.4013 | stalled |
| r96_n2_mri4/sample_012 | P-Flow | 196 | 190 | 7.4336 | 1.8810 | stalled |
| r96_n2_mri4/sample_000 | DLR D-Flow | 738 | 708 | 2.2326 | 5.2830 | stalled |
| r96_n2_mri4/sample_001 | DLR D-Flow | 472 | 458 | 4.7125 | 5.8118 | stalled |
| r96_n2_mri4/sample_002 | DLR D-Flow | 323 | 311 | 2.2836 | 5.1825 | stalled |
| r96_n2_mri4/sample_003 | DLR D-Flow | 159 | 153 | 2.4895 | 5.2300 | stalled |
| r96_n2_mri4/sample_004 | DLR D-Flow | 281 | 271 | 4.7395 | 5.8432 | stalled |
| r96_n2_mri4/sample_005 | DLR D-Flow | 167 | 161 | 1.6525 | 5.0723 | stalled |
| r96_n2_mri4/sample_006 | DLR D-Flow | 1000 | 971 | 74.3363 | 72.1076 | step_limit |
| r96_n2_mri4/sample_007 | DLR D-Flow | 606 | 585 | 62.1587 | 56.7555 | stalled |
| r96_n2_mri4/sample_008 | DLR D-Flow | 81 | 78 | 66.1056 | 65.8360 | stalled |
| r96_n2_mri4/sample_009 | DLR D-Flow | 92 | 86 | 76.3171 | 74.4294 | stalled |
| r96_n2_mri4/sample_010 | DLR D-Flow | 552 | 525 | 4.5475 | 6.0850 | stalled |
| r96_n2_mri4/sample_011 | DLR D-Flow | 865 | 832 | 1.6137 | 5.1497 | stalled |
| r96_n2_mri4/sample_012 | DLR D-Flow | 263 | 252 | 102.3674 | 100.8776 | stalled |
| r96_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 7.0686 | 1.8786 | step_limit |
| r96_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 2.0384 | 4.8136 | step_limit |
| r96_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 2.6275 | 4.1564 | step_limit |
| r96_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 3.6123 | 4.2915 | step_limit |
| r96_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 43.3955 | 41.7853 | step_limit |
| r96_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 6.1257 | 2.7206 | step_limit |
| r96_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 2.0912 | 4.6666 | step_limit |
| r96_n4_mri4/sample_007 | D-Flow | 818 | 816 | 6.9868 | 2.8314 | stalled |
| r96_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 6.2062 | 3.3281 | step_limit |
| r96_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 3.1437 | 3.8811 | step_limit |
| r96_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 2.0020 | 4.8165 | step_limit |
| r96_n4_mri4/sample_011 | D-Flow | 640 | 638 | 6.8082 | 2.7470 | stalled |
| r96_n4_mri4/sample_012 | D-Flow | 964 | 962 | 6.8167 | 2.3902 | stalled |
| r96_n4_mri4/sample_000 | P-Flow | 279 | 258 | 7.1755 | 1.8786 | stalled |
| r96_n4_mri4/sample_001 | P-Flow | 229 | 216 | 7.7592 | 2.5828 | stalled |
| r96_n4_mri4/sample_002 | P-Flow | 232 | 216 | 7.5283 | 2.4182 | stalled |
| r96_n4_mri4/sample_003 | P-Flow | 241 | 228 | 7.4329 | 2.5597 | stalled |
| r96_n4_mri4/sample_004 | P-Flow | 214 | 203 | 7.3856 | 2.4413 | stalled |
| r96_n4_mri4/sample_005 | P-Flow | 3 | 1 | 105.7077 | 89.8376 | stalled |
| r96_n4_mri4/sample_006 | P-Flow | 205 | 200 | 7.3550 | 1.9384 | stalled |
| r96_n4_mri4/sample_007 | P-Flow | 200 | 189 | 7.0905 | 2.8314 | stalled |
| r96_n4_mri4/sample_008 | P-Flow | 253 | 233 | 6.1305 | 3.3260 | stalled |
| r96_n4_mri4/sample_009 | P-Flow | 214 | 204 | 7.4588 | 2.1254 | stalled |
| r96_n4_mri4/sample_010 | P-Flow | 222 | 210 | 7.1905 | 2.7550 | stalled |
| r96_n4_mri4/sample_011 | P-Flow | 226 | 215 | 6.9874 | 2.7470 | stalled |
| r96_n4_mri4/sample_012 | P-Flow | 349 | 314 | 7.0000 | 2.3901 | stalled |
| r96_n4_mri4/sample_000 | DLR D-Flow | 262 | 256 | 18.2221 | 14.8331 | stalled |
| r96_n4_mri4/sample_001 | DLR D-Flow | 1000 | 968 | 31.8310 | 28.0651 | step_limit |
| r96_n4_mri4/sample_002 | DLR D-Flow | 795 | 765 | 7.5163 | 7.1738 | stalled |
| r96_n4_mri4/sample_003 | DLR D-Flow | 773 | 749 | 62.5658 | 59.0015 | stalled |
| r96_n4_mri4/sample_004 | DLR D-Flow | 1000 | 971 | 86.5096 | 77.4494 | step_limit |
| r96_n4_mri4/sample_005 | DLR D-Flow | 355 | 340 | 24.1336 | 19.7250 | stalled |
| r96_n4_mri4/sample_006 | DLR D-Flow | 787 | 750 | 15.0135 | 12.4595 | stalled |
| r96_n4_mri4/sample_007 | DLR D-Flow | 1000 | 968 | 7.0690 | 6.7425 | step_limit |
| r96_n4_mri4/sample_008 | DLR D-Flow | 533 | 505 | 13.3220 | 9.9220 | stalled |
| r96_n4_mri4/sample_009 | DLR D-Flow | 183 | 173 | 8.3188 | 7.6569 | stalled |
| r96_n4_mri4/sample_010 | DLR D-Flow | 350 | 337 | 15.6727 | 12.5463 | stalled |
| r96_n4_mri4/sample_011 | DLR D-Flow | 1000 | 974 | 68.7356 | 62.3188 | step_limit |
| r96_n4_mri4/sample_012 | DLR D-Flow | 734 | 697 | 82.5729 | 79.3119 | stalled |
| r128_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 19.0750 | 0.3391 | step_limit |
| r128_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 7.3058 | 2.7009 | step_limit |
| r128_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 18.9233 | 0.4768 | step_limit |
| r128_n1_ct10/sample_003 | D-Flow | 1000 | 1000 | 28.0513 | 0.3438 | step_limit |
| r128_n1_ct10/sample_004 | D-Flow | 675 | 672 | 25.0672 | 0.2518 | stalled |
| r128_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 9.6888 | 1.6715 | step_limit |
| r128_n1_ct10/sample_006 | D-Flow | 814 | 812 | 27.8760 | 0.3312 | stalled |
| r128_n1_ct10/sample_007 | D-Flow | 957 | 954 | 33.8507 | 0.2765 | stalled |
| r128_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 8.5448 | 1.7128 | step_limit |
| r128_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 22.3931 | 0.3048 | step_limit |
| r128_n1_ct10/sample_010 | D-Flow | 95 | 91 | 118.8740 | 111.1831 | stalled |
| r128_n1_ct10/sample_011 | D-Flow | 905 | 903 | 25.0209 | 0.3490 | stalled |
| r128_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 26.3769 | 0.3260 | step_limit |
| r128_n1_ct10/sample_013 | D-Flow | 1000 | 1000 | 24.9758 | 0.1811 | step_limit |
| r128_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 10.1409 | 0.9090 | step_limit |
| r128_n1_ct10/sample_015 | D-Flow | 1000 | 1000 | 23.8180 | 0.2422 | step_limit |
| r128_n1_ct10/sample_016 | D-Flow | 930 | 928 | 31.3260 | 0.2129 | stalled |
| r128_n1_ct10/sample_017 | D-Flow | 538 | 536 | 35.8692 | 0.1653 | stalled |
| r128_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 6.3416 | 2.9154 | step_limit |
| r128_n1_ct10/sample_019 | D-Flow | 466 | 464 | 28.9243 | 0.1766 | stalled |
| r128_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 6.8243 | 3.6835 | step_limit |
| r128_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 27.3041 | 0.3964 | step_limit |
| r128_n1_ct10/sample_022 | D-Flow | 1000 | 1000 | 26.0413 | 0.3293 | step_limit |
| r128_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 6.8486 | 2.7524 | step_limit |
| r128_n1_ct10/sample_024 | D-Flow | 929 | 926 | 24.4215 | 0.1376 | stalled |
| r128_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 25.9419 | 0.3429 | step_limit |
| r128_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 25.9048 | 0.3865 | step_limit |
| r128_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 7.4494 | 3.1061 | step_limit |
| r128_n1_ct10/sample_028 | D-Flow | 527 | 524 | 33.5565 | 0.2002 | stalled |
| r128_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 6.8367 | 2.8255 | step_limit |
| r128_n1_ct10/sample_000 | P-Flow | 1000 | 962 | 20.5212 | 0.4883 | step_limit |
| r128_n1_ct10/sample_001 | P-Flow | 52 | 45 | 1.9340 | 5.0539 | stalled |
| r128_n1_ct10/sample_002 | P-Flow | 79 | 69 | 4.5128 | 5.1325 | stalled |
| r128_n1_ct10/sample_003 | P-Flow | 688 | 562 | 3.1640 | 4.9976 | stalled |
| r128_n1_ct10/sample_004 | P-Flow | 1000 | 980 | 21.4012 | 0.3939 | step_limit |
| r128_n1_ct10/sample_005 | P-Flow | 803 | 720 | 24.0287 | 0.4524 | stalled |
| r128_n1_ct10/sample_006 | P-Flow | 726 | 654 | 25.4842 | 0.3987 | stalled |
| r128_n1_ct10/sample_007 | P-Flow | 21 | 17 | 4.6408 | 5.1449 | stalled |
| r128_n1_ct10/sample_008 | P-Flow | 187 | 149 | 3.4797 | 5.0616 | stalled |
| r128_n1_ct10/sample_009 | P-Flow | 31 | 26 | 3.7778 | 5.2287 | stalled |
| r128_n1_ct10/sample_010 | P-Flow | 80 | 65 | 3.7906 | 5.1102 | stalled |
| r128_n1_ct10/sample_011 | P-Flow | 34 | 29 | 3.4773 | 5.3574 | stalled |
| r128_n1_ct10/sample_012 | P-Flow | 124 | 108 | 2.3776 | 5.0469 | stalled |
| r128_n1_ct10/sample_013 | P-Flow | 50 | 42 | 4.5655 | 5.5729 | stalled |
| r128_n1_ct10/sample_014 | P-Flow | 22 | 19 | 13.3125 | 7.1194 | stalled |
| r128_n1_ct10/sample_015 | P-Flow | 36 | 29 | 4.5935 | 5.1455 | stalled |
| r128_n1_ct10/sample_016 | P-Flow | 79 | 69 | 2.4348 | 4.9853 | stalled |
| r128_n1_ct10/sample_017 | P-Flow | 461 | 446 | 32.2608 | 0.1670 | stalled |
| r128_n1_ct10/sample_018 | P-Flow | 26 | 21 | 2.7964 | 5.0763 | stalled |
| r128_n1_ct10/sample_019 | P-Flow | 277 | 224 | 2.4667 | 5.0261 | stalled |
| r128_n1_ct10/sample_020 | P-Flow | 38 | 32 | 5.4598 | 5.1528 | stalled |
| r128_n1_ct10/sample_021 | P-Flow | 45 | 38 | 2.1750 | 5.0956 | stalled |
| r128_n1_ct10/sample_022 | P-Flow | 1000 | 979 | 21.1309 | 0.4833 | step_limit |
| r128_n1_ct10/sample_023 | P-Flow | 41 | 33 | 12.8208 | 6.0839 | stalled |
| r128_n1_ct10/sample_024 | P-Flow | 1000 | 939 | 27.1491 | 0.1623 | step_limit |
| r128_n1_ct10/sample_025 | P-Flow | 42 | 34 | 4.1218 | 5.1300 | stalled |
| r128_n1_ct10/sample_026 | P-Flow | 36 | 30 | 3.5383 | 5.1270 | stalled |
| r128_n1_ct10/sample_027 | P-Flow | 56 | 50 | 2.0283 | 5.0156 | stalled |
| r128_n1_ct10/sample_028 | P-Flow | 131 | 109 | 2.5024 | 5.0123 | stalled |
| r128_n1_ct10/sample_029 | P-Flow | 890 | 828 | 27.1418 | 0.3650 | stalled |
| r128_n1_ct10/sample_000 | DLR D-Flow | 85 | 80 | 12.8830 | 7.4482 | stalled |
| r128_n1_ct10/sample_001 | DLR D-Flow | 190 | 180 | 2.1179 | 5.0034 | stalled |
| r128_n1_ct10/sample_002 | DLR D-Flow | 394 | 374 | 2.4524 | 4.9806 | stalled |
| r128_n1_ct10/sample_003 | DLR D-Flow | 371 | 359 | 2.0465 | 5.0015 | stalled |
| r128_n1_ct10/sample_004 | DLR D-Flow | 143 | 138 | 2.3687 | 5.0184 | stalled |
| r128_n1_ct10/sample_005 | DLR D-Flow | 88 | 84 | 2.6036 | 4.9803 | stalled |
| r128_n1_ct10/sample_006 | DLR D-Flow | 87 | 84 | 3.7454 | 5.0261 | stalled |
| r128_n1_ct10/sample_007 | DLR D-Flow | 539 | 513 | 3.6870 | 4.9749 | stalled |
| r128_n1_ct10/sample_008 | DLR D-Flow | 113 | 108 | 110.5116 | 94.3191 | stalled |
| r128_n1_ct10/sample_009 | DLR D-Flow | 348 | 334 | 1.6955 | 4.9938 | stalled |
| r128_n1_ct10/sample_010 | DLR D-Flow | 263 | 253 | 116.0430 | 112.1456 | stalled |
| r128_n1_ct10/sample_011 | DLR D-Flow | 565 | 543 | 2.2691 | 4.9999 | stalled |
| r128_n1_ct10/sample_012 | DLR D-Flow | 113 | 108 | 2.2944 | 5.0993 | stalled |
| r128_n1_ct10/sample_013 | DLR D-Flow | 429 | 405 | 2.1130 | 5.0202 | stalled |
| r128_n1_ct10/sample_014 | DLR D-Flow | 779 | 743 | 9.1214 | 5.7515 | stalled |
| r128_n1_ct10/sample_015 | DLR D-Flow | 67 | 61 | 4.1881 | 5.0834 | stalled |
| r128_n1_ct10/sample_016 | DLR D-Flow | 420 | 399 | 3.6522 | 4.9935 | stalled |
| r128_n1_ct10/sample_017 | DLR D-Flow | 221 | 211 | 1.9595 | 5.0018 | stalled |
| r128_n1_ct10/sample_018 | DLR D-Flow | 545 | 521 | 1.8601 | 4.9907 | stalled |
| r128_n1_ct10/sample_019 | DLR D-Flow | 525 | 503 | 2.4221 | 5.0016 | stalled |
| r128_n1_ct10/sample_020 | DLR D-Flow | 781 | 750 | 2.0612 | 4.9714 | stalled |
| r128_n1_ct10/sample_021 | DLR D-Flow | 545 | 497 | 1.7784 | 4.9905 | stalled |
| r128_n1_ct10/sample_022 | DLR D-Flow | 89 | 87 | 3.4103 | 4.9769 | stalled |
| r128_n1_ct10/sample_023 | DLR D-Flow | 341 | 322 | 5.0043 | 5.1810 | stalled |
| r128_n1_ct10/sample_024 | DLR D-Flow | 574 | 552 | 37.1655 | 16.3129 | stalled |
| r128_n1_ct10/sample_025 | DLR D-Flow | 556 | 529 | 2.4487 | 5.0156 | stalled |
| r128_n1_ct10/sample_026 | DLR D-Flow | 851 | 768 | 2.7549 | 5.0285 | stalled |
| r128_n1_ct10/sample_027 | DLR D-Flow | 167 | 161 | 2.3003 | 4.9798 | stalled |
| r128_n1_ct10/sample_028 | DLR D-Flow | 392 | 372 | 3.3817 | 4.9758 | stalled |
| r128_n1_ct10/sample_029 | DLR D-Flow | 605 | 583 | 1.8910 | 5.0019 | stalled |
| r128_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 3.5769 | 4.8173 | step_limit |
| r128_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 29.9964 | 0.1389 | step_limit |
| r128_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 5.4731 | 3.5188 | step_limit |
| r128_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 31.7063 | 0.4961 | step_limit |
| r128_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 18.1115 | 0.3376 | step_limit |
| r128_n2_ct10/sample_005 | D-Flow | 1000 | 1000 | 22.4436 | 0.6528 | step_limit |
| r128_n2_ct10/sample_006 | D-Flow | 599 | 597 | 32.3606 | 0.4378 | stalled |
| r128_n2_ct10/sample_007 | D-Flow | 698 | 695 | 34.9709 | 0.3735 | stalled |
| r128_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 26.9565 | 0.5575 | step_limit |
| r128_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 5.3606 | 4.7169 | step_limit |
| r128_n2_ct10/sample_010 | D-Flow | 822 | 820 | 32.4138 | 0.5838 | stalled |
| r128_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 25.1766 | 0.4964 | step_limit |
| r128_n2_ct10/sample_012 | D-Flow | 957 | 954 | 32.0588 | 0.2991 | stalled |
| r128_n2_ct10/sample_000 | P-Flow | 1000 | 811 | 3.0464 | 5.0365 | step_limit |
| r128_n2_ct10/sample_001 | P-Flow | 62 | 54 | 4.4385 | 5.0843 | stalled |
| r128_n2_ct10/sample_002 | P-Flow | 53 | 46 | 6.2728 | 5.2848 | stalled |
| r128_n2_ct10/sample_003 | P-Flow | 67 | 60 | 2.2163 | 5.0647 | stalled |
| r128_n2_ct10/sample_004 | P-Flow | 1000 | 816 | 4.3959 | 5.0631 | step_limit |
| r128_n2_ct10/sample_005 | P-Flow | 569 | 462 | 2.2993 | 5.0542 | stalled |
| r128_n2_ct10/sample_006 | P-Flow | 817 | 763 | 28.7347 | 0.4386 | stalled |
| r128_n2_ct10/sample_007 | P-Flow | 676 | 631 | 27.0228 | 0.3739 | stalled |
| r128_n2_ct10/sample_008 | P-Flow | 1000 | 919 | 30.2454 | 0.2617 | step_limit |
| r128_n2_ct10/sample_009 | P-Flow | 56 | 47 | 7.6124 | 5.4429 | stalled |
| r128_n2_ct10/sample_010 | P-Flow | 1000 | 849 | 21.8449 | 0.6738 | step_limit |
| r128_n2_ct10/sample_011 | P-Flow | 929 | 848 | 26.1874 | 0.1503 | stalled |
| r128_n2_ct10/sample_012 | P-Flow | 749 | 703 | 28.5482 | 0.3031 | stalled |
| r128_n2_ct10/sample_000 | DLR D-Flow | 53 | 50 | 114.2373 | 89.0319 | stalled |
| r128_n2_ct10/sample_001 | DLR D-Flow | 1000 | 970 | 4.1570 | 5.0763 | step_limit |
| r128_n2_ct10/sample_002 | DLR D-Flow | 799 | 763 | 98.6335 | 67.1696 | stalled |
| r128_n2_ct10/sample_003 | DLR D-Flow | 202 | 192 | 118.1708 | 85.1665 | stalled |
| r128_n2_ct10/sample_004 | DLR D-Flow | 1000 | 975 | 11.1988 | 7.1101 | step_limit |
| r128_n2_ct10/sample_005 | DLR D-Flow | 101 | 98 | 8.7536 | 5.3266 | stalled |
| r128_n2_ct10/sample_006 | DLR D-Flow | 333 | 314 | 2.4817 | 4.9782 | stalled |
| r128_n2_ct10/sample_007 | DLR D-Flow | 1000 | 971 | 2.6967 | 5.0026 | step_limit |
| r128_n2_ct10/sample_008 | DLR D-Flow | 386 | 370 | 2.3552 | 5.0273 | stalled |
| r128_n2_ct10/sample_009 | DLR D-Flow | 131 | 124 | 116.8830 | 86.3783 | stalled |
| r128_n2_ct10/sample_010 | DLR D-Flow | 189 | 180 | 2.5190 | 5.0201 | stalled |
| r128_n2_ct10/sample_011 | DLR D-Flow | 651 | 630 | 3.2105 | 5.0116 | stalled |
| r128_n2_ct10/sample_012 | DLR D-Flow | 71 | 67 | 4.4806 | 5.4931 | stalled |
| r128_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 24.1478 | 0.8414 | step_limit |
| r128_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 25.9509 | 0.3762 | step_limit |
| r128_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 52.5465 | 24.1672 | step_limit |
| r128_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 5.0490 | 4.8515 | step_limit |
| r128_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 7.2362 | 4.6833 | step_limit |
| r128_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 28.1328 | 0.6049 | step_limit |
| r128_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 6.0912 | 4.8306 | step_limit |
| r128_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 48.6637 | 26.2222 | step_limit |
| r128_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 26.6727 | 0.5368 | step_limit |
| r128_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 18.4084 | 2.6996 | step_limit |
| r128_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 12.4989 | 4.0295 | step_limit |
| r128_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 78.3675 | 42.3857 | step_limit |
| r128_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 6.9272 | 4.7115 | step_limit |
| r128_n4_ct10/sample_000 | P-Flow | 75 | 68 | 3.9452 | 4.9641 | stalled |
| r128_n4_ct10/sample_001 | P-Flow | 1000 | 884 | 19.2052 | 0.5015 | step_limit |
| r128_n4_ct10/sample_002 | P-Flow | 59 | 50 | 4.3050 | 5.0054 | stalled |
| r128_n4_ct10/sample_003 | P-Flow | 86 | 75 | 5.0651 | 5.0541 | stalled |
| r128_n4_ct10/sample_004 | P-Flow | 154 | 142 | 4.4168 | 4.5752 | stalled |
| r128_n4_ct10/sample_005 | P-Flow | 1000 | 862 | 20.3674 | 0.6443 | step_limit |
| r128_n4_ct10/sample_006 | P-Flow | 1000 | 902 | 20.7693 | 0.5448 | step_limit |
| r128_n4_ct10/sample_007 | P-Flow | 102 | 86 | 3.9334 | 4.8223 | stalled |
| r128_n4_ct10/sample_008 | P-Flow | 136 | 123 | 6.0535 | 4.4856 | stalled |
| r128_n4_ct10/sample_009 | P-Flow | 111 | 100 | 5.1933 | 4.5984 | stalled |
| r128_n4_ct10/sample_010 | P-Flow | 1000 | 880 | 20.7553 | 0.5874 | step_limit |
| r128_n4_ct10/sample_011 | P-Flow | 1000 | 901 | 20.7012 | 0.4945 | step_limit |
| r128_n4_ct10/sample_012 | P-Flow | 1000 | 898 | 22.7507 | 0.5454 | step_limit |
| r128_n4_ct10/sample_000 | DLR D-Flow | 479 | 456 | 76.5281 | 36.2224 | stalled |
| r128_n4_ct10/sample_001 | DLR D-Flow | 424 | 411 | 43.5958 | 17.4181 | stalled |
| r128_n4_ct10/sample_002 | DLR D-Flow | 423 | 411 | 59.1834 | 26.9574 | stalled |
| r128_n4_ct10/sample_003 | DLR D-Flow | 193 | 183 | 43.5197 | 20.1556 | stalled |
| r128_n4_ct10/sample_004 | DLR D-Flow | 110 | 105 | 71.0478 | 34.3823 | stalled |
| r128_n4_ct10/sample_005 | DLR D-Flow | 834 | 801 | 84.8728 | 43.9804 | stalled |
| r128_n4_ct10/sample_006 | DLR D-Flow | 478 | 461 | 51.7955 | 25.8346 | stalled |
| r128_n4_ct10/sample_007 | DLR D-Flow | 106 | 102 | 95.1189 | 57.0255 | stalled |
| r128_n4_ct10/sample_008 | DLR D-Flow | 1000 | 971 | 51.1334 | 26.8460 | step_limit |
| r128_n4_ct10/sample_009 | DLR D-Flow | 628 | 607 | 32.5126 | 11.8998 | stalled |
| r128_n4_ct10/sample_010 | DLR D-Flow | 438 | 418 | 53.7541 | 25.0825 | stalled |
| r128_n4_ct10/sample_011 | DLR D-Flow | 1000 | 969 | 97.2621 | 54.6081 | step_limit |
| r128_n4_ct10/sample_012 | DLR D-Flow | 1000 | 978 | 33.7712 | 13.1211 | step_limit |
| r128_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 4.4059 | 3.5445 | step_limit |
| r128_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 6.3499 | 2.9062 | step_limit |
| r128_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 5.8139 | 3.2127 | step_limit |
| r128_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 4.5634 | 3.9358 | step_limit |
| r128_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 4.1348 | 3.6897 | step_limit |
| r128_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 3.2927 | 4.2583 | step_limit |
| r128_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 4.8646 | 3.3773 | step_limit |
| r128_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 5.2450 | 3.1111 | step_limit |
| r128_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 2.9676 | 4.3454 | step_limit |
| r128_n1_mri4/sample_009 | D-Flow | 744 | 741 | 6.9183 | 2.5917 | stalled |
| r128_n1_mri4/sample_010 | D-Flow | 653 | 650 | 105.5938 | 102.6130 | stalled |
| r128_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 2.8319 | 4.2813 | step_limit |
| r128_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 5.8374 | 2.6510 | step_limit |
| r128_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 2.9995 | 4.1192 | step_limit |
| r128_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 5.7819 | 3.0175 | step_limit |
| r128_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 5.2108 | 2.8906 | step_limit |
| r128_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 6.3162 | 2.8541 | step_limit |
| r128_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 6.8354 | 2.4011 | step_limit |
| r128_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 2.5643 | 4.6402 | step_limit |
| r128_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 6.3804 | 2.3744 | step_limit |
| r128_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 4.7344 | 3.5861 | step_limit |
| r128_n1_mri4/sample_021 | D-Flow | 1000 | 999 | 6.3964 | 2.8795 | step_limit |
| r128_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 2.8186 | 3.9232 | step_limit |
| r128_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 4.8980 | 3.4662 | step_limit |
| r128_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 4.7912 | 3.6283 | step_limit |
| r128_n1_mri4/sample_025 | D-Flow | 921 | 919 | 7.1769 | 2.6151 | stalled |
| r128_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 6.5695 | 2.5824 | step_limit |
| r128_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 6.8564 | 2.8925 | step_limit |
| r128_n1_mri4/sample_028 | D-Flow | 879 | 876 | 6.8805 | 2.5113 | stalled |
| r128_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 5.3902 | 3.0257 | step_limit |
| r128_n1_mri4/sample_000 | P-Flow | 26 | 21 | 3.8857 | 5.6627 | stalled |
| r128_n1_mri4/sample_001 | P-Flow | 37 | 29 | 1.5914 | 5.1103 | stalled |
| r128_n1_mri4/sample_002 | P-Flow | 32 | 26 | 2.2707 | 5.2027 | stalled |
| r128_n1_mri4/sample_003 | P-Flow | 247 | 225 | 7.8683 | 2.7053 | stalled |
| r128_n1_mri4/sample_004 | P-Flow | 264 | 237 | 7.2145 | 2.7267 | stalled |
| r128_n1_mri4/sample_005 | P-Flow | 37 | 30 | 1.8606 | 5.1027 | stalled |
| r128_n1_mri4/sample_006 | P-Flow | 224 | 204 | 8.2987 | 2.3807 | stalled |
| r128_n1_mri4/sample_007 | P-Flow | 58 | 48 | 1.5936 | 5.1174 | stalled |
| r128_n1_mri4/sample_008 | P-Flow | 36 | 29 | 1.9730 | 5.2128 | stalled |
| r128_n1_mri4/sample_009 | P-Flow | 22 | 17 | 1.6558 | 5.1160 | stalled |
| r128_n1_mri4/sample_010 | P-Flow | 294 | 274 | 8.2357 | 2.4946 | stalled |
| r128_n1_mri4/sample_011 | P-Flow | 242 | 224 | 7.0588 | 2.7614 | stalled |
| r128_n1_mri4/sample_012 | P-Flow | 218 | 203 | 8.1929 | 2.3977 | stalled |
| r128_n1_mri4/sample_013 | P-Flow | 38 | 32 | 65.5350 | 62.2427 | stalled |
| r128_n1_mri4/sample_014 | P-Flow | 226 | 209 | 8.1894 | 2.4064 | stalled |
| r128_n1_mri4/sample_015 | P-Flow | 189 | 176 | 7.7033 | 2.4163 | stalled |
| r128_n1_mri4/sample_016 | P-Flow | 45 | 37 | 1.9082 | 5.1222 | stalled |
| r128_n1_mri4/sample_017 | P-Flow | 182 | 176 | 8.1268 | 2.3929 | stalled |
| r128_n1_mri4/sample_018 | P-Flow | 37 | 30 | 1.4405 | 5.0770 | stalled |
| r128_n1_mri4/sample_019 | P-Flow | 28 | 22 | 1.4326 | 5.1118 | stalled |
| r128_n1_mri4/sample_020 | P-Flow | 206 | 194 | 7.7504 | 2.8694 | stalled |
| r128_n1_mri4/sample_021 | P-Flow | 205 | 191 | 7.8290 | 2.8795 | stalled |
| r128_n1_mri4/sample_022 | P-Flow | 222 | 209 | 6.9294 | 2.5808 | stalled |
| r128_n1_mri4/sample_023 | P-Flow | 21 | 17 | 1.7661 | 5.1167 | stalled |
| r128_n1_mri4/sample_024 | P-Flow | 23 | 18 | 1.7939 | 5.1394 | stalled |
| r128_n1_mri4/sample_025 | P-Flow | 48 | 42 | 2.0271 | 5.1475 | stalled |
| r128_n1_mri4/sample_026 | P-Flow | 204 | 195 | 8.6254 | 2.3936 | stalled |
| r128_n1_mri4/sample_027 | P-Flow | 21 | 17 | 1.4265 | 5.0768 | stalled |
| r128_n1_mri4/sample_028 | P-Flow | 227 | 208 | 8.3024 | 2.5112 | stalled |
| r128_n1_mri4/sample_029 | P-Flow | 227 | 205 | 8.1357 | 2.3203 | stalled |
| r128_n1_mri4/sample_000 | DLR D-Flow | 80 | 73 | 60.9153 | 53.1155 | stalled |
| r128_n1_mri4/sample_001 | DLR D-Flow | 900 | 869 | 1.3564 | 5.0555 | stalled |
| r128_n1_mri4/sample_002 | DLR D-Flow | 223 | 216 | 59.3499 | 50.6323 | stalled |
| r128_n1_mri4/sample_003 | DLR D-Flow | 1000 | 964 | 100.7040 | 100.3285 | step_limit |
| r128_n1_mri4/sample_004 | DLR D-Flow | 106 | 102 | 2.4746 | 5.3273 | stalled |
| r128_n1_mri4/sample_005 | DLR D-Flow | 212 | 206 | 1.7187 | 5.0643 | stalled |
| r128_n1_mri4/sample_006 | DLR D-Flow | 156 | 153 | 6.3116 | 7.3721 | stalled |
| r128_n1_mri4/sample_007 | DLR D-Flow | 621 | 596 | 1.3617 | 5.0522 | stalled |
| r128_n1_mri4/sample_008 | DLR D-Flow | 31 | 26 | 104.4702 | 104.3930 | stalled |
| r128_n1_mri4/sample_009 | DLR D-Flow | 211 | 202 | 2.0472 | 5.2516 | stalled |
| r128_n1_mri4/sample_010 | DLR D-Flow | 134 | 129 | 105.7652 | 104.0801 | stalled |
| r128_n1_mri4/sample_011 | DLR D-Flow | 370 | 361 | 3.3127 | 5.6770 | stalled |
| r128_n1_mri4/sample_012 | DLR D-Flow | 26 | 22 | 109.8279 | 105.7369 | stalled |
| r128_n1_mri4/sample_013 | DLR D-Flow | 792 | 766 | 2.7958 | 5.3456 | stalled |
| r128_n1_mri4/sample_014 | DLR D-Flow | 425 | 410 | 4.3497 | 6.2752 | stalled |
| r128_n1_mri4/sample_015 | DLR D-Flow | 363 | 349 | 2.0597 | 5.2361 | stalled |
| r128_n1_mri4/sample_016 | DLR D-Flow | 393 | 356 | 1.9261 | 5.1120 | stalled |
| r128_n1_mri4/sample_017 | DLR D-Flow | 126 | 120 | 1.7830 | 5.1147 | stalled |
| r128_n1_mri4/sample_018 | DLR D-Flow | 380 | 366 | 1.4403 | 5.0568 | stalled |
| r128_n1_mri4/sample_019 | DLR D-Flow | 570 | 547 | 1.5497 | 5.1198 | stalled |
| r128_n1_mri4/sample_020 | DLR D-Flow | 477 | 455 | 1.3129 | 5.0401 | stalled |
| r128_n1_mri4/sample_021 | DLR D-Flow | 1000 | 957 | 1.5811 | 5.0920 | step_limit |
| r128_n1_mri4/sample_022 | DLR D-Flow | 73 | 69 | 2.1200 | 5.1706 | stalled |
| r128_n1_mri4/sample_023 | DLR D-Flow | 158 | 152 | 3.1652 | 5.6298 | stalled |
| r128_n1_mri4/sample_024 | DLR D-Flow | 66 | 62 | 5.6399 | 6.2161 | stalled |
| r128_n1_mri4/sample_025 | DLR D-Flow | 441 | 414 | 42.1579 | 37.8073 | stalled |
| r128_n1_mri4/sample_026 | DLR D-Flow | 437 | 418 | 2.8710 | 5.3613 | stalled |
| r128_n1_mri4/sample_027 | DLR D-Flow | 182 | 176 | 5.7825 | 7.3354 | stalled |
| r128_n1_mri4/sample_028 | DLR D-Flow | 461 | 441 | 1.6847 | 5.0717 | stalled |
| r128_n1_mri4/sample_029 | DLR D-Flow | 602 | 577 | 1.5989 | 5.0843 | stalled |
| r128_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 2.0606 | 4.7954 | step_limit |
| r128_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 2.1394 | 4.6358 | step_limit |
| r128_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 3.1962 | 4.3764 | step_limit |
| r128_n2_mri4/sample_003 | D-Flow | 604 | 600 | 71.8328 | 71.3991 | stalled |
| r128_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 2.8189 | 4.5420 | step_limit |
| r128_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 3.3069 | 4.4105 | step_limit |
| r128_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 2.4471 | 4.5989 | step_limit |
| r128_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 2.2662 | 4.6869 | step_limit |
| r128_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 5.5864 | 3.1269 | step_limit |
| r128_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.7320 | 5.0854 | step_limit |
| r128_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.9421 | 4.7565 | step_limit |
| r128_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.7229 | 4.8379 | step_limit |
| r128_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 2.5666 | 4.6156 | step_limit |
| r128_n2_mri4/sample_000 | P-Flow | 193 | 182 | 8.1355 | 2.4599 | stalled |
| r128_n2_mri4/sample_001 | P-Flow | 252 | 227 | 6.9459 | 2.2247 | stalled |
| r128_n2_mri4/sample_002 | P-Flow | 217 | 197 | 7.3577 | 2.8037 | stalled |
| r128_n2_mri4/sample_003 | P-Flow | 182 | 172 | 7.9088 | 2.4382 | stalled |
| r128_n2_mri4/sample_004 | P-Flow | 57 | 46 | 2.4047 | 5.1586 | stalled |
| r128_n2_mri4/sample_005 | P-Flow | 190 | 183 | 7.9587 | 2.4433 | stalled |
| r128_n2_mri4/sample_006 | P-Flow | 177 | 166 | 7.6951 | 2.4310 | stalled |
| r128_n2_mri4/sample_007 | P-Flow | 200 | 189 | 7.8969 | 2.4091 | stalled |
| r128_n2_mri4/sample_008 | P-Flow | 177 | 167 | 7.6476 | 2.4487 | stalled |
| r128_n2_mri4/sample_009 | P-Flow | 244 | 226 | 7.3063 | 2.7813 | stalled |
| r128_n2_mri4/sample_010 | P-Flow | 198 | 185 | 7.6737 | 2.4586 | stalled |
| r128_n2_mri4/sample_011 | P-Flow | 231 | 216 | 7.8192 | 2.5268 | stalled |
| r128_n2_mri4/sample_012 | P-Flow | 16 | 13 | 96.2618 | 63.1032 | stalled |
| r128_n2_mri4/sample_000 | DLR D-Flow | 397 | 380 | 79.6629 | 78.6586 | stalled |
| r128_n2_mri4/sample_001 | DLR D-Flow | 1000 | 962 | 6.5056 | 7.3251 | step_limit |
| r128_n2_mri4/sample_002 | DLR D-Flow | 959 | 927 | 1.9508 | 5.1593 | stalled |
| r128_n2_mri4/sample_003 | DLR D-Flow | 67 | 64 | 119.0044 | 118.5228 | stalled |
| r128_n2_mri4/sample_004 | DLR D-Flow | 201 | 195 | 102.2561 | 92.4349 | stalled |
| r128_n2_mri4/sample_005 | DLR D-Flow | 1000 | 974 | 50.5320 | 44.5537 | step_limit |
| r128_n2_mri4/sample_006 | DLR D-Flow | 213 | 202 | 1.6399 | 5.1200 | stalled |
| r128_n2_mri4/sample_007 | DLR D-Flow | 377 | 362 | 68.2496 | 66.8197 | stalled |
| r128_n2_mri4/sample_008 | DLR D-Flow | 618 | 597 | 3.7547 | 5.6976 | stalled |
| r128_n2_mri4/sample_009 | DLR D-Flow | 434 | 420 | 2.8734 | 5.3800 | stalled |
| r128_n2_mri4/sample_010 | DLR D-Flow | 311 | 302 | 2.4493 | 5.2385 | stalled |
| r128_n2_mri4/sample_011 | DLR D-Flow | 295 | 280 | 83.2161 | 82.1481 | stalled |
| r128_n2_mri4/sample_012 | DLR D-Flow | 77 | 72 | 93.2114 | 76.8167 | stalled |
| r128_n4_mri4/sample_000 | D-Flow | 666 | 664 | 42.9689 | 40.6357 | stalled |
| r128_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 2.2501 | 4.5255 | step_limit |
| r128_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 37.7851 | 37.6081 | step_limit |
| r128_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 2.4483 | 4.6134 | step_limit |
| r128_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.9769 | 4.7424 | step_limit |
| r128_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 4.3962 | 4.1295 | step_limit |
| r128_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 7.1135 | 2.0694 | step_limit |
| r128_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.6498 | 4.9541 | step_limit |
| r128_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 5.4159 | 3.6676 | step_limit |
| r128_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 6.2201 | 3.3897 | step_limit |
| r128_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.8849 | 4.7244 | step_limit |
| r128_n4_mri4/sample_011 | D-Flow | 811 | 809 | 1.6579 | 4.9341 | stalled |
| r128_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.9270 | 4.9026 | step_limit |
| r128_n4_mri4/sample_000 | P-Flow | 198 | 191 | 7.5305 | 2.0529 | stalled |
| r128_n4_mri4/sample_001 | P-Flow | 289 | 261 | 7.3657 | 2.5978 | stalled |
| r128_n4_mri4/sample_002 | P-Flow | 205 | 195 | 7.3610 | 2.5793 | stalled |
| r128_n4_mri4/sample_003 | P-Flow | 236 | 222 | 7.0234 | 2.8611 | stalled |
| r128_n4_mri4/sample_004 | P-Flow | 315 | 289 | 7.4806 | 2.4250 | stalled |
| r128_n4_mri4/sample_005 | P-Flow | 245 | 227 | 7.0042 | 2.8391 | stalled |
| r128_n4_mri4/sample_006 | P-Flow | 213 | 208 | 7.0032 | 2.0686 | stalled |
| r128_n4_mri4/sample_007 | P-Flow | 170 | 166 | 6.6465 | 2.7553 | stalled |
| r128_n4_mri4/sample_008 | P-Flow | 239 | 224 | 6.6201 | 2.7633 | stalled |
| r128_n4_mri4/sample_009 | P-Flow | 200 | 187 | 7.2196 | 2.5758 | stalled |
| r128_n4_mri4/sample_010 | P-Flow | 198 | 191 | 6.9955 | 2.3500 | stalled |
| r128_n4_mri4/sample_011 | P-Flow | 206 | 195 | 7.1560 | 2.2695 | stalled |
| r128_n4_mri4/sample_012 | P-Flow | 234 | 221 | 7.1056 | 2.7292 | stalled |
| r128_n4_mri4/sample_000 | DLR D-Flow | 73 | 69 | 68.6370 | 63.2790 | stalled |
| r128_n4_mri4/sample_001 | DLR D-Flow | 149 | 139 | 20.8338 | 18.1245 | stalled |
| r128_n4_mri4/sample_002 | DLR D-Flow | 1000 | 962 | 74.0377 | 71.8501 | step_limit |
| r128_n4_mri4/sample_003 | DLR D-Flow | 19 | 16 | 106.1334 | 104.7119 | stalled |
| r128_n4_mri4/sample_004 | DLR D-Flow | 42 | 39 | 73.4965 | 68.5896 | stalled |
| r128_n4_mri4/sample_005 | DLR D-Flow | 62 | 58 | 90.3272 | 88.2523 | stalled |
| r128_n4_mri4/sample_006 | DLR D-Flow | 959 | 923 | 19.4847 | 16.1110 | stalled |
| r128_n4_mri4/sample_007 | DLR D-Flow | 678 | 648 | 57.1956 | 55.8261 | stalled |
| r128_n4_mri4/sample_008 | DLR D-Flow | 285 | 271 | 41.3185 | 39.4610 | stalled |
| r128_n4_mri4/sample_009 | DLR D-Flow | 1000 | 964 | 41.0900 | 38.4907 | step_limit |
| r128_n4_mri4/sample_010 | DLR D-Flow | 208 | 197 | 81.1580 | 77.1057 | stalled |
| r128_n4_mri4/sample_011 | DLR D-Flow | 1000 | 976 | 66.1053 | 59.4659 | step_limit |
| r128_n4_mri4/sample_012 | DLR D-Flow | 851 | 811 | 64.8053 | 62.5705 | stalled |
| r160_n1_ct10/sample_000 | D-Flow | 839 | 837 | 30.0755 | 0.1519 | stalled |
| r160_n1_ct10/sample_001 | D-Flow | 941 | 939 | 28.5601 | 0.2911 | stalled |
| r160_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 19.5143 | 0.5426 | step_limit |
| r160_n1_ct10/sample_003 | D-Flow | 679 | 677 | 27.2115 | 0.2842 | stalled |
| r160_n1_ct10/sample_004 | D-Flow | 646 | 644 | 45.0791 | 0.1348 | stalled |
| r160_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 28.2938 | 0.1725 | step_limit |
| r160_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 8.0037 | 2.3641 | step_limit |
| r160_n1_ct10/sample_007 | D-Flow | 590 | 588 | 38.5388 | 0.3147 | stalled |
| r160_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 18.5280 | 0.4252 | step_limit |
| r160_n1_ct10/sample_009 | D-Flow | 556 | 554 | 36.1855 | 0.3509 | stalled |
| r160_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 7.6749 | 2.7571 | step_limit |
| r160_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 27.2125 | 0.1929 | step_limit |
| r160_n1_ct10/sample_012 | D-Flow | 1000 | 1000 | 29.0983 | 0.2822 | step_limit |
| r160_n1_ct10/sample_013 | D-Flow | 1000 | 1000 | 13.3282 | 0.8649 | step_limit |
| r160_n1_ct10/sample_014 | D-Flow | 79 | 77 | 2.2477 | 5.0142 | stalled |
| r160_n1_ct10/sample_015 | D-Flow | 122 | 119 | 98.5605 | 87.0097 | stalled |
| r160_n1_ct10/sample_016 | D-Flow | 861 | 859 | 27.0515 | 0.2463 | stalled |
| r160_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 21.6538 | 0.3125 | step_limit |
| r160_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 11.5531 | 2.2034 | step_limit |
| r160_n1_ct10/sample_019 | D-Flow | 440 | 438 | 40.5655 | 0.3728 | stalled |
| r160_n1_ct10/sample_020 | D-Flow | 25 | 23 | 101.9481 | 91.1401 | stalled |
| r160_n1_ct10/sample_021 | D-Flow | 586 | 584 | 40.5117 | 0.4134 | stalled |
| r160_n1_ct10/sample_022 | D-Flow | 100 | 98 | 93.7455 | 79.9910 | stalled |
| r160_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 7.5641 | 4.2018 | step_limit |
| r160_n1_ct10/sample_024 | D-Flow | 758 | 756 | 35.3803 | 0.2298 | stalled |
| r160_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 20.5013 | 0.4448 | step_limit |
| r160_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 30.5872 | 0.2205 | step_limit |
| r160_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 25.5795 | 0.3238 | step_limit |
| r160_n1_ct10/sample_028 | D-Flow | 333 | 330 | 95.1715 | 78.6067 | stalled |
| r160_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 7.4168 | 2.9287 | step_limit |
| r160_n1_ct10/sample_000 | P-Flow | 32 | 27 | 5.5620 | 5.9321 | stalled |
| r160_n1_ct10/sample_001 | P-Flow | 69 | 58 | 6.4361 | 6.0726 | stalled |
| r160_n1_ct10/sample_002 | P-Flow | 619 | 512 | 2.1110 | 5.0543 | stalled |
| r160_n1_ct10/sample_003 | P-Flow | 1000 | 875 | 21.6609 | 0.4366 | step_limit |
| r160_n1_ct10/sample_004 | P-Flow | 81 | 70 | 5.5905 | 5.1897 | stalled |
| r160_n1_ct10/sample_005 | P-Flow | 54 | 45 | 6.3579 | 5.5359 | stalled |
| r160_n1_ct10/sample_006 | P-Flow | 31 | 26 | 3.0927 | 5.0388 | stalled |
| r160_n1_ct10/sample_007 | P-Flow | 1000 | 893 | 25.0559 | 0.4402 | step_limit |
| r160_n1_ct10/sample_008 | P-Flow | 1000 | 966 | 20.0970 | 0.5590 | step_limit |
| r160_n1_ct10/sample_009 | P-Flow | 1000 | 855 | 24.7457 | 0.4403 | step_limit |
| r160_n1_ct10/sample_010 | P-Flow | 35 | 29 | 1.4880 | 4.9997 | stalled |
| r160_n1_ct10/sample_011 | P-Flow | 96 | 86 | 2.6764 | 5.0010 | stalled |
| r160_n1_ct10/sample_012 | P-Flow | 1000 | 871 | 23.5513 | 0.3909 | step_limit |
| r160_n1_ct10/sample_013 | P-Flow | 116 | 93 | 3.3771 | 4.9984 | stalled |
| r160_n1_ct10/sample_014 | P-Flow | 359 | 297 | 2.9921 | 5.2763 | stalled |
| r160_n1_ct10/sample_015 | P-Flow | 1000 | 947 | 19.9836 | 0.5302 | step_limit |
| r160_n1_ct10/sample_016 | P-Flow | 680 | 545 | 2.9265 | 4.9045 | stalled |
| r160_n1_ct10/sample_017 | P-Flow | 1000 | 819 | 4.2301 | 4.9945 | step_limit |
| r160_n1_ct10/sample_018 | P-Flow | 39 | 34 | 3.6069 | 5.0318 | stalled |
| r160_n1_ct10/sample_019 | P-Flow | 622 | 480 | 3.2233 | 5.0392 | stalled |
| r160_n1_ct10/sample_020 | P-Flow | 37 | 30 | 2.6800 | 5.0181 | stalled |
| r160_n1_ct10/sample_021 | P-Flow | 65 | 57 | 2.4635 | 5.0067 | stalled |
| r160_n1_ct10/sample_022 | P-Flow | 56 | 50 | 3.1118 | 4.9634 | stalled |
| r160_n1_ct10/sample_023 | P-Flow | 385 | 316 | 3.1233 | 5.0088 | stalled |
| r160_n1_ct10/sample_024 | P-Flow | 34 | 29 | 2.6389 | 5.0106 | stalled |
| r160_n1_ct10/sample_025 | P-Flow | 30 | 26 | 4.3347 | 5.3593 | stalled |
| r160_n1_ct10/sample_026 | P-Flow | 66 | 58 | 1.9830 | 4.9835 | stalled |
| r160_n1_ct10/sample_027 | P-Flow | 48 | 42 | 3.5958 | 4.9915 | stalled |
| r160_n1_ct10/sample_028 | P-Flow | 1000 | 965 | 21.4503 | 0.4718 | step_limit |
| r160_n1_ct10/sample_029 | P-Flow | 689 | 637 | 26.4206 | 0.3500 | stalled |
| r160_n1_ct10/sample_000 | DLR D-Flow | 447 | 435 | 4.1010 | 5.0439 | stalled |
| r160_n1_ct10/sample_001 | DLR D-Flow | 631 | 614 | 1.8482 | 4.9967 | stalled |
| r160_n1_ct10/sample_002 | DLR D-Flow | 113 | 108 | 97.2691 | 83.4356 | stalled |
| r160_n1_ct10/sample_003 | DLR D-Flow | 172 | 164 | 1.7256 | 5.0046 | stalled |
| r160_n1_ct10/sample_004 | DLR D-Flow | 360 | 349 | 2.6486 | 4.9782 | stalled |
| r160_n1_ct10/sample_005 | DLR D-Flow | 890 | 860 | 1.9730 | 4.9789 | stalled |
| r160_n1_ct10/sample_006 | DLR D-Flow | 648 | 617 | 2.1000 | 4.9928 | stalled |
| r160_n1_ct10/sample_007 | DLR D-Flow | 545 | 523 | 2.1588 | 4.9810 | stalled |
| r160_n1_ct10/sample_008 | DLR D-Flow | 472 | 456 | 3.3428 | 4.9768 | stalled |
| r160_n1_ct10/sample_009 | DLR D-Flow | 474 | 459 | 1.9039 | 4.9962 | stalled |
| r160_n1_ct10/sample_010 | DLR D-Flow | 514 | 494 | 4.5736 | 4.9791 | stalled |
| r160_n1_ct10/sample_011 | DLR D-Flow | 438 | 424 | 1.6575 | 4.9837 | stalled |
| r160_n1_ct10/sample_012 | DLR D-Flow | 312 | 301 | 3.2982 | 4.9695 | stalled |
| r160_n1_ct10/sample_013 | DLR D-Flow | 295 | 275 | 2.8475 | 5.0134 | stalled |
| r160_n1_ct10/sample_014 | DLR D-Flow | 1000 | 952 | 3.6790 | 5.1755 | step_limit |
| r160_n1_ct10/sample_015 | DLR D-Flow | 18 | 16 | 106.9884 | 91.6859 | stalled |
| r160_n1_ct10/sample_016 | DLR D-Flow | 384 | 368 | 2.1556 | 4.9752 | stalled |
| r160_n1_ct10/sample_017 | DLR D-Flow | 965 | 921 | 4.0322 | 5.0448 | stalled |
| r160_n1_ct10/sample_018 | DLR D-Flow | 315 | 302 | 2.9790 | 5.0286 | stalled |
| r160_n1_ct10/sample_019 | DLR D-Flow | 182 | 174 | 2.0498 | 5.0014 | stalled |
| r160_n1_ct10/sample_020 | DLR D-Flow | 191 | 184 | 2.0795 | 5.0132 | stalled |
| r160_n1_ct10/sample_021 | DLR D-Flow | 343 | 336 | 2.2182 | 4.9992 | stalled |
| r160_n1_ct10/sample_022 | DLR D-Flow | 24 | 22 | 93.9681 | 81.1974 | stalled |
| r160_n1_ct10/sample_023 | DLR D-Flow | 145 | 139 | 1.9755 | 4.9868 | stalled |
| r160_n1_ct10/sample_024 | DLR D-Flow | 339 | 324 | 2.2164 | 4.9908 | stalled |
| r160_n1_ct10/sample_025 | DLR D-Flow | 111 | 106 | 1.9497 | 4.9835 | stalled |
| r160_n1_ct10/sample_026 | DLR D-Flow | 403 | 384 | 4.3003 | 4.9956 | stalled |
| r160_n1_ct10/sample_027 | DLR D-Flow | 336 | 325 | 1.7661 | 4.9728 | stalled |
| r160_n1_ct10/sample_028 | DLR D-Flow | 177 | 169 | 2.0800 | 4.9911 | stalled |
| r160_n1_ct10/sample_029 | DLR D-Flow | 529 | 507 | 1.6692 | 4.9904 | stalled |
| r160_n2_ct10/sample_000 | D-Flow | 828 | 825 | 33.9185 | 0.2406 | stalled |
| r160_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 37.1654 | 0.1997 | step_limit |
| r160_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 25.7082 | 0.1711 | step_limit |
| r160_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 45.3917 | 0.4394 | step_limit |
| r160_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 30.7798 | 0.1087 | step_limit |
| r160_n2_ct10/sample_005 | D-Flow | 850 | 848 | 36.2346 | 0.1089 | stalled |
| r160_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 5.1519 | 4.8061 | step_limit |
| r160_n2_ct10/sample_007 | D-Flow | 674 | 671 | 33.7998 | 0.3600 | stalled |
| r160_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 24.2473 | 0.6487 | step_limit |
| r160_n2_ct10/sample_009 | D-Flow | 1000 | 1000 | 5.0241 | 3.6038 | step_limit |
| r160_n2_ct10/sample_010 | D-Flow | 986 | 984 | 34.4297 | 0.1531 | stalled |
| r160_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 27.7306 | 0.2350 | step_limit |
| r160_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 6.7888 | 4.7325 | step_limit |
| r160_n2_ct10/sample_000 | P-Flow | 1000 | 857 | 28.8816 | 0.2849 | step_limit |
| r160_n2_ct10/sample_001 | P-Flow | 26 | 23 | 43.0107 | 18.8416 | stalled |
| r160_n2_ct10/sample_002 | P-Flow | 1000 | 937 | 21.6712 | 0.3530 | step_limit |
| r160_n2_ct10/sample_003 | P-Flow | 764 | 707 | 26.5344 | 0.4497 | stalled |
| r160_n2_ct10/sample_004 | P-Flow | 1000 | 930 | 29.1671 | 0.0861 | step_limit |
| r160_n2_ct10/sample_005 | P-Flow | 1000 | 840 | 19.9460 | 1.0251 | step_limit |
| r160_n2_ct10/sample_006 | P-Flow | 756 | 602 | 4.9788 | 4.5821 | stalled |
| r160_n2_ct10/sample_007 | P-Flow | 1000 | 891 | 23.7493 | 0.4825 | step_limit |
| r160_n2_ct10/sample_008 | P-Flow | 4 | 2 | 98.5156 | 65.0941 | stalled |
| r160_n2_ct10/sample_009 | P-Flow | 802 | 742 | 26.8771 | 0.1258 | stalled |
| r160_n2_ct10/sample_010 | P-Flow | 1000 | 834 | 22.3846 | 0.3137 | step_limit |
| r160_n2_ct10/sample_011 | P-Flow | 1000 | 935 | 21.2846 | 0.4449 | step_limit |
| r160_n2_ct10/sample_012 | P-Flow | 104 | 89 | 6.0029 | 5.1612 | stalled |
| r160_n2_ct10/sample_000 | DLR D-Flow | 349 | 334 | 2.7927 | 4.9909 | stalled |
| r160_n2_ct10/sample_001 | DLR D-Flow | 694 | 665 | 4.0599 | 5.0312 | stalled |
| r160_n2_ct10/sample_002 | DLR D-Flow | 208 | 197 | 60.7095 | 41.5502 | stalled |
| r160_n2_ct10/sample_003 | DLR D-Flow | 65 | 57 | 76.4753 | 42.6629 | stalled |
| r160_n2_ct10/sample_004 | DLR D-Flow | 186 | 177 | 69.0057 | 47.9529 | stalled |
| r160_n2_ct10/sample_005 | DLR D-Flow | 1000 | 968 | 4.4161 | 5.0952 | step_limit |
| r160_n2_ct10/sample_006 | DLR D-Flow | 541 | 524 | 5.5423 | 5.1581 | stalled |
| r160_n2_ct10/sample_007 | DLR D-Flow | 418 | 405 | 5.8985 | 5.3444 | stalled |
| r160_n2_ct10/sample_008 | DLR D-Flow | 700 | 675 | 90.9496 | 56.5929 | stalled |
| r160_n2_ct10/sample_009 | DLR D-Flow | 183 | 174 | 2.3046 | 4.9889 | stalled |
| r160_n2_ct10/sample_010 | DLR D-Flow | 106 | 100 | 13.3356 | 5.9184 | stalled |
| r160_n2_ct10/sample_011 | DLR D-Flow | 347 | 337 | 48.4854 | 29.4098 | stalled |
| r160_n2_ct10/sample_012 | DLR D-Flow | 109 | 101 | 126.3249 | 107.0327 | stalled |
| r160_n4_ct10/sample_000 | D-Flow | 958 | 955 | 31.3055 | 0.0874 | stalled |
| r160_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 36.4392 | 0.2491 | step_limit |
| r160_n4_ct10/sample_002 | D-Flow | 712 | 710 | 31.0072 | 0.2493 | stalled |
| r160_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 38.5093 | 0.2741 | step_limit |
| r160_n4_ct10/sample_004 | D-Flow | 772 | 770 | 35.7802 | 0.3954 | stalled |
| r160_n4_ct10/sample_005 | D-Flow | 945 | 942 | 33.5462 | 0.3764 | stalled |
| r160_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 24.9530 | 0.5417 | step_limit |
| r160_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 25.8592 | 0.6752 | step_limit |
| r160_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 33.2863 | 0.4538 | step_limit |
| r160_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 30.5456 | 0.5112 | step_limit |
| r160_n4_ct10/sample_010 | D-Flow | 677 | 675 | 33.3752 | 0.2053 | stalled |
| r160_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 25.6512 | 0.5858 | step_limit |
| r160_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 26.8539 | 0.5022 | step_limit |
| r160_n4_ct10/sample_000 | P-Flow | 1000 | 891 | 22.5989 | 0.2778 | step_limit |
| r160_n4_ct10/sample_001 | P-Flow | 1000 | 926 | 24.2936 | 0.3750 | step_limit |
| r160_n4_ct10/sample_002 | P-Flow | 1000 | 876 | 20.8904 | 0.4090 | step_limit |
| r160_n4_ct10/sample_003 | P-Flow | 716 | 674 | 26.1790 | 0.2630 | stalled |
| r160_n4_ct10/sample_004 | P-Flow | 453 | 427 | 24.9145 | 0.4048 | stalled |
| r160_n4_ct10/sample_005 | P-Flow | 749 | 685 | 24.7741 | 0.4126 | stalled |
| r160_n4_ct10/sample_006 | P-Flow | 1000 | 876 | 23.2772 | 0.2822 | step_limit |
| r160_n4_ct10/sample_007 | P-Flow | 1000 | 895 | 26.3937 | 0.2281 | step_limit |
| r160_n4_ct10/sample_008 | P-Flow | 1000 | 925 | 27.8472 | 0.4705 | step_limit |
| r160_n4_ct10/sample_009 | P-Flow | 592 | 582 | 30.8784 | 0.4566 | stalled |
| r160_n4_ct10/sample_010 | P-Flow | 5 | 3 | 89.0216 | 47.7481 | stalled |
| r160_n4_ct10/sample_011 | P-Flow | 221 | 205 | 20.9169 | 0.5728 | stalled |
| r160_n4_ct10/sample_012 | P-Flow | 1000 | 896 | 20.5139 | 0.5369 | step_limit |
| r160_n4_ct10/sample_000 | DLR D-Flow | 1000 | 963 | 45.0321 | 19.8856 | step_limit |
| r160_n4_ct10/sample_001 | DLR D-Flow | 669 | 637 | 34.9219 | 15.0982 | stalled |
| r160_n4_ct10/sample_002 | DLR D-Flow | 1000 | 955 | 64.2210 | 30.2842 | step_limit |
| r160_n4_ct10/sample_003 | DLR D-Flow | 466 | 451 | 9.3049 | 5.3322 | stalled |
| r160_n4_ct10/sample_004 | DLR D-Flow | 343 | 331 | 5.9601 | 5.1171 | stalled |
| r160_n4_ct10/sample_005 | DLR D-Flow | 246 | 231 | 52.1189 | 27.1378 | stalled |
| r160_n4_ct10/sample_006 | DLR D-Flow | 1000 | 964 | 72.4191 | 38.2387 | step_limit |
| r160_n4_ct10/sample_007 | DLR D-Flow | 862 | 821 | 72.3254 | 34.4749 | stalled |
| r160_n4_ct10/sample_008 | DLR D-Flow | 592 | 569 | 28.0587 | 10.4447 | stalled |
| r160_n4_ct10/sample_009 | DLR D-Flow | 497 | 473 | 12.1118 | 5.6978 | stalled |
| r160_n4_ct10/sample_010 | DLR D-Flow | 1000 | 968 | 8.0124 | 5.3352 | step_limit |
| r160_n4_ct10/sample_011 | DLR D-Flow | 1000 | 960 | 52.1687 | 21.2673 | step_limit |
| r160_n4_ct10/sample_012 | DLR D-Flow | 465 | 444 | 12.0967 | 5.8965 | stalled |
| r160_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 4.0449 | 4.0859 | step_limit |
| r160_n1_mri4/sample_001 | D-Flow | 98 | 96 | 1.9304 | 5.1444 | stalled |
| r160_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 63.4605 | 62.0578 | step_limit |
| r160_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 4.3094 | 4.1941 | step_limit |
| r160_n1_mri4/sample_004 | D-Flow | 95 | 93 | 1.3660 | 5.0722 | stalled |
| r160_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 4.6966 | 3.6761 | step_limit |
| r160_n1_mri4/sample_006 | D-Flow | 999 | 997 | 6.1513 | 2.5685 | stalled |
| r160_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 5.9195 | 2.9119 | step_limit |
| r160_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 5.0649 | 3.0938 | step_limit |
| r160_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.8381 | 4.8050 | step_limit |
| r160_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 5.3706 | 2.9259 | step_limit |
| r160_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 3.2126 | 4.2359 | step_limit |
| r160_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 115.8501 | 41.9272 | step_limit |
| r160_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 5.5248 | 2.9066 | step_limit |
| r160_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 3.3508 | 4.2096 | step_limit |
| r160_n1_mri4/sample_015 | D-Flow | 927 | 925 | 14.1750 | 2.6621 | stalled |
| r160_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 5.4224 | 2.9470 | step_limit |
| r160_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 5.8427 | 3.0179 | step_limit |
| r160_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 4.7250 | 3.4012 | step_limit |
| r160_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 5.4460 | 2.9197 | step_limit |
| r160_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 4.7926 | 3.6101 | step_limit |
| r160_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 2.5499 | 4.4570 | step_limit |
| r160_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 3.8955 | 4.0866 | step_limit |
| r160_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 2.7021 | 4.0898 | step_limit |
| r160_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 2.4702 | 4.2528 | step_limit |
| r160_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 5.9035 | 2.5428 | step_limit |
| r160_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 4.8341 | 2.9697 | step_limit |
| r160_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 2.7645 | 4.1733 | step_limit |
| r160_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 5.0827 | 3.3040 | step_limit |
| r160_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 4.9720 | 3.9523 | step_limit |
| r160_n1_mri4/sample_000 | P-Flow | 284 | 260 | 8.1572 | 2.5190 | stalled |
| r160_n1_mri4/sample_001 | P-Flow | 223 | 204 | 8.1691 | 2.4045 | stalled |
| r160_n1_mri4/sample_002 | P-Flow | 64 | 58 | 2.0809 | 5.1914 | stalled |
| r160_n1_mri4/sample_003 | P-Flow | 234 | 212 | 8.2390 | 2.1689 | stalled |
| r160_n1_mri4/sample_004 | P-Flow | 235 | 222 | 6.6345 | 2.8132 | stalled |
| r160_n1_mri4/sample_005 | P-Flow | 125 | 106 | 1.6125 | 5.0476 | stalled |
| r160_n1_mri4/sample_006 | P-Flow | 210 | 194 | 7.0099 | 2.5685 | stalled |
| r160_n1_mri4/sample_007 | P-Flow | 233 | 216 | 7.4187 | 2.7957 | stalled |
| r160_n1_mri4/sample_008 | P-Flow | 303 | 276 | 7.7922 | 2.1794 | stalled |
| r160_n1_mri4/sample_009 | P-Flow | 207 | 196 | 6.9313 | 2.4912 | stalled |
| r160_n1_mri4/sample_010 | P-Flow | 217 | 204 | 7.5690 | 2.6469 | stalled |
| r160_n1_mri4/sample_011 | P-Flow | 278 | 252 | 7.4028 | 2.9415 | stalled |
| r160_n1_mri4/sample_012 | P-Flow | 243 | 224 | 6.2474 | 2.4170 | stalled |
| r160_n1_mri4/sample_013 | P-Flow | 342 | 305 | 7.4501 | 2.7348 | stalled |
| r160_n1_mri4/sample_014 | P-Flow | 47 | 40 | 2.9506 | 5.5476 | stalled |
| r160_n1_mri4/sample_015 | P-Flow | 299 | 266 | 7.6269 | 2.6621 | stalled |
| r160_n1_mri4/sample_016 | P-Flow | 241 | 223 | 7.8435 | 2.5008 | stalled |
| r160_n1_mri4/sample_017 | P-Flow | 316 | 283 | 8.3692 | 2.3269 | stalled |
| r160_n1_mri4/sample_018 | P-Flow | 305 | 271 | 7.8559 | 2.2185 | stalled |
| r160_n1_mri4/sample_019 | P-Flow | 190 | 182 | 6.6910 | 2.7869 | stalled |
| r160_n1_mri4/sample_020 | P-Flow | 304 | 273 | 7.2905 | 2.7565 | stalled |
| r160_n1_mri4/sample_021 | P-Flow | 11 | 8 | 81.9186 | 80.2160 | stalled |
| r160_n1_mri4/sample_022 | P-Flow | 195 | 183 | 8.0218 | 2.5228 | stalled |
| r160_n1_mri4/sample_023 | P-Flow | 181 | 174 | 6.0434 | 2.7794 | stalled |
| r160_n1_mri4/sample_024 | P-Flow | 185 | 177 | 7.0445 | 2.4185 | stalled |
| r160_n1_mri4/sample_025 | P-Flow | 196 | 188 | 8.2244 | 2.1349 | stalled |
| r160_n1_mri4/sample_026 | P-Flow | 246 | 224 | 7.0373 | 2.5065 | stalled |
| r160_n1_mri4/sample_027 | P-Flow | 239 | 219 | 6.7630 | 2.5303 | stalled |
| r160_n1_mri4/sample_028 | P-Flow | 162 | 155 | 7.8668 | 2.5913 | stalled |
| r160_n1_mri4/sample_029 | P-Flow | 301 | 272 | 7.6777 | 2.7182 | stalled |
| r160_n1_mri4/sample_000 | DLR D-Flow | 25 | 23 | 93.5753 | 93.2451 | stalled |
| r160_n1_mri4/sample_001 | DLR D-Flow | 148 | 139 | 106.6586 | 106.2309 | stalled |
| r160_n1_mri4/sample_002 | DLR D-Flow | 81 | 78 | 96.4294 | 96.0578 | stalled |
| r160_n1_mri4/sample_003 | DLR D-Flow | 442 | 424 | 1.4049 | 5.0398 | stalled |
| r160_n1_mri4/sample_004 | DLR D-Flow | 468 | 448 | 1.3203 | 5.0661 | stalled |
| r160_n1_mri4/sample_005 | DLR D-Flow | 149 | 141 | 7.2529 | 7.3501 | stalled |
| r160_n1_mri4/sample_006 | DLR D-Flow | 292 | 281 | 1.6223 | 5.0923 | stalled |
| r160_n1_mri4/sample_007 | DLR D-Flow | 163 | 155 | 1.2425 | 5.0421 | stalled |
| r160_n1_mri4/sample_008 | DLR D-Flow | 1000 | 959 | 2.2625 | 5.1813 | step_limit |
| r160_n1_mri4/sample_009 | DLR D-Flow | 323 | 315 | 2.1324 | 5.2354 | stalled |
| r160_n1_mri4/sample_010 | DLR D-Flow | 106 | 100 | 1.3319 | 5.0490 | stalled |
| r160_n1_mri4/sample_011 | DLR D-Flow | 1000 | 964 | 1.3798 | 5.0506 | step_limit |
| r160_n1_mri4/sample_012 | DLR D-Flow | 228 | 216 | 129.2611 | 55.3751 | stalled |
| r160_n1_mri4/sample_013 | DLR D-Flow | 1000 | 960 | 2.4209 | 5.2056 | step_limit |
| r160_n1_mri4/sample_014 | DLR D-Flow | 220 | 206 | 6.0310 | 6.9619 | stalled |
| r160_n1_mri4/sample_015 | DLR D-Flow | 658 | 629 | 2.2408 | 5.1355 | stalled |
| r160_n1_mri4/sample_016 | DLR D-Flow | 442 | 419 | 1.8282 | 5.1184 | stalled |
| r160_n1_mri4/sample_017 | DLR D-Flow | 78 | 72 | 98.4655 | 97.7334 | stalled |
| r160_n1_mri4/sample_018 | DLR D-Flow | 717 | 690 | 74.3425 | 70.4808 | stalled |
| r160_n1_mri4/sample_019 | DLR D-Flow | 477 | 457 | 1.3820 | 5.0831 | stalled |
| r160_n1_mri4/sample_020 | DLR D-Flow | 16 | 14 | 111.5383 | 110.4613 | stalled |
| r160_n1_mri4/sample_021 | DLR D-Flow | 727 | 703 | 3.8581 | 5.6130 | stalled |
| r160_n1_mri4/sample_022 | DLR D-Flow | 174 | 167 | 93.7686 | 93.4534 | stalled |
| r160_n1_mri4/sample_023 | DLR D-Flow | 763 | 733 | 1.5338 | 5.1135 | stalled |
| r160_n1_mri4/sample_024 | DLR D-Flow | 508 | 486 | 1.3783 | 5.0543 | stalled |
| r160_n1_mri4/sample_025 | DLR D-Flow | 478 | 459 | 1.3487 | 5.0399 | stalled |
| r160_n1_mri4/sample_026 | DLR D-Flow | 225 | 215 | 1.3684 | 5.0633 | stalled |
| r160_n1_mri4/sample_027 | DLR D-Flow | 245 | 235 | 2.1942 | 5.2255 | stalled |
| r160_n1_mri4/sample_028 | DLR D-Flow | 158 | 148 | 94.7532 | 94.4093 | stalled |
| r160_n1_mri4/sample_029 | DLR D-Flow | 274 | 263 | 1.2821 | 5.0393 | stalled |
| r160_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 2.4775 | 4.4632 | step_limit |
| r160_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.4357 | 5.0263 | step_limit |
| r160_n2_mri4/sample_002 | D-Flow | 345 | 343 | 54.2602 | 53.3578 | stalled |
| r160_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.7774 | 4.7854 | step_limit |
| r160_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.6556 | 4.8461 | step_limit |
| r160_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.5518 | 4.8851 | step_limit |
| r160_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.6234 | 4.7873 | step_limit |
| r160_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 6.6081 | 2.4740 | step_limit |
| r160_n2_mri4/sample_008 | D-Flow | 556 | 553 | 67.6810 | 67.6384 | stalled |
| r160_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 3.2833 | 4.1813 | step_limit |
| r160_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 3.4571 | 3.8027 | step_limit |
| r160_n2_mri4/sample_011 | D-Flow | 712 | 709 | 44.2293 | 42.5393 | stalled |
| r160_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.9619 | 4.7387 | step_limit |
| r160_n2_mri4/sample_000 | P-Flow | 179 | 167 | 7.8074 | 2.5963 | stalled |
| r160_n2_mri4/sample_001 | P-Flow | 303 | 274 | 6.9248 | 2.5892 | stalled |
| r160_n2_mri4/sample_002 | P-Flow | 198 | 183 | 7.2030 | 2.0029 | stalled |
| r160_n2_mri4/sample_003 | P-Flow | 175 | 167 | 7.9329 | 2.2573 | stalled |
| r160_n2_mri4/sample_004 | P-Flow | 173 | 165 | 7.5082 | 2.5579 | stalled |
| r160_n2_mri4/sample_005 | P-Flow | 169 | 165 | 7.2970 | 2.4505 | stalled |
| r160_n2_mri4/sample_006 | P-Flow | 186 | 172 | 6.7567 | 2.6634 | stalled |
| r160_n2_mri4/sample_007 | P-Flow | 193 | 182 | 7.8217 | 2.2798 | stalled |
| r160_n2_mri4/sample_008 | P-Flow | 177 | 166 | 7.9692 | 2.5521 | stalled |
| r160_n2_mri4/sample_009 | P-Flow | 208 | 194 | 7.7924 | 2.5753 | stalled |
| r160_n2_mri4/sample_010 | P-Flow | 196 | 185 | 7.1288 | 2.4328 | stalled |
| r160_n2_mri4/sample_011 | P-Flow | 243 | 216 | 7.1567 | 2.9172 | stalled |
| r160_n2_mri4/sample_012 | P-Flow | 226 | 212 | 8.2489 | 2.3493 | stalled |
| r160_n2_mri4/sample_000 | DLR D-Flow | 268 | 252 | 53.6919 | 51.0467 | stalled |
| r160_n2_mri4/sample_001 | DLR D-Flow | 160 | 152 | 1.8185 | 5.1412 | stalled |
| r160_n2_mri4/sample_002 | DLR D-Flow | 324 | 317 | 54.7734 | 53.9793 | stalled |
| r160_n2_mri4/sample_003 | DLR D-Flow | 309 | 298 | 49.9028 | 47.8601 | stalled |
| r160_n2_mri4/sample_004 | DLR D-Flow | 1000 | 976 | 64.0698 | 62.0504 | step_limit |
| r160_n2_mri4/sample_005 | DLR D-Flow | 1000 | 983 | 38.1105 | 35.9579 | step_limit |
| r160_n2_mri4/sample_006 | DLR D-Flow | 594 | 574 | 3.3166 | 5.5197 | stalled |
| r160_n2_mri4/sample_007 | DLR D-Flow | 696 | 668 | 2.7381 | 5.4171 | stalled |
| r160_n2_mri4/sample_008 | DLR D-Flow | 265 | 250 | 39.5462 | 36.7089 | stalled |
| r160_n2_mri4/sample_009 | DLR D-Flow | 99 | 96 | 2.5802 | 5.2509 | stalled |
| r160_n2_mri4/sample_010 | DLR D-Flow | 291 | 281 | 1.3908 | 5.0831 | stalled |
| r160_n2_mri4/sample_011 | DLR D-Flow | 233 | 223 | 45.0934 | 43.6372 | stalled |
| r160_n2_mri4/sample_012 | DLR D-Flow | 178 | 169 | 125.4872 | 124.1580 | stalled |
| r160_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 4.3050 | 4.0496 | step_limit |
| r160_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.8246 | 4.6586 | step_limit |
| r160_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 3.5591 | 4.4216 | step_limit |
| r160_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 5.5262 | 2.9470 | step_limit |
| r160_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.9381 | 4.7040 | step_limit |
| r160_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.3745 | 4.9733 | step_limit |
| r160_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.6761 | 4.8324 | step_limit |
| r160_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.3552 | 4.9158 | step_limit |
| r160_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.7630 | 4.6930 | step_limit |
| r160_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.2824 | 4.9624 | step_limit |
| r160_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 3.3807 | 4.4710 | step_limit |
| r160_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 27.9745 | 26.3795 | step_limit |
| r160_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.8952 | 4.5307 | step_limit |
| r160_n4_mri4/sample_000 | P-Flow | 180 | 174 | 7.0164 | 2.2142 | stalled |
| r160_n4_mri4/sample_001 | P-Flow | 199 | 190 | 6.2346 | 2.7179 | stalled |
| r160_n4_mri4/sample_002 | P-Flow | 194 | 185 | 7.3927 | 2.5562 | stalled |
| r160_n4_mri4/sample_003 | P-Flow | 178 | 171 | 6.3675 | 2.7467 | stalled |
| r160_n4_mri4/sample_004 | P-Flow | 190 | 182 | 6.9080 | 2.6734 | stalled |
| r160_n4_mri4/sample_005 | P-Flow | 198 | 190 | 7.3348 | 2.6485 | stalled |
| r160_n4_mri4/sample_006 | P-Flow | 222 | 213 | 7.2715 | 2.7597 | stalled |
| r160_n4_mri4/sample_007 | P-Flow | 262 | 243 | 7.0155 | 2.5444 | stalled |
| r160_n4_mri4/sample_008 | P-Flow | 258 | 237 | 6.6682 | 2.6772 | stalled |
| r160_n4_mri4/sample_009 | P-Flow | 234 | 216 | 6.6528 | 2.6881 | stalled |
| r160_n4_mri4/sample_010 | P-Flow | 194 | 186 | 6.8051 | 2.9348 | stalled |
| r160_n4_mri4/sample_011 | P-Flow | 181 | 174 | 6.8002 | 2.7744 | stalled |
| r160_n4_mri4/sample_012 | P-Flow | 192 | 184 | 6.5982 | 2.5277 | stalled |
| r160_n4_mri4/sample_000 | DLR D-Flow | 1000 | 969 | 18.4843 | 16.2471 | step_limit |
| r160_n4_mri4/sample_001 | DLR D-Flow | 534 | 510 | 62.2676 | 53.0827 | stalled |
| r160_n4_mri4/sample_002 | DLR D-Flow | 80 | 76 | 48.5464 | 44.3417 | stalled |
| r160_n4_mri4/sample_003 | DLR D-Flow | 81 | 77 | 76.6287 | 72.0031 | stalled |
| r160_n4_mri4/sample_004 | DLR D-Flow | 1000 | 969 | 52.6305 | 46.5739 | step_limit |
| r160_n4_mri4/sample_005 | DLR D-Flow | 744 | 718 | 15.1935 | 11.0361 | stalled |
| r160_n4_mri4/sample_006 | DLR D-Flow | 819 | 792 | 38.2774 | 36.8506 | stalled |
| r160_n4_mri4/sample_007 | DLR D-Flow | 610 | 593 | 71.9544 | 69.8919 | stalled |
| r160_n4_mri4/sample_008 | DLR D-Flow | 202 | 194 | 64.8898 | 56.6507 | stalled |
| r160_n4_mri4/sample_009 | DLR D-Flow | 1000 | 966 | 54.1543 | 49.2138 | step_limit |
| r160_n4_mri4/sample_010 | DLR D-Flow | 1000 | 970 | 54.9352 | 49.6020 | step_limit |
| r160_n4_mri4/sample_011 | DLR D-Flow | 1000 | 965 | 60.9095 | 56.5713 | step_limit |
| r160_n4_mri4/sample_012 | DLR D-Flow | 724 | 688 | 13.1474 | 10.9685 | stalled |
| r200_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 5.8238 | 3.2582 | step_limit |
| r200_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 9.2689 | 1.0719 | step_limit |
| r200_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 6.3251 | 2.7648 | step_limit |
| r200_n1_ct10/sample_003 | D-Flow | 759 | 757 | 32.2349 | 0.3486 | stalled |
| r200_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 10.3326 | 0.5376 | step_limit |
| r200_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 9.7531 | 0.7568 | step_limit |
| r200_n1_ct10/sample_006 | D-Flow | 1000 | 1000 | 10.3175 | 0.8194 | step_limit |
| r200_n1_ct10/sample_007 | D-Flow | 82 | 78 | 108.8873 | 95.4233 | stalled |
| r200_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 6.2604 | 3.0199 | step_limit |
| r200_n1_ct10/sample_009 | D-Flow | 898 | 896 | 23.5463 | 0.2104 | stalled |
| r200_n1_ct10/sample_010 | D-Flow | 1000 | 1000 | 32.4565 | 0.2264 | step_limit |
| r200_n1_ct10/sample_011 | D-Flow | 517 | 515 | 38.3645 | 0.2719 | stalled |
| r200_n1_ct10/sample_012 | D-Flow | 730 | 728 | 24.9720 | 0.2036 | stalled |
| r200_n1_ct10/sample_013 | D-Flow | 615 | 613 | 42.4129 | 0.2410 | stalled |
| r200_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 29.3925 | 0.2465 | step_limit |
| r200_n1_ct10/sample_015 | D-Flow | 583 | 581 | 25.5540 | 0.1374 | stalled |
| r200_n1_ct10/sample_016 | D-Flow | 634 | 632 | 37.7367 | 0.2397 | stalled |
| r200_n1_ct10/sample_017 | D-Flow | 586 | 584 | 33.3435 | 0.3331 | stalled |
| r200_n1_ct10/sample_018 | D-Flow | 852 | 850 | 35.9524 | 0.2719 | stalled |
| r200_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 28.9209 | 0.2527 | step_limit |
| r200_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 30.5689 | 0.2724 | step_limit |
| r200_n1_ct10/sample_021 | D-Flow | 102 | 99 | 106.4464 | 97.0021 | stalled |
| r200_n1_ct10/sample_022 | D-Flow | 937 | 935 | 29.0243 | 0.2921 | stalled |
| r200_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 10.5580 | 0.4854 | step_limit |
| r200_n1_ct10/sample_024 | D-Flow | 506 | 504 | 46.9445 | 0.1966 | stalled |
| r200_n1_ct10/sample_025 | D-Flow | 695 | 693 | 31.1947 | 0.2967 | stalled |
| r200_n1_ct10/sample_026 | D-Flow | 608 | 606 | 54.1970 | 0.3133 | stalled |
| r200_n1_ct10/sample_027 | D-Flow | 587 | 585 | 30.1257 | 0.3389 | stalled |
| r200_n1_ct10/sample_028 | D-Flow | 788 | 786 | 26.0739 | 0.1582 | stalled |
| r200_n1_ct10/sample_029 | D-Flow | 582 | 580 | 23.5495 | 0.1129 | stalled |
| r200_n1_ct10/sample_000 | P-Flow | 27 | 23 | 9.2427 | 5.6134 | stalled |
| r200_n1_ct10/sample_001 | P-Flow | 446 | 360 | 2.4875 | 4.9714 | stalled |
| r200_n1_ct10/sample_002 | P-Flow | 1000 | 952 | 21.4350 | 0.4598 | step_limit |
| r200_n1_ct10/sample_003 | P-Flow | 123 | 110 | 2.6449 | 4.9573 | stalled |
| r200_n1_ct10/sample_004 | P-Flow | 1000 | 964 | 20.0525 | 0.4907 | step_limit |
| r200_n1_ct10/sample_005 | P-Flow | 55 | 46 | 4.4610 | 5.1089 | stalled |
| r200_n1_ct10/sample_006 | P-Flow | 35 | 30 | 9.7938 | 7.3117 | stalled |
| r200_n1_ct10/sample_007 | P-Flow | 11 | 9 | 110.4350 | 94.9585 | stalled |
| r200_n1_ct10/sample_008 | P-Flow | 42 | 35 | 4.1400 | 5.1294 | stalled |
| r200_n1_ct10/sample_009 | P-Flow | 688 | 645 | 23.8212 | 0.2161 | stalled |
| r200_n1_ct10/sample_010 | P-Flow | 27 | 22 | 6.5053 | 5.3443 | stalled |
| r200_n1_ct10/sample_011 | P-Flow | 52 | 45 | 3.4682 | 5.0789 | stalled |
| r200_n1_ct10/sample_012 | P-Flow | 178 | 141 | 9.2544 | 5.6238 | stalled |
| r200_n1_ct10/sample_013 | P-Flow | 1000 | 792 | 2.1093 | 5.0226 | step_limit |
| r200_n1_ct10/sample_014 | P-Flow | 40 | 33 | 6.4274 | 6.4292 | stalled |
| r200_n1_ct10/sample_015 | P-Flow | 803 | 731 | 23.4850 | 0.1649 | stalled |
| r200_n1_ct10/sample_016 | P-Flow | 6 | 4 | 104.7074 | 97.1681 | stalled |
| r200_n1_ct10/sample_017 | P-Flow | 162 | 133 | 3.2621 | 5.0346 | stalled |
| r200_n1_ct10/sample_018 | P-Flow | 148 | 117 | 3.6378 | 5.0805 | stalled |
| r200_n1_ct10/sample_019 | P-Flow | 36 | 30 | 19.0340 | 6.7123 | stalled |
| r200_n1_ct10/sample_020 | P-Flow | 21 | 18 | 8.4420 | 5.5420 | stalled |
| r200_n1_ct10/sample_021 | P-Flow | 20 | 16 | 23.4740 | 9.4977 | stalled |
| r200_n1_ct10/sample_022 | P-Flow | 338 | 270 | 2.4440 | 5.0132 | stalled |
| r200_n1_ct10/sample_023 | P-Flow | 169 | 138 | 12.1328 | 6.8544 | stalled |
| r200_n1_ct10/sample_024 | P-Flow | 7 | 5 | 108.7626 | 94.0542 | stalled |
| r200_n1_ct10/sample_025 | P-Flow | 1000 | 947 | 22.9662 | 0.3456 | step_limit |
| r200_n1_ct10/sample_026 | P-Flow | 1000 | 889 | 22.1937 | 0.4478 | step_limit |
| r200_n1_ct10/sample_027 | P-Flow | 187 | 155 | 4.1299 | 5.3785 | stalled |
| r200_n1_ct10/sample_028 | P-Flow | 1000 | 947 | 21.3141 | 0.3045 | step_limit |
| r200_n1_ct10/sample_029 | P-Flow | 53 | 45 | 9.0945 | 6.3027 | stalled |
| r200_n1_ct10/sample_000 | DLR D-Flow | 233 | 222 | 1.8222 | 4.9732 | stalled |
| r200_n1_ct10/sample_001 | DLR D-Flow | 612 | 582 | 2.3955 | 4.9934 | stalled |
| r200_n1_ct10/sample_002 | DLR D-Flow | 189 | 180 | 1.4353 | 4.9744 | stalled |
| r200_n1_ct10/sample_003 | DLR D-Flow | 140 | 133 | 1.5982 | 4.9907 | stalled |
| r200_n1_ct10/sample_004 | DLR D-Flow | 542 | 517 | 2.3433 | 4.9953 | stalled |
| r200_n1_ct10/sample_005 | DLR D-Flow | 560 | 540 | 2.5898 | 4.9996 | stalled |
| r200_n1_ct10/sample_006 | DLR D-Flow | 346 | 335 | 6.8704 | 6.5540 | stalled |
| r200_n1_ct10/sample_007 | DLR D-Flow | 18 | 15 | 109.2969 | 96.2646 | stalled |
| r200_n1_ct10/sample_008 | DLR D-Flow | 283 | 264 | 1.5467 | 4.9925 | stalled |
| r200_n1_ct10/sample_009 | DLR D-Flow | 21 | 19 | 58.5996 | 30.7465 | stalled |
| r200_n1_ct10/sample_010 | DLR D-Flow | 539 | 515 | 1.4639 | 4.9847 | stalled |
| r200_n1_ct10/sample_011 | DLR D-Flow | 109 | 104 | 1.6713 | 4.9839 | stalled |
| r200_n1_ct10/sample_012 | DLR D-Flow | 388 | 367 | 1.9168 | 4.9927 | stalled |
| r200_n1_ct10/sample_013 | DLR D-Flow | 475 | 449 | 2.3372 | 4.9996 | stalled |
| r200_n1_ct10/sample_014 | DLR D-Flow | 20 | 17 | 111.8144 | 95.7671 | stalled |
| r200_n1_ct10/sample_015 | DLR D-Flow | 302 | 289 | 1.7879 | 4.9964 | stalled |
| r200_n1_ct10/sample_016 | DLR D-Flow | 304 | 290 | 1.5984 | 4.9974 | stalled |
| r200_n1_ct10/sample_017 | DLR D-Flow | 415 | 398 | 2.3205 | 4.9827 | stalled |
| r200_n1_ct10/sample_018 | DLR D-Flow | 250 | 237 | 1.9250 | 4.9799 | stalled |
| r200_n1_ct10/sample_019 | DLR D-Flow | 70 | 66 | 37.6939 | 16.0534 | stalled |
| r200_n1_ct10/sample_020 | DLR D-Flow | 594 | 567 | 1.9399 | 4.9916 | stalled |
| r200_n1_ct10/sample_021 | DLR D-Flow | 48 | 45 | 105.2266 | 97.9977 | stalled |
| r200_n1_ct10/sample_022 | DLR D-Flow | 371 | 354 | 1.7116 | 5.0092 | stalled |
| r200_n1_ct10/sample_023 | DLR D-Flow | 1000 | 954 | 3.1054 | 5.0856 | step_limit |
| r200_n1_ct10/sample_024 | DLR D-Flow | 397 | 373 | 2.1508 | 5.0017 | stalled |
| r200_n1_ct10/sample_025 | DLR D-Flow | 189 | 182 | 1.3015 | 4.9845 | stalled |
| r200_n1_ct10/sample_026 | DLR D-Flow | 517 | 493 | 2.2916 | 4.9869 | stalled |
| r200_n1_ct10/sample_027 | DLR D-Flow | 272 | 259 | 7.7080 | 6.3390 | stalled |
| r200_n1_ct10/sample_028 | DLR D-Flow | 304 | 291 | 2.9361 | 5.0400 | stalled |
| r200_n1_ct10/sample_029 | DLR D-Flow | 191 | 187 | 60.0751 | 38.1839 | stalled |
| r200_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 31.6420 | 0.1555 | step_limit |
| r200_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 16.0159 | 3.8624 | step_limit |
| r200_n2_ct10/sample_002 | D-Flow | 1000 | 1000 | 34.6405 | 0.3095 | step_limit |
| r200_n2_ct10/sample_003 | D-Flow | 66 | 64 | 91.7756 | 70.9918 | stalled |
| r200_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 5.2530 | 4.8313 | step_limit |
| r200_n2_ct10/sample_005 | D-Flow | 577 | 575 | 34.3729 | 0.2432 | stalled |
| r200_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 26.4947 | 0.3511 | step_limit |
| r200_n2_ct10/sample_007 | D-Flow | 792 | 790 | 30.0253 | 0.2263 | stalled |
| r200_n2_ct10/sample_008 | D-Flow | 777 | 775 | 32.3124 | 0.1819 | stalled |
| r200_n2_ct10/sample_009 | D-Flow | 686 | 684 | 34.8093 | 0.0973 | stalled |
| r200_n2_ct10/sample_010 | D-Flow | 1000 | 1000 | 29.2365 | 0.2981 | step_limit |
| r200_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 26.2085 | 0.5460 | step_limit |
| r200_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 28.1373 | 0.3969 | step_limit |
| r200_n2_ct10/sample_000 | P-Flow | 72 | 59 | 7.5397 | 5.4809 | stalled |
| r200_n2_ct10/sample_001 | P-Flow | 869 | 698 | 1.9044 | 4.9896 | stalled |
| r200_n2_ct10/sample_002 | P-Flow | 214 | 172 | 5.2065 | 5.2874 | stalled |
| r200_n2_ct10/sample_003 | P-Flow | 545 | 438 | 3.8156 | 5.0861 | stalled |
| r200_n2_ct10/sample_004 | P-Flow | 167 | 137 | 3.2998 | 4.9989 | stalled |
| r200_n2_ct10/sample_005 | P-Flow | 96 | 84 | 3.4053 | 4.9652 | stalled |
| r200_n2_ct10/sample_006 | P-Flow | 234 | 192 | 3.3175 | 5.0907 | stalled |
| r200_n2_ct10/sample_007 | P-Flow | 1000 | 836 | 19.9965 | 0.3884 | step_limit |
| r200_n2_ct10/sample_008 | P-Flow | 354 | 291 | 2.6417 | 5.0183 | stalled |
| r200_n2_ct10/sample_009 | P-Flow | 1000 | 933 | 22.6936 | 0.2720 | step_limit |
| r200_n2_ct10/sample_010 | P-Flow | 435 | 351 | 3.2212 | 4.9940 | stalled |
| r200_n2_ct10/sample_011 | P-Flow | 1000 | 856 | 19.5715 | 0.5873 | step_limit |
| r200_n2_ct10/sample_012 | P-Flow | 662 | 543 | 2.5308 | 5.0395 | stalled |
| r200_n2_ct10/sample_000 | DLR D-Flow | 412 | 397 | 5.3208 | 5.1086 | stalled |
| r200_n2_ct10/sample_001 | DLR D-Flow | 790 | 764 | 1.8778 | 4.9794 | stalled |
| r200_n2_ct10/sample_002 | DLR D-Flow | 1000 | 993 | 6.2728 | 5.1557 | step_limit |
| r200_n2_ct10/sample_003 | DLR D-Flow | 196 | 190 | 93.1483 | 68.6982 | stalled |
| r200_n2_ct10/sample_004 | DLR D-Flow | 1000 | 956 | 2.2292 | 5.0300 | step_limit |
| r200_n2_ct10/sample_005 | DLR D-Flow | 1000 | 971 | 2.1868 | 4.9917 | step_limit |
| r200_n2_ct10/sample_006 | DLR D-Flow | 1000 | 969 | 2.0917 | 5.0046 | step_limit |
| r200_n2_ct10/sample_007 | DLR D-Flow | 368 | 351 | 2.3141 | 4.9920 | stalled |
| r200_n2_ct10/sample_008 | DLR D-Flow | 677 | 654 | 32.3004 | 12.9911 | stalled |
| r200_n2_ct10/sample_009 | DLR D-Flow | 312 | 301 | 55.0361 | 36.5492 | stalled |
| r200_n2_ct10/sample_010 | DLR D-Flow | 422 | 403 | 2.2664 | 4.9909 | stalled |
| r200_n2_ct10/sample_011 | DLR D-Flow | 555 | 528 | 2.4740 | 5.0263 | stalled |
| r200_n2_ct10/sample_012 | DLR D-Flow | 346 | 332 | 2.7740 | 5.0000 | stalled |
| r200_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 13.0004 | 4.3322 | step_limit |
| r200_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 24.7699 | 0.6107 | step_limit |
| r200_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 31.0259 | 0.1496 | step_limit |
| r200_n4_ct10/sample_003 | D-Flow | 777 | 774 | 66.6433 | 35.4836 | stalled |
| r200_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 3.9214 | 4.9112 | step_limit |
| r200_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 12.1271 | 4.6320 | step_limit |
| r200_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 35.1577 | 0.4212 | step_limit |
| r200_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 32.4018 | 0.1331 | step_limit |
| r200_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 28.5976 | 10.6897 | step_limit |
| r200_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 3.0769 | 4.9793 | step_limit |
| r200_n4_ct10/sample_010 | D-Flow | 824 | 822 | 34.1384 | 0.2801 | stalled |
| r200_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 47.7612 | 27.9373 | step_limit |
| r200_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 26.6977 | 0.6246 | step_limit |
| r200_n4_ct10/sample_000 | P-Flow | 1000 | 907 | 20.3199 | 0.4617 | step_limit |
| r200_n4_ct10/sample_001 | P-Flow | 1000 | 879 | 19.8046 | 0.4884 | step_limit |
| r200_n4_ct10/sample_002 | P-Flow | 1000 | 912 | 21.5600 | 0.3084 | step_limit |
| r200_n4_ct10/sample_003 | P-Flow | 1000 | 901 | 20.1669 | 0.4002 | step_limit |
| r200_n4_ct10/sample_004 | P-Flow | 1000 | 906 | 20.8174 | 0.4847 | step_limit |
| r200_n4_ct10/sample_005 | P-Flow | 1000 | 930 | 23.1538 | 0.2141 | step_limit |
| r200_n4_ct10/sample_006 | P-Flow | 934 | 868 | 28.3475 | 0.4213 | stalled |
| r200_n4_ct10/sample_007 | P-Flow | 1000 | 904 | 22.2040 | 0.2916 | step_limit |
| r200_n4_ct10/sample_008 | P-Flow | 802 | 728 | 22.6502 | 0.3415 | stalled |
| r200_n4_ct10/sample_009 | P-Flow | 1000 | 949 | 20.2123 | 0.5422 | step_limit |
| r200_n4_ct10/sample_010 | P-Flow | 1000 | 932 | 22.5967 | 0.3175 | step_limit |
| r200_n4_ct10/sample_011 | P-Flow | 1000 | 877 | 20.0566 | 0.5878 | step_limit |
| r200_n4_ct10/sample_012 | P-Flow | 1000 | 975 | 19.8260 | 0.4342 | step_limit |
| r200_n4_ct10/sample_000 | DLR D-Flow | 1000 | 965 | 21.5581 | 7.7038 | step_limit |
| r200_n4_ct10/sample_001 | DLR D-Flow | 370 | 344 | 88.1719 | 48.0151 | stalled |
| r200_n4_ct10/sample_002 | DLR D-Flow | 1000 | 964 | 57.0361 | 27.0404 | step_limit |
| r200_n4_ct10/sample_003 | DLR D-Flow | 1000 | 965 | 70.8756 | 36.9400 | step_limit |
| r200_n4_ct10/sample_004 | DLR D-Flow | 1000 | 962 | 63.5934 | 31.9079 | step_limit |
| r200_n4_ct10/sample_005 | DLR D-Flow | 661 | 639 | 18.6243 | 7.1472 | stalled |
| r200_n4_ct10/sample_006 | DLR D-Flow | 1000 | 956 | 25.3088 | 8.2112 | step_limit |
| r200_n4_ct10/sample_007 | DLR D-Flow | 984 | 948 | 14.7878 | 6.1209 | stalled |
| r200_n4_ct10/sample_008 | DLR D-Flow | 577 | 553 | 65.5998 | 29.8984 | stalled |
| r200_n4_ct10/sample_009 | DLR D-Flow | 1000 | 954 | 76.2788 | 42.1786 | step_limit |
| r200_n4_ct10/sample_010 | DLR D-Flow | 152 | 146 | 89.6573 | 48.1960 | stalled |
| r200_n4_ct10/sample_011 | DLR D-Flow | 1000 | 960 | 71.6751 | 35.5848 | step_limit |
| r200_n4_ct10/sample_012 | DLR D-Flow | 1000 | 956 | 69.2424 | 37.9776 | step_limit |
| r200_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 4.2824 | 3.4100 | step_limit |
| r200_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 5.6998 | 2.9620 | step_limit |
| r200_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 3.4938 | 4.0763 | step_limit |
| r200_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 4.7977 | 3.1328 | step_limit |
| r200_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 3.5018 | 3.7659 | step_limit |
| r200_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 5.5399 | 2.8659 | step_limit |
| r200_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 2.3794 | 4.3616 | step_limit |
| r200_n1_mri4/sample_007 | D-Flow | 35 | 31 | 105.4951 | 103.9887 | stalled |
| r200_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 4.9062 | 2.7371 | step_limit |
| r200_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 2.4004 | 4.2517 | step_limit |
| r200_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 3.7632 | 3.7827 | step_limit |
| r200_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 4.6825 | 3.4419 | step_limit |
| r200_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 4.2722 | 3.6589 | step_limit |
| r200_n1_mri4/sample_013 | D-Flow | 65 | 63 | 79.5644 | 80.2494 | stalled |
| r200_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 2.1448 | 4.8634 | step_limit |
| r200_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 6.4208 | 2.4163 | step_limit |
| r200_n1_mri4/sample_016 | D-Flow | 66 | 64 | 100.3626 | 99.7323 | stalled |
| r200_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 5.5648 | 2.8684 | step_limit |
| r200_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 6.7451 | 2.3864 | step_limit |
| r200_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 4.4563 | 2.6648 | step_limit |
| r200_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 6.9123 | 2.4514 | step_limit |
| r200_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 6.9102 | 2.7853 | step_limit |
| r200_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 4.4314 | 3.7209 | step_limit |
| r200_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 3.2072 | 3.8502 | step_limit |
| r200_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 5.0504 | 2.7039 | step_limit |
| r200_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 3.0841 | 4.0599 | step_limit |
| r200_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 2.4298 | 4.2890 | step_limit |
| r200_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 4.5924 | 3.8371 | step_limit |
| r200_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 4.8494 | 2.9347 | step_limit |
| r200_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 2.4421 | 4.4968 | step_limit |
| r200_n1_mri4/sample_000 | P-Flow | 250 | 229 | 7.6190 | 2.6669 | stalled |
| r200_n1_mri4/sample_001 | P-Flow | 293 | 263 | 7.7812 | 2.3768 | stalled |
| r200_n1_mri4/sample_002 | P-Flow | 351 | 313 | 6.9805 | 2.4809 | stalled |
| r200_n1_mri4/sample_003 | P-Flow | 270 | 250 | 7.7817 | 2.5946 | stalled |
| r200_n1_mri4/sample_004 | P-Flow | 304 | 273 | 7.2619 | 2.3910 | stalled |
| r200_n1_mri4/sample_005 | P-Flow | 437 | 388 | 6.8377 | 2.5754 | stalled |
| r200_n1_mri4/sample_006 | P-Flow | 317 | 288 | 7.9622 | 2.4061 | stalled |
| r200_n1_mri4/sample_007 | P-Flow | 263 | 242 | 7.5951 | 2.6205 | stalled |
| r200_n1_mri4/sample_008 | P-Flow | 80 | 66 | 1.3132 | 5.0929 | stalled |
| r200_n1_mri4/sample_009 | P-Flow | 378 | 331 | 7.4429 | 2.6203 | stalled |
| r200_n1_mri4/sample_010 | P-Flow | 225 | 208 | 7.8112 | 2.4968 | stalled |
| r200_n1_mri4/sample_011 | P-Flow | 38 | 31 | 1.3710 | 5.0769 | stalled |
| r200_n1_mri4/sample_012 | P-Flow | 239 | 216 | 7.0374 | 2.5823 | stalled |
| r200_n1_mri4/sample_013 | P-Flow | 247 | 223 | 7.0088 | 2.5717 | stalled |
| r200_n1_mri4/sample_014 | P-Flow | 344 | 305 | 8.0521 | 2.7165 | stalled |
| r200_n1_mri4/sample_015 | P-Flow | 268 | 244 | 7.0437 | 2.4142 | stalled |
| r200_n1_mri4/sample_016 | P-Flow | 342 | 308 | 7.7523 | 2.5071 | stalled |
| r200_n1_mri4/sample_017 | P-Flow | 183 | 174 | 7.6599 | 2.7202 | stalled |
| r200_n1_mri4/sample_018 | P-Flow | 216 | 198 | 7.9460 | 2.3861 | stalled |
| r200_n1_mri4/sample_019 | P-Flow | 248 | 229 | 5.8924 | 2.3135 | stalled |
| r200_n1_mri4/sample_020 | P-Flow | 30 | 24 | 2.9496 | 5.5516 | stalled |
| r200_n1_mri4/sample_021 | P-Flow | 241 | 221 | 8.2609 | 2.2648 | stalled |
| r200_n1_mri4/sample_022 | P-Flow | 208 | 199 | 8.0921 | 2.5084 | stalled |
| r200_n1_mri4/sample_023 | P-Flow | 528 | 466 | 7.1674 | 2.2814 | stalled |
| r200_n1_mri4/sample_024 | P-Flow | 241 | 221 | 7.1442 | 2.4080 | stalled |
| r200_n1_mri4/sample_025 | P-Flow | 194 | 184 | 7.9757 | 2.2875 | stalled |
| r200_n1_mri4/sample_026 | P-Flow | 211 | 195 | 7.2297 | 2.3962 | stalled |
| r200_n1_mri4/sample_027 | P-Flow | 249 | 233 | 7.6744 | 2.7161 | stalled |
| r200_n1_mri4/sample_028 | P-Flow | 187 | 176 | 6.8301 | 2.5622 | stalled |
| r200_n1_mri4/sample_029 | P-Flow | 307 | 275 | 7.0774 | 2.7063 | stalled |
| r200_n1_mri4/sample_000 | DLR D-Flow | 337 | 322 | 1.2255 | 5.0332 | stalled |
| r200_n1_mri4/sample_001 | DLR D-Flow | 1000 | 961 | 1.5810 | 5.0734 | step_limit |
| r200_n1_mri4/sample_002 | DLR D-Flow | 245 | 232 | 1.1778 | 5.0463 | stalled |
| r200_n1_mri4/sample_003 | DLR D-Flow | 253 | 243 | 1.2049 | 5.0349 | stalled |
| r200_n1_mri4/sample_004 | DLR D-Flow | 365 | 350 | 44.7331 | 42.9860 | stalled |
| r200_n1_mri4/sample_005 | DLR D-Flow | 612 | 583 | 4.7815 | 6.4885 | stalled |
| r200_n1_mri4/sample_006 | DLR D-Flow | 462 | 451 | 5.9003 | 7.2827 | stalled |
| r200_n1_mri4/sample_007 | DLR D-Flow | 20 | 16 | 104.7011 | 102.9934 | stalled |
| r200_n1_mri4/sample_008 | DLR D-Flow | 590 | 577 | 1.5613 | 5.1344 | stalled |
| r200_n1_mri4/sample_009 | DLR D-Flow | 264 | 259 | 78.3401 | 77.5376 | stalled |
| r200_n1_mri4/sample_010 | DLR D-Flow | 374 | 362 | 1.2727 | 5.0484 | stalled |
| r200_n1_mri4/sample_011 | DLR D-Flow | 376 | 361 | 1.5609 | 5.0717 | stalled |
| r200_n1_mri4/sample_012 | DLR D-Flow | 166 | 156 | 99.5599 | 98.5845 | stalled |
| r200_n1_mri4/sample_013 | DLR D-Flow | 81 | 78 | 80.8478 | 81.6326 | stalled |
| r200_n1_mri4/sample_014 | DLR D-Flow | 94 | 90 | 101.6109 | 100.0563 | stalled |
| r200_n1_mri4/sample_015 | DLR D-Flow | 526 | 511 | 1.1910 | 5.0532 | stalled |
| r200_n1_mri4/sample_016 | DLR D-Flow | 319 | 308 | 1.9656 | 5.2368 | stalled |
| r200_n1_mri4/sample_017 | DLR D-Flow | 1000 | 961 | 2.1388 | 5.1883 | step_limit |
| r200_n1_mri4/sample_018 | DLR D-Flow | 245 | 233 | 1.2201 | 5.0381 | stalled |
| r200_n1_mri4/sample_019 | DLR D-Flow | 1000 | 965 | 2.3655 | 5.4376 | step_limit |
| r200_n1_mri4/sample_020 | DLR D-Flow | 463 | 443 | 1.2708 | 5.0476 | stalled |
| r200_n1_mri4/sample_021 | DLR D-Flow | 212 | 204 | 101.1160 | 99.9796 | stalled |
| r200_n1_mri4/sample_022 | DLR D-Flow | 10 | 8 | 101.5546 | 100.5336 | stalled |
| r200_n1_mri4/sample_023 | DLR D-Flow | 238 | 227 | 6.7271 | 7.8701 | stalled |
| r200_n1_mri4/sample_024 | DLR D-Flow | 490 | 469 | 1.2855 | 5.0589 | stalled |
| r200_n1_mri4/sample_025 | DLR D-Flow | 322 | 312 | 2.5713 | 5.4228 | stalled |
| r200_n1_mri4/sample_026 | DLR D-Flow | 985 | 950 | 1.8142 | 5.1389 | stalled |
| r200_n1_mri4/sample_027 | DLR D-Flow | 1000 | 949 | 1.3800 | 5.0488 | step_limit |
| r200_n1_mri4/sample_028 | DLR D-Flow | 218 | 215 | 14.5373 | 13.3961 | stalled |
| r200_n1_mri4/sample_029 | DLR D-Flow | 56 | 52 | 64.4308 | 65.4529 | stalled |
| r200_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.6133 | 4.8393 | step_limit |
| r200_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 51.2783 | 48.8413 | step_limit |
| r200_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 6.2726 | 3.3723 | step_limit |
| r200_n2_mri4/sample_003 | D-Flow | 911 | 908 | 58.1226 | 57.4403 | stalled |
| r200_n2_mri4/sample_004 | D-Flow | 817 | 814 | 59.2569 | 57.6158 | stalled |
| r200_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.5551 | 4.8441 | step_limit |
| r200_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.9902 | 4.5701 | step_limit |
| r200_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.5374 | 4.7132 | step_limit |
| r200_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.6186 | 5.1184 | step_limit |
| r200_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 2.5577 | 4.5602 | step_limit |
| r200_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.5954 | 4.9806 | step_limit |
| r200_n2_mri4/sample_011 | D-Flow | 180 | 180 | 3.1774 | 5.9031 | paused |
| r200_n2_mri4/sample_000 | P-Flow | 235 | 224 | 7.2918 | 2.6131 | stalled |
| r200_n2_mri4/sample_001 | P-Flow | 187 | 178 | 6.5717 | 2.6073 | stalled |
| r200_n2_mri4/sample_002 | P-Flow | 219 | 207 | 7.5497 | 2.5135 | stalled |
| r200_n2_mri4/sample_003 | P-Flow | 251 | 230 | 7.6346 | 2.7116 | stalled |
| r200_n2_mri4/sample_004 | P-Flow | 207 | 197 | 7.2545 | 2.7038 | stalled |
| r200_n2_mri4/sample_005 | P-Flow | 182 | 175 | 7.6615 | 2.5195 | stalled |
| r200_n2_mri4/sample_006 | P-Flow | 237 | 220 | 7.7198 | 2.3759 | stalled |
| r200_n2_mri4/sample_007 | P-Flow | 210 | 199 | 7.0339 | 2.4039 | stalled |
| r200_n2_mri4/sample_008 | P-Flow | 220 | 204 | 7.0958 | 2.7996 | stalled |
| r200_n2_mri4/sample_009 | P-Flow | 191 | 181 | 7.5806 | 2.4329 | stalled |
| r200_n2_mri4/sample_010 | P-Flow | 202 | 190 | 7.8953 | 2.4483 | stalled |
| r200_n2_mri4/sample_000 | DLR D-Flow | 158 | 153 | 6.3704 | 7.3201 | stalled |
| r200_n2_mri4/sample_001 | DLR D-Flow | 290 | 278 | 1.2714 | 5.0745 | stalled |
| r200_n2_mri4/sample_002 | DLR D-Flow | 1000 | 969 | 3.9780 | 5.8758 | step_limit |
| r200_n2_mri4/sample_003 | DLR D-Flow | 874 | 850 | 64.1578 | 62.3462 | stalled |
| r200_n2_mri4/sample_004 | DLR D-Flow | 616 | 596 | 61.3682 | 60.5120 | stalled |
| r200_n2_mri4/sample_005 | DLR D-Flow | 717 | 693 | 1.7698 | 5.0981 | stalled |
| r200_n2_mri4/sample_006 | DLR D-Flow | 153 | 145 | 78.2662 | 74.7808 | stalled |
| r200_n2_mri4/sample_007 | DLR D-Flow | 402 | 388 | 1.2310 | 5.0450 | stalled |
| r200_n2_mri4/sample_008 | DLR D-Flow | 633 | 592 | 4.3534 | 5.9597 | stalled |
| r200_n2_mri4/sample_009 | DLR D-Flow | 322 | 309 | 1.3219 | 5.0570 | stalled |
| r200_n2_mri4/sample_010 | DLR D-Flow | 566 | 531 | 2.6185 | 5.3455 | stalled |
| r200_n2_mri4/sample_011 | DLR D-Flow | 689 | 666 | 3.7711 | 6.0231 | stalled |
| r200_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.8553 | 4.6837 | step_limit |
| r200_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.4856 | 5.0672 | step_limit |
| r200_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.6879 | 4.8786 | step_limit |
| r200_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.6184 | 4.9095 | step_limit |
| r200_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.4077 | 4.9064 | step_limit |
| r200_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.5593 | 4.8829 | step_limit |
| r200_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.4536 | 4.9081 | step_limit |
| r200_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 5.0799 | 3.9528 | step_limit |
| r200_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.5388 | 4.8679 | step_limit |
| r200_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.4309 | 4.9783 | step_limit |
| r200_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.2668 | 4.9719 | step_limit |
| r200_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 15.6928 | 13.5412 | step_limit |
| r200_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.3209 | 5.0393 | step_limit |
| r200_n4_mri4/sample_013 | D-Flow | 349 | 349 | 2.0428 | 4.7350 | paused |
| r200_n4_mri4/sample_000 | P-Flow | 265 | 247 | 7.5587 | 2.4772 | stalled |
| r200_n4_mri4/sample_001 | P-Flow | 268 | 258 | 7.1758 | 2.4256 | stalled |
| r200_n4_mri4/sample_002 | P-Flow | 216 | 208 | 7.5354 | 2.5383 | stalled |
| r200_n4_mri4/sample_003 | P-Flow | 188 | 178 | 7.0810 | 2.9535 | stalled |
| r200_n4_mri4/sample_004 | P-Flow | 215 | 202 | 7.4097 | 2.1556 | stalled |
| r200_n4_mri4/sample_005 | P-Flow | 194 | 190 | 7.5034 | 2.2773 | stalled |
| r200_n4_mri4/sample_006 | P-Flow | 226 | 217 | 6.7631 | 3.0450 | stalled |
| r200_n4_mri4/sample_007 | P-Flow | 30 | 23 | 1.6125 | 5.0849 | stalled |
| r200_n4_mri4/sample_008 | P-Flow | 177 | 173 | 7.0681 | 2.2864 | stalled |
| r200_n4_mri4/sample_009 | P-Flow | 181 | 173 | 7.1076 | 2.9420 | stalled |
| r200_n4_mri4/sample_010 | P-Flow | 187 | 178 | 7.5986 | 2.5036 | stalled |
| r200_n4_mri4/sample_011 | P-Flow | 213 | 205 | 7.5340 | 2.6733 | stalled |
| r200_n4_mri4/sample_012 | P-Flow | 265 | 239 | 7.0214 | 2.6288 | stalled |
| r200_n4_mri4/sample_013 | P-Flow | 215 | 206 | 7.4773 | 2.2961 | stalled |
| r200_n4_mri4/sample_000 | DLR D-Flow | 419 | 409 | 14.1560 | 11.2990 | stalled |
| r200_n4_mri4/sample_001 | DLR D-Flow | 212 | 204 | 64.2972 | 58.5439 | stalled |
| r200_n4_mri4/sample_002 | DLR D-Flow | 645 | 623 | 63.3947 | 62.5895 | stalled |
| r200_n4_mri4/sample_003 | DLR D-Flow | 1000 | 974 | 68.5773 | 67.5175 | step_limit |
| r200_n4_mri4/sample_004 | DLR D-Flow | 895 | 863 | 64.4025 | 62.2130 | stalled |
| r200_n4_mri4/sample_005 | DLR D-Flow | 1000 | 958 | 23.9433 | 21.1919 | step_limit |
| r200_n4_mri4/sample_006 | DLR D-Flow | 247 | 235 | 28.3791 | 25.9489 | stalled |
| r200_n4_mri4/sample_007 | DLR D-Flow | 225 | 220 | 17.4105 | 14.0601 | stalled |
| r200_n4_mri4/sample_008 | DLR D-Flow | 1000 | 963 | 38.8975 | 35.2561 | step_limit |
| r200_n4_mri4/sample_009 | DLR D-Flow | 1000 | 967 | 60.6709 | 58.4388 | step_limit |
| r200_n4_mri4/sample_010 | DLR D-Flow | 1000 | 967 | 69.9303 | 67.8769 | step_limit |
| r200_n4_mri4/sample_011 | DLR D-Flow | 1000 | 964 | 79.1405 | 77.6425 | step_limit |
| r200_n4_mri4/sample_012 | DLR D-Flow | 1000 | 963 | 65.8299 | 65.0572 | step_limit |
| r200_n4_mri4/sample_013 | DLR D-Flow | 792 | 755 | 44.8607 | 40.1083 | stalled |
| r250_n1_ct10/sample_000 | D-Flow | 889 | 886 | 34.4868 | 0.1998 | stalled |
| r250_n1_ct10/sample_001 | D-Flow | 858 | 856 | 34.6633 | 0.1546 | stalled |
| r250_n1_ct10/sample_002 | D-Flow | 653 | 651 | 31.0051 | 0.2315 | stalled |
| r250_n1_ct10/sample_003 | D-Flow | 916 | 914 | 29.2915 | 0.2207 | stalled |
| r250_n1_ct10/sample_004 | D-Flow | 525 | 523 | 37.5471 | 0.2308 | stalled |
| r250_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 6.0736 | 3.0724 | step_limit |
| r250_n1_ct10/sample_006 | D-Flow | 724 | 722 | 46.6452 | 0.0843 | stalled |
| r250_n1_ct10/sample_007 | D-Flow | 20 | 18 | 99.7851 | 97.3519 | stalled |
| r250_n1_ct10/sample_008 | D-Flow | 537 | 535 | 40.1599 | 0.2640 | stalled |
| r250_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 5.9700 | 2.7403 | step_limit |
| r250_n1_ct10/sample_010 | D-Flow | 663 | 661 | 31.1336 | 0.2798 | stalled |
| r250_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 6.7765 | 2.3325 | step_limit |
| r250_n1_ct10/sample_012 | D-Flow | 9 | 7 | 99.7294 | 97.7011 | stalled |
| r250_n1_ct10/sample_013 | D-Flow | 552 | 550 | 33.6835 | 0.2678 | stalled |
| r250_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 24.8202 | 0.2167 | step_limit |
| r250_n1_ct10/sample_015 | D-Flow | 1000 | 1000 | 9.0310 | 4.7456 | step_limit |
| r250_n1_ct10/sample_016 | D-Flow | 449 | 447 | 41.2329 | 0.3196 | stalled |
| r250_n1_ct10/sample_017 | D-Flow | 659 | 656 | 29.0979 | 0.2191 | stalled |
| r250_n1_ct10/sample_018 | D-Flow | 210 | 206 | 107.2222 | 91.5524 | stalled |
| r250_n1_ct10/sample_019 | D-Flow | 1000 | 1000 | 5.7721 | 2.7846 | step_limit |
| r250_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 32.6517 | 0.2635 | step_limit |
| r250_n1_ct10/sample_021 | D-Flow | 1000 | 1000 | 9.9087 | 0.6323 | step_limit |
| r250_n1_ct10/sample_022 | D-Flow | 1000 | 1000 | 22.6799 | 0.2037 | step_limit |
| r250_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 8.9957 | 1.4248 | step_limit |
| r250_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 5.5901 | 2.9629 | step_limit |
| r250_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 24.0750 | 0.3256 | step_limit |
| r250_n1_ct10/sample_026 | D-Flow | 905 | 903 | 29.0356 | 0.2905 | stalled |
| r250_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 11.7710 | 0.3454 | step_limit |
| r250_n1_ct10/sample_028 | D-Flow | 750 | 748 | 27.0383 | 0.2302 | stalled |
| r250_n1_ct10/sample_029 | D-Flow | 561 | 558 | 37.0526 | 0.1908 | stalled |
| r250_n1_ct10/sample_000 | P-Flow | 1000 | 953 | 23.2626 | 0.2610 | step_limit |
| r250_n1_ct10/sample_001 | P-Flow | 43 | 38 | 2.1291 | 5.0078 | stalled |
| r250_n1_ct10/sample_002 | P-Flow | 90 | 71 | 6.7093 | 5.3113 | stalled |
| r250_n1_ct10/sample_003 | P-Flow | 266 | 215 | 1.9525 | 5.0034 | stalled |
| r250_n1_ct10/sample_004 | P-Flow | 1000 | 878 | 21.2890 | 0.3851 | step_limit |
| r250_n1_ct10/sample_005 | P-Flow | 6 | 4 | 112.0556 | 95.6561 | stalled |
| r250_n1_ct10/sample_006 | P-Flow | 1000 | 794 | 2.4818 | 4.9581 | step_limit |
| r250_n1_ct10/sample_007 | P-Flow | 252 | 214 | 2.9010 | 5.0397 | stalled |
| r250_n1_ct10/sample_008 | P-Flow | 864 | 763 | 21.6792 | 0.2952 | stalled |
| r250_n1_ct10/sample_009 | P-Flow | 1000 | 804 | 2.6054 | 5.0067 | step_limit |
| r250_n1_ct10/sample_010 | P-Flow | 25 | 20 | 3.1429 | 5.0157 | stalled |
| r250_n1_ct10/sample_011 | P-Flow | 100 | 83 | 4.0589 | 5.3936 | stalled |
| r250_n1_ct10/sample_012 | P-Flow | 1000 | 923 | 18.7602 | 0.4878 | step_limit |
| r250_n1_ct10/sample_013 | P-Flow | 59 | 50 | 1.6053 | 4.9887 | stalled |
| r250_n1_ct10/sample_014 | P-Flow | 30 | 25 | 2.5900 | 5.0163 | stalled |
| r250_n1_ct10/sample_015 | P-Flow | 803 | 744 | 25.0360 | 0.2278 | stalled |
| r250_n1_ct10/sample_016 | P-Flow | 1000 | 872 | 26.7132 | 0.3207 | step_limit |
| r250_n1_ct10/sample_017 | P-Flow | 287 | 234 | 2.0822 | 5.0049 | stalled |
| r250_n1_ct10/sample_018 | P-Flow | 440 | 347 | 2.8785 | 5.0037 | stalled |
| r250_n1_ct10/sample_019 | P-Flow | 1000 | 947 | 20.7643 | 0.2431 | step_limit |
| r250_n1_ct10/sample_020 | P-Flow | 88 | 67 | 3.1990 | 5.0831 | stalled |
| r250_n1_ct10/sample_021 | P-Flow | 33 | 27 | 2.5915 | 5.0111 | stalled |
| r250_n1_ct10/sample_022 | P-Flow | 1000 | 964 | 21.0529 | 0.2061 | step_limit |
| r250_n1_ct10/sample_023 | P-Flow | 23 | 19 | 2.0920 | 5.0138 | stalled |
| r250_n1_ct10/sample_024 | P-Flow | 1000 | 789 | 2.1674 | 5.0186 | step_limit |
| r250_n1_ct10/sample_025 | P-Flow | 298 | 243 | 3.2405 | 5.0459 | stalled |
| r250_n1_ct10/sample_026 | P-Flow | 117 | 92 | 3.3059 | 5.1168 | stalled |
| r250_n1_ct10/sample_027 | P-Flow | 301 | 240 | 3.6875 | 5.0472 | stalled |
| r250_n1_ct10/sample_028 | P-Flow | 36 | 30 | 12.6573 | 6.0914 | stalled |
| r250_n1_ct10/sample_029 | P-Flow | 1000 | 880 | 21.2949 | 0.2930 | step_limit |
| r250_n1_ct10/sample_000 | DLR D-Flow | 108 | 105 | 2.3534 | 4.9797 | stalled |
| r250_n1_ct10/sample_001 | DLR D-Flow | 123 | 117 | 8.5605 | 5.0427 | stalled |
| r250_n1_ct10/sample_002 | DLR D-Flow | 283 | 274 | 1.7136 | 4.9918 | stalled |
| r250_n1_ct10/sample_003 | DLR D-Flow | 574 | 548 | 2.9655 | 5.0040 | stalled |
| r250_n1_ct10/sample_004 | DLR D-Flow | 364 | 348 | 1.8188 | 4.9851 | stalled |
| r250_n1_ct10/sample_005 | DLR D-Flow | 567 | 543 | 1.4640 | 4.9940 | stalled |
| r250_n1_ct10/sample_006 | DLR D-Flow | 242 | 234 | 1.5377 | 4.9893 | stalled |
| r250_n1_ct10/sample_007 | DLR D-Flow | 14 | 12 | 99.6343 | 96.9283 | stalled |
| r250_n1_ct10/sample_008 | DLR D-Flow | 295 | 283 | 1.7775 | 4.9914 | stalled |
| r250_n1_ct10/sample_009 | DLR D-Flow | 497 | 475 | 1.8375 | 4.9885 | stalled |
| r250_n1_ct10/sample_010 | DLR D-Flow | 1000 | 967 | 7.4133 | 5.5052 | step_limit |
| r250_n1_ct10/sample_011 | DLR D-Flow | 446 | 429 | 1.8929 | 4.9842 | stalled |
| r250_n1_ct10/sample_012 | DLR D-Flow | 24 | 19 | 99.7313 | 97.7003 | stalled |
| r250_n1_ct10/sample_013 | DLR D-Flow | 203 | 191 | 1.7282 | 4.9982 | stalled |
| r250_n1_ct10/sample_014 | DLR D-Flow | 40 | 37 | 103.4251 | 98.0094 | stalled |
| r250_n1_ct10/sample_015 | DLR D-Flow | 110 | 106 | 2.3225 | 4.9955 | stalled |
| r250_n1_ct10/sample_016 | DLR D-Flow | 396 | 375 | 2.9822 | 4.9966 | stalled |
| r250_n1_ct10/sample_017 | DLR D-Flow | 818 | 787 | 1.6153 | 4.9986 | stalled |
| r250_n1_ct10/sample_018 | DLR D-Flow | 23 | 21 | 108.3840 | 93.6047 | stalled |
| r250_n1_ct10/sample_019 | DLR D-Flow | 373 | 360 | 1.3430 | 4.9940 | stalled |
| r250_n1_ct10/sample_020 | DLR D-Flow | 505 | 487 | 1.8129 | 4.9939 | stalled |
| r250_n1_ct10/sample_021 | DLR D-Flow | 375 | 363 | 2.2353 | 4.9940 | stalled |
| r250_n1_ct10/sample_022 | DLR D-Flow | 319 | 306 | 1.8280 | 4.9761 | stalled |
| r250_n1_ct10/sample_023 | DLR D-Flow | 70 | 63 | 111.4124 | 91.5981 | stalled |
| r250_n1_ct10/sample_024 | DLR D-Flow | 159 | 152 | 1.3879 | 4.9861 | stalled |
| r250_n1_ct10/sample_025 | DLR D-Flow | 379 | 364 | 1.5798 | 4.9967 | stalled |
| r250_n1_ct10/sample_026 | DLR D-Flow | 242 | 232 | 1.7429 | 4.9941 | stalled |
| r250_n1_ct10/sample_027 | DLR D-Flow | 115 | 110 | 5.4132 | 7.5505 | stalled |
| r250_n1_ct10/sample_028 | DLR D-Flow | 171 | 167 | 2.2202 | 5.0448 | stalled |
| r250_n1_ct10/sample_029 | DLR D-Flow | 418 | 403 | 1.9272 | 4.9847 | stalled |
| r250_n2_ct10/sample_000 | D-Flow | 536 | 534 | 44.7600 | 0.1375 | stalled |
| r250_n2_ct10/sample_001 | D-Flow | 744 | 741 | 33.3247 | 0.2340 | stalled |
| r250_n2_ct10/sample_002 | D-Flow | 365 | 363 | 50.8977 | 0.2429 | stalled |
| r250_n2_ct10/sample_003 | D-Flow | 585 | 583 | 35.5017 | 0.2149 | stalled |
| r250_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 4.8708 | 4.8718 | step_limit |
| r250_n2_ct10/sample_005 | D-Flow | 479 | 477 | 59.3933 | 0.2908 | stalled |
| r250_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 6.5241 | 3.7380 | step_limit |
| r250_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 8.7142 | 4.3273 | step_limit |
| r250_n2_ct10/sample_008 | D-Flow | 462 | 460 | 38.9023 | 0.2247 | stalled |
| r250_n2_ct10/sample_009 | D-Flow | 490 | 488 | 34.0390 | 0.2588 | stalled |
| r250_n2_ct10/sample_000 | P-Flow | 1000 | 902 | 25.5774 | 0.1564 | step_limit |
| r250_n2_ct10/sample_001 | P-Flow | 141 | 123 | 2.1284 | 5.0244 | stalled |
| r250_n2_ct10/sample_002 | P-Flow | 1000 | 927 | 23.3958 | 0.2627 | step_limit |
| r250_n2_ct10/sample_003 | P-Flow | 1000 | 936 | 21.7582 | 0.3199 | step_limit |
| r250_n2_ct10/sample_004 | P-Flow | 1000 | 799 | 2.6083 | 5.0231 | step_limit |
| r250_n2_ct10/sample_005 | P-Flow | 39 | 33 | 3.4179 | 5.1350 | stalled |
| r250_n2_ct10/sample_006 | P-Flow | 1000 | 856 | 22.3396 | 0.6523 | step_limit |
| r250_n2_ct10/sample_007 | P-Flow | 242 | 196 | 2.7799 | 4.9727 | stalled |
| r250_n2_ct10/sample_008 | P-Flow | 477 | 391 | 2.1474 | 4.9131 | stalled |
| r250_n2_ct10/sample_009 | P-Flow | 515 | 498 | 25.3275 | 0.2600 | stalled |
| r250_n2_ct10/sample_000 | DLR D-Flow | 293 | 280 | 6.3138 | 5.1395 | stalled |
| r250_n2_ct10/sample_001 | DLR D-Flow | 481 | 463 | 2.5891 | 4.9896 | stalled |
| r250_n2_ct10/sample_002 | DLR D-Flow | 294 | 284 | 9.3443 | 5.2719 | stalled |
| r250_n2_ct10/sample_003 | DLR D-Flow | 1000 | 979 | 3.2767 | 5.0487 | step_limit |
| r250_n2_ct10/sample_004 | DLR D-Flow | 737 | 714 | 2.5596 | 5.0125 | stalled |
| r250_n2_ct10/sample_005 | DLR D-Flow | 506 | 493 | 1.9327 | 5.0149 | stalled |
| r250_n2_ct10/sample_006 | DLR D-Flow | 606 | 587 | 2.7104 | 5.0769 | stalled |
| r250_n2_ct10/sample_007 | DLR D-Flow | 541 | 524 | 32.1867 | 15.0152 | stalled |
| r250_n2_ct10/sample_008 | DLR D-Flow | 624 | 600 | 2.2781 | 5.0204 | stalled |
| r250_n2_ct10/sample_009 | DLR D-Flow | 433 | 416 | 1.7401 | 4.9906 | stalled |
| r250_n4_ct10/sample_000 | D-Flow | 791 | 789 | 36.4709 | 0.2296 | stalled |
| r250_n4_ct10/sample_001 | D-Flow | 1000 | 1000 | 40.4305 | 0.2218 | step_limit |
| r250_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.5173 | 4.9749 | step_limit |
| r250_n4_ct10/sample_003 | D-Flow | 737 | 734 | 38.1310 | 0.2138 | stalled |
| r250_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 4.9505 | 4.9008 | step_limit |
| r250_n4_ct10/sample_005 | D-Flow | 754 | 752 | 44.0740 | 0.0586 | stalled |
| r250_n4_ct10/sample_006 | D-Flow | 704 | 702 | 40.4068 | 0.1569 | stalled |
| r250_n4_ct10/sample_007 | D-Flow | 847 | 844 | 36.2975 | 0.1796 | stalled |
| r250_n4_ct10/sample_008 | D-Flow | 813 | 811 | 35.1415 | 0.1909 | stalled |
| r250_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 33.0952 | 0.1708 | step_limit |
| r250_n4_ct10/sample_000 | P-Flow | 52 | 43 | 4.6097 | 4.9860 | stalled |
| r250_n4_ct10/sample_001 | P-Flow | 1000 | 880 | 21.3936 | 0.4581 | step_limit |
| r250_n4_ct10/sample_002 | P-Flow | 1000 | 947 | 22.9009 | 0.4135 | step_limit |
| r250_n4_ct10/sample_003 | P-Flow | 107 | 93 | 3.2797 | 4.8145 | stalled |
| r250_n4_ct10/sample_004 | P-Flow | 1000 | 978 | 21.4615 | 0.3571 | step_limit |
| r250_n4_ct10/sample_005 | P-Flow | 1000 | 928 | 22.3003 | 0.2079 | step_limit |
| r250_n4_ct10/sample_006 | P-Flow | 1000 | 908 | 24.3883 | 0.2193 | step_limit |
| r250_n4_ct10/sample_007 | P-Flow | 1000 | 897 | 21.9077 | 0.3125 | step_limit |
| r250_n4_ct10/sample_008 | P-Flow | 1000 | 909 | 22.1550 | 0.3250 | step_limit |
| r250_n4_ct10/sample_009 | P-Flow | 519 | 492 | 22.9530 | 0.2463 | stalled |
| r250_n4_ct10/sample_000 | DLR D-Flow | 515 | 495 | 52.3107 | 19.6429 | stalled |
| r250_n4_ct10/sample_001 | DLR D-Flow | 300 | 287 | 61.8976 | 30.6930 | stalled |
| r250_n4_ct10/sample_002 | DLR D-Flow | 1000 | 949 | 53.1505 | 22.6141 | step_limit |
| r250_n4_ct10/sample_003 | DLR D-Flow | 622 | 592 | 31.9029 | 13.0896 | stalled |
| r250_n4_ct10/sample_004 | DLR D-Flow | 699 | 672 | 48.9997 | 22.8646 | stalled |
| r250_n4_ct10/sample_005 | DLR D-Flow | 1000 | 959 | 25.6749 | 9.7535 | step_limit |
| r250_n4_ct10/sample_006 | DLR D-Flow | 941 | 898 | 30.4497 | 10.9487 | stalled |
| r250_n4_ct10/sample_007 | DLR D-Flow | 880 | 849 | 11.8344 | 5.6458 | stalled |
| r250_n4_ct10/sample_008 | DLR D-Flow | 259 | 251 | 86.5972 | 43.2792 | stalled |
| r250_n4_ct10/sample_009 | DLR D-Flow | 785 | 758 | 70.1546 | 32.6365 | stalled |
| r250_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 6.4191 | 2.5147 | step_limit |
| r250_n1_mri4/sample_001 | D-Flow | 1000 | 1000 | 6.5209 | 2.8828 | step_limit |
| r250_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 3.8746 | 3.9131 | step_limit |
| r250_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 6.7608 | 2.5278 | step_limit |
| r250_n1_mri4/sample_004 | D-Flow | 1000 | 1000 | 6.0199 | 2.4824 | step_limit |
| r250_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.9184 | 4.5399 | step_limit |
| r250_n1_mri4/sample_006 | D-Flow | 1000 | 1000 | 6.6053 | 2.7261 | step_limit |
| r250_n1_mri4/sample_007 | D-Flow | 43 | 41 | 99.6306 | 99.5907 | stalled |
| r250_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 2.6902 | 4.5074 | step_limit |
| r250_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.8983 | 4.5206 | step_limit |
| r250_n1_mri4/sample_010 | D-Flow | 1000 | 1000 | 2.6689 | 4.2907 | step_limit |
| r250_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 2.2375 | 4.6121 | step_limit |
| r250_n1_mri4/sample_012 | D-Flow | 13 | 10 | 99.5923 | 99.5382 | stalled |
| r250_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 4.1606 | 3.9577 | step_limit |
| r250_n1_mri4/sample_014 | D-Flow | 351 | 349 | 98.4362 | 97.7847 | stalled |
| r250_n1_mri4/sample_015 | D-Flow | 1000 | 1000 | 5.0458 | 4.0435 | step_limit |
| r250_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 6.1034 | 2.6558 | step_limit |
| r250_n1_mri4/sample_017 | D-Flow | 15 | 13 | 99.5248 | 99.4920 | stalled |
| r250_n1_mri4/sample_018 | D-Flow | 1000 | 1000 | 99.7364 | 99.7088 | step_limit |
| r250_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 6.2591 | 2.7060 | step_limit |
| r250_n1_mri4/sample_020 | D-Flow | 17 | 15 | 99.3698 | 99.3391 | stalled |
| r250_n1_mri4/sample_021 | D-Flow | 24 | 22 | 99.5007 | 99.4701 | stalled |
| r250_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 2.0149 | 4.4180 | step_limit |
| r250_n1_mri4/sample_023 | D-Flow | 77 | 75 | 99.3683 | 99.3359 | stalled |
| r250_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 4.8519 | 4.0144 | step_limit |
| r250_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 4.4921 | 3.9622 | step_limit |
| r250_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 4.7224 | 3.9191 | step_limit |
| r250_n1_mri4/sample_027 | D-Flow | 134 | 132 | 99.6480 | 99.6220 | stalled |
| r250_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 2.8561 | 4.4600 | step_limit |
| r250_n1_mri4/sample_029 | D-Flow | 11 | 9 | 99.5092 | 99.3422 | stalled |
| r250_n1_mri4/sample_000 | P-Flow | 308 | 273 | 7.6330 | 2.5097 | stalled |
| r250_n1_mri4/sample_001 | P-Flow | 279 | 255 | 7.3236 | 2.7844 | stalled |
| r250_n1_mri4/sample_002 | P-Flow | 39 | 31 | 1.4428 | 5.0884 | stalled |
| r250_n1_mri4/sample_003 | P-Flow | 232 | 216 | 7.6706 | 2.5078 | stalled |
| r250_n1_mri4/sample_004 | P-Flow | 270 | 243 | 7.0639 | 2.3247 | stalled |
| r250_n1_mri4/sample_005 | P-Flow | 20 | 16 | 1.7479 | 5.1669 | stalled |
| r250_n1_mri4/sample_006 | P-Flow | 268 | 242 | 7.4147 | 2.7235 | stalled |
| r250_n1_mri4/sample_007 | P-Flow | 279 | 250 | 8.4236 | 2.4306 | stalled |
| r250_n1_mri4/sample_008 | P-Flow | 43 | 35 | 1.5093 | 5.0696 | stalled |
| r250_n1_mri4/sample_009 | P-Flow | 57 | 47 | 1.3249 | 5.0657 | stalled |
| r250_n1_mri4/sample_010 | P-Flow | 466 | 413 | 6.9113 | 2.7515 | stalled |
| r250_n1_mri4/sample_011 | P-Flow | 238 | 215 | 7.4027 | 2.7372 | stalled |
| r250_n1_mri4/sample_012 | P-Flow | 450 | 402 | 8.0642 | 2.6015 | stalled |
| r250_n1_mri4/sample_013 | P-Flow | 216 | 199 | 6.7940 | 2.7051 | stalled |
| r250_n1_mri4/sample_014 | P-Flow | 263 | 236 | 7.8971 | 2.4865 | stalled |
| r250_n1_mri4/sample_015 | P-Flow | 668 | 579 | 7.4583 | 2.7220 | stalled |
| r250_n1_mri4/sample_016 | P-Flow | 361 | 312 | 7.4381 | 2.6532 | stalled |
| r250_n1_mri4/sample_017 | P-Flow | 380 | 336 | 7.9621 | 2.3839 | stalled |
| r250_n1_mri4/sample_018 | P-Flow | 313 | 292 | 8.3438 | 2.2464 | stalled |
| r250_n1_mri4/sample_019 | P-Flow | 247 | 225 | 7.0159 | 2.4815 | stalled |
| r250_n1_mri4/sample_020 | P-Flow | 96 | 81 | 1.3607 | 5.0512 | stalled |
| r250_n1_mri4/sample_021 | P-Flow | 33 | 25 | 1.6540 | 5.0899 | stalled |
| r250_n1_mri4/sample_022 | P-Flow | 37 | 31 | 1.5587 | 5.1207 | stalled |
| r250_n1_mri4/sample_023 | P-Flow | 219 | 207 | 8.2839 | 2.3458 | stalled |
| r250_n1_mri4/sample_024 | P-Flow | 250 | 236 | 7.3632 | 2.6682 | stalled |
| r250_n1_mri4/sample_025 | P-Flow | 22 | 18 | 1.8961 | 5.1533 | stalled |
| r250_n1_mri4/sample_026 | P-Flow | 401 | 359 | 7.7991 | 2.5699 | stalled |
| r250_n1_mri4/sample_027 | P-Flow | 436 | 382 | 8.3282 | 2.6749 | stalled |
| r250_n1_mri4/sample_028 | P-Flow | 390 | 332 | 7.8323 | 2.5058 | stalled |
| r250_n1_mri4/sample_029 | P-Flow | 40 | 32 | 1.1302 | 5.0413 | stalled |
| r250_n1_mri4/sample_000 | DLR D-Flow | 424 | 405 | 1.2247 | 5.0304 | stalled |
| r250_n1_mri4/sample_001 | DLR D-Flow | 264 | 254 | 1.6262 | 5.0950 | stalled |
| r250_n1_mri4/sample_002 | DLR D-Flow | 227 | 217 | 1.2532 | 5.0450 | stalled |
| r250_n1_mri4/sample_003 | DLR D-Flow | 496 | 470 | 1.4124 | 5.0623 | stalled |
| r250_n1_mri4/sample_004 | DLR D-Flow | 379 | 363 | 1.1888 | 5.0399 | stalled |
| r250_n1_mri4/sample_005 | DLR D-Flow | 408 | 397 | 4.1955 | 5.7568 | stalled |
| r250_n1_mri4/sample_006 | DLR D-Flow | 108 | 104 | 1.0901 | 5.0336 | stalled |
| r250_n1_mri4/sample_007 | DLR D-Flow | 19 | 16 | 99.4738 | 99.4213 | stalled |
| r250_n1_mri4/sample_008 | DLR D-Flow | 432 | 415 | 1.5002 | 5.0608 | stalled |
| r250_n1_mri4/sample_009 | DLR D-Flow | 288 | 276 | 1.3360 | 5.0453 | stalled |
| r250_n1_mri4/sample_010 | DLR D-Flow | 127 | 122 | 3.6822 | 5.9675 | stalled |
| r250_n1_mri4/sample_011 | DLR D-Flow | 442 | 424 | 1.3805 | 5.0586 | stalled |
| r250_n1_mri4/sample_012 | DLR D-Flow | 18 | 15 | 99.5921 | 99.5380 | stalled |
| r250_n1_mri4/sample_013 | DLR D-Flow | 388 | 367 | 1.2696 | 5.0495 | stalled |
| r250_n1_mri4/sample_014 | DLR D-Flow | 39 | 35 | 99.9711 | 99.8155 | stalled |
| r250_n1_mri4/sample_015 | DLR D-Flow | 331 | 318 | 1.1205 | 5.0287 | stalled |
| r250_n1_mri4/sample_016 | DLR D-Flow | 317 | 300 | 1.7405 | 5.1185 | stalled |
| r250_n1_mri4/sample_017 | DLR D-Flow | 21 | 19 | 99.5097 | 99.4764 | stalled |
| r250_n1_mri4/sample_018 | DLR D-Flow | 99 | 94 | 99.7383 | 99.4109 | stalled |
| r250_n1_mri4/sample_019 | DLR D-Flow | 706 | 688 | 1.7597 | 5.1965 | stalled |
| r250_n1_mri4/sample_020 | DLR D-Flow | 742 | 718 | 1.2416 | 5.0322 | stalled |
| r250_n1_mri4/sample_021 | DLR D-Flow | 14 | 11 | 97.8121 | 97.6283 | stalled |
| r250_n1_mri4/sample_022 | DLR D-Flow | 425 | 410 | 1.4850 | 5.0927 | stalled |
| r250_n1_mri4/sample_023 | DLR D-Flow | 20 | 18 | 99.2806 | 99.2421 | stalled |
| r250_n1_mri4/sample_024 | DLR D-Flow | 292 | 281 | 1.2012 | 5.0380 | stalled |
| r250_n1_mri4/sample_025 | DLR D-Flow | 362 | 352 | 1.4151 | 5.0563 | stalled |
| r250_n1_mri4/sample_026 | DLR D-Flow | 203 | 194 | 1.3572 | 5.0541 | stalled |
| r250_n1_mri4/sample_027 | DLR D-Flow | 13 | 10 | 99.6201 | 99.5929 | stalled |
| r250_n1_mri4/sample_028 | DLR D-Flow | 41 | 39 | 2.1000 | 5.2589 | stalled |
| r250_n1_mri4/sample_029 | DLR D-Flow | 383 | 363 | 1.1867 | 5.0424 | stalled |
| r250_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.3819 | 4.9223 | step_limit |
| r250_n2_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.5856 | 4.8286 | step_limit |
| r250_n2_mri4/sample_002 | D-Flow | 938 | 934 | 32.3074 | 30.6291 | stalled |
| r250_n2_mri4/sample_003 | D-Flow | 408 | 404 | 47.5446 | 47.2850 | stalled |
| r250_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 35.8418 | 34.8432 | step_limit |
| r250_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.8647 | 4.6399 | step_limit |
| r250_n2_mri4/sample_006 | D-Flow | 953 | 950 | 6.4463 | 2.5796 | stalled |
| r250_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 2.1524 | 4.3788 | step_limit |
| r250_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.9356 | 4.4390 | step_limit |
| r250_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 3.9448 | 4.2832 | step_limit |
| r250_n2_mri4/sample_000 | P-Flow | 191 | 179 | 7.2044 | 2.5935 | stalled |
| r250_n2_mri4/sample_001 | P-Flow | 206 | 194 | 7.4499 | 2.4921 | stalled |
| r250_n2_mri4/sample_002 | P-Flow | 210 | 198 | 7.6666 | 2.5181 | stalled |
| r250_n2_mri4/sample_003 | P-Flow | 177 | 169 | 7.5245 | 2.6355 | stalled |
| r250_n2_mri4/sample_004 | P-Flow | 190 | 179 | 7.4271 | 2.6531 | stalled |
| r250_n2_mri4/sample_005 | P-Flow | 205 | 193 | 6.5884 | 2.5593 | stalled |
| r250_n2_mri4/sample_006 | P-Flow | 246 | 227 | 7.6016 | 2.5796 | stalled |
| r250_n2_mri4/sample_007 | P-Flow | 175 | 167 | 7.3988 | 2.7112 | stalled |
| r250_n2_mri4/sample_008 | P-Flow | 242 | 223 | 6.7976 | 2.2237 | stalled |
| r250_n2_mri4/sample_009 | P-Flow | 253 | 237 | 7.8069 | 2.3455 | stalled |
| r250_n2_mri4/sample_000 | DLR D-Flow | 281 | 273 | 36.1556 | 30.2782 | stalled |
| r250_n2_mri4/sample_001 | DLR D-Flow | 205 | 199 | 1.4513 | 5.0790 | stalled |
| r250_n2_mri4/sample_002 | DLR D-Flow | 186 | 179 | 44.8866 | 43.8782 | stalled |
| r250_n2_mri4/sample_003 | DLR D-Flow | 251 | 241 | 48.4345 | 48.3066 | stalled |
| r250_n2_mri4/sample_004 | DLR D-Flow | 124 | 118 | 2.9982 | 5.6016 | stalled |
| r250_n2_mri4/sample_005 | DLR D-Flow | 175 | 168 | 52.7420 | 50.9235 | stalled |
| r250_n2_mri4/sample_006 | DLR D-Flow | 1000 | 975 | 28.1692 | 24.7700 | step_limit |
| r250_n2_mri4/sample_007 | DLR D-Flow | 39 | 36 | 71.3295 | 68.1266 | stalled |
| r250_n2_mri4/sample_008 | DLR D-Flow | 329 | 316 | 3.7107 | 5.9414 | stalled |
| r250_n2_mri4/sample_009 | DLR D-Flow | 334 | 323 | 1.5086 | 5.1014 | stalled |
| r250_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.6728 | 4.7358 | step_limit |
| r250_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.5961 | 4.8130 | step_limit |
| r250_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.3495 | 4.9583 | step_limit |
| r250_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.9407 | 4.6308 | step_limit |
| r250_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.8363 | 4.7859 | step_limit |
| r250_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.5791 | 4.7792 | step_limit |
| r250_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 3.6332 | 4.5488 | step_limit |
| r250_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.7366 | 4.7683 | step_limit |
| r250_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.3845 | 4.8816 | step_limit |
| r250_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.4804 | 4.8576 | step_limit |
| r250_n4_mri4/sample_000 | P-Flow | 597 | 512 | 7.0455 | 2.5870 | stalled |
| r250_n4_mri4/sample_001 | P-Flow | 341 | 309 | 7.5117 | 2.6629 | stalled |
| r250_n4_mri4/sample_002 | P-Flow | 196 | 188 | 7.3936 | 2.8158 | stalled |
| r250_n4_mri4/sample_003 | P-Flow | 248 | 240 | 7.1148 | 2.6616 | stalled |
| r250_n4_mri4/sample_004 | P-Flow | 256 | 241 | 7.6712 | 2.6521 | stalled |
| r250_n4_mri4/sample_005 | P-Flow | 253 | 236 | 6.8110 | 2.8855 | stalled |
| r250_n4_mri4/sample_006 | P-Flow | 336 | 318 | 7.5167 | 2.6415 | stalled |
| r250_n4_mri4/sample_007 | P-Flow | 227 | 219 | 7.7638 | 2.5528 | stalled |
| r250_n4_mri4/sample_008 | P-Flow | 364 | 338 | 7.5763 | 2.7281 | stalled |
| r250_n4_mri4/sample_009 | P-Flow | 463 | 415 | 7.6600 | 2.7234 | stalled |
| r250_n4_mri4/sample_000 | DLR D-Flow | 766 | 733 | 20.7505 | 19.2443 | stalled |
| r250_n4_mri4/sample_001 | DLR D-Flow | 595 | 569 | 12.2399 | 10.1628 | stalled |
| r250_n4_mri4/sample_002 | DLR D-Flow | 689 | 662 | 51.1084 | 49.2440 | stalled |
| r250_n4_mri4/sample_003 | DLR D-Flow | 961 | 914 | 28.2149 | 26.4135 | stalled |
| r250_n4_mri4/sample_004 | DLR D-Flow | 386 | 376 | 40.2909 | 38.7065 | stalled |
| r250_n4_mri4/sample_005 | DLR D-Flow | 1000 | 941 | 49.0464 | 46.3452 | step_limit |
| r250_n4_mri4/sample_006 | DLR D-Flow | 1000 | 964 | 54.6677 | 50.7947 | step_limit |
| r250_n4_mri4/sample_007 | DLR D-Flow | 825 | 790 | 7.0921 | 7.3991 | stalled |
| r250_n4_mri4/sample_008 | DLR D-Flow | 86 | 81 | 77.6081 | 75.7026 | stalled |
| r250_n4_mri4/sample_009 | DLR D-Flow | 90 | 87 | 64.4342 | 62.1278 | stalled |

Terminal method runs: 1953/3240. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.
