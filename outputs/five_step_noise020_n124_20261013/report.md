# Five-step Euler pilot

20% relative measurement noise; CT10 and Cartesian MRI4; 30 test samples per acquisition at each resolution; circle count in [1, 2, 4]. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 8, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_n1_ct10/sample_000 | D-Flow | 854 | 852 | 180.4075 | 1.0330 | stalled |
| r64_n1_ct10/sample_001 | D-Flow | 451 | 449 | 225.0214 | 1.9716 | stalled |
| r64_n1_ct10/sample_002 | D-Flow | 869 | 867 | 91.3818 | 2.4184 | stalled |
| r64_n1_ct10/sample_003 | D-Flow | 888 | 886 | 119.7866 | 0.7809 | stalled |
| r64_n1_ct10/sample_004 | D-Flow | 1000 | 1000 | 106.4501 | 1.4572 | step_limit |
| r64_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 127.6398 | 1.7628 | step_limit |
| r64_n1_ct10/sample_006 | D-Flow | 718 | 716 | 121.1938 | 2.0672 | stalled |
| r64_n1_ct10/sample_007 | D-Flow | 712 | 710 | 145.9722 | 1.7441 | stalled |
| r64_n1_ct10/sample_008 | D-Flow | 920 | 918 | 126.6869 | 0.6078 | stalled |
| r64_n1_ct10/sample_009 | D-Flow | 1000 | 1000 | 108.2194 | 2.7277 | step_limit |
| r64_n1_ct10/sample_010 | D-Flow | 848 | 845 | 143.8713 | 1.1768 | stalled |
| r64_n1_ct10/sample_011 | D-Flow | 450 | 448 | 10.1140 | 17.4179 | stalled |
| r64_n1_ct10/sample_012 | D-Flow | 573 | 571 | 146.5852 | 1.2254 | stalled |
| r64_n1_ct10/sample_013 | D-Flow | 949 | 947 | 95.3637 | 1.2118 | stalled |
| r64_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 128.0788 | 1.4790 | step_limit |
| r64_n1_ct10/sample_015 | D-Flow | 750 | 747 | 100.4292 | 88.8949 | stalled |
| r64_n1_ct10/sample_016 | D-Flow | 1000 | 1000 | 93.2773 | 2.0401 | step_limit |
| r64_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 105.0790 | 1.6397 | step_limit |
| r64_n1_ct10/sample_018 | D-Flow | 780 | 778 | 153.5706 | 1.9196 | stalled |
| r64_n1_ct10/sample_019 | D-Flow | 557 | 554 | 9.7444 | 17.0941 | stalled |
| r64_n1_ct10/sample_020 | D-Flow | 384 | 382 | 8.8751 | 16.9469 | stalled |
| r64_n1_ct10/sample_021 | D-Flow | 948 | 946 | 131.5313 | 2.1315 | stalled |
| r64_n1_ct10/sample_022 | D-Flow | 628 | 626 | 178.9511 | 0.9015 | stalled |
| r64_n1_ct10/sample_023 | D-Flow | 558 | 556 | 12.0300 | 16.6622 | stalled |
| r64_n1_ct10/sample_024 | D-Flow | 634 | 632 | 182.4999 | 1.7571 | stalled |
| r64_n1_ct10/sample_025 | D-Flow | 759 | 756 | 134.6566 | 1.2662 | stalled |
| r64_n1_ct10/sample_026 | D-Flow | 912 | 910 | 120.7834 | 2.2164 | stalled |
| r64_n1_ct10/sample_027 | D-Flow | 476 | 473 | 8.3446 | 17.1703 | stalled |
| r64_n1_ct10/sample_028 | D-Flow | 1000 | 1000 | 100.2953 | 1.7024 | step_limit |
| r64_n1_ct10/sample_029 | D-Flow | 774 | 772 | 127.2457 | 1.2522 | stalled |
| r64_n1_ct10/sample_000 | P-Flow | 621 | 507 | 23.4913 | 21.7093 | stalled |
| r64_n1_ct10/sample_001 | P-Flow | 811 | 773 | 127.3556 | 1.9716 | stalled |
| r64_n1_ct10/sample_002 | P-Flow | 284 | 237 | 10.6565 | 17.6684 | stalled |
| r64_n1_ct10/sample_003 | P-Flow | 29 | 25 | 5.0077 | 19.5897 | stalled |
| r64_n1_ct10/sample_004 | P-Flow | 677 | 655 | 110.9155 | 1.0723 | stalled |
| r64_n1_ct10/sample_005 | P-Flow | 32 | 26 | 9.5130 | 19.5896 | stalled |
| r64_n1_ct10/sample_006 | P-Flow | 584 | 576 | 116.0167 | 2.0672 | stalled |
| r64_n1_ct10/sample_007 | P-Flow | 587 | 581 | 135.7424 | 1.7441 | stalled |
| r64_n1_ct10/sample_008 | P-Flow | 1000 | 910 | 105.6197 | 0.7671 | step_limit |
| r64_n1_ct10/sample_009 | P-Flow | 1000 | 926 | 72.1533 | 3.0874 | step_limit |
| r64_n1_ct10/sample_010 | P-Flow | 50 | 43 | 12.5287 | 20.4948 | stalled |
| r64_n1_ct10/sample_011 | P-Flow | 572 | 565 | 145.0612 | 1.5085 | stalled |
| r64_n1_ct10/sample_012 | P-Flow | 616 | 587 | 112.9850 | 1.2263 | stalled |
| r64_n1_ct10/sample_013 | P-Flow | 57 | 48 | 4.7369 | 19.2199 | stalled |
| r64_n1_ct10/sample_014 | P-Flow | 29 | 24 | 5.8670 | 19.7612 | stalled |
| r64_n1_ct10/sample_015 | P-Flow | 583 | 565 | 88.7000 | 1.4450 | stalled |
| r64_n1_ct10/sample_016 | P-Flow | 52 | 44 | 6.8091 | 20.0843 | stalled |
| r64_n1_ct10/sample_017 | P-Flow | 137 | 121 | 8.8424 | 18.5850 | stalled |
| r64_n1_ct10/sample_018 | P-Flow | 41 | 35 | 3.0396 | 19.7297 | stalled |
| r64_n1_ct10/sample_019 | P-Flow | 41 | 35 | 3.4606 | 19.5527 | stalled |
| r64_n1_ct10/sample_020 | P-Flow | 26 | 22 | 6.7582 | 19.4440 | stalled |
| r64_n1_ct10/sample_021 | P-Flow | 84 | 70 | 6.1886 | 19.5135 | stalled |
| r64_n1_ct10/sample_022 | P-Flow | 406 | 328 | 8.9420 | 19.7726 | stalled |
| r64_n1_ct10/sample_023 | P-Flow | 563 | 556 | 144.3848 | 1.2264 | stalled |
| r64_n1_ct10/sample_024 | P-Flow | 640 | 626 | 136.1438 | 1.7571 | stalled |
| r64_n1_ct10/sample_025 | P-Flow | 693 | 671 | 123.4874 | 1.2665 | stalled |
| r64_n1_ct10/sample_026 | P-Flow | 59 | 49 | 4.0624 | 19.8623 | stalled |
| r64_n1_ct10/sample_027 | P-Flow | 49 | 40 | 6.1105 | 19.4265 | stalled |
| r64_n1_ct10/sample_028 | P-Flow | 1000 | 833 | 71.4148 | 2.5764 | step_limit |
| r64_n1_ct10/sample_029 | P-Flow | 24 | 20 | 4.9431 | 19.7923 | stalled |
| r64_n1_ct10/sample_000 | DLR D-Flow | 332 | 320 | 10.8007 | 19.9844 | stalled |
| r64_n1_ct10/sample_001 | DLR D-Flow | 153 | 140 | 4.3302 | 19.5204 | stalled |
| r64_n1_ct10/sample_002 | DLR D-Flow | 215 | 209 | 14.0449 | 19.4972 | stalled |
| r64_n1_ct10/sample_003 | DLR D-Flow | 497 | 477 | 4.7612 | 19.6519 | stalled |
| r64_n1_ct10/sample_004 | DLR D-Flow | 46 | 40 | 99.6034 | 90.7883 | stalled |
| r64_n1_ct10/sample_005 | DLR D-Flow | 337 | 322 | 5.2775 | 19.5003 | stalled |
| r64_n1_ct10/sample_006 | DLR D-Flow | 265 | 256 | 5.9149 | 19.8275 | stalled |
| r64_n1_ct10/sample_007 | DLR D-Flow | 233 | 222 | 9.6024 | 19.4081 | stalled |
| r64_n1_ct10/sample_008 | DLR D-Flow | 1000 | 897 | 3.1237 | 19.6965 | step_limit |
| r64_n1_ct10/sample_009 | DLR D-Flow | 359 | 343 | 14.4396 | 19.3889 | stalled |
| r64_n1_ct10/sample_010 | DLR D-Flow | 424 | 406 | 4.7006 | 19.8368 | stalled |
| r64_n1_ct10/sample_011 | DLR D-Flow | 149 | 146 | 13.1524 | 19.4962 | stalled |
| r64_n1_ct10/sample_012 | DLR D-Flow | 166 | 161 | 8.0544 | 19.5389 | stalled |
| r64_n1_ct10/sample_013 | DLR D-Flow | 771 | 726 | 7.8040 | 19.1710 | stalled |
| r64_n1_ct10/sample_014 | DLR D-Flow | 1000 | 968 | 4.7952 | 19.6822 | step_limit |
| r64_n1_ct10/sample_015 | DLR D-Flow | 1000 | 963 | 3.4081 | 19.7262 | step_limit |
| r64_n1_ct10/sample_016 | DLR D-Flow | 174 | 163 | 3.8166 | 19.4903 | stalled |
| r64_n1_ct10/sample_017 | DLR D-Flow | 304 | 290 | 4.6045 | 19.7068 | stalled |
| r64_n1_ct10/sample_018 | DLR D-Flow | 328 | 318 | 4.5418 | 19.8779 | stalled |
| r64_n1_ct10/sample_019 | DLR D-Flow | 405 | 388 | 5.3665 | 19.3439 | stalled |
| r64_n1_ct10/sample_020 | DLR D-Flow | 89 | 84 | 3.1682 | 19.3792 | stalled |
| r64_n1_ct10/sample_021 | DLR D-Flow | 202 | 188 | 102.4986 | 87.8423 | stalled |
| r64_n1_ct10/sample_022 | DLR D-Flow | 534 | 500 | 4.9334 | 19.2305 | stalled |
| r64_n1_ct10/sample_023 | DLR D-Flow | 114 | 112 | 13.6447 | 19.1202 | stalled |
| r64_n1_ct10/sample_024 | DLR D-Flow | 189 | 179 | 6.1905 | 19.4857 | stalled |
| r64_n1_ct10/sample_025 | DLR D-Flow | 213 | 203 | 20.2910 | 19.2955 | stalled |
| r64_n1_ct10/sample_026 | DLR D-Flow | 605 | 569 | 2.8401 | 19.7219 | stalled |
| r64_n1_ct10/sample_027 | DLR D-Flow | 216 | 198 | 4.1901 | 19.4068 | stalled |
| r64_n1_ct10/sample_028 | DLR D-Flow | 168 | 163 | 8.0877 | 19.7928 | stalled |
| r64_n1_ct10/sample_029 | DLR D-Flow | 56 | 52 | 2.9298 | 19.6567 | stalled |
| r64_n2_ct10/sample_000 | D-Flow | 620 | 618 | 136.9099 | 1.6824 | stalled |
| r64_n2_ct10/sample_001 | D-Flow | 825 | 822 | 168.3964 | 1.4176 | stalled |
| r64_n2_ct10/sample_002 | D-Flow | 930 | 928 | 99.3156 | 1.2498 | stalled |
| r64_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 127.3970 | 1.5220 | step_limit |
| r64_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 30.5216 | 16.4798 | step_limit |
| r64_n2_ct10/sample_005 | D-Flow | 716 | 713 | 160.5842 | 1.6339 | stalled |
| r64_n2_ct10/sample_006 | D-Flow | 989 | 986 | 129.8783 | 1.9724 | stalled |
| r64_n2_ct10/sample_007 | D-Flow | 1000 | 1000 | 72.2994 | 3.9376 | step_limit |
| r64_n2_ct10/sample_008 | D-Flow | 784 | 782 | 125.9633 | 1.1634 | stalled |
| r64_n2_ct10/sample_009 | D-Flow | 798 | 796 | 172.2946 | 0.8974 | stalled |
| r64_n2_ct10/sample_010 | D-Flow | 748 | 745 | 146.1544 | 2.1834 | stalled |
| r64_n2_ct10/sample_000 | P-Flow | 601 | 483 | 19.2284 | 15.7040 | stalled |
| r64_n2_ct10/sample_001 | P-Flow | 618 | 602 | 131.9341 | 1.4176 | stalled |
| r64_n2_ct10/sample_002 | P-Flow | 1000 | 895 | 96.2936 | 1.3007 | step_limit |
| r64_n2_ct10/sample_003 | P-Flow | 10 | 8 | 113.9825 | 71.2942 | stalled |
| r64_n2_ct10/sample_004 | P-Flow | 22 | 19 | 57.6747 | 34.1329 | stalled |
| r64_n2_ct10/sample_005 | P-Flow | 17 | 13 | 78.7297 | 48.4307 | stalled |
| r64_n2_ct10/sample_006 | P-Flow | 47 | 41 | 8.2941 | 19.2839 | stalled |
| r64_n2_ct10/sample_007 | P-Flow | 50 | 43 | 8.9593 | 19.5973 | stalled |
| r64_n2_ct10/sample_008 | P-Flow | 583 | 566 | 116.8439 | 1.1633 | stalled |
| r64_n2_ct10/sample_009 | P-Flow | 668 | 655 | 132.5715 | 0.8974 | stalled |
| r64_n2_ct10/sample_010 | P-Flow | 112 | 99 | 9.3748 | 19.0586 | stalled |
| r64_n2_ct10/sample_000 | DLR D-Flow | 302 | 285 | 87.6368 | 54.6122 | stalled |
| r64_n2_ct10/sample_001 | DLR D-Flow | 230 | 223 | 8.0109 | 19.5760 | stalled |
| r64_n2_ct10/sample_002 | DLR D-Flow | 77 | 72 | 14.0250 | 20.0560 | stalled |
| r64_n2_ct10/sample_003 | DLR D-Flow | 132 | 129 | 9.1485 | 19.4956 | stalled |
| r64_n2_ct10/sample_004 | DLR D-Flow | 591 | 568 | 85.5792 | 57.3310 | stalled |
| r64_n2_ct10/sample_005 | DLR D-Flow | 895 | 860 | 6.6546 | 19.3823 | stalled |
| r64_n2_ct10/sample_006 | DLR D-Flow | 150 | 142 | 75.3207 | 58.9538 | stalled |
| r64_n2_ct10/sample_007 | DLR D-Flow | 778 | 757 | 78.5085 | 66.3633 | stalled |
| r64_n2_ct10/sample_008 | DLR D-Flow | 152 | 145 | 75.5804 | 50.8297 | stalled |
| r64_n2_ct10/sample_009 | DLR D-Flow | 421 | 404 | 4.5854 | 19.6547 | stalled |
| r64_n2_ct10/sample_010 | DLR D-Flow | 263 | 251 | 4.8719 | 19.4082 | stalled |
| r64_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 149.7830 | 2.2533 | step_limit |
| r64_n4_ct10/sample_001 | D-Flow | 925 | 923 | 133.9624 | 1.7011 | stalled |
| r64_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 85.2168 | 2.0413 | step_limit |
| r64_n4_ct10/sample_003 | D-Flow | 977 | 975 | 108.0558 | 1.8598 | stalled |
| r64_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 136.5880 | 2.0317 | step_limit |
| r64_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 122.4381 | 1.8110 | step_limit |
| r64_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 159.1969 | 1.2764 | step_limit |
| r64_n4_ct10/sample_007 | D-Flow | 883 | 881 | 120.0385 | 2.4011 | stalled |
| r64_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 80.3985 | 3.2502 | step_limit |
| r64_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 128.8020 | 2.4167 | step_limit |
| r64_n4_ct10/sample_010 | D-Flow | 689 | 686 | 158.0028 | 2.7604 | stalled |
| r64_n4_ct10/sample_000 | P-Flow | 1000 | 926 | 93.7119 | 2.9842 | step_limit |
| r64_n4_ct10/sample_001 | P-Flow | 1000 | 917 | 78.4215 | 2.5784 | step_limit |
| r64_n4_ct10/sample_002 | P-Flow | 150 | 129 | 14.5178 | 18.5522 | stalled |
| r64_n4_ct10/sample_003 | P-Flow | 13 | 11 | 58.6373 | 34.0832 | stalled |
| r64_n4_ct10/sample_004 | P-Flow | 17 | 14 | 61.8653 | 35.8755 | stalled |
| r64_n4_ct10/sample_005 | P-Flow | 1000 | 863 | 62.8398 | 3.4001 | step_limit |
| r64_n4_ct10/sample_006 | P-Flow | 62 | 54 | 13.2688 | 19.5776 | stalled |
| r64_n4_ct10/sample_007 | P-Flow | 752 | 715 | 109.4859 | 2.4070 | stalled |
| r64_n4_ct10/sample_008 | P-Flow | 1000 | 926 | 97.7037 | 2.3498 | step_limit |
| r64_n4_ct10/sample_009 | P-Flow | 19 | 15 | 60.2762 | 35.3106 | stalled |
| r64_n4_ct10/sample_010 | P-Flow | 717 | 690 | 113.5383 | 2.7608 | stalled |
| r64_n4_ct10/sample_000 | DLR D-Flow | 222 | 217 | 67.7752 | 35.9065 | stalled |
| r64_n4_ct10/sample_001 | DLR D-Flow | 704 | 673 | 23.1514 | 21.4245 | stalled |
| r64_n4_ct10/sample_002 | DLR D-Flow | 335 | 324 | 84.4475 | 50.8279 | stalled |
| r64_n4_ct10/sample_003 | DLR D-Flow | 367 | 357 | 31.8062 | 23.2633 | stalled |
| r64_n4_ct10/sample_004 | DLR D-Flow | 1000 | 959 | 17.9692 | 20.2955 | step_limit |
| r64_n4_ct10/sample_005 | DLR D-Flow | 172 | 162 | 77.4311 | 48.6281 | stalled |
| r64_n4_ct10/sample_006 | DLR D-Flow | 670 | 656 | 18.4074 | 20.4478 | stalled |
| r64_n4_ct10/sample_007 | DLR D-Flow | 818 | 788 | 71.1061 | 45.0396 | stalled |
| r64_n4_ct10/sample_008 | DLR D-Flow | 1000 | 975 | 79.6215 | 44.6193 | step_limit |
| r64_n4_ct10/sample_009 | DLR D-Flow | 1000 | 965 | 40.7275 | 26.0125 | step_limit |
| r64_n4_ct10/sample_010 | DLR D-Flow | 241 | 233 | 16.8144 | 20.7466 | stalled |
| r64_n1_mri4/sample_000 | D-Flow | 844 | 842 | 26.3341 | 8.9603 | stalled |
| r64_n1_mri4/sample_001 | D-Flow | 453 | 451 | 22.6328 | 10.6431 | stalled |
| r64_n1_mri4/sample_002 | D-Flow | 355 | 353 | 26.4649 | 8.1694 | stalled |
| r64_n1_mri4/sample_003 | D-Flow | 692 | 690 | 10.2439 | 14.0287 | stalled |
| r64_n1_mri4/sample_004 | D-Flow | 818 | 814 | 100.3519 | 94.8726 | stalled |
| r64_n1_mri4/sample_005 | D-Flow | 392 | 390 | 28.6008 | 7.7473 | stalled |
| r64_n1_mri4/sample_006 | D-Flow | 396 | 394 | 8.1041 | 15.4351 | stalled |
| r64_n1_mri4/sample_007 | D-Flow | 446 | 444 | 25.6848 | 8.8948 | stalled |
| r64_n1_mri4/sample_008 | D-Flow | 338 | 336 | 32.7357 | 8.8911 | stalled |
| r64_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 10.2432 | 14.0554 | step_limit |
| r64_n1_mri4/sample_010 | D-Flow | 602 | 599 | 28.7975 | 9.0086 | stalled |
| r64_n1_mri4/sample_011 | D-Flow | 491 | 489 | 8.0544 | 15.5950 | stalled |
| r64_n1_mri4/sample_012 | D-Flow | 431 | 428 | 27.4389 | 10.4799 | stalled |
| r64_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 12.3240 | 13.9022 | step_limit |
| r64_n1_mri4/sample_014 | D-Flow | 509 | 507 | 22.6708 | 12.4414 | stalled |
| r64_n1_mri4/sample_015 | D-Flow | 828 | 826 | 10.1128 | 14.2216 | stalled |
| r64_n1_mri4/sample_016 | D-Flow | 528 | 526 | 23.1439 | 10.7265 | stalled |
| r64_n1_mri4/sample_017 | D-Flow | 996 | 992 | 10.0080 | 14.8755 | stalled |
| r64_n1_mri4/sample_018 | D-Flow | 555 | 553 | 23.5277 | 10.0456 | stalled |
| r64_n1_mri4/sample_019 | D-Flow | 693 | 690 | 8.4595 | 15.6201 | stalled |
| r64_n1_mri4/sample_020 | D-Flow | 319 | 316 | 28.2577 | 10.5818 | stalled |
| r64_n1_mri4/sample_021 | D-Flow | 931 | 929 | 11.1450 | 13.6354 | stalled |
| r64_n1_mri4/sample_022 | D-Flow | 439 | 437 | 24.7337 | 10.9378 | stalled |
| r64_n1_mri4/sample_023 | D-Flow | 471 | 469 | 8.4405 | 15.4948 | stalled |
| r64_n1_mri4/sample_024 | D-Flow | 498 | 496 | 27.2546 | 9.3245 | stalled |
| r64_n1_mri4/sample_025 | D-Flow | 782 | 780 | 9.9982 | 13.6462 | stalled |
| r64_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 11.9957 | 13.2791 | step_limit |
| r64_n1_mri4/sample_027 | D-Flow | 487 | 485 | 25.8792 | 9.3470 | stalled |
| r64_n1_mri4/sample_028 | D-Flow | 554 | 552 | 24.7291 | 11.8321 | stalled |
| r64_n1_mri4/sample_029 | D-Flow | 293 | 291 | 32.5872 | 10.0927 | stalled |
| r64_n1_mri4/sample_000 | P-Flow | 196 | 186 | 25.0075 | 8.9603 | stalled |
| r64_n1_mri4/sample_001 | P-Flow | 169 | 163 | 22.1587 | 10.6431 | stalled |
| r64_n1_mri4/sample_002 | P-Flow | 177 | 172 | 25.1529 | 8.1694 | stalled |
| r64_n1_mri4/sample_003 | P-Flow | 25 | 20 | 3.1011 | 19.7192 | stalled |
| r64_n1_mri4/sample_004 | P-Flow | 209 | 204 | 22.5009 | 11.1692 | stalled |
| r64_n1_mri4/sample_005 | P-Flow | 222 | 208 | 24.5047 | 7.7473 | stalled |
| r64_n1_mri4/sample_006 | P-Flow | 186 | 182 | 25.6044 | 9.9039 | stalled |
| r64_n1_mri4/sample_007 | P-Flow | 199 | 197 | 27.0685 | 8.8948 | stalled |
| r64_n1_mri4/sample_008 | P-Flow | 232 | 217 | 27.2523 | 8.8911 | stalled |
| r64_n1_mri4/sample_009 | P-Flow | 189 | 180 | 22.1504 | 10.4077 | stalled |
| r64_n1_mri4/sample_010 | P-Flow | 193 | 190 | 24.4228 | 9.0086 | stalled |
| r64_n1_mri4/sample_011 | P-Flow | 216 | 212 | 28.6630 | 7.2383 | stalled |
| r64_n1_mri4/sample_012 | P-Flow | 178 | 175 | 25.3705 | 10.4799 | stalled |
| r64_n1_mri4/sample_013 | P-Flow | 172 | 167 | 24.2415 | 11.7705 | stalled |
| r64_n1_mri4/sample_014 | P-Flow | 165 | 162 | 23.2806 | 12.4414 | stalled |
| r64_n1_mri4/sample_015 | P-Flow | 208 | 193 | 23.4563 | 11.3493 | stalled |
| r64_n1_mri4/sample_016 | P-Flow | 177 | 172 | 22.6067 | 10.7265 | stalled |
| r64_n1_mri4/sample_017 | P-Flow | 22 | 17 | 2.1301 | 19.5582 | stalled |
| r64_n1_mri4/sample_018 | P-Flow | 201 | 193 | 26.3338 | 10.0455 | stalled |
| r64_n1_mri4/sample_019 | P-Flow | 35 | 29 | 2.7859 | 19.5618 | stalled |
| r64_n1_mri4/sample_020 | P-Flow | 273 | 256 | 25.2577 | 10.5818 | stalled |
| r64_n1_mri4/sample_021 | P-Flow | 176 | 172 | 27.6899 | 7.8470 | stalled |
| r64_n1_mri4/sample_022 | P-Flow | 42 | 35 | 2.2963 | 19.5937 | stalled |
| r64_n1_mri4/sample_023 | P-Flow | 171 | 167 | 23.6429 | 11.2426 | stalled |
| r64_n1_mri4/sample_024 | P-Flow | 57 | 48 | 2.3858 | 19.6594 | stalled |
| r64_n1_mri4/sample_025 | P-Flow | 165 | 162 | 24.9115 | 7.4331 | stalled |
| r64_n1_mri4/sample_026 | P-Flow | 178 | 170 | 26.1357 | 9.1435 | stalled |
| r64_n1_mri4/sample_027 | P-Flow | 46 | 37 | 2.0875 | 19.7323 | stalled |
| r64_n1_mri4/sample_028 | P-Flow | 173 | 169 | 24.1999 | 11.8321 | stalled |
| r64_n1_mri4/sample_029 | P-Flow | 162 | 159 | 25.3860 | 10.0927 | stalled |
| r64_n1_mri4/sample_000 | DLR D-Flow | 170 | 163 | 77.5281 | 77.9229 | stalled |
| r64_n1_mri4/sample_001 | DLR D-Flow | 27 | 24 | 16.7409 | 24.2501 | stalled |
| r64_n1_mri4/sample_002 | DLR D-Flow | 170 | 165 | 2.3440 | 19.5348 | stalled |
| r64_n1_mri4/sample_003 | DLR D-Flow | 111 | 108 | 2.7200 | 19.4977 | stalled |
| r64_n1_mri4/sample_004 | DLR D-Flow | 197 | 191 | 100.4752 | 95.7448 | stalled |
| r64_n1_mri4/sample_005 | DLR D-Flow | 319 | 303 | 2.7540 | 19.7409 | stalled |
| r64_n1_mri4/sample_006 | DLR D-Flow | 175 | 172 | 3.8326 | 19.7652 | stalled |
| r64_n1_mri4/sample_007 | DLR D-Flow | 440 | 409 | 1.8537 | 19.6746 | stalled |
| r64_n1_mri4/sample_008 | DLR D-Flow | 231 | 223 | 37.3972 | 36.9413 | stalled |
| r64_n1_mri4/sample_009 | DLR D-Flow | 117 | 111 | 3.4548 | 19.8473 | stalled |
| r64_n1_mri4/sample_010 | DLR D-Flow | 72 | 68 | 13.6872 | 22.0133 | stalled |
| r64_n1_mri4/sample_011 | DLR D-Flow | 82 | 78 | 2.8113 | 19.5547 | stalled |
| r64_n1_mri4/sample_012 | DLR D-Flow | 562 | 538 | 2.5965 | 19.7206 | stalled |
| r64_n1_mri4/sample_013 | DLR D-Flow | 138 | 132 | 3.2043 | 19.5289 | stalled |
| r64_n1_mri4/sample_014 | DLR D-Flow | 197 | 189 | 2.9430 | 19.6805 | stalled |
| r64_n1_mri4/sample_015 | DLR D-Flow | 237 | 229 | 3.2796 | 19.6480 | stalled |
| r64_n1_mri4/sample_016 | DLR D-Flow | 379 | 364 | 3.5044 | 19.7593 | stalled |
| r64_n1_mri4/sample_017 | DLR D-Flow | 995 | 939 | 2.8331 | 19.4755 | stalled |
| r64_n1_mri4/sample_018 | DLR D-Flow | 689 | 655 | 2.1865 | 19.5281 | stalled |
| r64_n1_mri4/sample_019 | DLR D-Flow | 183 | 176 | 2.6833 | 19.5520 | stalled |
| r64_n1_mri4/sample_020 | DLR D-Flow | 200 | 191 | 2.0300 | 19.6804 | stalled |
| r64_n1_mri4/sample_021 | DLR D-Flow | 513 | 493 | 2.8154 | 19.4222 | stalled |
| r64_n1_mri4/sample_022 | DLR D-Flow | 384 | 349 | 2.6645 | 19.5744 | stalled |
| r64_n1_mri4/sample_023 | DLR D-Flow | 106 | 101 | 2.4357 | 19.4613 | stalled |
| r64_n1_mri4/sample_024 | DLR D-Flow | 150 | 142 | 1.9901 | 19.6714 | stalled |
| r64_n1_mri4/sample_025 | DLR D-Flow | 200 | 188 | 2.2365 | 19.6436 | stalled |
| r64_n1_mri4/sample_026 | DLR D-Flow | 1000 | 950 | 2.2639 | 19.7133 | step_limit |
| r64_n1_mri4/sample_027 | DLR D-Flow | 150 | 144 | 3.3023 | 19.8220 | stalled |
| r64_n1_mri4/sample_028 | DLR D-Flow | 463 | 448 | 3.1660 | 19.5968 | stalled |
| r64_n1_mri4/sample_029 | DLR D-Flow | 102 | 98 | 3.4677 | 19.5637 | stalled |
| r64_n2_mri4/sample_000 | D-Flow | 439 | 437 | 26.7085 | 10.3778 | stalled |
| r64_n2_mri4/sample_001 | D-Flow | 505 | 503 | 19.5231 | 10.6197 | stalled |
| r64_n2_mri4/sample_002 | D-Flow | 485 | 481 | 25.5982 | 10.2064 | stalled |
| r64_n2_mri4/sample_003 | D-Flow | 498 | 496 | 24.3823 | 9.3093 | stalled |
| r64_n2_mri4/sample_004 | D-Flow | 863 | 861 | 26.2832 | 9.1451 | stalled |
| r64_n2_mri4/sample_005 | D-Flow | 669 | 667 | 28.0290 | 7.2140 | stalled |
| r64_n2_mri4/sample_006 | D-Flow | 712 | 710 | 22.5991 | 9.5732 | stalled |
| r64_n2_mri4/sample_007 | D-Flow | 389 | 387 | 26.9183 | 8.5906 | stalled |
| r64_n2_mri4/sample_008 | D-Flow | 671 | 669 | 26.9968 | 7.7462 | stalled |
| r64_n2_mri4/sample_009 | D-Flow | 385 | 383 | 27.0762 | 8.9635 | stalled |
| r64_n2_mri4/sample_010 | D-Flow | 719 | 716 | 26.9097 | 8.9050 | stalled |
| r64_n2_mri4/sample_000 | P-Flow | 184 | 177 | 26.5366 | 10.3778 | stalled |
| r64_n2_mri4/sample_001 | P-Flow | 199 | 194 | 22.8659 | 10.6197 | stalled |
| r64_n2_mri4/sample_002 | P-Flow | 175 | 172 | 26.9412 | 10.2064 | stalled |
| r64_n2_mri4/sample_003 | P-Flow | 10 | 7 | 108.6724 | 108.5827 | stalled |
| r64_n2_mri4/sample_004 | P-Flow | 246 | 234 | 27.9821 | 9.1451 | stalled |
| r64_n2_mri4/sample_005 | P-Flow | 204 | 198 | 28.8054 | 7.2140 | stalled |
| r64_n2_mri4/sample_006 | P-Flow | 175 | 166 | 25.7936 | 9.5732 | stalled |
| r64_n2_mri4/sample_007 | P-Flow | 175 | 170 | 28.4957 | 8.5906 | stalled |
| r64_n2_mri4/sample_008 | P-Flow | 187 | 181 | 27.9280 | 7.7462 | stalled |
| r64_n2_mri4/sample_009 | P-Flow | 193 | 187 | 27.9188 | 8.9635 | stalled |
| r64_n2_mri4/sample_010 | P-Flow | 5 | 3 | 112.4476 | 107.9871 | stalled |
| r64_n2_mri4/sample_000 | DLR D-Flow | 900 | 874 | 41.8699 | 42.2512 | stalled |
| r64_n2_mri4/sample_001 | DLR D-Flow | 1000 | 959 | 3.7564 | 19.6257 | step_limit |
| r64_n2_mri4/sample_002 | DLR D-Flow | 116 | 114 | 5.1857 | 19.9971 | stalled |
| r64_n2_mri4/sample_003 | DLR D-Flow | 180 | 170 | 77.7020 | 77.1543 | stalled |
| r64_n2_mri4/sample_004 | DLR D-Flow | 173 | 164 | 38.0125 | 38.1703 | stalled |
| r64_n2_mri4/sample_005 | DLR D-Flow | 335 | 319 | 77.2502 | 73.3479 | stalled |
| r64_n2_mri4/sample_006 | DLR D-Flow | 1000 | 975 | 68.4290 | 69.1985 | step_limit |
| r64_n2_mri4/sample_007 | DLR D-Flow | 1000 | 963 | 71.6470 | 73.0514 | step_limit |
| r64_n2_mri4/sample_008 | DLR D-Flow | 1000 | 968 | 1.9187 | 19.4849 | step_limit |
| r64_n2_mri4/sample_009 | DLR D-Flow | 142 | 138 | 3.6824 | 19.5352 | stalled |
| r64_n2_mri4/sample_010 | DLR D-Flow | 1000 | 973 | 93.8484 | 93.0402 | step_limit |
| r64_n4_mri4/sample_000 | D-Flow | 464 | 461 | 25.1605 | 11.2168 | stalled |
| r64_n4_mri4/sample_001 | D-Flow | 545 | 543 | 24.7222 | 10.4461 | stalled |
| r64_n4_mri4/sample_002 | D-Flow | 841 | 838 | 23.4515 | 10.2720 | stalled |
| r64_n4_mri4/sample_003 | D-Flow | 757 | 754 | 5.4067 | 17.0099 | stalled |
| r64_n4_mri4/sample_004 | D-Flow | 520 | 518 | 26.2866 | 10.1497 | stalled |
| r64_n4_mri4/sample_005 | D-Flow | 585 | 583 | 6.5676 | 16.5782 | stalled |
| r64_n4_mri4/sample_006 | D-Flow | 745 | 743 | 25.5388 | 11.3107 | stalled |
| r64_n4_mri4/sample_007 | D-Flow | 747 | 745 | 26.4511 | 7.7502 | stalled |
| r64_n4_mri4/sample_008 | D-Flow | 538 | 536 | 25.3184 | 11.4290 | stalled |
| r64_n4_mri4/sample_009 | D-Flow | 584 | 582 | 21.8536 | 11.3387 | stalled |
| r64_n4_mri4/sample_010 | D-Flow | 439 | 437 | 25.1452 | 8.1385 | stalled |
| r64_n4_mri4/sample_000 | P-Flow | 186 | 184 | 24.3146 | 11.2168 | stalled |
| r64_n4_mri4/sample_001 | P-Flow | 232 | 216 | 24.4149 | 10.4461 | stalled |
| r64_n4_mri4/sample_002 | P-Flow | 191 | 185 | 23.5259 | 10.2720 | stalled |
| r64_n4_mri4/sample_003 | P-Flow | 12 | 9 | 62.2331 | 59.5516 | stalled |
| r64_n4_mri4/sample_004 | P-Flow | 214 | 206 | 26.1792 | 10.1497 | stalled |
| r64_n4_mri4/sample_005 | P-Flow | 260 | 243 | 24.0873 | 11.7123 | stalled |
| r64_n4_mri4/sample_006 | P-Flow | 186 | 183 | 24.8180 | 11.3107 | stalled |
| r64_n4_mri4/sample_007 | P-Flow | 188 | 186 | 28.5850 | 7.7502 | stalled |
| r64_n4_mri4/sample_008 | P-Flow | 187 | 180 | 24.2217 | 11.4290 | stalled |
| r64_n4_mri4/sample_009 | P-Flow | 26 | 22 | 47.7653 | 45.5219 | stalled |
| r64_n4_mri4/sample_010 | P-Flow | 184 | 182 | 25.6983 | 8.1385 | stalled |
| r64_n4_mri4/sample_000 | DLR D-Flow | 379 | 364 | 15.1488 | 22.7947 | stalled |
| r64_n4_mri4/sample_001 | DLR D-Flow | 54 | 51 | 29.6494 | 30.4381 | stalled |
| r64_n4_mri4/sample_002 | DLR D-Flow | 1000 | 976 | 45.7252 | 39.7746 | step_limit |
| r64_n4_mri4/sample_003 | DLR D-Flow | 1000 | 944 | 20.0707 | 25.2650 | step_limit |
| r64_n4_mri4/sample_004 | DLR D-Flow | 455 | 432 | 18.4548 | 22.7620 | stalled |
| r64_n4_mri4/sample_005 | DLR D-Flow | 1000 | 953 | 79.9505 | 78.2877 | step_limit |
| r64_n4_mri4/sample_006 | DLR D-Flow | 694 | 665 | 89.9597 | 85.5274 | stalled |
| r64_n4_mri4/sample_007 | DLR D-Flow | 826 | 802 | 63.7724 | 64.6887 | stalled |
| r64_n4_mri4/sample_008 | DLR D-Flow | 1000 | 982 | 54.4668 | 51.8635 | step_limit |
| r64_n4_mri4/sample_009 | DLR D-Flow | 269 | 265 | 16.2871 | 22.7371 | stalled |
| r64_n4_mri4/sample_010 | DLR D-Flow | 100 | 94 | 19.7024 | 24.6446 | stalled |
| r96_n1_ct10/sample_000 | D-Flow | 321 | 319 | 10.2383 | 15.4148 | stalled |
| r96_n1_ct10/sample_001 | D-Flow | 926 | 924 | 105.1373 | 1.5564 | stalled |
| r96_n1_ct10/sample_002 | D-Flow | 494 | 492 | 9.1105 | 16.1034 | stalled |
| r96_n1_ct10/sample_003 | D-Flow | 642 | 640 | 121.4570 | 1.2552 | stalled |
| r96_n1_ct10/sample_004 | D-Flow | 49 | 47 | 116.1304 | 107.3743 | stalled |
| r96_n1_ct10/sample_005 | D-Flow | 489 | 486 | 151.3346 | 1.0890 | stalled |
| r96_n1_ct10/sample_006 | D-Flow | 649 | 646 | 124.4889 | 0.8226 | stalled |
| r96_n1_ct10/sample_007 | D-Flow | 665 | 663 | 188.4236 | 0.8380 | stalled |
| r96_n1_ct10/sample_008 | D-Flow | 1000 | 1000 | 43.5268 | 15.9941 | step_limit |
| r96_n1_ct10/sample_009 | D-Flow | 478 | 476 | 11.3460 | 15.5906 | stalled |
| r96_n1_ct10/sample_010 | D-Flow | 548 | 546 | 155.9066 | 1.5183 | stalled |
| r96_n1_ct10/sample_011 | D-Flow | 525 | 523 | 195.3032 | 1.0689 | stalled |
| r96_n1_ct10/sample_012 | D-Flow | 385 | 382 | 8.4849 | 17.2392 | stalled |
| r96_n1_ct10/sample_013 | D-Flow | 513 | 511 | 105.9095 | 105.0793 | stalled |
| r96_n1_ct10/sample_014 | D-Flow | 255 | 253 | 115.7081 | 105.7682 | stalled |
| r96_n1_ct10/sample_015 | D-Flow | 734 | 731 | 88.9832 | 1.4408 | stalled |
| r96_n1_ct10/sample_016 | D-Flow | 681 | 679 | 112.5918 | 1.1658 | stalled |
| r96_n1_ct10/sample_017 | D-Flow | 646 | 644 | 148.4574 | 1.2238 | stalled |
| r96_n1_ct10/sample_018 | D-Flow | 445 | 443 | 11.3497 | 16.2415 | stalled |
| r96_n1_ct10/sample_019 | D-Flow | 767 | 765 | 131.5891 | 1.4067 | stalled |
| r96_n1_ct10/sample_020 | D-Flow | 670 | 668 | 100.3326 | 0.9103 | stalled |
| r96_n1_ct10/sample_021 | D-Flow | 597 | 595 | 103.6831 | 1.2102 | stalled |
| r96_n1_ct10/sample_022 | D-Flow | 571 | 568 | 124.3328 | 1.8250 | stalled |
| r96_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 176.3031 | 1.4399 | step_limit |
| r96_n1_ct10/sample_024 | D-Flow | 728 | 725 | 100.2148 | 1.8354 | stalled |
| r96_n1_ct10/sample_025 | D-Flow | 581 | 579 | 118.7220 | 1.3183 | stalled |
| r96_n1_ct10/sample_026 | D-Flow | 575 | 572 | 93.4104 | 1.3185 | stalled |
| r96_n1_ct10/sample_027 | D-Flow | 542 | 540 | 329.2921 | 0.6443 | stalled |
| r96_n1_ct10/sample_028 | D-Flow | 882 | 880 | 136.5168 | 1.2712 | stalled |
| r96_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 103.8114 | 0.8784 | step_limit |
| r96_n1_ct10/sample_000 | P-Flow | 46 | 38 | 6.9514 | 19.3805 | stalled |
| r96_n1_ct10/sample_001 | P-Flow | 117 | 100 | 4.8140 | 19.7276 | stalled |
| r96_n1_ct10/sample_002 | P-Flow | 51 | 41 | 7.0722 | 19.4418 | stalled |
| r96_n1_ct10/sample_003 | P-Flow | 78 | 66 | 4.8622 | 19.6540 | stalled |
| r96_n1_ct10/sample_004 | P-Flow | 1000 | 894 | 65.0601 | 1.6473 | step_limit |
| r96_n1_ct10/sample_005 | P-Flow | 54 | 49 | 6.1504 | 19.6293 | stalled |
| r96_n1_ct10/sample_006 | P-Flow | 922 | 855 | 105.5468 | 0.8227 | stalled |
| r96_n1_ct10/sample_007 | P-Flow | 224 | 182 | 6.4714 | 19.3378 | stalled |
| r96_n1_ct10/sample_008 | P-Flow | 571 | 554 | 83.2032 | 1.5968 | stalled |
| r96_n1_ct10/sample_009 | P-Flow | 115 | 93 | 4.1239 | 19.6844 | stalled |
| r96_n1_ct10/sample_010 | P-Flow | 726 | 697 | 102.8044 | 1.5184 | stalled |
| r96_n1_ct10/sample_011 | P-Flow | 501 | 496 | 133.5241 | 1.0689 | stalled |
| r96_n1_ct10/sample_012 | P-Flow | 502 | 492 | 121.5353 | 1.1385 | stalled |
| r96_n1_ct10/sample_013 | P-Flow | 570 | 539 | 96.4692 | 1.3542 | stalled |
| r96_n1_ct10/sample_014 | P-Flow | 1000 | 889 | 69.2772 | 1.7257 | step_limit |
| r96_n1_ct10/sample_015 | P-Flow | 1000 | 824 | 3.7995 | 19.8063 | step_limit |
| r96_n1_ct10/sample_016 | P-Flow | 22 | 19 | 8.5685 | 20.0072 | stalled |
| r96_n1_ct10/sample_017 | P-Flow | 28 | 22 | 3.8579 | 19.6625 | stalled |
| r96_n1_ct10/sample_018 | P-Flow | 26 | 20 | 9.7172 | 19.4462 | stalled |
| r96_n1_ct10/sample_019 | P-Flow | 90 | 78 | 4.0874 | 19.5621 | stalled |
| r96_n1_ct10/sample_020 | P-Flow | 41 | 35 | 4.2442 | 19.7237 | stalled |
| r96_n1_ct10/sample_021 | P-Flow | 437 | 420 | 91.2947 | 1.2109 | stalled |
| r96_n1_ct10/sample_022 | P-Flow | 1000 | 835 | 57.2876 | 4.2468 | step_limit |
| r96_n1_ct10/sample_023 | P-Flow | 527 | 521 | 96.3363 | 1.4398 | stalled |
| r96_n1_ct10/sample_024 | P-Flow | 41 | 34 | 12.2740 | 20.5662 | stalled |
| r96_n1_ct10/sample_025 | P-Flow | 29 | 24 | 5.7657 | 19.5295 | stalled |
| r96_n1_ct10/sample_026 | P-Flow | 488 | 398 | 5.5235 | 19.4914 | stalled |
| r96_n1_ct10/sample_027 | P-Flow | 561 | 543 | 106.7321 | 0.6445 | stalled |
| r96_n1_ct10/sample_028 | P-Flow | 1000 | 887 | 100.1186 | 1.2764 | step_limit |
| r96_n1_ct10/sample_029 | P-Flow | 39 | 31 | 4.3636 | 19.7459 | stalled |
| r96_n1_ct10/sample_000 | DLR D-Flow | 173 | 166 | 5.0891 | 19.3589 | stalled |
| r96_n1_ct10/sample_001 | DLR D-Flow | 22 | 19 | 113.3081 | 101.1097 | stalled |
| r96_n1_ct10/sample_002 | DLR D-Flow | 237 | 227 | 5.9576 | 19.3838 | stalled |
| r96_n1_ct10/sample_003 | DLR D-Flow | 152 | 145 | 3.8292 | 19.6490 | stalled |
| r96_n1_ct10/sample_004 | DLR D-Flow | 1000 | 945 | 4.3244 | 19.5686 | step_limit |
| r96_n1_ct10/sample_005 | DLR D-Flow | 240 | 230 | 3.1438 | 19.6147 | stalled |
| r96_n1_ct10/sample_006 | DLR D-Flow | 97 | 93 | 5.1993 | 19.4934 | stalled |
| r96_n1_ct10/sample_007 | DLR D-Flow | 231 | 219 | 6.9318 | 19.3583 | stalled |
| r96_n1_ct10/sample_008 | DLR D-Flow | 68 | 65 | 114.4540 | 102.5939 | stalled |
| r96_n1_ct10/sample_009 | DLR D-Flow | 206 | 197 | 119.2955 | 104.4228 | stalled |
| r96_n1_ct10/sample_010 | DLR D-Flow | 491 | 475 | 6.4935 | 19.5784 | stalled |
| r96_n1_ct10/sample_011 | DLR D-Flow | 157 | 152 | 9.0875 | 19.5095 | stalled |
| r96_n1_ct10/sample_012 | DLR D-Flow | 960 | 928 | 6.1818 | 19.4209 | stalled |
| r96_n1_ct10/sample_013 | DLR D-Flow | 40 | 34 | 123.8219 | 112.0225 | stalled |
| r96_n1_ct10/sample_014 | DLR D-Flow | 43 | 36 | 116.5012 | 107.4848 | stalled |
| r96_n1_ct10/sample_015 | DLR D-Flow | 840 | 802 | 62.5477 | 43.3507 | stalled |
| r96_n1_ct10/sample_016 | DLR D-Flow | 104 | 101 | 9.2384 | 19.7873 | stalled |
| r96_n1_ct10/sample_017 | DLR D-Flow | 1000 | 902 | 5.8878 | 19.6805 | step_limit |
| r96_n1_ct10/sample_018 | DLR D-Flow | 391 | 371 | 8.0989 | 19.2706 | stalled |
| r96_n1_ct10/sample_019 | DLR D-Flow | 117 | 112 | 95.8624 | 69.5182 | stalled |
| r96_n1_ct10/sample_020 | DLR D-Flow | 296 | 284 | 111.8223 | 101.6697 | stalled |
| r96_n1_ct10/sample_021 | DLR D-Flow | 941 | 909 | 2.9264 | 19.4826 | stalled |
| r96_n1_ct10/sample_022 | DLR D-Flow | 131 | 126 | 6.6090 | 19.6231 | stalled |
| r96_n1_ct10/sample_023 | DLR D-Flow | 688 | 668 | 6.8657 | 19.2819 | stalled |
| r96_n1_ct10/sample_024 | DLR D-Flow | 584 | 555 | 4.1043 | 19.6369 | stalled |
| r96_n1_ct10/sample_025 | DLR D-Flow | 412 | 395 | 4.1309 | 19.5017 | stalled |
| r96_n1_ct10/sample_026 | DLR D-Flow | 114 | 109 | 124.6962 | 102.5234 | stalled |
| r96_n1_ct10/sample_027 | DLR D-Flow | 142 | 135 | 11.7778 | 19.2429 | stalled |
| r96_n1_ct10/sample_028 | DLR D-Flow | 105 | 97 | 120.1462 | 99.8835 | stalled |
| r96_n1_ct10/sample_029 | DLR D-Flow | 818 | 783 | 4.8227 | 19.7274 | stalled |
| r96_n2_ct10/sample_000 | D-Flow | 834 | 831 | 128.9501 | 1.0408 | stalled |
| r96_n2_ct10/sample_001 | D-Flow | 680 | 678 | 132.8051 | 1.5384 | stalled |
| r96_n2_ct10/sample_002 | D-Flow | 502 | 500 | 150.6685 | 1.1578 | stalled |
| r96_n2_ct10/sample_003 | D-Flow | 659 | 656 | 134.9272 | 0.8066 | stalled |
| r96_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 160.1823 | 1.1134 | step_limit |
| r96_n2_ct10/sample_005 | D-Flow | 675 | 672 | 109.0353 | 1.3293 | stalled |
| r96_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 62.5931 | 7.6398 | step_limit |
| r96_n2_ct10/sample_007 | D-Flow | 213 | 211 | 60.2090 | 39.5954 | stalled |
| r96_n2_ct10/sample_008 | D-Flow | 1000 | 1000 | 84.9134 | 2.3742 | step_limit |
| r96_n2_ct10/sample_009 | D-Flow | 30 | 28 | 83.5308 | 65.6135 | stalled |
| r96_n2_ct10/sample_010 | D-Flow | 833 | 831 | 113.2356 | 1.7379 | stalled |
| r96_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 105.8685 | 1.9228 | step_limit |
| r96_n2_ct10/sample_012 | D-Flow | 1000 | 1000 | 159.4978 | 1.5451 | step_limit |
| r96_n2_ct10/sample_000 | P-Flow | 656 | 598 | 94.9913 | 1.4230 | stalled |
| r96_n2_ct10/sample_001 | P-Flow | 1000 | 923 | 80.4066 | 2.2150 | step_limit |
| r96_n2_ct10/sample_002 | P-Flow | 494 | 478 | 105.1333 | 1.1582 | stalled |
| r96_n2_ct10/sample_003 | P-Flow | 1000 | 926 | 109.9450 | 0.8695 | step_limit |
| r96_n2_ct10/sample_004 | P-Flow | 80 | 61 | 8.9481 | 19.7033 | stalled |
| r96_n2_ct10/sample_005 | P-Flow | 494 | 482 | 101.8015 | 1.3295 | stalled |
| r96_n2_ct10/sample_006 | P-Flow | 53 | 42 | 6.3053 | 19.5805 | stalled |
| r96_n2_ct10/sample_007 | P-Flow | 123 | 109 | 9.6527 | 19.1147 | stalled |
| r96_n2_ct10/sample_008 | P-Flow | 727 | 694 | 89.0469 | 1.5203 | stalled |
| r96_n2_ct10/sample_009 | P-Flow | 456 | 379 | 11.6665 | 18.6608 | stalled |
| r96_n2_ct10/sample_010 | P-Flow | 26 | 22 | 13.8793 | 20.1856 | stalled |
| r96_n2_ct10/sample_011 | P-Flow | 754 | 718 | 130.9490 | 0.6236 | stalled |
| r96_n2_ct10/sample_012 | P-Flow | 14 | 11 | 74.0134 | 47.9612 | stalled |
| r96_n2_ct10/sample_000 | DLR D-Flow | 1000 | 969 | 61.4543 | 48.0826 | step_limit |
| r96_n2_ct10/sample_001 | DLR D-Flow | 1000 | 963 | 6.8464 | 19.6001 | step_limit |
| r96_n2_ct10/sample_002 | DLR D-Flow | 884 | 850 | 5.5648 | 19.5362 | stalled |
| r96_n2_ct10/sample_003 | DLR D-Flow | 305 | 294 | 8.7767 | 19.7344 | stalled |
| r96_n2_ct10/sample_004 | DLR D-Flow | 360 | 346 | 6.2730 | 19.4688 | stalled |
| r96_n2_ct10/sample_005 | DLR D-Flow | 265 | 252 | 5.2068 | 19.5465 | stalled |
| r96_n2_ct10/sample_006 | DLR D-Flow | 69 | 66 | 102.8755 | 73.9904 | stalled |
| r96_n2_ct10/sample_007 | DLR D-Flow | 164 | 157 | 62.6486 | 39.8916 | stalled |
| r96_n2_ct10/sample_008 | DLR D-Flow | 949 | 904 | 4.4355 | 19.7872 | stalled |
| r96_n2_ct10/sample_009 | DLR D-Flow | 164 | 159 | 78.1843 | 66.1789 | stalled |
| r96_n2_ct10/sample_010 | DLR D-Flow | 558 | 533 | 6.4422 | 19.7697 | stalled |
| r96_n2_ct10/sample_011 | DLR D-Flow | 912 | 882 | 4.0997 | 19.6627 | stalled |
| r96_n2_ct10/sample_012 | DLR D-Flow | 1000 | 972 | 81.7358 | 55.0794 | step_limit |
| r96_n4_ct10/sample_000 | D-Flow | 844 | 842 | 319.5132 | 1.6196 | stalled |
| r96_n4_ct10/sample_001 | D-Flow | 832 | 830 | 143.2725 | 1.4313 | stalled |
| r96_n4_ct10/sample_002 | D-Flow | 765 | 762 | 102.8521 | 0.8873 | stalled |
| r96_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 108.5750 | 0.7944 | step_limit |
| r96_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 36.4506 | 25.2179 | step_limit |
| r96_n4_ct10/sample_005 | D-Flow | 758 | 756 | 122.9536 | 1.3250 | stalled |
| r96_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 123.2375 | 1.6482 | step_limit |
| r96_n4_ct10/sample_007 | D-Flow | 852 | 850 | 477.0957 | 1.2729 | stalled |
| r96_n4_ct10/sample_008 | D-Flow | 768 | 766 | 122.2116 | 0.8399 | stalled |
| r96_n4_ct10/sample_009 | D-Flow | 728 | 726 | 152.3553 | 1.0115 | stalled |
| r96_n4_ct10/sample_010 | D-Flow | 677 | 675 | 241.3809 | 0.4657 | stalled |
| r96_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 13.2199 | 19.3679 | step_limit |
| r96_n4_ct10/sample_012 | D-Flow | 679 | 677 | 155.8852 | 2.5100 | stalled |
| r96_n4_ct10/sample_000 | P-Flow | 64 | 58 | 13.9901 | 19.4457 | stalled |
| r96_n4_ct10/sample_001 | P-Flow | 104 | 91 | 11.4127 | 19.4182 | stalled |
| r96_n4_ct10/sample_002 | P-Flow | 219 | 187 | 8.2395 | 19.4137 | stalled |
| r96_n4_ct10/sample_003 | P-Flow | 26 | 22 | 53.0849 | 29.4053 | stalled |
| r96_n4_ct10/sample_004 | P-Flow | 484 | 478 | 101.8258 | 2.3199 | stalled |
| r96_n4_ct10/sample_005 | P-Flow | 114 | 97 | 13.3945 | 19.5651 | stalled |
| r96_n4_ct10/sample_006 | P-Flow | 737 | 717 | 100.1647 | 1.6244 | stalled |
| r96_n4_ct10/sample_007 | P-Flow | 78 | 71 | 9.6935 | 19.4676 | stalled |
| r96_n4_ct10/sample_008 | P-Flow | 728 | 699 | 104.1334 | 0.8403 | stalled |
| r96_n4_ct10/sample_009 | P-Flow | 838 | 793 | 101.7246 | 1.0115 | stalled |
| r96_n4_ct10/sample_010 | P-Flow | 502 | 496 | 105.3059 | 0.4658 | stalled |
| r96_n4_ct10/sample_011 | P-Flow | 864 | 839 | 116.0160 | 1.1190 | stalled |
| r96_n4_ct10/sample_012 | P-Flow | 757 | 726 | 113.6567 | 2.5101 | stalled |
| r96_n4_ct10/sample_000 | DLR D-Flow | 1000 | 967 | 44.3166 | 27.5802 | step_limit |
| r96_n4_ct10/sample_001 | DLR D-Flow | 1000 | 967 | 35.3702 | 23.5022 | step_limit |
| r96_n4_ct10/sample_002 | DLR D-Flow | 342 | 336 | 23.9399 | 21.2472 | stalled |
| r96_n4_ct10/sample_003 | DLR D-Flow | 381 | 365 | 65.9156 | 42.9031 | stalled |
| r96_n4_ct10/sample_004 | DLR D-Flow | 326 | 314 | 92.0416 | 52.6228 | stalled |
| r96_n4_ct10/sample_005 | DLR D-Flow | 1000 | 963 | 52.1496 | 30.6375 | step_limit |
| r96_n4_ct10/sample_006 | DLR D-Flow | 975 | 934 | 54.4506 | 30.1290 | stalled |
| r96_n4_ct10/sample_007 | DLR D-Flow | 452 | 441 | 17.8964 | 19.7772 | stalled |
| r96_n4_ct10/sample_008 | DLR D-Flow | 378 | 368 | 26.0707 | 21.5856 | stalled |
| r96_n4_ct10/sample_009 | DLR D-Flow | 1000 | 964 | 52.6899 | 31.8954 | step_limit |
| r96_n4_ct10/sample_010 | DLR D-Flow | 849 | 816 | 58.7808 | 39.9529 | stalled |
| r96_n4_ct10/sample_011 | DLR D-Flow | 687 | 667 | 78.2919 | 41.4685 | stalled |
| r96_n4_ct10/sample_012 | DLR D-Flow | 646 | 622 | 40.7533 | 27.7848 | stalled |
| r96_n1_mri4/sample_000 | D-Flow | 445 | 442 | 25.4396 | 9.3779 | stalled |
| r96_n1_mri4/sample_001 | D-Flow | 981 | 977 | 26.0978 | 9.2787 | stalled |
| r96_n1_mri4/sample_002 | D-Flow | 664 | 662 | 23.6254 | 10.1220 | stalled |
| r96_n1_mri4/sample_003 | D-Flow | 404 | 402 | 8.2482 | 15.4154 | stalled |
| r96_n1_mri4/sample_004 | D-Flow | 729 | 727 | 11.5962 | 12.2331 | stalled |
| r96_n1_mri4/sample_005 | D-Flow | 527 | 524 | 22.5636 | 10.1382 | stalled |
| r96_n1_mri4/sample_006 | D-Flow | 465 | 463 | 24.7147 | 9.3069 | stalled |
| r96_n1_mri4/sample_007 | D-Flow | 488 | 486 | 24.0291 | 8.9307 | stalled |
| r96_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 23.7032 | 11.1461 | step_limit |
| r96_n1_mri4/sample_009 | D-Flow | 722 | 720 | 103.0997 | 101.2012 | stalled |
| r96_n1_mri4/sample_010 | D-Flow | 459 | 457 | 28.5014 | 10.1544 | stalled |
| r96_n1_mri4/sample_011 | D-Flow | 442 | 440 | 27.1604 | 10.2496 | stalled |
| r96_n1_mri4/sample_012 | D-Flow | 462 | 460 | 27.0036 | 8.4685 | stalled |
| r96_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 26.4147 | 8.5212 | step_limit |
| r96_n1_mri4/sample_014 | D-Flow | 392 | 390 | 103.6310 | 100.8754 | stalled |
| r96_n1_mri4/sample_015 | D-Flow | 704 | 702 | 8.7068 | 14.0184 | stalled |
| r96_n1_mri4/sample_016 | D-Flow | 498 | 495 | 23.9694 | 9.2328 | stalled |
| r96_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 4.8915 | 17.7587 | step_limit |
| r96_n1_mri4/sample_018 | D-Flow | 406 | 404 | 26.2515 | 8.6669 | stalled |
| r96_n1_mri4/sample_019 | D-Flow | 899 | 897 | 24.0166 | 9.7017 | stalled |
| r96_n1_mri4/sample_020 | D-Flow | 417 | 414 | 8.9765 | 14.7415 | stalled |
| r96_n1_mri4/sample_021 | D-Flow | 840 | 837 | 9.2040 | 14.7210 | stalled |
| r96_n1_mri4/sample_022 | D-Flow | 386 | 384 | 26.0211 | 11.7942 | stalled |
| r96_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 21.4691 | 14.8201 | step_limit |
| r96_n1_mri4/sample_024 | D-Flow | 789 | 787 | 20.7386 | 11.7405 | stalled |
| r96_n1_mri4/sample_025 | D-Flow | 595 | 593 | 23.2743 | 10.0053 | stalled |
| r96_n1_mri4/sample_026 | D-Flow | 93 | 91 | 103.5928 | 103.0689 | stalled |
| r96_n1_mri4/sample_027 | D-Flow | 554 | 551 | 8.7028 | 16.7893 | stalled |
| r96_n1_mri4/sample_028 | D-Flow | 614 | 611 | 26.6051 | 9.2756 | stalled |
| r96_n1_mri4/sample_029 | D-Flow | 481 | 479 | 8.1238 | 14.6696 | stalled |
| r96_n1_mri4/sample_000 | P-Flow | 269 | 246 | 27.1111 | 9.3779 | stalled |
| r96_n1_mri4/sample_001 | P-Flow | 13 | 10 | 117.1950 | 115.4056 | stalled |
| r96_n1_mri4/sample_002 | P-Flow | 177 | 170 | 27.2977 | 10.1220 | stalled |
| r96_n1_mri4/sample_003 | P-Flow | 178 | 171 | 26.4307 | 10.8859 | stalled |
| r96_n1_mri4/sample_004 | P-Flow | 40 | 32 | 3.3937 | 19.6436 | stalled |
| r96_n1_mri4/sample_005 | P-Flow | 172 | 165 | 24.0516 | 10.1382 | stalled |
| r96_n1_mri4/sample_006 | P-Flow | 191 | 177 | 25.8444 | 9.3069 | stalled |
| r96_n1_mri4/sample_007 | P-Flow | 185 | 180 | 24.5128 | 8.9307 | stalled |
| r96_n1_mri4/sample_008 | P-Flow | 170 | 165 | 25.0718 | 11.1457 | stalled |
| r96_n1_mri4/sample_009 | P-Flow | 18 | 15 | 120.5968 | 111.6085 | stalled |
| r96_n1_mri4/sample_010 | P-Flow | 204 | 201 | 26.2888 | 10.1544 | stalled |
| r96_n1_mri4/sample_011 | P-Flow | 174 | 170 | 26.6389 | 10.2496 | stalled |
| r96_n1_mri4/sample_012 | P-Flow | 38 | 33 | 2.5942 | 19.5560 | stalled |
| r96_n1_mri4/sample_013 | P-Flow | 167 | 161 | 28.9742 | 8.5212 | stalled |
| r96_n1_mri4/sample_014 | P-Flow | 6 | 4 | 136.2263 | 128.2934 | stalled |
| r96_n1_mri4/sample_015 | P-Flow | 12 | 10 | 112.9083 | 112.3953 | stalled |
| r96_n1_mri4/sample_016 | P-Flow | 245 | 217 | 27.7457 | 9.2328 | stalled |
| r96_n1_mri4/sample_017 | P-Flow | 29 | 24 | 1.6578 | 19.5929 | stalled |
| r96_n1_mri4/sample_018 | P-Flow | 178 | 173 | 25.8368 | 8.6669 | stalled |
| r96_n1_mri4/sample_019 | P-Flow | 173 | 166 | 24.5849 | 9.7017 | stalled |
| r96_n1_mri4/sample_020 | P-Flow | 20 | 16 | 2.1429 | 19.6368 | stalled |
| r96_n1_mri4/sample_021 | P-Flow | 39 | 30 | 2.5621 | 19.6887 | stalled |
| r96_n1_mri4/sample_022 | P-Flow | 183 | 176 | 24.7555 | 11.7942 | stalled |
| r96_n1_mri4/sample_023 | P-Flow | 197 | 190 | 24.7361 | 8.4931 | stalled |
| r96_n1_mri4/sample_024 | P-Flow | 77 | 67 | 3.4536 | 19.7060 | stalled |
| r96_n1_mri4/sample_025 | P-Flow | 169 | 165 | 27.7157 | 10.0053 | stalled |
| r96_n1_mri4/sample_026 | P-Flow | 55 | 48 | 2.6442 | 19.6619 | stalled |
| r96_n1_mri4/sample_027 | P-Flow | 26 | 21 | 1.4561 | 19.6191 | stalled |
| r96_n1_mri4/sample_028 | P-Flow | 70 | 56 | 2.0420 | 19.5987 | stalled |
| r96_n1_mri4/sample_029 | P-Flow | 21 | 17 | 2.4208 | 19.5529 | stalled |
| r96_n1_mri4/sample_000 | DLR D-Flow | 778 | 743 | 1.8822 | 19.6766 | stalled |
| r96_n1_mri4/sample_001 | DLR D-Flow | 185 | 180 | 102.5484 | 102.2149 | stalled |
| r96_n1_mri4/sample_002 | DLR D-Flow | 525 | 494 | 1.8444 | 19.5774 | stalled |
| r96_n1_mri4/sample_003 | DLR D-Flow | 89 | 83 | 1.9620 | 19.6652 | stalled |
| r96_n1_mri4/sample_004 | DLR D-Flow | 1000 | 961 | 3.9207 | 19.7067 | step_limit |
| r96_n1_mri4/sample_005 | DLR D-Flow | 402 | 383 | 1.6485 | 19.5883 | stalled |
| r96_n1_mri4/sample_006 | DLR D-Flow | 91 | 87 | 1.8870 | 19.6464 | stalled |
| r96_n1_mri4/sample_007 | DLR D-Flow | 1000 | 947 | 1.7244 | 19.6229 | step_limit |
| r96_n1_mri4/sample_008 | DLR D-Flow | 105 | 98 | 102.9053 | 102.7618 | stalled |
| r96_n1_mri4/sample_009 | DLR D-Flow | 137 | 131 | 103.7768 | 101.9136 | stalled |
| r96_n1_mri4/sample_010 | DLR D-Flow | 519 | 500 | 1.6579 | 19.6858 | stalled |
| r96_n1_mri4/sample_011 | DLR D-Flow | 403 | 384 | 1.7194 | 19.6066 | stalled |
| r96_n1_mri4/sample_012 | DLR D-Flow | 353 | 340 | 1.9197 | 19.6040 | stalled |
| r96_n1_mri4/sample_013 | DLR D-Flow | 26 | 21 | 124.5343 | 122.9813 | stalled |
| r96_n1_mri4/sample_014 | DLR D-Flow | 47 | 44 | 119.1145 | 112.4775 | stalled |
| r96_n1_mri4/sample_015 | DLR D-Flow | 580 | 553 | 2.7735 | 19.7129 | stalled |
| r96_n1_mri4/sample_016 | DLR D-Flow | 191 | 181 | 2.2420 | 19.6201 | stalled |
| r96_n1_mri4/sample_017 | DLR D-Flow | 740 | 651 | 1.9174 | 19.6032 | stalled |
| r96_n1_mri4/sample_018 | DLR D-Flow | 539 | 516 | 2.2856 | 19.6120 | stalled |
| r96_n1_mri4/sample_019 | DLR D-Flow | 90 | 84 | 102.4225 | 102.6796 | stalled |
| r96_n1_mri4/sample_020 | DLR D-Flow | 33 | 30 | 115.7159 | 114.8359 | stalled |
| r96_n1_mri4/sample_021 | DLR D-Flow | 1000 | 897 | 2.2160 | 19.7156 | step_limit |
| r96_n1_mri4/sample_022 | DLR D-Flow | 269 | 258 | 2.3478 | 19.7212 | stalled |
| r96_n1_mri4/sample_023 | DLR D-Flow | 674 | 649 | 1.5822 | 19.6742 | stalled |
| r96_n1_mri4/sample_024 | DLR D-Flow | 237 | 231 | 54.8470 | 51.1951 | stalled |
| r96_n1_mri4/sample_025 | DLR D-Flow | 142 | 130 | 4.1182 | 19.8410 | stalled |
| r96_n1_mri4/sample_026 | DLR D-Flow | 160 | 155 | 103.5809 | 103.1091 | stalled |
| r96_n1_mri4/sample_027 | DLR D-Flow | 155 | 145 | 1.4309 | 19.5979 | stalled |
| r96_n1_mri4/sample_028 | DLR D-Flow | 127 | 121 | 121.3192 | 117.6391 | stalled |
| r96_n1_mri4/sample_029 | DLR D-Flow | 79 | 74 | 2.2022 | 19.5086 | stalled |
| r96_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 5.5727 | 16.0649 | step_limit |
| r96_n2_mri4/sample_001 | D-Flow | 414 | 412 | 25.6954 | 10.2438 | stalled |
| r96_n2_mri4/sample_002 | D-Flow | 423 | 421 | 26.1272 | 7.5733 | stalled |
| r96_n2_mri4/sample_003 | D-Flow | 690 | 687 | 28.0215 | 10.0163 | stalled |
| r96_n2_mri4/sample_004 | D-Flow | 395 | 393 | 25.9733 | 10.2351 | stalled |
| r96_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 4.6472 | 16.9336 | step_limit |
| r96_n2_mri4/sample_006 | D-Flow | 937 | 934 | 27.2171 | 9.3236 | stalled |
| r96_n2_mri4/sample_007 | D-Flow | 817 | 814 | 51.5892 | 53.0237 | stalled |
| r96_n2_mri4/sample_008 | D-Flow | 623 | 621 | 22.0825 | 11.5609 | stalled |
| r96_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 5.7421 | 16.7461 | step_limit |
| r96_n2_mri4/sample_010 | D-Flow | 377 | 374 | 26.4749 | 11.0660 | stalled |
| r96_n2_mri4/sample_011 | D-Flow | 1000 | 1000 | 24.3092 | 9.4535 | step_limit |
| r96_n2_mri4/sample_012 | D-Flow | 639 | 637 | 28.1021 | 7.4054 | stalled |
| r96_n2_mri4/sample_000 | P-Flow | 170 | 165 | 22.4673 | 10.2103 | stalled |
| r96_n2_mri4/sample_001 | P-Flow | 199 | 187 | 26.3948 | 10.2438 | stalled |
| r96_n2_mri4/sample_002 | P-Flow | 205 | 192 | 27.9470 | 7.5733 | stalled |
| r96_n2_mri4/sample_003 | P-Flow | 174 | 168 | 26.9809 | 10.0163 | stalled |
| r96_n2_mri4/sample_004 | P-Flow | 183 | 177 | 26.1483 | 10.2351 | stalled |
| r96_n2_mri4/sample_005 | P-Flow | 177 | 174 | 24.4428 | 9.2309 | stalled |
| r96_n2_mri4/sample_006 | P-Flow | 49 | 41 | 7.0605 | 20.1372 | stalled |
| r96_n2_mri4/sample_007 | P-Flow | 189 | 181 | 28.0148 | 8.5002 | stalled |
| r96_n2_mri4/sample_008 | P-Flow | 226 | 210 | 24.6939 | 11.5609 | stalled |
| r96_n2_mri4/sample_009 | P-Flow | 215 | 204 | 24.7309 | 11.6406 | stalled |
| r96_n2_mri4/sample_010 | P-Flow | 232 | 219 | 24.4127 | 11.0660 | stalled |
| r96_n2_mri4/sample_011 | P-Flow | 176 | 171 | 24.0362 | 9.4535 | stalled |
| r96_n2_mri4/sample_012 | P-Flow | 173 | 170 | 26.0790 | 7.4054 | stalled |
| r96_n2_mri4/sample_000 | DLR D-Flow | 1000 | 970 | 73.4164 | 72.7034 | step_limit |
| r96_n2_mri4/sample_001 | DLR D-Flow | 620 | 597 | 2.8027 | 19.5975 | stalled |
| r96_n2_mri4/sample_002 | DLR D-Flow | 1000 | 974 | 84.4623 | 80.5401 | step_limit |
| r96_n2_mri4/sample_003 | DLR D-Flow | 1000 | 965 | 2.2943 | 19.5680 | step_limit |
| r96_n2_mri4/sample_004 | DLR D-Flow | 550 | 520 | 2.3660 | 19.5523 | stalled |
| r96_n2_mri4/sample_005 | DLR D-Flow | 148 | 142 | 1.9913 | 19.5947 | stalled |
| r96_n2_mri4/sample_006 | DLR D-Flow | 516 | 490 | 77.8595 | 75.6254 | stalled |
| r96_n2_mri4/sample_007 | DLR D-Flow | 722 | 703 | 2.4319 | 19.6259 | stalled |
| r96_n2_mri4/sample_008 | DLR D-Flow | 1000 | 975 | 4.4943 | 19.9244 | step_limit |
| r96_n2_mri4/sample_009 | DLR D-Flow | 158 | 150 | 75.3315 | 74.4716 | stalled |
| r96_n2_mri4/sample_010 | DLR D-Flow | 750 | 723 | 2.8005 | 19.7642 | stalled |
| r96_n2_mri4/sample_011 | DLR D-Flow | 77 | 71 | 54.1293 | 53.8625 | stalled |
| r96_n2_mri4/sample_012 | DLR D-Flow | 428 | 411 | 102.6232 | 100.9613 | stalled |
| r96_n4_mri4/sample_000 | D-Flow | 538 | 536 | 26.3717 | 7.3910 | stalled |
| r96_n4_mri4/sample_001 | D-Flow | 971 | 969 | 27.4408 | 10.1558 | stalled |
| r96_n4_mri4/sample_002 | D-Flow | 807 | 804 | 25.8835 | 9.4934 | stalled |
| r96_n4_mri4/sample_003 | D-Flow | 551 | 549 | 26.9817 | 10.0533 | stalled |
| r96_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 44.7865 | 45.8400 | step_limit |
| r96_n4_mri4/sample_005 | D-Flow | 587 | 585 | 24.3364 | 9.1550 | stalled |
| r96_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 26.1794 | 7.5911 | step_limit |
| r96_n4_mri4/sample_007 | D-Flow | 449 | 447 | 25.6070 | 11.1463 | stalled |
| r96_n4_mri4/sample_008 | D-Flow | 467 | 465 | 20.5055 | 13.0247 | stalled |
| r96_n4_mri4/sample_009 | D-Flow | 823 | 821 | 9.0308 | 16.7701 | stalled |
| r96_n4_mri4/sample_010 | D-Flow | 904 | 902 | 25.4776 | 10.8160 | stalled |
| r96_n4_mri4/sample_011 | D-Flow | 537 | 535 | 23.9872 | 10.7648 | stalled |
| r96_n4_mri4/sample_012 | D-Flow | 916 | 914 | 24.8982 | 9.3667 | stalled |
| r96_n4_mri4/sample_000 | P-Flow | 203 | 200 | 25.1431 | 7.3910 | stalled |
| r96_n4_mri4/sample_001 | P-Flow | 228 | 220 | 26.3011 | 10.1558 | stalled |
| r96_n4_mri4/sample_002 | P-Flow | 201 | 196 | 26.1184 | 9.4933 | stalled |
| r96_n4_mri4/sample_003 | P-Flow | 225 | 217 | 25.6787 | 10.0533 | stalled |
| r96_n4_mri4/sample_004 | P-Flow | 204 | 194 | 25.6980 | 9.5658 | stalled |
| r96_n4_mri4/sample_005 | P-Flow | 5 | 3 | 105.7077 | 90.0633 | stalled |
| r96_n4_mri4/sample_006 | P-Flow | 212 | 205 | 25.7784 | 7.5872 | stalled |
| r96_n4_mri4/sample_007 | P-Flow | 195 | 190 | 24.2533 | 11.1463 | stalled |
| r96_n4_mri4/sample_008 | P-Flow | 180 | 177 | 20.3093 | 13.0247 | stalled |
| r96_n4_mri4/sample_009 | P-Flow | 203 | 198 | 26.2269 | 8.3548 | stalled |
| r96_n4_mri4/sample_010 | P-Flow | 181 | 179 | 24.6966 | 10.8160 | stalled |
| r96_n4_mri4/sample_011 | P-Flow | 203 | 199 | 23.9675 | 10.7648 | stalled |
| r96_n4_mri4/sample_012 | P-Flow | 201 | 198 | 24.2045 | 9.3667 | stalled |
| r96_n4_mri4/sample_000 | DLR D-Flow | 179 | 169 | 63.1636 | 59.5479 | stalled |
| r96_n4_mri4/sample_001 | DLR D-Flow | 1000 | 959 | 39.4389 | 39.1219 | step_limit |
| r96_n4_mri4/sample_002 | DLR D-Flow | 290 | 278 | 34.7688 | 34.5495 | stalled |
| r96_n4_mri4/sample_003 | DLR D-Flow | 218 | 213 | 34.5341 | 34.5545 | stalled |
| r96_n4_mri4/sample_004 | DLR D-Flow | 317 | 305 | 84.3913 | 77.0240 | stalled |
| r96_n4_mri4/sample_005 | DLR D-Flow | 670 | 643 | 19.6859 | 24.4951 | stalled |
| r96_n4_mri4/sample_006 | DLR D-Flow | 474 | 456 | 48.8259 | 47.7351 | stalled |
| r96_n4_mri4/sample_007 | DLR D-Flow | 274 | 266 | 12.7235 | 21.9107 | stalled |
| r96_n4_mri4/sample_008 | DLR D-Flow | 1000 | 970 | 30.0487 | 32.0255 | step_limit |
| r96_n4_mri4/sample_009 | DLR D-Flow | 756 | 728 | 39.5133 | 42.7512 | stalled |
| r96_n4_mri4/sample_010 | DLR D-Flow | 76 | 72 | 91.6102 | 88.6582 | stalled |
| r96_n4_mri4/sample_011 | DLR D-Flow | 1000 | 970 | 58.7429 | 57.0507 | step_limit |
| r96_n4_mri4/sample_012 | DLR D-Flow | 1000 | 953 | 71.8456 | 70.5935 | step_limit |
| r128_n1_ct10/sample_000 | D-Flow | 466 | 464 | 104.6293 | 1.2631 | stalled |
| r128_n1_ct10/sample_001 | D-Flow | 1000 | 1000 | 12.4732 | 16.6640 | step_limit |
| r128_n1_ct10/sample_002 | D-Flow | 893 | 891 | 98.3433 | 1.6805 | stalled |
| r128_n1_ct10/sample_003 | D-Flow | 504 | 502 | 131.8717 | 1.3515 | stalled |
| r128_n1_ct10/sample_004 | D-Flow | 501 | 499 | 110.9609 | 0.9921 | stalled |
| r128_n1_ct10/sample_005 | D-Flow | 1000 | 1000 | 13.2549 | 14.4501 | step_limit |
| r128_n1_ct10/sample_006 | D-Flow | 496 | 494 | 195.1156 | 1.2984 | stalled |
| r128_n1_ct10/sample_007 | D-Flow | 1000 | 1000 | 147.1667 | 1.0882 | step_limit |
| r128_n1_ct10/sample_008 | D-Flow | 695 | 693 | 181.4716 | 0.8915 | stalled |
| r128_n1_ct10/sample_009 | D-Flow | 586 | 583 | 112.4893 | 1.1961 | stalled |
| r128_n1_ct10/sample_010 | D-Flow | 146 | 144 | 118.2757 | 111.1796 | stalled |
| r128_n1_ct10/sample_011 | D-Flow | 486 | 484 | 164.6320 | 1.3723 | stalled |
| r128_n1_ct10/sample_012 | D-Flow | 439 | 437 | 162.2666 | 1.2087 | stalled |
| r128_n1_ct10/sample_013 | D-Flow | 539 | 537 | 114.8684 | 0.5774 | stalled |
| r128_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 13.8632 | 12.2055 | step_limit |
| r128_n1_ct10/sample_015 | D-Flow | 1000 | 1000 | 11.2111 | 15.8689 | step_limit |
| r128_n1_ct10/sample_016 | D-Flow | 664 | 662 | 108.9863 | 0.8338 | stalled |
| r128_n1_ct10/sample_017 | D-Flow | 502 | 500 | 194.0091 | 0.6500 | stalled |
| r128_n1_ct10/sample_018 | D-Flow | 1000 | 1000 | 9.9309 | 17.2940 | step_limit |
| r128_n1_ct10/sample_019 | D-Flow | 628 | 626 | 117.6944 | 0.6987 | stalled |
| r128_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 9.6025 | 16.5957 | step_limit |
| r128_n1_ct10/sample_021 | D-Flow | 480 | 477 | 139.0037 | 1.5587 | stalled |
| r128_n1_ct10/sample_022 | D-Flow | 396 | 394 | 190.8659 | 1.2221 | stalled |
| r128_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 9.1095 | 16.6751 | step_limit |
| r128_n1_ct10/sample_024 | D-Flow | 569 | 567 | 100.4533 | 0.5404 | stalled |
| r128_n1_ct10/sample_025 | D-Flow | 669 | 667 | 96.4176 | 1.2179 | stalled |
| r128_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 13.9814 | 12.5858 | step_limit |
| r128_n1_ct10/sample_027 | D-Flow | 476 | 474 | 194.1226 | 0.6881 | stalled |
| r128_n1_ct10/sample_028 | D-Flow | 578 | 576 | 352.5402 | 0.7822 | stalled |
| r128_n1_ct10/sample_029 | D-Flow | 1000 | 1000 | 9.9358 | 16.8906 | step_limit |
| r128_n1_ct10/sample_000 | P-Flow | 517 | 501 | 77.5709 | 1.2653 | stalled |
| r128_n1_ct10/sample_001 | P-Flow | 29 | 24 | 3.2860 | 19.6678 | stalled |
| r128_n1_ct10/sample_002 | P-Flow | 42 | 36 | 5.5830 | 19.5625 | stalled |
| r128_n1_ct10/sample_003 | P-Flow | 1000 | 820 | 6.9217 | 19.5162 | step_limit |
| r128_n1_ct10/sample_004 | P-Flow | 1000 | 971 | 79.8865 | 1.0077 | step_limit |
| r128_n1_ct10/sample_005 | P-Flow | 412 | 399 | 95.7640 | 1.4430 | stalled |
| r128_n1_ct10/sample_006 | P-Flow | 436 | 425 | 98.9248 | 1.2986 | stalled |
| r128_n1_ct10/sample_007 | P-Flow | 21 | 17 | 5.3415 | 19.6138 | stalled |
| r128_n1_ct10/sample_008 | P-Flow | 122 | 99 | 5.2704 | 19.5811 | stalled |
| r128_n1_ct10/sample_009 | P-Flow | 45 | 37 | 3.7232 | 19.6636 | stalled |
| r128_n1_ct10/sample_010 | P-Flow | 38 | 31 | 5.1630 | 19.7190 | stalled |
| r128_n1_ct10/sample_011 | P-Flow | 53 | 43 | 4.7580 | 19.7288 | stalled |
| r128_n1_ct10/sample_012 | P-Flow | 34 | 28 | 6.9195 | 19.8471 | stalled |
| r128_n1_ct10/sample_013 | P-Flow | 49 | 40 | 32.9530 | 23.6734 | stalled |
| r128_n1_ct10/sample_014 | P-Flow | 26 | 21 | 12.6265 | 20.2489 | stalled |
| r128_n1_ct10/sample_015 | P-Flow | 34 | 28 | 4.9555 | 19.5687 | stalled |
| r128_n1_ct10/sample_016 | P-Flow | 1000 | 841 | 70.8199 | 1.8452 | step_limit |
| r128_n1_ct10/sample_017 | P-Flow | 424 | 418 | 107.0895 | 0.6500 | stalled |
| r128_n1_ct10/sample_018 | P-Flow | 44 | 38 | 2.1366 | 19.7479 | stalled |
| r128_n1_ct10/sample_019 | P-Flow | 62 | 53 | 4.4292 | 19.7365 | stalled |
| r128_n1_ct10/sample_020 | P-Flow | 43 | 37 | 9.4271 | 19.4682 | stalled |
| r128_n1_ct10/sample_021 | P-Flow | 560 | 537 | 90.3929 | 1.5587 | stalled |
| r128_n1_ct10/sample_022 | P-Flow | 457 | 446 | 85.0696 | 1.2221 | stalled |
| r128_n1_ct10/sample_023 | P-Flow | 41 | 33 | 13.1343 | 19.8760 | stalled |
| r128_n1_ct10/sample_024 | P-Flow | 30 | 25 | 4.0367 | 19.6740 | stalled |
| r128_n1_ct10/sample_025 | P-Flow | 377 | 371 | 97.0310 | 1.2182 | stalled |
| r128_n1_ct10/sample_026 | P-Flow | 19 | 15 | 7.1614 | 19.6811 | stalled |
| r128_n1_ct10/sample_027 | P-Flow | 50 | 43 | 5.0371 | 19.5413 | stalled |
| r128_n1_ct10/sample_028 | P-Flow | 567 | 539 | 106.2390 | 0.7822 | stalled |
| r128_n1_ct10/sample_029 | P-Flow | 554 | 543 | 108.8070 | 1.1405 | stalled |
| r128_n1_ct10/sample_000 | DLR D-Flow | 251 | 234 | 17.2743 | 21.2800 | stalled |
| r128_n1_ct10/sample_001 | DLR D-Flow | 159 | 150 | 5.9023 | 19.6067 | stalled |
| r128_n1_ct10/sample_002 | DLR D-Flow | 240 | 233 | 4.3131 | 19.5621 | stalled |
| r128_n1_ct10/sample_003 | DLR D-Flow | 929 | 896 | 9.9071 | 19.6390 | stalled |
| r128_n1_ct10/sample_004 | DLR D-Flow | 112 | 107 | 4.8552 | 19.7510 | stalled |
| r128_n1_ct10/sample_005 | DLR D-Flow | 168 | 161 | 6.9309 | 19.5063 | stalled |
| r128_n1_ct10/sample_006 | DLR D-Flow | 190 | 182 | 7.0059 | 19.5316 | stalled |
| r128_n1_ct10/sample_007 | DLR D-Flow | 54 | 50 | 7.9147 | 19.6652 | stalled |
| r128_n1_ct10/sample_008 | DLR D-Flow | 32 | 29 | 109.8840 | 99.7004 | stalled |
| r128_n1_ct10/sample_009 | DLR D-Flow | 427 | 407 | 6.5926 | 19.5712 | stalled |
| r128_n1_ct10/sample_010 | DLR D-Flow | 336 | 318 | 117.6983 | 110.8889 | stalled |
| r128_n1_ct10/sample_011 | DLR D-Flow | 374 | 358 | 5.1894 | 19.6579 | stalled |
| r128_n1_ct10/sample_012 | DLR D-Flow | 242 | 231 | 4.2686 | 19.7030 | stalled |
| r128_n1_ct10/sample_013 | DLR D-Flow | 146 | 138 | 4.5086 | 19.6337 | stalled |
| r128_n1_ct10/sample_014 | DLR D-Flow | 101 | 95 | 30.4987 | 22.6982 | stalled |
| r128_n1_ct10/sample_015 | DLR D-Flow | 434 | 413 | 10.6633 | 19.4980 | stalled |
| r128_n1_ct10/sample_016 | DLR D-Flow | 416 | 397 | 4.1349 | 19.5519 | stalled |
| r128_n1_ct10/sample_017 | DLR D-Flow | 461 | 439 | 6.8445 | 19.6283 | stalled |
| r128_n1_ct10/sample_018 | DLR D-Flow | 169 | 159 | 3.3916 | 19.7165 | stalled |
| r128_n1_ct10/sample_019 | DLR D-Flow | 621 | 592 | 4.5928 | 19.7323 | stalled |
| r128_n1_ct10/sample_020 | DLR D-Flow | 702 | 682 | 7.3314 | 19.3712 | stalled |
| r128_n1_ct10/sample_021 | DLR D-Flow | 302 | 288 | 7.9073 | 19.5713 | stalled |
| r128_n1_ct10/sample_022 | DLR D-Flow | 74 | 71 | 5.7660 | 19.3392 | stalled |
| r128_n1_ct10/sample_023 | DLR D-Flow | 418 | 401 | 6.1278 | 19.5616 | stalled |
| r128_n1_ct10/sample_024 | DLR D-Flow | 371 | 354 | 4.3956 | 19.5973 | stalled |
| r128_n1_ct10/sample_025 | DLR D-Flow | 198 | 188 | 42.0413 | 33.9038 | stalled |
| r128_n1_ct10/sample_026 | DLR D-Flow | 69 | 66 | 13.7920 | 20.3053 | stalled |
| r128_n1_ct10/sample_027 | DLR D-Flow | 112 | 110 | 8.0580 | 19.4572 | stalled |
| r128_n1_ct10/sample_028 | DLR D-Flow | 208 | 201 | 15.2536 | 19.4054 | stalled |
| r128_n1_ct10/sample_029 | DLR D-Flow | 266 | 250 | 4.5026 | 19.6412 | stalled |
| r128_n2_ct10/sample_000 | D-Flow | 1000 | 1000 | 14.8699 | 18.7329 | step_limit |
| r128_n2_ct10/sample_001 | D-Flow | 575 | 572 | 141.8223 | 0.5451 | stalled |
| r128_n2_ct10/sample_002 | D-Flow | 626 | 624 | 146.6182 | 1.2446 | stalled |
| r128_n2_ct10/sample_003 | D-Flow | 504 | 502 | 209.5359 | 1.9562 | stalled |
| r128_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 91.1749 | 1.2206 | step_limit |
| r128_n2_ct10/sample_005 | D-Flow | 647 | 644 | 149.0047 | 1.3313 | stalled |
| r128_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 126.4871 | 1.7265 | step_limit |
| r128_n2_ct10/sample_007 | D-Flow | 481 | 479 | 183.1040 | 1.4639 | stalled |
| r128_n2_ct10/sample_008 | D-Flow | 579 | 577 | 146.8785 | 0.9345 | stalled |
| r128_n2_ct10/sample_009 | D-Flow | 868 | 866 | 89.1182 | 1.3079 | stalled |
| r128_n2_ct10/sample_010 | D-Flow | 619 | 617 | 114.3197 | 2.2896 | stalled |
| r128_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 101.3678 | 0.5888 | step_limit |
| r128_n2_ct10/sample_012 | D-Flow | 569 | 567 | 164.8110 | 1.1732 | stalled |
| r128_n2_ct10/sample_000 | P-Flow | 46 | 40 | 4.2159 | 19.4210 | stalled |
| r128_n2_ct10/sample_001 | P-Flow | 480 | 469 | 88.9816 | 0.5451 | stalled |
| r128_n2_ct10/sample_002 | P-Flow | 60 | 50 | 7.9606 | 19.6824 | stalled |
| r128_n2_ct10/sample_003 | P-Flow | 282 | 244 | 9.0308 | 18.2713 | stalled |
| r128_n2_ct10/sample_004 | P-Flow | 1000 | 866 | 72.8624 | 1.2774 | step_limit |
| r128_n2_ct10/sample_005 | P-Flow | 108 | 94 | 3.5462 | 19.7182 | stalled |
| r128_n2_ct10/sample_006 | P-Flow | 1000 | 931 | 93.7012 | 1.7409 | step_limit |
| r128_n2_ct10/sample_007 | P-Flow | 439 | 431 | 89.0879 | 1.4639 | stalled |
| r128_n2_ct10/sample_008 | P-Flow | 17 | 14 | 44.8227 | 26.1282 | stalled |
| r128_n2_ct10/sample_009 | P-Flow | 41 | 33 | 8.3883 | 19.7232 | stalled |
| r128_n2_ct10/sample_010 | P-Flow | 507 | 495 | 91.9359 | 2.2896 | stalled |
| r128_n2_ct10/sample_011 | P-Flow | 291 | 245 | 8.1495 | 19.4988 | stalled |
| r128_n2_ct10/sample_012 | P-Flow | 570 | 558 | 99.7936 | 1.1732 | stalled |
| r128_n2_ct10/sample_000 | DLR D-Flow | 42 | 38 | 114.2726 | 89.8387 | stalled |
| r128_n2_ct10/sample_001 | DLR D-Flow | 1000 | 955 | 70.0700 | 44.6684 | step_limit |
| r128_n2_ct10/sample_002 | DLR D-Flow | 309 | 300 | 66.7594 | 42.8597 | stalled |
| r128_n2_ct10/sample_003 | DLR D-Flow | 447 | 430 | 117.8625 | 86.3227 | stalled |
| r128_n2_ct10/sample_004 | DLR D-Flow | 1000 | 963 | 4.7319 | 19.4776 | step_limit |
| r128_n2_ct10/sample_005 | DLR D-Flow | 115 | 110 | 4.5179 | 19.7392 | stalled |
| r128_n2_ct10/sample_006 | DLR D-Flow | 287 | 266 | 9.2954 | 19.6174 | stalled |
| r128_n2_ct10/sample_007 | DLR D-Flow | 207 | 202 | 72.8794 | 45.8984 | stalled |
| r128_n2_ct10/sample_008 | DLR D-Flow | 863 | 838 | 8.2705 | 19.5705 | stalled |
| r128_n2_ct10/sample_009 | DLR D-Flow | 99 | 92 | 119.8456 | 90.5721 | stalled |
| r128_n2_ct10/sample_010 | DLR D-Flow | 679 | 657 | 4.6096 | 19.5556 | stalled |
| r128_n2_ct10/sample_011 | DLR D-Flow | 151 | 144 | 87.1800 | 60.7125 | stalled |
| r128_n2_ct10/sample_012 | DLR D-Flow | 189 | 186 | 10.6837 | 19.9750 | stalled |
| r128_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 50.5452 | 31.5329 | step_limit |
| r128_n4_ct10/sample_001 | D-Flow | 942 | 940 | 113.3108 | 0.8406 | stalled |
| r128_n4_ct10/sample_002 | D-Flow | 1000 | 1000 | 46.3419 | 28.7246 | step_limit |
| r128_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 42.9113 | 30.9914 | step_limit |
| r128_n4_ct10/sample_004 | D-Flow | 932 | 930 | 122.1596 | 0.6318 | stalled |
| r128_n4_ct10/sample_005 | D-Flow | 1000 | 1000 | 61.6796 | 36.6621 | step_limit |
| r128_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 25.2658 | 18.6574 | step_limit |
| r128_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 48.6822 | 32.0449 | step_limit |
| r128_n4_ct10/sample_008 | D-Flow | 1000 | 1000 | 14.9876 | 19.1840 | step_limit |
| r128_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 100.4381 | 1.3401 | step_limit |
| r128_n4_ct10/sample_010 | D-Flow | 1000 | 1000 | 110.0831 | 1.4210 | step_limit |
| r128_n4_ct10/sample_011 | D-Flow | 519 | 516 | 61.6497 | 38.3614 | stalled |
| r128_n4_ct10/sample_012 | D-Flow | 1000 | 1000 | 51.0751 | 37.7907 | step_limit |
| r128_n4_ct10/sample_000 | P-Flow | 78 | 70 | 11.3284 | 19.3670 | stalled |
| r128_n4_ct10/sample_001 | P-Flow | 118 | 105 | 11.3913 | 19.3005 | stalled |
| r128_n4_ct10/sample_002 | P-Flow | 43 | 39 | 10.8520 | 19.5318 | stalled |
| r128_n4_ct10/sample_003 | P-Flow | 63 | 57 | 8.6730 | 19.6378 | stalled |
| r128_n4_ct10/sample_004 | P-Flow | 98 | 87 | 11.5639 | 19.6570 | stalled |
| r128_n4_ct10/sample_005 | P-Flow | 1000 | 844 | 59.6923 | 4.2733 | step_limit |
| r128_n4_ct10/sample_006 | P-Flow | 1000 | 918 | 73.8892 | 1.8637 | step_limit |
| r128_n4_ct10/sample_007 | P-Flow | 99 | 88 | 10.2382 | 19.6157 | stalled |
| r128_n4_ct10/sample_008 | P-Flow | 1000 | 901 | 75.9241 | 1.8121 | step_limit |
| r128_n4_ct10/sample_009 | P-Flow | 207 | 201 | 57.7210 | 3.5400 | stalled |
| r128_n4_ct10/sample_010 | P-Flow | 1000 | 882 | 64.3820 | 2.6961 | step_limit |
| r128_n4_ct10/sample_011 | P-Flow | 1000 | 905 | 79.9735 | 1.2356 | step_limit |
| r128_n4_ct10/sample_012 | P-Flow | 1000 | 936 | 96.5501 | 1.6977 | step_limit |
| r128_n4_ct10/sample_000 | DLR D-Flow | 1000 | 962 | 15.4837 | 19.9350 | step_limit |
| r128_n4_ct10/sample_001 | DLR D-Flow | 1000 | 954 | 27.8713 | 22.4531 | step_limit |
| r128_n4_ct10/sample_002 | DLR D-Flow | 56 | 52 | 95.5240 | 53.4533 | stalled |
| r128_n4_ct10/sample_003 | DLR D-Flow | 700 | 671 | 61.8272 | 38.5506 | stalled |
| r128_n4_ct10/sample_004 | DLR D-Flow | 58 | 55 | 74.2543 | 40.7515 | stalled |
| r128_n4_ct10/sample_005 | DLR D-Flow | 135 | 130 | 83.9706 | 48.0627 | stalled |
| r128_n4_ct10/sample_006 | DLR D-Flow | 684 | 658 | 10.0923 | 19.5477 | stalled |
| r128_n4_ct10/sample_007 | DLR D-Flow | 83 | 76 | 98.3431 | 59.3963 | stalled |
| r128_n4_ct10/sample_008 | DLR D-Flow | 1000 | 967 | 19.6327 | 20.5983 | step_limit |
| r128_n4_ct10/sample_009 | DLR D-Flow | 237 | 230 | 17.3644 | 20.2359 | stalled |
| r128_n4_ct10/sample_010 | DLR D-Flow | 1000 | 970 | 57.3952 | 31.5425 | step_limit |
| r128_n4_ct10/sample_011 | DLR D-Flow | 869 | 836 | 95.2529 | 56.1365 | stalled |
| r128_n4_ct10/sample_012 | DLR D-Flow | 95 | 89 | 76.1152 | 48.3062 | stalled |
| r128_n1_mri4/sample_000 | D-Flow | 528 | 525 | 25.8547 | 8.7701 | stalled |
| r128_n1_mri4/sample_001 | D-Flow | 607 | 604 | 25.3902 | 10.4848 | stalled |
| r128_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 9.7968 | 14.0458 | step_limit |
| r128_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 8.3242 | 15.0562 | step_limit |
| r128_n1_mri4/sample_004 | D-Flow | 465 | 461 | 21.6191 | 10.7224 | stalled |
| r128_n1_mri4/sample_005 | D-Flow | 449 | 446 | 23.4889 | 10.8370 | stalled |
| r128_n1_mri4/sample_006 | D-Flow | 509 | 507 | 25.2048 | 9.3616 | stalled |
| r128_n1_mri4/sample_007 | D-Flow | 1000 | 1000 | 7.8900 | 15.6489 | step_limit |
| r128_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 9.2539 | 14.0997 | step_limit |
| r128_n1_mri4/sample_009 | D-Flow | 385 | 383 | 24.5706 | 10.1460 | stalled |
| r128_n1_mri4/sample_010 | D-Flow | 256 | 252 | 105.0004 | 102.5678 | stalled |
| r128_n1_mri4/sample_011 | D-Flow | 526 | 524 | 22.6004 | 10.8811 | stalled |
| r128_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 7.9796 | 15.0201 | step_limit |
| r128_n1_mri4/sample_013 | D-Flow | 715 | 713 | 24.8050 | 7.9589 | stalled |
| r128_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 13.6336 | 11.7270 | step_limit |
| r128_n1_mri4/sample_015 | D-Flow | 488 | 486 | 22.4600 | 9.5197 | stalled |
| r128_n1_mri4/sample_016 | D-Flow | 1000 | 1000 | 10.7200 | 13.8648 | step_limit |
| r128_n1_mri4/sample_017 | D-Flow | 449 | 447 | 30.5595 | 9.3900 | stalled |
| r128_n1_mri4/sample_018 | D-Flow | 428 | 425 | 27.0683 | 10.5047 | stalled |
| r128_n1_mri4/sample_019 | D-Flow | 362 | 360 | 24.1300 | 9.3364 | stalled |
| r128_n1_mri4/sample_020 | D-Flow | 1000 | 1000 | 7.5722 | 15.7555 | step_limit |
| r128_n1_mri4/sample_021 | D-Flow | 293 | 291 | 23.6393 | 11.2941 | stalled |
| r128_n1_mri4/sample_022 | D-Flow | 1000 | 1000 | 13.0260 | 11.2174 | step_limit |
| r128_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 7.1760 | 15.9075 | step_limit |
| r128_n1_mri4/sample_024 | D-Flow | 383 | 380 | 26.2424 | 9.0080 | stalled |
| r128_n1_mri4/sample_025 | D-Flow | 549 | 547 | 22.6952 | 10.2941 | stalled |
| r128_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 13.0694 | 12.2885 | step_limit |
| r128_n1_mri4/sample_027 | D-Flow | 571 | 569 | 29.9019 | 11.3495 | stalled |
| r128_n1_mri4/sample_028 | D-Flow | 459 | 456 | 26.1077 | 9.8774 | stalled |
| r128_n1_mri4/sample_029 | D-Flow | 562 | 559 | 26.1612 | 9.1328 | stalled |
| r128_n1_mri4/sample_000 | P-Flow | 171 | 165 | 28.3655 | 8.7701 | stalled |
| r128_n1_mri4/sample_001 | P-Flow | 26 | 21 | 1.7010 | 19.6861 | stalled |
| r128_n1_mri4/sample_002 | P-Flow | 38 | 30 | 2.3033 | 19.6409 | stalled |
| r128_n1_mri4/sample_003 | P-Flow | 170 | 161 | 27.0506 | 10.6348 | stalled |
| r128_n1_mri4/sample_004 | P-Flow | 160 | 151 | 24.5840 | 10.7224 | stalled |
| r128_n1_mri4/sample_005 | P-Flow | 176 | 165 | 27.4381 | 10.8370 | stalled |
| r128_n1_mri4/sample_006 | P-Flow | 156 | 151 | 28.7730 | 9.3616 | stalled |
| r128_n1_mri4/sample_007 | P-Flow | 189 | 180 | 29.0072 | 9.6684 | stalled |
| r128_n1_mri4/sample_008 | P-Flow | 31 | 24 | 2.0185 | 19.7143 | stalled |
| r128_n1_mri4/sample_009 | P-Flow | 153 | 148 | 28.7625 | 10.1460 | stalled |
| r128_n1_mri4/sample_010 | P-Flow | 223 | 199 | 27.8560 | 9.7810 | stalled |
| r128_n1_mri4/sample_011 | P-Flow | 158 | 155 | 24.2515 | 10.8811 | stalled |
| r128_n1_mri4/sample_012 | P-Flow | 40 | 34 | 1.9091 | 19.6851 | stalled |
| r128_n1_mri4/sample_013 | P-Flow | 194 | 186 | 26.1820 | 7.9589 | stalled |
| r128_n1_mri4/sample_014 | P-Flow | 168 | 164 | 28.7444 | 9.4398 | stalled |
| r128_n1_mri4/sample_015 | P-Flow | 157 | 151 | 26.5754 | 9.5197 | stalled |
| r128_n1_mri4/sample_016 | P-Flow | 155 | 149 | 29.2624 | 10.1218 | stalled |
| r128_n1_mri4/sample_017 | P-Flow | 169 | 166 | 28.3802 | 9.3899 | stalled |
| r128_n1_mri4/sample_018 | P-Flow | 19 | 15 | 2.5463 | 19.5636 | stalled |
| r128_n1_mri4/sample_019 | P-Flow | 160 | 154 | 25.1634 | 9.3364 | stalled |
| r128_n1_mri4/sample_020 | P-Flow | 163 | 160 | 26.5815 | 11.2898 | stalled |
| r128_n1_mri4/sample_021 | P-Flow | 173 | 161 | 26.6404 | 11.2941 | stalled |
| r128_n1_mri4/sample_022 | P-Flow | 157 | 152 | 23.7261 | 10.1240 | stalled |
| r128_n1_mri4/sample_023 | P-Flow | 25 | 20 | 1.6477 | 19.6300 | stalled |
| r128_n1_mri4/sample_024 | P-Flow | 35 | 28 | 2.2625 | 19.5752 | stalled |
| r128_n1_mri4/sample_025 | P-Flow | 46 | 36 | 2.5748 | 19.7147 | stalled |
| r128_n1_mri4/sample_026 | P-Flow | 22 | 16 | 2.9026 | 19.7324 | stalled |
| r128_n1_mri4/sample_027 | P-Flow | 171 | 165 | 26.7413 | 11.3495 | stalled |
| r128_n1_mri4/sample_028 | P-Flow | 161 | 158 | 28.6219 | 9.8774 | stalled |
| r128_n1_mri4/sample_029 | P-Flow | 158 | 154 | 28.5853 | 9.1328 | stalled |
| r128_n1_mri4/sample_000 | DLR D-Flow | 458 | 440 | 5.3003 | 20.0780 | stalled |
| r128_n1_mri4/sample_001 | DLR D-Flow | 478 | 461 | 1.5786 | 19.6674 | stalled |
| r128_n1_mri4/sample_002 | DLR D-Flow | 89 | 86 | 2.8722 | 19.6755 | stalled |
| r128_n1_mri4/sample_003 | DLR D-Flow | 1000 | 972 | 101.2912 | 100.5855 | step_limit |
| r128_n1_mri4/sample_004 | DLR D-Flow | 124 | 118 | 2.2591 | 19.6725 | stalled |
| r128_n1_mri4/sample_005 | DLR D-Flow | 665 | 634 | 1.9427 | 19.6684 | stalled |
| r128_n1_mri4/sample_006 | DLR D-Flow | 173 | 165 | 2.0547 | 19.6826 | stalled |
| r128_n1_mri4/sample_007 | DLR D-Flow | 383 | 366 | 1.6753 | 19.6056 | stalled |
| r128_n1_mri4/sample_008 | DLR D-Flow | 21 | 16 | 104.4751 | 104.2350 | stalled |
| r128_n1_mri4/sample_009 | DLR D-Flow | 253 | 246 | 3.1575 | 19.7020 | stalled |
| r128_n1_mri4/sample_010 | DLR D-Flow | 130 | 124 | 106.4353 | 104.6582 | stalled |
| r128_n1_mri4/sample_011 | DLR D-Flow | 136 | 131 | 2.3172 | 19.7487 | stalled |
| r128_n1_mri4/sample_012 | DLR D-Flow | 213 | 204 | 2.2678 | 19.6795 | stalled |
| r128_n1_mri4/sample_013 | DLR D-Flow | 1000 | 965 | 55.2762 | 55.2907 | step_limit |
| r128_n1_mri4/sample_014 | DLR D-Flow | 461 | 440 | 5.5893 | 20.2204 | stalled |
| r128_n1_mri4/sample_015 | DLR D-Flow | 183 | 175 | 2.5330 | 19.7803 | stalled |
| r128_n1_mri4/sample_016 | DLR D-Flow | 381 | 363 | 2.0923 | 19.6773 | stalled |
| r128_n1_mri4/sample_017 | DLR D-Flow | 120 | 114 | 3.9258 | 19.7671 | stalled |
| r128_n1_mri4/sample_018 | DLR D-Flow | 572 | 545 | 1.6247 | 19.5001 | stalled |
| r128_n1_mri4/sample_019 | DLR D-Flow | 383 | 371 | 8.1807 | 20.7872 | stalled |
| r128_n1_mri4/sample_020 | DLR D-Flow | 1000 | 971 | 1.9122 | 19.6767 | step_limit |
| r128_n1_mri4/sample_021 | DLR D-Flow | 914 | 867 | 1.5692 | 19.5860 | stalled |
| r128_n1_mri4/sample_022 | DLR D-Flow | 75 | 71 | 2.3723 | 19.6128 | stalled |
| r128_n1_mri4/sample_023 | DLR D-Flow | 156 | 151 | 2.6797 | 19.6985 | stalled |
| r128_n1_mri4/sample_024 | DLR D-Flow | 201 | 192 | 5.5474 | 20.0083 | stalled |
| r128_n1_mri4/sample_025 | DLR D-Flow | 89 | 84 | 50.9517 | 49.4917 | stalled |
| r128_n1_mri4/sample_026 | DLR D-Flow | 173 | 160 | 2.8668 | 19.7300 | stalled |
| r128_n1_mri4/sample_027 | DLR D-Flow | 656 | 629 | 2.0903 | 19.6811 | stalled |
| r128_n1_mri4/sample_028 | DLR D-Flow | 355 | 342 | 1.7831 | 19.6680 | stalled |
| r128_n1_mri4/sample_029 | DLR D-Flow | 328 | 309 | 1.5941 | 19.6733 | stalled |
| r128_n2_mri4/sample_000 | D-Flow | 1000 | 1000 | 26.0441 | 9.6680 | step_limit |
| r128_n2_mri4/sample_001 | D-Flow | 851 | 848 | 22.2922 | 8.7419 | stalled |
| r128_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 5.6092 | 18.4524 | step_limit |
| r128_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 27.0962 | 9.5849 | step_limit |
| r128_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 3.8959 | 18.9618 | step_limit |
| r128_n2_mri4/sample_005 | D-Flow | 835 | 833 | 26.8614 | 9.6029 | stalled |
| r128_n2_mri4/sample_006 | D-Flow | 495 | 493 | 25.3988 | 9.5550 | stalled |
| r128_n2_mri4/sample_007 | D-Flow | 1000 | 1000 | 2.2810 | 19.5991 | step_limit |
| r128_n2_mri4/sample_008 | D-Flow | 421 | 419 | 25.2082 | 9.6240 | stalled |
| r128_n2_mri4/sample_009 | D-Flow | 692 | 690 | 22.0035 | 10.9173 | stalled |
| r128_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 5.4991 | 18.2518 | step_limit |
| r128_n2_mri4/sample_011 | D-Flow | 736 | 733 | 40.5189 | 44.0546 | stalled |
| r128_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 81.5992 | 12.1252 | step_limit |
| r128_n2_mri4/sample_000 | P-Flow | 178 | 169 | 28.4488 | 9.6649 | stalled |
| r128_n2_mri4/sample_001 | P-Flow | 176 | 172 | 24.3740 | 8.7419 | stalled |
| r128_n2_mri4/sample_002 | P-Flow | 189 | 184 | 25.4414 | 10.9820 | stalled |
| r128_n2_mri4/sample_003 | P-Flow | 170 | 165 | 27.6769 | 9.5822 | stalled |
| r128_n2_mri4/sample_004 | P-Flow | 168 | 162 | 27.3335 | 10.7935 | stalled |
| r128_n2_mri4/sample_005 | P-Flow | 165 | 161 | 27.9400 | 9.6029 | stalled |
| r128_n2_mri4/sample_006 | P-Flow | 190 | 182 | 27.0899 | 9.5550 | stalled |
| r128_n2_mri4/sample_007 | P-Flow | 174 | 166 | 27.6961 | 9.4753 | stalled |
| r128_n2_mri4/sample_008 | P-Flow | 163 | 160 | 26.7231 | 9.6240 | stalled |
| r128_n2_mri4/sample_009 | P-Flow | 168 | 162 | 25.2668 | 10.9173 | stalled |
| r128_n2_mri4/sample_010 | P-Flow | 178 | 173 | 26.8042 | 9.6469 | stalled |
| r128_n2_mri4/sample_011 | P-Flow | 184 | 175 | 27.2932 | 9.9027 | stalled |
| r128_n2_mri4/sample_012 | P-Flow | 21 | 17 | 96.1658 | 64.6447 | stalled |
| r128_n2_mri4/sample_000 | DLR D-Flow | 1000 | 974 | 2.7346 | 19.6778 | step_limit |
| r128_n2_mri4/sample_001 | DLR D-Flow | 125 | 118 | 3.9938 | 19.8699 | stalled |
| r128_n2_mri4/sample_002 | DLR D-Flow | 922 | 890 | 3.7973 | 19.8487 | stalled |
| r128_n2_mri4/sample_003 | DLR D-Flow | 30 | 27 | 119.6418 | 118.5255 | stalled |
| r128_n2_mri4/sample_004 | DLR D-Flow | 887 | 863 | 5.9208 | 19.9408 | stalled |
| r128_n2_mri4/sample_005 | DLR D-Flow | 111 | 103 | 70.4975 | 68.4846 | stalled |
| r128_n2_mri4/sample_006 | DLR D-Flow | 181 | 173 | 1.8207 | 19.6516 | stalled |
| r128_n2_mri4/sample_007 | DLR D-Flow | 228 | 221 | 68.2530 | 68.1676 | stalled |
| r128_n2_mri4/sample_008 | DLR D-Flow | 821 | 788 | 1.4266 | 19.6392 | stalled |
| r128_n2_mri4/sample_009 | DLR D-Flow | 38 | 34 | 127.1474 | 120.4045 | stalled |
| r128_n2_mri4/sample_010 | DLR D-Flow | 286 | 272 | 9.7687 | 21.0953 | stalled |
| r128_n2_mri4/sample_011 | DLR D-Flow | 61 | 57 | 91.1001 | 90.3485 | stalled |
| r128_n2_mri4/sample_012 | DLR D-Flow | 25 | 22 | 89.6710 | 81.3818 | stalled |
| r128_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 41.7780 | 42.5700 | step_limit |
| r128_n4_mri4/sample_001 | D-Flow | 667 | 665 | 25.7375 | 10.2334 | stalled |
| r128_n4_mri4/sample_002 | D-Flow | 919 | 915 | 56.3937 | 56.7600 | stalled |
| r128_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 6.0483 | 19.1343 | step_limit |
| r128_n4_mri4/sample_004 | D-Flow | 618 | 614 | 25.9753 | 9.5041 | stalled |
| r128_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 40.7866 | 44.1845 | step_limit |
| r128_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 36.2584 | 38.0518 | step_limit |
| r128_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 2.6687 | 19.5006 | step_limit |
| r128_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 22.7928 | 10.9582 | step_limit |
| r128_n4_mri4/sample_009 | D-Flow | 700 | 698 | 26.2887 | 10.1219 | stalled |
| r128_n4_mri4/sample_010 | D-Flow | 1000 | 1000 | 18.6818 | 14.8129 | step_limit |
| r128_n4_mri4/sample_011 | D-Flow | 1000 | 1000 | 25.4794 | 8.8986 | step_limit |
| r128_n4_mri4/sample_012 | D-Flow | 1000 | 1000 | 46.8320 | 49.7323 | step_limit |
| r128_n4_mri4/sample_000 | P-Flow | 203 | 200 | 26.6224 | 8.0618 | stalled |
| r128_n4_mri4/sample_001 | P-Flow | 57 | 46 | 2.4091 | 19.7191 | stalled |
| r128_n4_mri4/sample_002 | P-Flow | 192 | 188 | 25.6594 | 10.1106 | stalled |
| r128_n4_mri4/sample_003 | P-Flow | 241 | 226 | 24.1696 | 11.2360 | stalled |
| r128_n4_mri4/sample_004 | P-Flow | 198 | 192 | 26.0967 | 9.5041 | stalled |
| r128_n4_mri4/sample_005 | P-Flow | 257 | 245 | 24.2419 | 11.1229 | stalled |
| r128_n4_mri4/sample_006 | P-Flow | 205 | 201 | 24.7873 | 8.1311 | stalled |
| r128_n4_mri4/sample_007 | P-Flow | 184 | 181 | 22.8871 | 10.8141 | stalled |
| r128_n4_mri4/sample_008 | P-Flow | 201 | 197 | 23.0995 | 10.8408 | stalled |
| r128_n4_mri4/sample_009 | P-Flow | 190 | 184 | 25.2169 | 10.1219 | stalled |
| r128_n4_mri4/sample_010 | P-Flow | 212 | 206 | 24.4955 | 9.2193 | stalled |
| r128_n4_mri4/sample_011 | P-Flow | 225 | 217 | 25.2267 | 8.8986 | stalled |
| r128_n4_mri4/sample_012 | P-Flow | 210 | 205 | 24.7672 | 10.7170 | stalled |
| r128_n4_mri4/sample_000 | DLR D-Flow | 1000 | 968 | 74.1068 | 67.0572 | step_limit |
| r128_n4_mri4/sample_001 | DLR D-Flow | 272 | 261 | 21.6687 | 26.1071 | stalled |
| r128_n4_mri4/sample_002 | DLR D-Flow | 67 | 61 | 104.8606 | 103.0889 | stalled |
| r128_n4_mri4/sample_003 | DLR D-Flow | 19 | 16 | 105.7942 | 104.2336 | stalled |
| r128_n4_mri4/sample_004 | DLR D-Flow | 550 | 534 | 32.3449 | 31.0602 | stalled |
| r128_n4_mri4/sample_005 | DLR D-Flow | 995 | 965 | 84.6439 | 82.6623 | stalled |
| r128_n4_mri4/sample_006 | DLR D-Flow | 1000 | 957 | 47.1193 | 47.4832 | step_limit |
| r128_n4_mri4/sample_007 | DLR D-Flow | 309 | 301 | 45.8575 | 47.4117 | stalled |
| r128_n4_mri4/sample_008 | DLR D-Flow | 882 | 846 | 43.6629 | 45.2812 | stalled |
| r128_n4_mri4/sample_009 | DLR D-Flow | 331 | 321 | 14.8346 | 22.2939 | stalled |
| r128_n4_mri4/sample_010 | DLR D-Flow | 244 | 237 | 72.3604 | 70.1686 | stalled |
| r128_n4_mri4/sample_011 | DLR D-Flow | 1000 | 932 | 55.8119 | 52.6205 | step_limit |
| r128_n4_mri4/sample_012 | DLR D-Flow | 1000 | 964 | 64.7829 | 64.5454 | step_limit |
| r160_n1_ct10/sample_000 | D-Flow | 356 | 354 | 141.6539 | 0.5965 | stalled |
| r160_n1_ct10/sample_001 | D-Flow | 522 | 520 | 119.3861 | 1.1456 | stalled |
| r160_n1_ct10/sample_002 | D-Flow | 466 | 464 | 109.8818 | 1.3190 | stalled |
| r160_n1_ct10/sample_003 | D-Flow | 381 | 378 | 133.7664 | 1.1214 | stalled |
| r160_n1_ct10/sample_004 | D-Flow | 537 | 535 | 125.0033 | 0.5299 | stalled |
| r160_n1_ct10/sample_005 | D-Flow | 460 | 458 | 127.3237 | 0.5699 | stalled |
| r160_n1_ct10/sample_006 | D-Flow | 502 | 500 | 255.8991 | 0.7212 | stalled |
| r160_n1_ct10/sample_007 | D-Flow | 496 | 494 | 218.2039 | 1.2350 | stalled |
| r160_n1_ct10/sample_008 | D-Flow | 456 | 454 | 166.4762 | 0.7093 | stalled |
| r160_n1_ct10/sample_009 | D-Flow | 565 | 563 | 155.3110 | 1.3784 | stalled |
| r160_n1_ct10/sample_010 | D-Flow | 592 | 590 | 22.7845 | 16.4219 | stalled |
| r160_n1_ct10/sample_011 | D-Flow | 533 | 531 | 139.5813 | 0.7600 | stalled |
| r160_n1_ct10/sample_012 | D-Flow | 466 | 464 | 20.5532 | 15.9056 | stalled |
| r160_n1_ct10/sample_013 | D-Flow | 412 | 410 | 21.3059 | 10.3511 | stalled |
| r160_n1_ct10/sample_014 | D-Flow | 1000 | 1000 | 83.5313 | 1.4803 | step_limit |
| r160_n1_ct10/sample_015 | D-Flow | 300 | 296 | 99.9945 | 86.9007 | stalled |
| r160_n1_ct10/sample_016 | D-Flow | 511 | 509 | 104.6842 | 0.9669 | stalled |
| r160_n1_ct10/sample_017 | D-Flow | 1000 | 1000 | 85.0198 | 1.2435 | step_limit |
| r160_n1_ct10/sample_018 | D-Flow | 789 | 787 | 117.4557 | 0.9432 | stalled |
| r160_n1_ct10/sample_019 | D-Flow | 486 | 484 | 150.9597 | 1.4633 | stalled |
| r160_n1_ct10/sample_020 | D-Flow | 1000 | 1000 | 66.9230 | 53.8022 | step_limit |
| r160_n1_ct10/sample_021 | D-Flow | 515 | 512 | 209.9546 | 1.6248 | stalled |
| r160_n1_ct10/sample_022 | D-Flow | 158 | 156 | 93.5129 | 80.4049 | stalled |
| r160_n1_ct10/sample_023 | D-Flow | 650 | 648 | 137.7524 | 1.2943 | stalled |
| r160_n1_ct10/sample_024 | D-Flow | 424 | 422 | 20.5760 | 15.3180 | stalled |
| r160_n1_ct10/sample_025 | D-Flow | 690 | 687 | 140.1985 | 0.5692 | stalled |
| r160_n1_ct10/sample_026 | D-Flow | 1000 | 1000 | 108.7505 | 0.7340 | step_limit |
| r160_n1_ct10/sample_027 | D-Flow | 837 | 835 | 154.9708 | 1.2563 | stalled |
| r160_n1_ct10/sample_028 | D-Flow | 166 | 163 | 94.3668 | 80.0902 | stalled |
| r160_n1_ct10/sample_029 | D-Flow | 603 | 601 | 20.6467 | 16.2079 | stalled |
| r160_n1_ct10/sample_000 | P-Flow | 1000 | 893 | 77.6109 | 1.1278 | step_limit |
| r160_n1_ct10/sample_001 | P-Flow | 23 | 19 | 4.9074 | 19.9402 | stalled |
| r160_n1_ct10/sample_002 | P-Flow | 70 | 59 | 6.1829 | 19.9197 | stalled |
| r160_n1_ct10/sample_003 | P-Flow | 607 | 573 | 80.2925 | 1.1240 | stalled |
| r160_n1_ct10/sample_004 | P-Flow | 53 | 47 | 11.7379 | 19.5590 | stalled |
| r160_n1_ct10/sample_005 | P-Flow | 85 | 71 | 5.6547 | 19.5406 | stalled |
| r160_n1_ct10/sample_006 | P-Flow | 41 | 35 | 6.7684 | 19.6840 | stalled |
| r160_n1_ct10/sample_007 | P-Flow | 841 | 797 | 114.3138 | 1.2351 | stalled |
| r160_n1_ct10/sample_008 | P-Flow | 1000 | 952 | 67.5382 | 1.4047 | step_limit |
| r160_n1_ct10/sample_009 | P-Flow | 648 | 598 | 94.6571 | 1.3785 | stalled |
| r160_n1_ct10/sample_010 | P-Flow | 465 | 457 | 94.5502 | 2.0695 | stalled |
| r160_n1_ct10/sample_011 | P-Flow | 704 | 585 | 12.6752 | 18.4636 | stalled |
| r160_n1_ct10/sample_012 | P-Flow | 640 | 614 | 92.1205 | 1.1019 | stalled |
| r160_n1_ct10/sample_013 | P-Flow | 36 | 29 | 4.8613 | 19.6286 | stalled |
| r160_n1_ct10/sample_014 | P-Flow | 853 | 694 | 4.1955 | 19.6858 | stalled |
| r160_n1_ct10/sample_015 | P-Flow | 1000 | 918 | 76.3736 | 1.1934 | step_limit |
| r160_n1_ct10/sample_016 | P-Flow | 1000 | 851 | 65.6253 | 1.8847 | step_limit |
| r160_n1_ct10/sample_017 | P-Flow | 1000 | 893 | 63.0230 | 1.7103 | step_limit |
| r160_n1_ct10/sample_018 | P-Flow | 44 | 37 | 3.9133 | 19.5427 | stalled |
| r160_n1_ct10/sample_019 | P-Flow | 29 | 24 | 5.8767 | 19.6445 | stalled |
| r160_n1_ct10/sample_020 | P-Flow | 1000 | 975 | 94.3778 | 0.2668 | step_limit |
| r160_n1_ct10/sample_021 | P-Flow | 38 | 32 | 6.5316 | 19.7113 | stalled |
| r160_n1_ct10/sample_022 | P-Flow | 39 | 36 | 8.8261 | 19.3294 | stalled |
| r160_n1_ct10/sample_023 | P-Flow | 25 | 21 | 6.0902 | 19.8073 | stalled |
| r160_n1_ct10/sample_024 | P-Flow | 794 | 740 | 108.0487 | 0.9044 | stalled |
| r160_n1_ct10/sample_025 | P-Flow | 50 | 43 | 4.3011 | 19.6010 | stalled |
| r160_n1_ct10/sample_026 | P-Flow | 59 | 50 | 4.1344 | 19.6140 | stalled |
| r160_n1_ct10/sample_027 | P-Flow | 135 | 111 | 7.9426 | 19.3897 | stalled |
| r160_n1_ct10/sample_028 | P-Flow | 489 | 474 | 77.0559 | 1.4070 | stalled |
| r160_n1_ct10/sample_029 | P-Flow | 449 | 438 | 91.4941 | 1.3284 | stalled |
| r160_n1_ct10/sample_000 | DLR D-Flow | 595 | 574 | 4.0928 | 19.6046 | stalled |
| r160_n1_ct10/sample_001 | DLR D-Flow | 280 | 267 | 5.1111 | 19.6378 | stalled |
| r160_n1_ct10/sample_002 | DLR D-Flow | 75 | 69 | 97.0635 | 84.2702 | stalled |
| r160_n1_ct10/sample_003 | DLR D-Flow | 368 | 358 | 5.2526 | 19.6990 | stalled |
| r160_n1_ct10/sample_004 | DLR D-Flow | 202 | 193 | 7.8887 | 19.5520 | stalled |
| r160_n1_ct10/sample_005 | DLR D-Flow | 162 | 156 | 6.9949 | 19.5370 | stalled |
| r160_n1_ct10/sample_006 | DLR D-Flow | 381 | 363 | 5.6931 | 19.6831 | stalled |
| r160_n1_ct10/sample_007 | DLR D-Flow | 391 | 374 | 7.6168 | 19.5344 | stalled |
| r160_n1_ct10/sample_008 | DLR D-Flow | 478 | 455 | 10.5945 | 19.4252 | stalled |
| r160_n1_ct10/sample_009 | DLR D-Flow | 281 | 272 | 13.6987 | 19.6074 | stalled |
| r160_n1_ct10/sample_010 | DLR D-Flow | 175 | 168 | 4.4671 | 19.5898 | stalled |
| r160_n1_ct10/sample_011 | DLR D-Flow | 449 | 433 | 4.7413 | 19.6606 | stalled |
| r160_n1_ct10/sample_012 | DLR D-Flow | 815 | 781 | 8.0230 | 19.4187 | stalled |
| r160_n1_ct10/sample_013 | DLR D-Flow | 972 | 930 | 5.8651 | 19.6095 | stalled |
| r160_n1_ct10/sample_014 | DLR D-Flow | 219 | 208 | 5.9085 | 20.2293 | stalled |
| r160_n1_ct10/sample_015 | DLR D-Flow | 25 | 21 | 107.0433 | 91.6163 | stalled |
| r160_n1_ct10/sample_016 | DLR D-Flow | 223 | 212 | 4.7250 | 19.5559 | stalled |
| r160_n1_ct10/sample_017 | DLR D-Flow | 165 | 157 | 6.9888 | 19.5686 | stalled |
| r160_n1_ct10/sample_018 | DLR D-Flow | 631 | 601 | 4.6748 | 19.4530 | stalled |
| r160_n1_ct10/sample_019 | DLR D-Flow | 570 | 540 | 7.2160 | 19.5708 | stalled |
| r160_n1_ct10/sample_020 | DLR D-Flow | 36 | 32 | 114.3481 | 98.5924 | stalled |
| r160_n1_ct10/sample_021 | DLR D-Flow | 101 | 94 | 6.6453 | 19.6197 | stalled |
| r160_n1_ct10/sample_022 | DLR D-Flow | 32 | 27 | 94.1483 | 82.2030 | stalled |
| r160_n1_ct10/sample_023 | DLR D-Flow | 619 | 596 | 3.8884 | 19.7017 | stalled |
| r160_n1_ct10/sample_024 | DLR D-Flow | 642 | 595 | 6.6316 | 19.6131 | stalled |
| r160_n1_ct10/sample_025 | DLR D-Flow | 555 | 535 | 6.9212 | 19.5356 | stalled |
| r160_n1_ct10/sample_026 | DLR D-Flow | 709 | 680 | 17.7571 | 19.6211 | stalled |
| r160_n1_ct10/sample_027 | DLR D-Flow | 156 | 149 | 3.7548 | 19.3870 | stalled |
| r160_n1_ct10/sample_028 | DLR D-Flow | 60 | 57 | 95.0301 | 81.8778 | stalled |
| r160_n1_ct10/sample_029 | DLR D-Flow | 409 | 394 | 4.3012 | 19.6380 | stalled |
| r160_n2_ct10/sample_000 | D-Flow | 738 | 736 | 135.3406 | 0.9427 | stalled |
| r160_n2_ct10/sample_001 | D-Flow | 549 | 546 | 327.1999 | 0.6917 | stalled |
| r160_n2_ct10/sample_002 | D-Flow | 601 | 599 | 97.8203 | 0.6705 | stalled |
| r160_n2_ct10/sample_003 | D-Flow | 530 | 528 | 304.1152 | 1.7249 | stalled |
| r160_n2_ct10/sample_004 | D-Flow | 621 | 619 | 132.5658 | 0.2848 | stalled |
| r160_n2_ct10/sample_005 | D-Flow | 518 | 515 | 149.9449 | 0.4283 | stalled |
| r160_n2_ct10/sample_006 | D-Flow | 827 | 825 | 99.9107 | 0.4824 | stalled |
| r160_n2_ct10/sample_007 | D-Flow | 598 | 595 | 136.7854 | 1.4075 | stalled |
| r160_n2_ct10/sample_008 | D-Flow | 801 | 799 | 113.6562 | 1.4026 | stalled |
| r160_n2_ct10/sample_009 | D-Flow | 457 | 455 | 13.8857 | 17.9939 | stalled |
| r160_n2_ct10/sample_010 | D-Flow | 462 | 460 | 167.8896 | 0.6046 | stalled |
| r160_n2_ct10/sample_011 | D-Flow | 643 | 641 | 233.0475 | 0.6692 | stalled |
| r160_n2_ct10/sample_012 | D-Flow | 110 | 106 | 124.7402 | 104.9657 | stalled |
| r160_n2_ct10/sample_000 | P-Flow | 505 | 499 | 107.1816 | 0.9427 | stalled |
| r160_n2_ct10/sample_001 | P-Flow | 30 | 26 | 8.6304 | 19.7167 | stalled |
| r160_n2_ct10/sample_002 | P-Flow | 1000 | 892 | 77.3225 | 0.7659 | step_limit |
| r160_n2_ct10/sample_003 | P-Flow | 487 | 470 | 95.6088 | 1.7249 | stalled |
| r160_n2_ct10/sample_004 | P-Flow | 808 | 772 | 98.0938 | 0.2848 | stalled |
| r160_n2_ct10/sample_005 | P-Flow | 1000 | 858 | 84.8774 | 0.9418 | step_limit |
| r160_n2_ct10/sample_006 | P-Flow | 60 | 51 | 10.5545 | 19.4327 | stalled |
| r160_n2_ct10/sample_007 | P-Flow | 588 | 576 | 96.6757 | 1.4076 | stalled |
| r160_n2_ct10/sample_008 | P-Flow | 7 | 5 | 98.5157 | 66.5252 | stalled |
| r160_n2_ct10/sample_009 | P-Flow | 428 | 423 | 90.1152 | 0.4809 | stalled |
| r160_n2_ct10/sample_010 | P-Flow | 592 | 576 | 86.8518 | 0.6046 | stalled |
| r160_n2_ct10/sample_011 | P-Flow | 79 | 70 | 10.1959 | 19.3461 | stalled |
| r160_n2_ct10/sample_012 | P-Flow | 48 | 39 | 10.5995 | 19.7823 | stalled |
| r160_n2_ct10/sample_000 | DLR D-Flow | 698 | 673 | 7.5507 | 19.4844 | stalled |
| r160_n2_ct10/sample_001 | DLR D-Flow | 139 | 132 | 6.9682 | 19.6269 | stalled |
| r160_n2_ct10/sample_002 | DLR D-Flow | 560 | 542 | 59.8452 | 45.0016 | stalled |
| r160_n2_ct10/sample_003 | DLR D-Flow | 89 | 84 | 76.4456 | 46.1693 | stalled |
| r160_n2_ct10/sample_004 | DLR D-Flow | 152 | 144 | 72.6734 | 51.6259 | stalled |
| r160_n2_ct10/sample_005 | DLR D-Flow | 431 | 413 | 9.3779 | 19.7458 | stalled |
| r160_n2_ct10/sample_006 | DLR D-Flow | 405 | 392 | 6.8715 | 19.3199 | stalled |
| r160_n2_ct10/sample_007 | DLR D-Flow | 502 | 492 | 62.5016 | 36.6057 | stalled |
| r160_n2_ct10/sample_008 | DLR D-Flow | 771 | 742 | 10.2941 | 19.7963 | stalled |
| r160_n2_ct10/sample_009 | DLR D-Flow | 46 | 42 | 16.5671 | 19.8698 | stalled |
| r160_n2_ct10/sample_010 | DLR D-Flow | 491 | 476 | 5.6700 | 19.7216 | stalled |
| r160_n2_ct10/sample_011 | DLR D-Flow | 641 | 627 | 51.5168 | 34.3149 | stalled |
| r160_n2_ct10/sample_012 | DLR D-Flow | 198 | 191 | 125.0521 | 105.3793 | stalled |
| r160_n4_ct10/sample_000 | D-Flow | 816 | 814 | 116.1269 | 0.3424 | stalled |
| r160_n4_ct10/sample_001 | D-Flow | 705 | 703 | 129.0217 | 0.8248 | stalled |
| r160_n4_ct10/sample_002 | D-Flow | 882 | 879 | 123.5747 | 0.9780 | stalled |
| r160_n4_ct10/sample_003 | D-Flow | 678 | 676 | 131.3379 | 0.9965 | stalled |
| r160_n4_ct10/sample_004 | D-Flow | 519 | 517 | 180.3712 | 1.5391 | stalled |
| r160_n4_ct10/sample_005 | D-Flow | 691 | 689 | 123.8810 | 1.4839 | stalled |
| r160_n4_ct10/sample_006 | D-Flow | 799 | 797 | 116.2593 | 0.3094 | stalled |
| r160_n4_ct10/sample_007 | D-Flow | 1000 | 1000 | 116.3514 | 0.6369 | step_limit |
| r160_n4_ct10/sample_008 | D-Flow | 704 | 701 | 145.7591 | 1.7772 | stalled |
| r160_n4_ct10/sample_009 | D-Flow | 866 | 864 | 165.3739 | 1.7832 | stalled |
| r160_n4_ct10/sample_010 | D-Flow | 787 | 784 | 118.8727 | 0.8098 | stalled |
| r160_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 81.9784 | 9.4736 | step_limit |
| r160_n4_ct10/sample_012 | D-Flow | 992 | 990 | 113.9171 | 1.3661 | stalled |
| r160_n4_ct10/sample_000 | P-Flow | 1000 | 924 | 88.4403 | 0.3477 | step_limit |
| r160_n4_ct10/sample_001 | P-Flow | 172 | 144 | 11.7274 | 19.2875 | stalled |
| r160_n4_ct10/sample_002 | P-Flow | 606 | 595 | 83.1373 | 0.9780 | stalled |
| r160_n4_ct10/sample_003 | P-Flow | 596 | 586 | 93.1223 | 0.9966 | stalled |
| r160_n4_ct10/sample_004 | P-Flow | 482 | 472 | 90.4813 | 1.5391 | stalled |
| r160_n4_ct10/sample_005 | P-Flow | 1000 | 926 | 93.5679 | 1.5100 | step_limit |
| r160_n4_ct10/sample_006 | P-Flow | 1000 | 874 | 95.4297 | 0.3975 | step_limit |
| r160_n4_ct10/sample_007 | P-Flow | 765 | 729 | 98.8864 | 0.6376 | stalled |
| r160_n4_ct10/sample_008 | P-Flow | 617 | 608 | 102.3841 | 1.7772 | stalled |
| r160_n4_ct10/sample_009 | P-Flow | 32 | 26 | 10.0569 | 19.5508 | stalled |
| r160_n4_ct10/sample_010 | P-Flow | 4 | 2 | 89.0216 | 50.9870 | stalled |
| r160_n4_ct10/sample_011 | P-Flow | 77 | 68 | 11.9677 | 19.5505 | stalled |
| r160_n4_ct10/sample_012 | P-Flow | 8 | 6 | 93.6300 | 56.2324 | stalled |
| r160_n4_ct10/sample_000 | DLR D-Flow | 881 | 845 | 7.5704 | 19.6500 | stalled |
| r160_n4_ct10/sample_001 | DLR D-Flow | 555 | 534 | 29.7123 | 21.7171 | stalled |
| r160_n4_ct10/sample_002 | DLR D-Flow | 1000 | 959 | 26.4263 | 20.9681 | step_limit |
| r160_n4_ct10/sample_003 | DLR D-Flow | 1000 | 967 | 27.6257 | 21.5310 | step_limit |
| r160_n4_ct10/sample_004 | DLR D-Flow | 1000 | 961 | 32.9711 | 21.6485 | step_limit |
| r160_n4_ct10/sample_005 | DLR D-Flow | 578 | 563 | 38.7017 | 24.5674 | stalled |
| r160_n4_ct10/sample_006 | DLR D-Flow | 158 | 150 | 76.7833 | 45.9738 | stalled |
| r160_n4_ct10/sample_007 | DLR D-Flow | 216 | 203 | 76.6358 | 39.2099 | stalled |
| r160_n4_ct10/sample_008 | DLR D-Flow | 544 | 521 | 18.8498 | 20.0390 | stalled |
| r160_n4_ct10/sample_009 | DLR D-Flow | 222 | 210 | 44.7209 | 27.9952 | stalled |
| r160_n4_ct10/sample_010 | DLR D-Flow | 329 | 318 | 13.5571 | 20.0026 | stalled |
| r160_n4_ct10/sample_011 | DLR D-Flow | 783 | 754 | 62.0620 | 31.8626 | stalled |
| r160_n4_ct10/sample_012 | DLR D-Flow | 252 | 245 | 32.9316 | 22.8040 | stalled |
| r160_n1_mri4/sample_000 | D-Flow | 939 | 937 | 15.2860 | 11.8839 | stalled |
| r160_n1_mri4/sample_001 | D-Flow | 152 | 150 | 1.9828 | 19.5962 | stalled |
| r160_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 15.2348 | 11.8464 | step_limit |
| r160_n1_mri4/sample_003 | D-Flow | 708 | 705 | 26.2435 | 8.5122 | stalled |
| r160_n1_mri4/sample_004 | D-Flow | 767 | 765 | 11.5437 | 14.1777 | stalled |
| r160_n1_mri4/sample_005 | D-Flow | 579 | 577 | 24.5142 | 9.9271 | stalled |
| r160_n1_mri4/sample_006 | D-Flow | 442 | 440 | 23.1796 | 10.1036 | stalled |
| r160_n1_mri4/sample_007 | D-Flow | 600 | 598 | 23.0112 | 10.9853 | stalled |
| r160_n1_mri4/sample_008 | D-Flow | 634 | 631 | 25.8358 | 8.5505 | stalled |
| r160_n1_mri4/sample_009 | D-Flow | 683 | 681 | 11.2873 | 14.3568 | stalled |
| r160_n1_mri4/sample_010 | D-Flow | 910 | 907 | 23.1297 | 10.4188 | stalled |
| r160_n1_mri4/sample_011 | D-Flow | 691 | 689 | 23.9020 | 11.5407 | stalled |
| r160_n1_mri4/sample_012 | D-Flow | 1000 | 1000 | 113.7022 | 32.3706 | step_limit |
| r160_n1_mri4/sample_013 | D-Flow | 613 | 611 | 21.5684 | 10.7351 | stalled |
| r160_n1_mri4/sample_014 | D-Flow | 1000 | 1000 | 17.0898 | 10.3985 | step_limit |
| r160_n1_mri4/sample_015 | D-Flow | 61 | 58 | 100.6527 | 100.1245 | stalled |
| r160_n1_mri4/sample_016 | D-Flow | 673 | 671 | 24.6627 | 9.8071 | stalled |
| r160_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 15.8700 | 13.5097 | step_limit |
| r160_n1_mri4/sample_018 | D-Flow | 786 | 784 | 24.3800 | 8.7213 | stalled |
| r160_n1_mri4/sample_019 | D-Flow | 554 | 552 | 21.4453 | 10.9447 | stalled |
| r160_n1_mri4/sample_020 | D-Flow | 32 | 29 | 99.0075 | 98.4006 | stalled |
| r160_n1_mri4/sample_021 | D-Flow | 666 | 663 | 24.5223 | 10.1207 | stalled |
| r160_n1_mri4/sample_022 | D-Flow | 568 | 565 | 25.8358 | 9.9111 | stalled |
| r160_n1_mri4/sample_023 | D-Flow | 1000 | 1000 | 17.1375 | 11.8402 | step_limit |
| r160_n1_mri4/sample_024 | D-Flow | 552 | 550 | 24.2313 | 9.5085 | stalled |
| r160_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 12.5905 | 14.1326 | step_limit |
| r160_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 12.3438 | 13.0827 | step_limit |
| r160_n1_mri4/sample_027 | D-Flow | 696 | 694 | 21.4897 | 9.9110 | stalled |
| r160_n1_mri4/sample_028 | D-Flow | 306 | 303 | 90.1676 | 89.5675 | stalled |
| r160_n1_mri4/sample_029 | D-Flow | 570 | 568 | 24.9996 | 10.6622 | stalled |
| r160_n1_mri4/sample_000 | P-Flow | 190 | 180 | 28.9397 | 9.8884 | stalled |
| r160_n1_mri4/sample_001 | P-Flow | 195 | 181 | 28.7944 | 9.4318 | stalled |
| r160_n1_mri4/sample_002 | P-Flow | 47 | 41 | 2.6949 | 19.6590 | stalled |
| r160_n1_mri4/sample_003 | P-Flow | 198 | 188 | 29.4330 | 8.5122 | stalled |
| r160_n1_mri4/sample_004 | P-Flow | 187 | 180 | 23.0199 | 11.0238 | stalled |
| r160_n1_mri4/sample_005 | P-Flow | 208 | 194 | 28.3247 | 9.9271 | stalled |
| r160_n1_mri4/sample_006 | P-Flow | 175 | 169 | 24.6172 | 10.1036 | stalled |
| r160_n1_mri4/sample_007 | P-Flow | 192 | 184 | 25.6955 | 10.9853 | stalled |
| r160_n1_mri4/sample_008 | P-Flow | 201 | 186 | 28.0473 | 8.5505 | stalled |
| r160_n1_mri4/sample_009 | P-Flow | 187 | 179 | 24.2418 | 9.7574 | stalled |
| r160_n1_mri4/sample_010 | P-Flow | 167 | 163 | 26.5081 | 10.4188 | stalled |
| r160_n1_mri4/sample_011 | P-Flow | 185 | 176 | 25.7302 | 11.5407 | stalled |
| r160_n1_mri4/sample_012 | P-Flow | 158 | 154 | 21.8760 | 9.4761 | stalled |
| r160_n1_mri4/sample_013 | P-Flow | 189 | 179 | 26.1082 | 10.7351 | stalled |
| r160_n1_mri4/sample_014 | P-Flow | 96 | 83 | 2.6907 | 19.6514 | stalled |
| r160_n1_mri4/sample_015 | P-Flow | 191 | 185 | 26.9677 | 10.4447 | stalled |
| r160_n1_mri4/sample_016 | P-Flow | 157 | 151 | 27.8118 | 9.8071 | stalled |
| r160_n1_mri4/sample_017 | P-Flow | 169 | 160 | 29.9131 | 9.1636 | stalled |
| r160_n1_mri4/sample_018 | P-Flow | 240 | 220 | 27.9648 | 8.7212 | stalled |
| r160_n1_mri4/sample_019 | P-Flow | 165 | 161 | 23.0404 | 10.9447 | stalled |
| r160_n1_mri4/sample_020 | P-Flow | 160 | 156 | 25.8790 | 10.8106 | stalled |
| r160_n1_mri4/sample_021 | P-Flow | 12 | 9 | 81.6513 | 80.6969 | stalled |
| r160_n1_mri4/sample_022 | P-Flow | 177 | 172 | 28.1121 | 9.9111 | stalled |
| r160_n1_mri4/sample_023 | P-Flow | 155 | 152 | 21.1874 | 10.9329 | stalled |
| r160_n1_mri4/sample_024 | P-Flow | 160 | 155 | 24.8354 | 9.5085 | stalled |
| r160_n1_mri4/sample_025 | P-Flow | 173 | 170 | 29.2334 | 8.3796 | stalled |
| r160_n1_mri4/sample_026 | P-Flow | 167 | 164 | 24.7314 | 9.8241 | stalled |
| r160_n1_mri4/sample_027 | P-Flow | 194 | 186 | 23.7500 | 9.9110 | stalled |
| r160_n1_mri4/sample_028 | P-Flow | 147 | 144 | 27.7745 | 10.1540 | stalled |
| r160_n1_mri4/sample_029 | P-Flow | 161 | 156 | 26.7240 | 10.6622 | stalled |
| r160_n1_mri4/sample_000 | DLR D-Flow | 49 | 43 | 88.6622 | 88.6897 | stalled |
| r160_n1_mri4/sample_001 | DLR D-Flow | 105 | 95 | 107.1040 | 106.4317 | stalled |
| r160_n1_mri4/sample_002 | DLR D-Flow | 13 | 11 | 96.9872 | 96.7034 | stalled |
| r160_n1_mri4/sample_003 | DLR D-Flow | 404 | 388 | 1.5106 | 19.5930 | stalled |
| r160_n1_mri4/sample_004 | DLR D-Flow | 624 | 599 | 1.4777 | 19.5756 | stalled |
| r160_n1_mri4/sample_005 | DLR D-Flow | 75 | 72 | 7.1835 | 20.3115 | stalled |
| r160_n1_mri4/sample_006 | DLR D-Flow | 239 | 227 | 1.7528 | 19.6767 | stalled |
| r160_n1_mri4/sample_007 | DLR D-Flow | 547 | 530 | 1.9172 | 19.6312 | stalled |
| r160_n1_mri4/sample_008 | DLR D-Flow | 758 | 720 | 2.6218 | 19.6482 | stalled |
| r160_n1_mri4/sample_009 | DLR D-Flow | 192 | 187 | 2.5624 | 19.6255 | stalled |
| r160_n1_mri4/sample_010 | DLR D-Flow | 245 | 234 | 1.7963 | 19.6674 | stalled |
| r160_n1_mri4/sample_011 | DLR D-Flow | 1000 | 969 | 1.3166 | 19.5895 | step_limit |
| r160_n1_mri4/sample_012 | DLR D-Flow | 429 | 416 | 121.1830 | 56.1718 | stalled |
| r160_n1_mri4/sample_013 | DLR D-Flow | 740 | 701 | 2.5649 | 19.6502 | stalled |
| r160_n1_mri4/sample_014 | DLR D-Flow | 308 | 294 | 4.3716 | 19.9140 | stalled |
| r160_n1_mri4/sample_015 | DLR D-Flow | 64 | 60 | 98.5653 | 98.1080 | stalled |
| r160_n1_mri4/sample_016 | DLR D-Flow | 787 | 742 | 1.8380 | 19.6075 | stalled |
| r160_n1_mri4/sample_017 | DLR D-Flow | 622 | 586 | 2.6466 | 19.7294 | stalled |
| r160_n1_mri4/sample_018 | DLR D-Flow | 1000 | 965 | 44.9954 | 44.3194 | step_limit |
| r160_n1_mri4/sample_019 | DLR D-Flow | 428 | 412 | 1.4241 | 19.6266 | stalled |
| r160_n1_mri4/sample_020 | DLR D-Flow | 22 | 19 | 111.6232 | 110.0957 | stalled |
| r160_n1_mri4/sample_021 | DLR D-Flow | 239 | 228 | 2.4184 | 19.6671 | stalled |
| r160_n1_mri4/sample_022 | DLR D-Flow | 46 | 42 | 94.1212 | 94.1187 | stalled |
| r160_n1_mri4/sample_023 | DLR D-Flow | 18 | 15 | 102.9997 | 103.4958 | stalled |
| r160_n1_mri4/sample_024 | DLR D-Flow | 488 | 461 | 1.5052 | 19.6451 | stalled |
| r160_n1_mri4/sample_025 | DLR D-Flow | 331 | 320 | 1.7101 | 19.6034 | stalled |
| r160_n1_mri4/sample_026 | DLR D-Flow | 441 | 425 | 1.5553 | 19.5714 | stalled |
| r160_n1_mri4/sample_027 | DLR D-Flow | 296 | 283 | 1.5063 | 19.5489 | stalled |
| r160_n1_mri4/sample_028 | DLR D-Flow | 36 | 32 | 94.8701 | 94.7176 | stalled |
| r160_n1_mri4/sample_029 | DLR D-Flow | 245 | 238 | 1.4540 | 19.5895 | stalled |
| r160_n2_mri4/sample_000 | D-Flow | 462 | 459 | 25.9638 | 10.2113 | stalled |
| r160_n2_mri4/sample_001 | D-Flow | 991 | 988 | 23.1950 | 10.1479 | stalled |
| r160_n2_mri4/sample_002 | D-Flow | 1000 | 1000 | 53.4036 | 54.8430 | step_limit |
| r160_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 27.7716 | 8.8769 | step_limit |
| r160_n2_mri4/sample_004 | D-Flow | 215 | 213 | 60.4130 | 61.9511 | stalled |
| r160_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 20.3667 | 14.1925 | step_limit |
| r160_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 6.8883 | 19.1688 | step_limit |
| r160_n2_mri4/sample_007 | D-Flow | 598 | 596 | 27.4694 | 8.9636 | stalled |
| r160_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 30.2469 | 30.3159 | step_limit |
| r160_n2_mri4/sample_009 | D-Flow | 1000 | 1000 | 5.3718 | 16.9409 | step_limit |
| r160_n2_mri4/sample_010 | D-Flow | 458 | 456 | 22.5509 | 9.5774 | stalled |
| r160_n2_mri4/sample_011 | D-Flow | 858 | 856 | 23.3635 | 11.4910 | stalled |
| r160_n2_mri4/sample_012 | D-Flow | 378 | 373 | 73.7480 | 72.1029 | stalled |
| r160_n2_mri4/sample_000 | P-Flow | 166 | 162 | 27.3014 | 10.2113 | stalled |
| r160_n2_mri4/sample_001 | P-Flow | 196 | 189 | 24.3127 | 10.1479 | stalled |
| r160_n2_mri4/sample_002 | P-Flow | 167 | 161 | 25.7627 | 7.8643 | stalled |
| r160_n2_mri4/sample_003 | P-Flow | 188 | 183 | 27.9732 | 8.8658 | stalled |
| r160_n2_mri4/sample_004 | P-Flow | 166 | 163 | 26.2694 | 10.0271 | stalled |
| r160_n2_mri4/sample_005 | P-Flow | 184 | 174 | 25.6894 | 9.6405 | stalled |
| r160_n2_mri4/sample_006 | P-Flow | 170 | 166 | 23.5830 | 10.4644 | stalled |
| r160_n2_mri4/sample_007 | P-Flow | 168 | 165 | 27.6757 | 8.9636 | stalled |
| r160_n2_mri4/sample_008 | P-Flow | 168 | 163 | 27.8285 | 10.0353 | stalled |
| r160_n2_mri4/sample_009 | P-Flow | 169 | 165 | 27.2199 | 10.1144 | stalled |
| r160_n2_mri4/sample_010 | P-Flow | 162 | 157 | 25.1066 | 9.5774 | stalled |
| r160_n2_mri4/sample_011 | P-Flow | 164 | 160 | 24.7734 | 11.4910 | stalled |
| r160_n2_mri4/sample_012 | P-Flow | 207 | 194 | 29.1083 | 9.2285 | stalled |
| r160_n2_mri4/sample_000 | DLR D-Flow | 348 | 339 | 50.6055 | 51.4446 | stalled |
| r160_n2_mri4/sample_001 | DLR D-Flow | 518 | 495 | 45.4291 | 37.0986 | stalled |
| r160_n2_mri4/sample_002 | DLR D-Flow | 553 | 536 | 54.6103 | 56.3175 | stalled |
| r160_n2_mri4/sample_003 | DLR D-Flow | 1000 | 965 | 54.4080 | 54.0249 | step_limit |
| r160_n2_mri4/sample_004 | DLR D-Flow | 31 | 28 | 102.1392 | 94.4365 | stalled |
| r160_n2_mri4/sample_005 | DLR D-Flow | 783 | 759 | 56.5051 | 55.3981 | stalled |
| r160_n2_mri4/sample_006 | DLR D-Flow | 473 | 456 | 1.8029 | 19.6424 | stalled |
| r160_n2_mri4/sample_007 | DLR D-Flow | 690 | 664 | 3.0820 | 19.7398 | stalled |
| r160_n2_mri4/sample_008 | DLR D-Flow | 660 | 643 | 68.1872 | 69.6132 | stalled |
| r160_n2_mri4/sample_009 | DLR D-Flow | 233 | 222 | 2.9541 | 19.6844 | stalled |
| r160_n2_mri4/sample_010 | DLR D-Flow | 884 | 858 | 1.5267 | 19.6884 | stalled |
| r160_n2_mri4/sample_011 | DLR D-Flow | 806 | 791 | 44.8057 | 46.5837 | stalled |
| r160_n2_mri4/sample_012 | DLR D-Flow | 294 | 286 | 125.6484 | 123.7040 | stalled |
| r160_n4_mri4/sample_000 | D-Flow | 577 | 574 | 24.9544 | 8.6976 | stalled |
| r160_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 20.4096 | 10.7703 | step_limit |
| r160_n4_mri4/sample_002 | D-Flow | 475 | 473 | 25.7173 | 10.0130 | stalled |
| r160_n4_mri4/sample_003 | D-Flow | 537 | 535 | 21.8992 | 10.7959 | stalled |
| r160_n4_mri4/sample_004 | D-Flow | 887 | 884 | 23.8356 | 10.4944 | stalled |
| r160_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 12.5703 | 18.0809 | step_limit |
| r160_n4_mri4/sample_006 | D-Flow | 1000 | 1000 | 4.5899 | 18.6589 | step_limit |
| r160_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 2.4637 | 19.5904 | step_limit |
| r160_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 4.9848 | 19.2693 | step_limit |
| r160_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 23.1142 | 10.5519 | step_limit |
| r160_n4_mri4/sample_010 | D-Flow | 598 | 596 | 24.1157 | 11.5215 | stalled |
| r160_n4_mri4/sample_011 | D-Flow | 739 | 737 | 23.6819 | 10.8843 | stalled |
| r160_n4_mri4/sample_012 | D-Flow | 625 | 623 | 21.2229 | 9.9265 | stalled |
| r160_n4_mri4/sample_000 | P-Flow | 180 | 177 | 25.0012 | 8.6976 | stalled |
| r160_n4_mri4/sample_001 | P-Flow | 189 | 185 | 21.8964 | 10.6647 | stalled |
| r160_n4_mri4/sample_002 | P-Flow | 193 | 186 | 26.2018 | 10.0130 | stalled |
| r160_n4_mri4/sample_003 | P-Flow | 180 | 178 | 22.1679 | 10.7959 | stalled |
| r160_n4_mri4/sample_004 | P-Flow | 169 | 166 | 24.2010 | 10.4944 | stalled |
| r160_n4_mri4/sample_005 | P-Flow | 187 | 181 | 25.7960 | 10.3924 | stalled |
| r160_n4_mri4/sample_006 | P-Flow | 198 | 192 | 25.4067 | 10.8277 | stalled |
| r160_n4_mri4/sample_007 | P-Flow | 183 | 179 | 24.5376 | 9.9806 | stalled |
| r160_n4_mri4/sample_008 | P-Flow | 53 | 47 | 1.9867 | 19.5991 | stalled |
| r160_n4_mri4/sample_009 | P-Flow | 185 | 183 | 23.3534 | 10.5518 | stalled |
| r160_n4_mri4/sample_010 | P-Flow | 191 | 185 | 23.6336 | 11.5215 | stalled |
| r160_n4_mri4/sample_011 | P-Flow | 192 | 184 | 23.7250 | 10.8843 | stalled |
| r160_n4_mri4/sample_012 | P-Flow | 179 | 177 | 23.0706 | 9.9265 | stalled |
| r160_n4_mri4/sample_000 | DLR D-Flow | 1000 | 962 | 17.5903 | 24.1753 | step_limit |
| r160_n4_mri4/sample_001 | DLR D-Flow | 471 | 454 | 58.5581 | 52.6466 | stalled |
| r160_n4_mri4/sample_002 | DLR D-Flow | 220 | 211 | 14.7719 | 22.2890 | stalled |
| r160_n4_mri4/sample_003 | DLR D-Flow | 1000 | 960 | 56.9474 | 55.9076 | step_limit |
| r160_n4_mri4/sample_004 | DLR D-Flow | 323 | 313 | 20.9040 | 26.5736 | stalled |
| r160_n4_mri4/sample_005 | DLR D-Flow | 195 | 185 | 91.1388 | 89.5139 | stalled |
| r160_n4_mri4/sample_006 | DLR D-Flow | 1000 | 970 | 45.3809 | 47.4562 | step_limit |
| r160_n4_mri4/sample_007 | DLR D-Flow | 259 | 242 | 80.8525 | 79.2923 | stalled |
| r160_n4_mri4/sample_008 | DLR D-Flow | 61 | 57 | 63.9881 | 57.7126 | stalled |
| r160_n4_mri4/sample_009 | DLR D-Flow | 1000 | 960 | 34.7053 | 36.9277 | step_limit |
| r160_n4_mri4/sample_010 | DLR D-Flow | 1000 | 976 | 42.8704 | 42.5583 | step_limit |
| r160_n4_mri4/sample_011 | DLR D-Flow | 1000 | 964 | 42.7792 | 43.1226 | step_limit |
| r160_n4_mri4/sample_012 | DLR D-Flow | 332 | 324 | 18.9747 | 25.6740 | stalled |
| r200_n1_ct10/sample_000 | D-Flow | 1000 | 1000 | 14.0511 | 17.3862 | step_limit |
| r200_n1_ct10/sample_001 | D-Flow | 923 | 921 | 15.4513 | 14.9659 | stalled |
| r200_n1_ct10/sample_002 | D-Flow | 1000 | 1000 | 13.2567 | 16.8059 | step_limit |
| r200_n1_ct10/sample_003 | D-Flow | 548 | 546 | 153.3840 | 1.3695 | stalled |
| r200_n1_ct10/sample_004 | D-Flow | 437 | 435 | 95.4909 | 1.3049 | stalled |
| r200_n1_ct10/sample_005 | D-Flow | 867 | 863 | 13.8145 | 14.5948 | stalled |
| r200_n1_ct10/sample_006 | D-Flow | 600 | 597 | 112.0301 | 1.0169 | stalled |
| r200_n1_ct10/sample_007 | D-Flow | 61 | 59 | 108.9314 | 95.8130 | stalled |
| r200_n1_ct10/sample_008 | D-Flow | 562 | 560 | 162.2355 | 1.2333 | stalled |
| r200_n1_ct10/sample_009 | D-Flow | 396 | 394 | 165.8815 | 0.8268 | stalled |
| r200_n1_ct10/sample_010 | D-Flow | 902 | 900 | 244.1974 | 0.7992 | stalled |
| r200_n1_ct10/sample_011 | D-Flow | 411 | 408 | 212.2636 | 1.0679 | stalled |
| r200_n1_ct10/sample_012 | D-Flow | 472 | 470 | 180.0774 | 0.8000 | stalled |
| r200_n1_ct10/sample_013 | D-Flow | 555 | 553 | 244.1589 | 0.9499 | stalled |
| r200_n1_ct10/sample_014 | D-Flow | 870 | 868 | 109.4688 | 0.9604 | stalled |
| r200_n1_ct10/sample_015 | D-Flow | 543 | 541 | 196.8976 | 0.5426 | stalled |
| r200_n1_ct10/sample_016 | D-Flow | 511 | 508 | 191.9492 | 0.9438 | stalled |
| r200_n1_ct10/sample_017 | D-Flow | 437 | 435 | 242.6131 | 1.3044 | stalled |
| r200_n1_ct10/sample_018 | D-Flow | 447 | 445 | 181.7001 | 1.0620 | stalled |
| r200_n1_ct10/sample_019 | D-Flow | 641 | 639 | 130.7133 | 0.9811 | stalled |
| r200_n1_ct10/sample_020 | D-Flow | 594 | 592 | 352.7011 | 1.0290 | stalled |
| r200_n1_ct10/sample_021 | D-Flow | 54 | 51 | 108.0383 | 96.7400 | stalled |
| r200_n1_ct10/sample_022 | D-Flow | 463 | 461 | 130.1113 | 1.1565 | stalled |
| r200_n1_ct10/sample_023 | D-Flow | 533 | 531 | 95.7211 | 0.8750 | stalled |
| r200_n1_ct10/sample_024 | D-Flow | 451 | 449 | 258.0185 | 0.7719 | stalled |
| r200_n1_ct10/sample_025 | D-Flow | 1000 | 1000 | 12.5495 | 17.5667 | step_limit |
| r200_n1_ct10/sample_026 | D-Flow | 471 | 468 | 263.4781 | 1.2281 | stalled |
| r200_n1_ct10/sample_027 | D-Flow | 476 | 474 | 128.9231 | 1.3391 | stalled |
| r200_n1_ct10/sample_028 | D-Flow | 454 | 452 | 209.1887 | 0.6220 | stalled |
| r200_n1_ct10/sample_029 | D-Flow | 477 | 474 | 170.1451 | 0.4430 | stalled |
| r200_n1_ct10/sample_000 | P-Flow | 48 | 41 | 8.0123 | 19.6269 | stalled |
| r200_n1_ct10/sample_001 | P-Flow | 1000 | 817 | 4.9142 | 19.6563 | step_limit |
| r200_n1_ct10/sample_002 | P-Flow | 489 | 476 | 87.5153 | 0.9661 | stalled |
| r200_n1_ct10/sample_003 | P-Flow | 80 | 66 | 7.2767 | 19.6164 | stalled |
| r200_n1_ct10/sample_004 | P-Flow | 1000 | 882 | 67.1164 | 1.3867 | step_limit |
| r200_n1_ct10/sample_005 | P-Flow | 35 | 28 | 6.7280 | 19.5673 | stalled |
| r200_n1_ct10/sample_006 | P-Flow | 23 | 19 | 29.9656 | 23.2613 | stalled |
| r200_n1_ct10/sample_007 | P-Flow | 15 | 13 | 110.7858 | 95.2103 | stalled |
| r200_n1_ct10/sample_008 | P-Flow | 53 | 46 | 5.6360 | 19.7057 | stalled |
| r200_n1_ct10/sample_009 | P-Flow | 420 | 408 | 76.2001 | 0.8278 | stalled |
| r200_n1_ct10/sample_010 | P-Flow | 35 | 28 | 4.2941 | 19.6090 | stalled |
| r200_n1_ct10/sample_011 | P-Flow | 940 | 837 | 86.0532 | 1.0679 | stalled |
| r200_n1_ct10/sample_012 | P-Flow | 148 | 119 | 7.8658 | 19.7192 | stalled |
| r200_n1_ct10/sample_013 | P-Flow | 1000 | 887 | 81.7238 | 0.9734 | step_limit |
| r200_n1_ct10/sample_014 | P-Flow | 54 | 45 | 6.1860 | 19.7334 | stalled |
| r200_n1_ct10/sample_015 | P-Flow | 452 | 428 | 80.1520 | 0.5426 | stalled |
| r200_n1_ct10/sample_016 | P-Flow | 3 | 1 | 104.7073 | 97.1926 | stalled |
| r200_n1_ct10/sample_017 | P-Flow | 104 | 83 | 7.4650 | 19.5630 | stalled |
| r200_n1_ct10/sample_018 | P-Flow | 82 | 67 | 8.0028 | 19.4595 | stalled |
| r200_n1_ct10/sample_019 | P-Flow | 32 | 26 | 21.9240 | 20.3319 | stalled |
| r200_n1_ct10/sample_020 | P-Flow | 1000 | 785 | 3.3337 | 19.6557 | step_limit |
| r200_n1_ct10/sample_021 | P-Flow | 708 | 650 | 68.1635 | 1.4852 | stalled |
| r200_n1_ct10/sample_022 | P-Flow | 156 | 129 | 6.2978 | 19.6237 | stalled |
| r200_n1_ct10/sample_023 | P-Flow | 99 | 82 | 11.4897 | 20.2540 | stalled |
| r200_n1_ct10/sample_024 | P-Flow | 7 | 5 | 108.7754 | 94.2335 | stalled |
| r200_n1_ct10/sample_025 | P-Flow | 37 | 31 | 2.7484 | 19.5593 | stalled |
| r200_n1_ct10/sample_026 | P-Flow | 493 | 479 | 88.2511 | 1.2282 | stalled |
| r200_n1_ct10/sample_027 | P-Flow | 232 | 192 | 4.1152 | 19.7941 | stalled |
| r200_n1_ct10/sample_028 | P-Flow | 793 | 729 | 75.3919 | 0.6238 | stalled |
| r200_n1_ct10/sample_029 | P-Flow | 42 | 37 | 6.2808 | 19.8246 | stalled |
| r200_n1_ct10/sample_000 | DLR D-Flow | 236 | 229 | 4.3273 | 19.4469 | stalled |
| r200_n1_ct10/sample_001 | DLR D-Flow | 377 | 361 | 2.7105 | 19.6507 | stalled |
| r200_n1_ct10/sample_002 | DLR D-Flow | 145 | 135 | 2.8357 | 19.4357 | stalled |
| r200_n1_ct10/sample_003 | DLR D-Flow | 382 | 363 | 5.3414 | 19.5919 | stalled |
| r200_n1_ct10/sample_004 | DLR D-Flow | 1000 | 967 | 4.7597 | 19.5618 | step_limit |
| r200_n1_ct10/sample_005 | DLR D-Flow | 482 | 460 | 3.1193 | 19.5343 | stalled |
| r200_n1_ct10/sample_006 | DLR D-Flow | 233 | 229 | 10.8476 | 20.2610 | stalled |
| r200_n1_ct10/sample_007 | DLR D-Flow | 15 | 13 | 107.8022 | 96.4785 | stalled |
| r200_n1_ct10/sample_008 | DLR D-Flow | 221 | 211 | 3.9681 | 19.6685 | stalled |
| r200_n1_ct10/sample_009 | DLR D-Flow | 24 | 22 | 93.4525 | 65.1602 | stalled |
| r200_n1_ct10/sample_010 | DLR D-Flow | 597 | 575 | 3.7940 | 19.5253 | stalled |
| r200_n1_ct10/sample_011 | DLR D-Flow | 223 | 214 | 4.6433 | 19.5769 | stalled |
| r200_n1_ct10/sample_012 | DLR D-Flow | 502 | 484 | 3.6068 | 19.6053 | stalled |
| r200_n1_ct10/sample_013 | DLR D-Flow | 653 | 639 | 4.6440 | 19.6766 | stalled |
| r200_n1_ct10/sample_014 | DLR D-Flow | 22 | 19 | 111.7959 | 95.8335 | stalled |
| r200_n1_ct10/sample_015 | DLR D-Flow | 163 | 156 | 5.3811 | 19.7110 | stalled |
| r200_n1_ct10/sample_016 | DLR D-Flow | 149 | 142 | 4.2640 | 19.6799 | stalled |
| r200_n1_ct10/sample_017 | DLR D-Flow | 481 | 444 | 4.8503 | 19.5231 | stalled |
| r200_n1_ct10/sample_018 | DLR D-Flow | 265 | 254 | 6.1917 | 19.4135 | stalled |
| r200_n1_ct10/sample_019 | DLR D-Flow | 1000 | 975 | 20.3911 | 20.1571 | step_limit |
| r200_n1_ct10/sample_020 | DLR D-Flow | 641 | 613 | 6.2843 | 19.6672 | stalled |
| r200_n1_ct10/sample_021 | DLR D-Flow | 286 | 278 | 105.1359 | 97.1117 | stalled |
| r200_n1_ct10/sample_022 | DLR D-Flow | 1000 | 976 | 43.5739 | 28.5453 | step_limit |
| r200_n1_ct10/sample_023 | DLR D-Flow | 1000 | 963 | 5.6353 | 19.7001 | step_limit |
| r200_n1_ct10/sample_024 | DLR D-Flow | 500 | 488 | 4.0165 | 19.5845 | stalled |
| r200_n1_ct10/sample_025 | DLR D-Flow | 232 | 222 | 2.7375 | 19.5201 | stalled |
| r200_n1_ct10/sample_026 | DLR D-Flow | 266 | 250 | 3.9593 | 19.5226 | stalled |
| r200_n1_ct10/sample_027 | DLR D-Flow | 416 | 404 | 3.7181 | 19.7309 | stalled |
| r200_n1_ct10/sample_028 | DLR D-Flow | 178 | 171 | 7.8132 | 19.6097 | stalled |
| r200_n1_ct10/sample_029 | DLR D-Flow | 293 | 285 | 3.2237 | 19.5897 | stalled |
| r200_n2_ct10/sample_000 | D-Flow | 639 | 637 | 121.2997 | 0.6034 | stalled |
| r200_n2_ct10/sample_001 | D-Flow | 1000 | 1000 | 48.7371 | 33.6872 | step_limit |
| r200_n2_ct10/sample_002 | D-Flow | 651 | 649 | 134.1416 | 1.2079 | stalled |
| r200_n2_ct10/sample_003 | D-Flow | 1000 | 1000 | 55.1671 | 39.9164 | step_limit |
| r200_n2_ct10/sample_004 | D-Flow | 1000 | 1000 | 71.8252 | 9.3158 | step_limit |
| r200_n2_ct10/sample_005 | D-Flow | 638 | 636 | 128.4275 | 0.9540 | stalled |
| r200_n2_ct10/sample_006 | D-Flow | 476 | 474 | 100.4230 | 1.3778 | stalled |
| r200_n2_ct10/sample_007 | D-Flow | 578 | 576 | 112.5352 | 0.8876 | stalled |
| r200_n2_ct10/sample_008 | D-Flow | 604 | 602 | 137.0824 | 0.7165 | stalled |
| r200_n2_ct10/sample_009 | D-Flow | 919 | 917 | 130.7480 | 0.3817 | stalled |
| r200_n2_ct10/sample_010 | D-Flow | 899 | 897 | 110.8498 | 1.1026 | stalled |
| r200_n2_ct10/sample_011 | D-Flow | 1000 | 1000 | 9.4117 | 19.6175 | step_limit |
| r200_n2_ct10/sample_012 | D-Flow | 548 | 546 | 152.4804 | 0.7396 | stalled |
| r200_n2_ct10/sample_000 | P-Flow | 71 | 60 | 10.9973 | 19.6415 | stalled |
| r200_n2_ct10/sample_001 | P-Flow | 666 | 631 | 78.1869 | 1.3944 | stalled |
| r200_n2_ct10/sample_002 | P-Flow | 1000 | 908 | 98.5652 | 1.2114 | step_limit |
| r200_n2_ct10/sample_003 | P-Flow | 1000 | 860 | 62.2648 | 1.9618 | step_limit |
| r200_n2_ct10/sample_004 | P-Flow | 109 | 92 | 8.5427 | 19.5446 | stalled |
| r200_n2_ct10/sample_005 | P-Flow | 620 | 501 | 9.0276 | 19.5378 | stalled |
| r200_n2_ct10/sample_006 | P-Flow | 107 | 85 | 5.1706 | 19.6694 | stalled |
| r200_n2_ct10/sample_007 | P-Flow | 1000 | 920 | 68.4559 | 1.1289 | step_limit |
| r200_n2_ct10/sample_008 | P-Flow | 45 | 38 | 7.3565 | 19.6902 | stalled |
| r200_n2_ct10/sample_009 | P-Flow | 26 | 22 | 6.3911 | 19.6056 | stalled |
| r200_n2_ct10/sample_010 | P-Flow | 852 | 690 | 9.1775 | 19.1927 | stalled |
| r200_n2_ct10/sample_011 | P-Flow | 171 | 148 | 5.8630 | 19.5781 | stalled |
| r200_n2_ct10/sample_012 | P-Flow | 205 | 171 | 7.1865 | 19.5540 | stalled |
| r200_n2_ct10/sample_000 | DLR D-Flow | 149 | 145 | 6.8709 | 19.5203 | stalled |
| r200_n2_ct10/sample_001 | DLR D-Flow | 99 | 95 | 54.5987 | 41.6865 | stalled |
| r200_n2_ct10/sample_002 | DLR D-Flow | 1000 | 968 | 67.1369 | 43.1569 | step_limit |
| r200_n2_ct10/sample_003 | DLR D-Flow | 106 | 100 | 102.9758 | 74.0600 | stalled |
| r200_n2_ct10/sample_004 | DLR D-Flow | 100 | 96 | 72.1565 | 46.7149 | stalled |
| r200_n2_ct10/sample_005 | DLR D-Flow | 1000 | 966 | 6.1078 | 19.6257 | step_limit |
| r200_n2_ct10/sample_006 | DLR D-Flow | 228 | 218 | 5.2517 | 19.5478 | stalled |
| r200_n2_ct10/sample_007 | DLR D-Flow | 472 | 453 | 5.3160 | 19.5456 | stalled |
| r200_n2_ct10/sample_008 | DLR D-Flow | 155 | 146 | 14.4722 | 19.8762 | stalled |
| r200_n2_ct10/sample_009 | DLR D-Flow | 116 | 110 | 22.7870 | 21.0450 | stalled |
| r200_n2_ct10/sample_010 | DLR D-Flow | 707 | 674 | 5.3188 | 19.6786 | stalled |
| r200_n2_ct10/sample_011 | DLR D-Flow | 513 | 487 | 6.5913 | 19.7577 | stalled |
| r200_n2_ct10/sample_012 | DLR D-Flow | 278 | 269 | 22.7491 | 21.2733 | stalled |
| r200_n4_ct10/sample_000 | D-Flow | 912 | 909 | 100.0605 | 1.1181 | stalled |
| r200_n4_ct10/sample_001 | D-Flow | 676 | 674 | 96.2072 | 1.6342 | stalled |
| r200_n4_ct10/sample_002 | D-Flow | 824 | 822 | 118.9226 | 0.2491 | stalled |
| r200_n4_ct10/sample_003 | D-Flow | 1000 | 1000 | 107.2629 | 0.5469 | step_limit |
| r200_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 10.9065 | 19.4537 | step_limit |
| r200_n4_ct10/sample_005 | D-Flow | 634 | 632 | 127.7395 | 0.6247 | stalled |
| r200_n4_ct10/sample_006 | D-Flow | 1000 | 1000 | 137.5673 | 1.6474 | step_limit |
| r200_n4_ct10/sample_007 | D-Flow | 762 | 760 | 126.6795 | 0.4557 | stalled |
| r200_n4_ct10/sample_008 | D-Flow | 703 | 701 | 48.0422 | 28.0418 | stalled |
| r200_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 42.8563 | 17.2209 | step_limit |
| r200_n4_ct10/sample_010 | D-Flow | 598 | 596 | 137.5473 | 1.0983 | stalled |
| r200_n4_ct10/sample_011 | D-Flow | 1000 | 1000 | 124.8290 | 1.1942 | step_limit |
| r200_n4_ct10/sample_012 | D-Flow | 672 | 669 | 119.3332 | 1.0950 | stalled |
| r200_n4_ct10/sample_013 | D-Flow | 784 | 782 | 126.9763 | 1.1591 | stalled |
| r200_n4_ct10/sample_000 | P-Flow | 1000 | 896 | 68.4246 | 1.6127 | step_limit |
| r200_n4_ct10/sample_001 | P-Flow | 1000 | 889 | 65.8533 | 1.7933 | step_limit |
| r200_n4_ct10/sample_002 | P-Flow | 99 | 86 | 7.8508 | 19.4659 | stalled |
| r200_n4_ct10/sample_003 | P-Flow | 1000 | 897 | 68.0317 | 1.1845 | step_limit |
| r200_n4_ct10/sample_004 | P-Flow | 1000 | 916 | 77.4850 | 1.4875 | step_limit |
| r200_n4_ct10/sample_005 | P-Flow | 603 | 591 | 87.0243 | 0.6248 | stalled |
| r200_n4_ct10/sample_006 | P-Flow | 62 | 53 | 10.9616 | 19.5336 | stalled |
| r200_n4_ct10/sample_007 | P-Flow | 175 | 147 | 12.7921 | 19.4964 | stalled |
| r200_n4_ct10/sample_008 | P-Flow | 70 | 62 | 11.0140 | 19.5971 | stalled |
| r200_n4_ct10/sample_009 | P-Flow | 1000 | 899 | 66.7276 | 2.0338 | step_limit |
| r200_n4_ct10/sample_010 | P-Flow | 261 | 217 | 7.1644 | 19.4487 | stalled |
| r200_n4_ct10/sample_011 | P-Flow | 1000 | 883 | 65.7263 | 2.2981 | step_limit |
| r200_n4_ct10/sample_012 | P-Flow | 555 | 535 | 74.7076 | 1.0957 | stalled |
| r200_n4_ct10/sample_013 | P-Flow | 927 | 867 | 79.2290 | 1.2018 | stalled |
| r200_n4_ct10/sample_000 | DLR D-Flow | 991 | 946 | 18.4269 | 20.3732 | stalled |
| r200_n4_ct10/sample_001 | DLR D-Flow | 578 | 552 | 12.5593 | 19.6844 | stalled |
| r200_n4_ct10/sample_002 | DLR D-Flow | 841 | 811 | 43.9539 | 27.8060 | stalled |
| r200_n4_ct10/sample_003 | DLR D-Flow | 204 | 200 | 78.8741 | 46.1287 | stalled |
| r200_n4_ct10/sample_004 | DLR D-Flow | 202 | 194 | 76.0556 | 41.5455 | stalled |
| r200_n4_ct10/sample_005 | DLR D-Flow | 351 | 338 | 50.6746 | 27.9068 | stalled |
| r200_n4_ct10/sample_006 | DLR D-Flow | 910 | 870 | 28.2926 | 21.0042 | stalled |
| r200_n4_ct10/sample_007 | DLR D-Flow | 680 | 653 | 9.7822 | 19.6247 | stalled |
| r200_n4_ct10/sample_008 | DLR D-Flow | 572 | 550 | 65.2705 | 34.4994 | stalled |
| r200_n4_ct10/sample_009 | DLR D-Flow | 1000 | 958 | 69.6349 | 41.7322 | step_limit |
| r200_n4_ct10/sample_010 | DLR D-Flow | 771 | 748 | 54.0760 | 30.8396 | stalled |
| r200_n4_ct10/sample_011 | DLR D-Flow | 541 | 515 | 90.3916 | 51.9443 | stalled |
| r200_n4_ct10/sample_012 | DLR D-Flow | 1000 | 969 | 65.4396 | 38.8685 | step_limit |
| r200_n4_ct10/sample_013 | DLR D-Flow | 1000 | 961 | 43.8755 | 27.1061 | step_limit |
| r200_n4_ct10/sample_014 | DLR D-Flow | 651 | 625 | 28.4849 | 21.4612 | paused |
| r200_n1_mri4/sample_000 | D-Flow | 1000 | 1000 | 7.4818 | 16.1425 | step_limit |
| r200_n1_mri4/sample_001 | D-Flow | 852 | 850 | 24.5146 | 9.3390 | stalled |
| r200_n1_mri4/sample_002 | D-Flow | 1000 | 1000 | 7.9664 | 15.0242 | step_limit |
| r200_n1_mri4/sample_003 | D-Flow | 1000 | 1000 | 7.7290 | 15.9750 | step_limit |
| r200_n1_mri4/sample_004 | D-Flow | 559 | 556 | 22.2088 | 9.4025 | stalled |
| r200_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 10.9094 | 12.5490 | step_limit |
| r200_n1_mri4/sample_006 | D-Flow | 804 | 801 | 25.2073 | 9.4472 | stalled |
| r200_n1_mri4/sample_007 | D-Flow | 190 | 187 | 105.7283 | 104.2062 | stalled |
| r200_n1_mri4/sample_008 | D-Flow | 467 | 464 | 23.8414 | 10.2574 | stalled |
| r200_n1_mri4/sample_009 | D-Flow | 519 | 517 | 23.7039 | 10.2849 | stalled |
| r200_n1_mri4/sample_010 | D-Flow | 679 | 676 | 26.6051 | 9.7971 | stalled |
| r200_n1_mri4/sample_011 | D-Flow | 513 | 511 | 26.7123 | 10.2430 | stalled |
| r200_n1_mri4/sample_012 | D-Flow | 470 | 468 | 22.1372 | 10.1328 | stalled |
| r200_n1_mri4/sample_013 | D-Flow | 561 | 559 | 24.1886 | 10.1123 | stalled |
| r200_n1_mri4/sample_014 | D-Flow | 749 | 747 | 24.5150 | 10.6815 | stalled |
| r200_n1_mri4/sample_015 | D-Flow | 472 | 470 | 21.8036 | 9.4752 | stalled |
| r200_n1_mri4/sample_016 | D-Flow | 57 | 55 | 100.5445 | 99.8369 | stalled |
| r200_n1_mri4/sample_017 | D-Flow | 1000 | 1000 | 8.6005 | 15.2555 | step_limit |
| r200_n1_mri4/sample_018 | D-Flow | 384 | 382 | 25.3695 | 9.3739 | stalled |
| r200_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 7.6254 | 13.6012 | step_limit |
| r200_n1_mri4/sample_020 | D-Flow | 402 | 400 | 25.0288 | 9.5505 | stalled |
| r200_n1_mri4/sample_021 | D-Flow | 1000 | 1000 | 101.5326 | 100.3448 | step_limit |
| r200_n1_mri4/sample_022 | D-Flow | 22 | 20 | 101.0728 | 100.0122 | stalled |
| r200_n1_mri4/sample_023 | D-Flow | 899 | 897 | 22.0861 | 8.9612 | stalled |
| r200_n1_mri4/sample_024 | D-Flow | 1000 | 1000 | 7.7828 | 15.1066 | step_limit |
| r200_n1_mri4/sample_025 | D-Flow | 548 | 546 | 27.5715 | 8.9866 | stalled |
| r200_n1_mri4/sample_026 | D-Flow | 687 | 685 | 23.8507 | 9.4160 | stalled |
| r200_n1_mri4/sample_027 | D-Flow | 1000 | 1000 | 9.7483 | 14.6326 | step_limit |
| r200_n1_mri4/sample_028 | D-Flow | 627 | 623 | 20.3638 | 10.0609 | stalled |
| r200_n1_mri4/sample_029 | D-Flow | 1000 | 1000 | 23.7979 | 10.6344 | step_limit |
| r200_n1_mri4/sample_000 | P-Flow | 24 | 19 | 2.9423 | 19.7391 | stalled |
| r200_n1_mri4/sample_001 | P-Flow | 229 | 207 | 27.5231 | 9.3389 | stalled |
| r200_n1_mri4/sample_002 | P-Flow | 231 | 212 | 24.5579 | 9.7285 | stalled |
| r200_n1_mri4/sample_003 | P-Flow | 50 | 41 | 1.6143 | 19.6040 | stalled |
| r200_n1_mri4/sample_004 | P-Flow | 187 | 175 | 25.5919 | 9.4025 | stalled |
| r200_n1_mri4/sample_005 | P-Flow | 282 | 259 | 24.1473 | 10.1185 | stalled |
| r200_n1_mri4/sample_006 | P-Flow | 200 | 193 | 28.1829 | 9.4472 | stalled |
| r200_n1_mri4/sample_007 | P-Flow | 234 | 209 | 26.5207 | 10.2900 | stalled |
| r200_n1_mri4/sample_008 | P-Flow | 18 | 13 | 1.6408 | 19.6621 | stalled |
| r200_n1_mri4/sample_009 | P-Flow | 165 | 160 | 26.1255 | 10.2849 | stalled |
| r200_n1_mri4/sample_010 | P-Flow | 185 | 177 | 27.3620 | 9.7971 | stalled |
| r200_n1_mri4/sample_011 | P-Flow | 27 | 22 | 1.5077 | 19.6203 | stalled |
| r200_n1_mri4/sample_012 | P-Flow | 181 | 172 | 24.8075 | 10.1328 | stalled |
| r200_n1_mri4/sample_013 | P-Flow | 169 | 163 | 24.2988 | 10.1123 | stalled |
| r200_n1_mri4/sample_014 | P-Flow | 179 | 175 | 28.1342 | 10.6815 | stalled |
| r200_n1_mri4/sample_015 | P-Flow | 25 | 21 | 1.5809 | 19.6362 | stalled |
| r200_n1_mri4/sample_016 | P-Flow | 196 | 187 | 27.1485 | 9.8443 | stalled |
| r200_n1_mri4/sample_017 | P-Flow | 164 | 159 | 26.6650 | 10.6664 | stalled |
| r200_n1_mri4/sample_018 | P-Flow | 162 | 157 | 27.8947 | 9.3739 | stalled |
| r200_n1_mri4/sample_019 | P-Flow | 160 | 155 | 20.8093 | 9.0925 | stalled |
| r200_n1_mri4/sample_020 | P-Flow | 41 | 32 | 2.4836 | 19.6867 | stalled |
| r200_n1_mri4/sample_021 | P-Flow | 204 | 191 | 29.2446 | 8.8951 | stalled |
| r200_n1_mri4/sample_022 | P-Flow | 163 | 161 | 28.5902 | 9.8552 | stalled |
| r200_n1_mri4/sample_023 | P-Flow | 69 | 56 | 3.8004 | 19.8906 | stalled |
| r200_n1_mri4/sample_024 | P-Flow | 177 | 170 | 25.1890 | 9.4626 | stalled |
| r200_n1_mri4/sample_025 | P-Flow | 172 | 167 | 28.0610 | 8.9866 | stalled |
| r200_n1_mri4/sample_026 | P-Flow | 182 | 172 | 25.4293 | 9.4160 | stalled |
| r200_n1_mri4/sample_027 | P-Flow | 179 | 172 | 27.0186 | 10.6821 | stalled |
| r200_n1_mri4/sample_028 | P-Flow | 159 | 152 | 23.8563 | 10.0609 | stalled |
| r200_n1_mri4/sample_029 | P-Flow | 171 | 167 | 24.7570 | 10.6319 | stalled |
| r200_n1_mri4/sample_000 | DLR D-Flow | 195 | 190 | 1.7246 | 19.6515 | stalled |
| r200_n1_mri4/sample_001 | DLR D-Flow | 689 | 663 | 1.6312 | 19.6477 | stalled |
| r200_n1_mri4/sample_002 | DLR D-Flow | 298 | 280 | 1.2944 | 19.5813 | stalled |
| r200_n1_mri4/sample_003 | DLR D-Flow | 311 | 295 | 1.2402 | 19.5934 | stalled |
| r200_n1_mri4/sample_004 | DLR D-Flow | 438 | 421 | 43.0597 | 44.4808 | stalled |
| r200_n1_mri4/sample_005 | DLR D-Flow | 594 | 568 | 1.7503 | 19.6441 | stalled |
| r200_n1_mri4/sample_006 | DLR D-Flow | 493 | 469 | 6.1493 | 20.3458 | stalled |
| r200_n1_mri4/sample_007 | DLR D-Flow | 20 | 16 | 104.7358 | 102.9341 | stalled |
| r200_n1_mri4/sample_008 | DLR D-Flow | 519 | 498 | 1.4618 | 19.6237 | stalled |
| r200_n1_mri4/sample_009 | DLR D-Flow | 752 | 722 | 1.7563 | 19.6128 | stalled |
| r200_n1_mri4/sample_010 | DLR D-Flow | 342 | 329 | 1.3157 | 19.6023 | stalled |
| r200_n1_mri4/sample_011 | DLR D-Flow | 575 | 548 | 1.2748 | 19.5858 | stalled |
| r200_n1_mri4/sample_012 | DLR D-Flow | 60 | 55 | 100.3903 | 99.3283 | stalled |
| r200_n1_mri4/sample_013 | DLR D-Flow | 33 | 30 | 80.8302 | 82.2937 | stalled |
| r200_n1_mri4/sample_014 | DLR D-Flow | 179 | 168 | 97.0773 | 94.9880 | stalled |
| r200_n1_mri4/sample_015 | DLR D-Flow | 469 | 453 | 1.2327 | 19.6072 | stalled |
| r200_n1_mri4/sample_016 | DLR D-Flow | 507 | 487 | 1.5695 | 19.6381 | stalled |
| r200_n1_mri4/sample_017 | DLR D-Flow | 134 | 127 | 100.7372 | 100.0434 | stalled |
| r200_n1_mri4/sample_018 | DLR D-Flow | 356 | 335 | 1.3280 | 19.6307 | stalled |
| r200_n1_mri4/sample_019 | DLR D-Flow | 713 | 695 | 1.6924 | 19.6613 | stalled |
| r200_n1_mri4/sample_020 | DLR D-Flow | 654 | 627 | 1.2743 | 19.6151 | stalled |
| r200_n1_mri4/sample_021 | DLR D-Flow | 286 | 277 | 100.7476 | 99.6290 | stalled |
| r200_n1_mri4/sample_022 | DLR D-Flow | 32 | 27 | 101.0262 | 99.9702 | stalled |
| r200_n1_mri4/sample_023 | DLR D-Flow | 185 | 175 | 6.7074 | 20.5236 | stalled |
| r200_n1_mri4/sample_024 | DLR D-Flow | 534 | 516 | 1.3231 | 19.6382 | stalled |
| r200_n1_mri4/sample_025 | DLR D-Flow | 222 | 213 | 1.3004 | 19.6314 | stalled |
| r200_n1_mri4/sample_026 | DLR D-Flow | 646 | 627 | 1.3550 | 19.6476 | stalled |
| r200_n1_mri4/sample_027 | DLR D-Flow | 178 | 170 | 3.4988 | 19.9031 | stalled |
| r200_n1_mri4/sample_028 | DLR D-Flow | 354 | 335 | 1.4816 | 19.6330 | stalled |
| r200_n1_mri4/sample_029 | DLR D-Flow | 99 | 93 | 65.6699 | 68.2054 | stalled |
| r200_n2_mri4/sample_000 | D-Flow | 777 | 775 | 24.9791 | 10.2436 | stalled |
| r200_n2_mri4/sample_001 | D-Flow | 598 | 595 | 51.3542 | 51.7129 | stalled |
| r200_n2_mri4/sample_002 | D-Flow | 578 | 576 | 26.0522 | 9.8736 | stalled |
| r200_n2_mri4/sample_003 | D-Flow | 415 | 412 | 62.7074 | 63.2887 | stalled |
| r200_n2_mri4/sample_004 | D-Flow | 157 | 155 | 61.0268 | 61.6835 | stalled |
| r200_n2_mri4/sample_005 | D-Flow | 824 | 822 | 26.9398 | 9.8923 | stalled |
| r200_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 6.1708 | 19.0709 | step_limit |
| r200_n2_mri4/sample_007 | D-Flow | 227 | 223 | 57.0658 | 58.0994 | stalled |
| r200_n2_mri4/sample_008 | D-Flow | 1000 | 1000 | 5.9153 | 19.1695 | step_limit |
| r200_n2_mri4/sample_009 | D-Flow | 682 | 680 | 26.2628 | 9.5482 | stalled |
| r200_n2_mri4/sample_010 | D-Flow | 1000 | 1000 | 24.6020 | 11.3355 | step_limit |
| r200_n2_mri4/sample_011 | D-Flow | 261 | 258 | 67.2335 | 69.1543 | stalled |
| r200_n2_mri4/sample_012 | D-Flow | 1000 | 1000 | 2.9527 | 19.5148 | step_limit |
| r200_n2_mri4/sample_013 | D-Flow | 498 | 495 | 26.9609 | 10.3029 | stalled |
| r200_n2_mri4/sample_000 | P-Flow | 231 | 225 | 25.5990 | 10.2436 | stalled |
| r200_n2_mri4/sample_001 | P-Flow | 198 | 190 | 23.1102 | 10.2527 | stalled |
| r200_n2_mri4/sample_002 | P-Flow | 175 | 170 | 26.7353 | 9.8736 | stalled |
| r200_n2_mri4/sample_003 | P-Flow | 177 | 172 | 26.7620 | 10.6265 | stalled |
| r200_n2_mri4/sample_004 | P-Flow | 233 | 215 | 25.4055 | 10.6078 | stalled |
| r200_n2_mri4/sample_005 | P-Flow | 193 | 189 | 26.9352 | 9.8923 | stalled |
| r200_n2_mri4/sample_006 | P-Flow | 177 | 169 | 27.1153 | 9.3275 | stalled |
| r200_n2_mri4/sample_007 | P-Flow | 176 | 171 | 24.7851 | 9.4266 | stalled |
| r200_n2_mri4/sample_008 | P-Flow | 205 | 193 | 24.7428 | 10.9846 | stalled |
| r200_n2_mri4/sample_009 | P-Flow | 181 | 178 | 26.8348 | 9.5482 | stalled |
| r200_n2_mri4/sample_010 | P-Flow | 177 | 170 | 27.6614 | 9.6304 | stalled |
| r200_n2_mri4/sample_011 | P-Flow | 186 | 180 | 24.1858 | 11.1030 | stalled |
| r200_n2_mri4/sample_012 | P-Flow | 196 | 186 | 26.1847 | 10.6451 | stalled |
| r200_n2_mri4/sample_013 | P-Flow | 198 | 189 | 26.8818 | 10.3029 | stalled |
| r200_n2_mri4/sample_000 | DLR D-Flow | 1000 | 980 | 3.3020 | 19.7329 | step_limit |
| r200_n2_mri4/sample_001 | DLR D-Flow | 1000 | 968 | 51.6778 | 52.3303 | step_limit |
| r200_n2_mri4/sample_002 | DLR D-Flow | 921 | 894 | 4.6190 | 19.8926 | stalled |
| r200_n2_mri4/sample_003 | DLR D-Flow | 929 | 906 | 64.9168 | 64.6924 | stalled |
| r200_n2_mri4/sample_004 | DLR D-Flow | 1000 | 970 | 62.1104 | 63.0669 | step_limit |
| r200_n2_mri4/sample_005 | DLR D-Flow | 582 | 563 | 1.9985 | 19.6377 | stalled |
| r200_n2_mri4/sample_006 | DLR D-Flow | 145 | 139 | 78.0745 | 75.7105 | stalled |
| r200_n2_mri4/sample_007 | DLR D-Flow | 513 | 497 | 1.5324 | 19.5918 | stalled |
| r200_n2_mri4/sample_008 | DLR D-Flow | 1000 | 978 | 1.8707 | 19.6117 | step_limit |
| r200_n2_mri4/sample_009 | DLR D-Flow | 87 | 84 | 2.3706 | 19.6510 | stalled |
| r200_n2_mri4/sample_010 | DLR D-Flow | 323 | 315 | 2.4395 | 19.7137 | stalled |
| r200_n2_mri4/sample_011 | DLR D-Flow | 1000 | 973 | 3.3456 | 19.7479 | step_limit |
| r200_n2_mri4/sample_012 | DLR D-Flow | 439 | 423 | 52.8384 | 54.7733 | stalled |
| r200_n2_mri4/sample_013 | DLR D-Flow | 572 | 554 | 2.8552 | 19.6580 | stalled |
| r200_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 2.7706 | 19.4676 | step_limit |
| r200_n4_mri4/sample_001 | D-Flow | 771 | 766 | 48.0440 | 48.9338 | stalled |
| r200_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 2.2309 | 19.5444 | step_limit |
| r200_n4_mri4/sample_003 | D-Flow | 1000 | 1000 | 20.7968 | 13.8671 | step_limit |
| r200_n4_mri4/sample_004 | D-Flow | 806 | 804 | 25.3827 | 8.4628 | stalled |
| r200_n4_mri4/sample_005 | D-Flow | 1000 | 1000 | 7.3070 | 19.1365 | step_limit |
| r200_n4_mri4/sample_006 | D-Flow | 785 | 783 | 23.0726 | 11.9574 | stalled |
| r200_n4_mri4/sample_007 | D-Flow | 856 | 854 | 25.4239 | 9.7873 | stalled |
| r200_n4_mri4/sample_008 | D-Flow | 727 | 725 | 24.6281 | 8.9846 | stalled |
| r200_n4_mri4/sample_009 | D-Flow | 1000 | 1000 | 3.8388 | 19.4857 | step_limit |
| r200_n4_mri4/sample_000 | P-Flow | 208 | 195 | 26.4796 | 9.7257 | stalled |
| r200_n4_mri4/sample_001 | P-Flow | 192 | 184 | 24.9434 | 9.5207 | stalled |
| r200_n4_mri4/sample_002 | P-Flow | 186 | 183 | 26.1182 | 9.9728 | stalled |
| r200_n4_mri4/sample_003 | P-Flow | 169 | 166 | 24.1724 | 11.5796 | stalled |
| r200_n4_mri4/sample_004 | P-Flow | 184 | 179 | 26.1083 | 8.4628 | stalled |
| r200_n4_mri4/sample_005 | P-Flow | 191 | 188 | 26.3819 | 8.9474 | stalled |
| r200_n4_mri4/sample_006 | P-Flow | 197 | 191 | 23.1257 | 11.9574 | stalled |
| r200_n4_mri4/sample_007 | P-Flow | 185 | 181 | 25.7466 | 9.7873 | stalled |
| r200_n4_mri4/sample_008 | P-Flow | 190 | 184 | 24.8028 | 8.9846 | stalled |
| r200_n4_mri4/sample_009 | P-Flow | 189 | 182 | 24.2576 | 11.5718 | stalled |
| r200_n4_mri4/sample_010 | P-Flow | 183 | 180 | 26.5085 | 9.8347 | stalled |
| r200_n4_mri4/sample_000 | DLR D-Flow | 270 | 259 | 21.3060 | 25.1638 | stalled |
| r200_n4_mri4/sample_001 | DLR D-Flow | 474 | 446 | 53.7834 | 53.3109 | stalled |
| r200_n4_mri4/sample_002 | DLR D-Flow | 129 | 122 | 69.1893 | 69.0097 | stalled |
| r200_n4_mri4/sample_003 | DLR D-Flow | 354 | 342 | 50.0751 | 52.5843 | stalled |
| r200_n4_mri4/sample_004 | DLR D-Flow | 1000 | 971 | 29.2237 | 30.5728 | step_limit |
| r200_n4_mri4/sample_005 | DLR D-Flow | 1000 | 953 | 31.5301 | 33.9858 | step_limit |
| r200_n4_mri4/sample_006 | DLR D-Flow | 214 | 204 | 45.5729 | 46.9656 | stalled |
| r200_n4_mri4/sample_007 | DLR D-Flow | 1000 | 969 | 19.1630 | 24.2855 | step_limit |
| r200_n4_mri4/sample_008 | DLR D-Flow | 207 | 198 | 56.5281 | 54.3070 | stalled |
| r200_n4_mri4/sample_009 | DLR D-Flow | 1000 | 965 | 11.9222 | 21.2680 | step_limit |
| r200_n4_mri4/sample_010 | DLR D-Flow | 25 | 25 | 80.4429 | 77.9674 | paused |
| r250_n1_ct10/sample_000 | D-Flow | 575 | 573 | 205.6643 | 0.7813 | stalled |
| r250_n1_ct10/sample_001 | D-Flow | 616 | 614 | 148.5193 | 0.6079 | stalled |
| r250_n1_ct10/sample_002 | D-Flow | 635 | 633 | 107.9091 | 0.9154 | stalled |
| r250_n1_ct10/sample_003 | D-Flow | 593 | 590 | 137.6718 | 0.8679 | stalled |
| r250_n1_ct10/sample_004 | D-Flow | 501 | 499 | 151.4661 | 0.9069 | stalled |
| r250_n1_ct10/sample_005 | D-Flow | 691 | 689 | 116.5686 | 0.9851 | stalled |
| r250_n1_ct10/sample_006 | D-Flow | 533 | 530 | 175.4889 | 0.3316 | stalled |
| r250_n1_ct10/sample_007 | D-Flow | 33 | 31 | 99.8154 | 97.3974 | stalled |
| r250_n1_ct10/sample_008 | D-Flow | 790 | 788 | 10.1393 | 15.6677 | stalled |
| r250_n1_ct10/sample_009 | D-Flow | 903 | 900 | 6.9112 | 19.5639 | stalled |
| r250_n1_ct10/sample_010 | D-Flow | 475 | 473 | 146.9791 | 1.0973 | stalled |
| r250_n1_ct10/sample_011 | D-Flow | 1000 | 1000 | 8.2957 | 16.5417 | step_limit |
| r250_n1_ct10/sample_012 | D-Flow | 13 | 11 | 99.7273 | 97.7951 | stalled |
| r250_n1_ct10/sample_013 | D-Flow | 521 | 519 | 131.7103 | 1.0537 | stalled |
| r250_n1_ct10/sample_014 | D-Flow | 786 | 784 | 13.0547 | 13.9595 | stalled |
| r250_n1_ct10/sample_015 | D-Flow | 450 | 448 | 190.5088 | 0.8640 | stalled |
| r250_n1_ct10/sample_016 | D-Flow | 445 | 442 | 152.4074 | 1.2545 | stalled |
| r250_n1_ct10/sample_017 | D-Flow | 170 | 166 | 104.5735 | 94.5633 | stalled |
| r250_n1_ct10/sample_018 | D-Flow | 839 | 836 | 92.4406 | 1.1011 | stalled |
| r250_n1_ct10/sample_019 | D-Flow | 624 | 622 | 146.9908 | 0.7486 | stalled |
| r250_n1_ct10/sample_020 | D-Flow | 521 | 518 | 175.6158 | 1.0343 | stalled |
| r250_n1_ct10/sample_021 | D-Flow | 956 | 952 | 10.8129 | 14.4863 | stalled |
| r250_n1_ct10/sample_022 | D-Flow | 486 | 484 | 131.6849 | 0.6391 | stalled |
| r250_n1_ct10/sample_023 | D-Flow | 1000 | 1000 | 9.9766 | 15.1128 | step_limit |
| r250_n1_ct10/sample_024 | D-Flow | 1000 | 1000 | 8.0253 | 17.3398 | step_limit |
| r250_n1_ct10/sample_025 | D-Flow | 941 | 938 | 116.2053 | 0.9919 | stalled |
| r250_n1_ct10/sample_026 | D-Flow | 555 | 553 | 124.7193 | 1.1398 | stalled |
| r250_n1_ct10/sample_027 | D-Flow | 1000 | 1000 | 14.6423 | 12.8604 | step_limit |
| r250_n1_ct10/sample_028 | D-Flow | 915 | 913 | 10.3129 | 16.6765 | stalled |
| r250_n1_ct10/sample_029 | D-Flow | 466 | 464 | 135.7037 | 0.7476 | stalled |
| r250_n1_ct10/sample_000 | P-Flow | 498 | 489 | 89.7622 | 0.7814 | stalled |
| r250_n1_ct10/sample_001 | P-Flow | 633 | 593 | 90.9258 | 0.6080 | stalled |
| r250_n1_ct10/sample_002 | P-Flow | 21 | 17 | 6.5556 | 19.7719 | stalled |
| r250_n1_ct10/sample_003 | P-Flow | 462 | 381 | 3.0809 | 19.6166 | stalled |
| r250_n1_ct10/sample_004 | P-Flow | 611 | 581 | 80.6805 | 0.9166 | stalled |
| r250_n1_ct10/sample_005 | P-Flow | 9 | 7 | 112.0824 | 95.6440 | stalled |
| r250_n1_ct10/sample_006 | P-Flow | 579 | 556 | 99.5912 | 0.3317 | stalled |
| r250_n1_ct10/sample_007 | P-Flow | 838 | 693 | 3.3884 | 19.7000 | stalled |
| r250_n1_ct10/sample_008 | P-Flow | 550 | 510 | 71.5962 | 1.0332 | stalled |
| r250_n1_ct10/sample_009 | P-Flow | 652 | 623 | 86.2629 | 0.3204 | stalled |
| r250_n1_ct10/sample_010 | P-Flow | 403 | 394 | 78.4662 | 1.0975 | stalled |
| r250_n1_ct10/sample_011 | P-Flow | 93 | 75 | 5.0711 | 19.6427 | stalled |
| r250_n1_ct10/sample_012 | P-Flow | 46 | 37 | 4.0837 | 19.6576 | stalled |
| r250_n1_ct10/sample_013 | P-Flow | 570 | 525 | 77.6911 | 1.0538 | stalled |
| r250_n1_ct10/sample_014 | P-Flow | 68 | 63 | 8.3099 | 18.4353 | stalled |
| r250_n1_ct10/sample_015 | P-Flow | 399 | 391 | 85.5207 | 0.8640 | stalled |
| r250_n1_ct10/sample_016 | P-Flow | 398 | 390 | 92.3444 | 1.2546 | stalled |
| r250_n1_ct10/sample_017 | P-Flow | 406 | 327 | 2.5264 | 19.5670 | stalled |
| r250_n1_ct10/sample_018 | P-Flow | 580 | 538 | 68.2740 | 1.1011 | stalled |
| r250_n1_ct10/sample_019 | P-Flow | 22 | 17 | 6.4538 | 19.7033 | stalled |
| r250_n1_ct10/sample_020 | P-Flow | 28 | 24 | 3.6589 | 19.6810 | stalled |
| r250_n1_ct10/sample_021 | P-Flow | 87 | 78 | 10.1325 | 18.4924 | stalled |
| r250_n1_ct10/sample_022 | P-Flow | 674 | 658 | 73.1445 | 0.6395 | stalled |
| r250_n1_ct10/sample_023 | P-Flow | 1000 | 855 | 65.4349 | 0.6321 | step_limit |
| r250_n1_ct10/sample_024 | P-Flow | 258 | 212 | 2.7243 | 19.5489 | stalled |
| r250_n1_ct10/sample_025 | P-Flow | 298 | 242 | 3.8073 | 19.6873 | stalled |
| r250_n1_ct10/sample_026 | P-Flow | 77 | 63 | 2.1159 | 19.6582 | stalled |
| r250_n1_ct10/sample_027 | P-Flow | 147 | 117 | 2.9819 | 19.6552 | stalled |
| r250_n1_ct10/sample_028 | P-Flow | 131 | 110 | 4.6927 | 19.6799 | stalled |
| r250_n1_ct10/sample_029 | P-Flow | 448 | 436 | 78.4742 | 0.7479 | stalled |
| r250_n1_ct10/sample_000 | DLR D-Flow | 323 | 312 | 6.0025 | 19.4522 | stalled |
| r250_n1_ct10/sample_001 | DLR D-Flow | 392 | 374 | 6.4925 | 19.5943 | stalled |
| r250_n1_ct10/sample_002 | DLR D-Flow | 147 | 143 | 4.5904 | 19.7364 | stalled |
| r250_n1_ct10/sample_003 | DLR D-Flow | 206 | 199 | 2.3799 | 19.6360 | stalled |
| r250_n1_ct10/sample_004 | DLR D-Flow | 285 | 271 | 4.5119 | 19.6065 | stalled |
| r250_n1_ct10/sample_005 | DLR D-Flow | 267 | 255 | 3.2093 | 19.6634 | stalled |
| r250_n1_ct10/sample_006 | DLR D-Flow | 112 | 109 | 16.0055 | 19.6571 | stalled |
| r250_n1_ct10/sample_007 | DLR D-Flow | 22 | 19 | 99.6421 | 96.9741 | stalled |
| r250_n1_ct10/sample_008 | DLR D-Flow | 371 | 359 | 4.0467 | 19.5168 | stalled |
| r250_n1_ct10/sample_009 | DLR D-Flow | 249 | 240 | 3.5775 | 19.6534 | stalled |
| r250_n1_ct10/sample_010 | DLR D-Flow | 368 | 353 | 13.4189 | 19.5416 | stalled |
| r250_n1_ct10/sample_011 | DLR D-Flow | 257 | 248 | 4.2647 | 19.4789 | stalled |
| r250_n1_ct10/sample_012 | DLR D-Flow | 13 | 11 | 99.7303 | 97.7943 | stalled |
| r250_n1_ct10/sample_013 | DLR D-Flow | 806 | 769 | 13.6514 | 19.6140 | stalled |
| r250_n1_ct10/sample_014 | DLR D-Flow | 25 | 23 | 99.5346 | 97.0971 | stalled |
| r250_n1_ct10/sample_015 | DLR D-Flow | 326 | 309 | 4.7703 | 19.5800 | stalled |
| r250_n1_ct10/sample_016 | DLR D-Flow | 256 | 245 | 4.1064 | 19.5802 | stalled |
| r250_n1_ct10/sample_017 | DLR D-Flow | 819 | 796 | 2.5440 | 19.5632 | stalled |
| r250_n1_ct10/sample_018 | DLR D-Flow | 69 | 62 | 108.1672 | 93.8404 | stalled |
| r250_n1_ct10/sample_019 | DLR D-Flow | 204 | 195 | 1.6925 | 19.5834 | stalled |
| r250_n1_ct10/sample_020 | DLR D-Flow | 425 | 407 | 4.9215 | 19.6274 | stalled |
| r250_n1_ct10/sample_021 | DLR D-Flow | 291 | 280 | 3.8501 | 19.5389 | stalled |
| r250_n1_ct10/sample_022 | DLR D-Flow | 363 | 347 | 5.7413 | 19.4727 | stalled |
| r250_n1_ct10/sample_023 | DLR D-Flow | 29 | 25 | 111.8835 | 92.0728 | stalled |
| r250_n1_ct10/sample_024 | DLR D-Flow | 224 | 215 | 3.1607 | 19.5277 | stalled |
| r250_n1_ct10/sample_025 | DLR D-Flow | 42 | 40 | 3.1796 | 19.7061 | stalled |
| r250_n1_ct10/sample_026 | DLR D-Flow | 511 | 490 | 3.4530 | 19.5740 | stalled |
| r250_n1_ct10/sample_027 | DLR D-Flow | 562 | 545 | 18.4665 | 20.9925 | stalled |
| r250_n1_ct10/sample_028 | DLR D-Flow | 233 | 228 | 2.0684 | 19.6128 | stalled |
| r250_n1_ct10/sample_029 | DLR D-Flow | 257 | 242 | 6.5237 | 19.5175 | stalled |
| r250_n2_ct10/sample_000 | D-Flow | 503 | 500 | 208.2113 | 0.5376 | stalled |
| r250_n2_ct10/sample_001 | D-Flow | 551 | 549 | 246.9994 | 0.9209 | stalled |
| r250_n2_ct10/sample_002 | D-Flow | 513 | 511 | 26.5790 | 16.7647 | stalled |
| r250_n2_ct10/sample_003 | D-Flow | 689 | 687 | 26.3151 | 16.6597 | stalled |
| r250_n2_ct10/sample_004 | D-Flow | 729 | 727 | 163.3842 | 0.6045 | stalled |
| r250_n2_ct10/sample_005 | D-Flow | 517 | 514 | 164.0044 | 1.1445 | stalled |
| r250_n2_ct10/sample_006 | D-Flow | 1000 | 1000 | 24.7540 | 17.3446 | step_limit |
| r250_n2_ct10/sample_007 | D-Flow | 337 | 335 | 25.5529 | 12.5090 | stalled |
| r250_n2_ct10/sample_008 | D-Flow | 487 | 485 | 142.6706 | 0.8856 | stalled |
| r250_n2_ct10/sample_009 | D-Flow | 439 | 436 | 159.9241 | 1.0139 | stalled |
| r250_n2_ct10/sample_000 | P-Flow | 473 | 468 | 90.1004 | 0.5377 | stalled |
| r250_n2_ct10/sample_001 | P-Flow | 127 | 110 | 5.7277 | 19.4983 | stalled |
| r250_n2_ct10/sample_002 | P-Flow | 618 | 576 | 81.6235 | 0.9505 | stalled |
| r250_n2_ct10/sample_003 | P-Flow | 564 | 551 | 80.9186 | 0.8428 | stalled |
| r250_n2_ct10/sample_004 | P-Flow | 627 | 599 | 82.8382 | 0.6045 | stalled |
| r250_n2_ct10/sample_005 | P-Flow | 99 | 87 | 7.1591 | 19.3218 | stalled |
| r250_n2_ct10/sample_006 | P-Flow | 53 | 43 | 9.8796 | 19.7141 | stalled |
| r250_n2_ct10/sample_007 | P-Flow | 361 | 292 | 6.2091 | 19.2642 | stalled |
| r250_n2_ct10/sample_008 | P-Flow | 464 | 368 | 4.2776 | 19.6165 | stalled |
| r250_n2_ct10/sample_009 | P-Flow | 418 | 405 | 84.1931 | 1.0142 | stalled |
| r250_n2_ct10/sample_000 | DLR D-Flow | 256 | 240 | 14.8958 | 20.1199 | stalled |
| r250_n2_ct10/sample_001 | DLR D-Flow | 187 | 179 | 6.8619 | 19.6304 | stalled |
| r250_n2_ct10/sample_002 | DLR D-Flow | 339 | 327 | 10.3677 | 19.7380 | stalled |
| r250_n2_ct10/sample_003 | DLR D-Flow | 494 | 473 | 5.2940 | 19.5582 | stalled |
| r250_n2_ct10/sample_004 | DLR D-Flow | 325 | 314 | 6.2668 | 19.5981 | stalled |
| r250_n2_ct10/sample_005 | DLR D-Flow | 429 | 422 | 23.2801 | 20.7069 | stalled |
| r250_n2_ct10/sample_006 | DLR D-Flow | 341 | 328 | 4.7769 | 19.5768 | stalled |
| r250_n2_ct10/sample_007 | DLR D-Flow | 706 | 687 | 42.5292 | 27.1629 | stalled |
| r250_n2_ct10/sample_008 | DLR D-Flow | 171 | 163 | 19.6677 | 20.3835 | stalled |
| r250_n2_ct10/sample_009 | DLR D-Flow | 108 | 102 | 16.7359 | 20.2215 | stalled |
| r250_n4_ct10/sample_000 | D-Flow | 1000 | 1000 | 165.7614 | 7.6665 | step_limit |
| r250_n4_ct10/sample_001 | D-Flow | 685 | 682 | 203.3043 | 0.7847 | stalled |
| r250_n4_ct10/sample_002 | D-Flow | 1000 | 998 | 132.1841 | 1.1005 | stalled |
| r250_n4_ct10/sample_003 | D-Flow | 822 | 820 | 127.3132 | 0.8376 | stalled |
| r250_n4_ct10/sample_004 | D-Flow | 1000 | 1000 | 88.0310 | 7.6665 | step_limit |
| r250_n4_ct10/sample_005 | D-Flow | 540 | 538 | 260.6793 | 0.2309 | stalled |
| r250_n4_ct10/sample_006 | D-Flow | 626 | 624 | 271.3155 | 0.6179 | stalled |
| r250_n4_ct10/sample_007 | D-Flow | 572 | 570 | 186.1675 | 0.7071 | stalled |
| r250_n4_ct10/sample_008 | D-Flow | 875 | 872 | 135.5229 | 0.7476 | stalled |
| r250_n4_ct10/sample_009 | D-Flow | 1000 | 1000 | 42.2599 | 25.3866 | step_limit |
| r250_n4_ct10/sample_000 | P-Flow | 76 | 66 | 14.2210 | 19.5620 | stalled |
| r250_n4_ct10/sample_001 | P-Flow | 914 | 870 | 90.3773 | 0.7859 | stalled |
| r250_n4_ct10/sample_002 | P-Flow | 67 | 59 | 10.6680 | 19.5764 | stalled |
| r250_n4_ct10/sample_003 | P-Flow | 98 | 90 | 11.6916 | 19.3277 | stalled |
| r250_n4_ct10/sample_004 | P-Flow | 48 | 40 | 9.2129 | 19.6665 | stalled |
| r250_n4_ct10/sample_005 | P-Flow | 1000 | 921 | 79.1985 | 0.2697 | step_limit |
| r250_n4_ct10/sample_006 | P-Flow | 745 | 720 | 90.6307 | 0.6186 | stalled |
| r250_n4_ct10/sample_007 | P-Flow | 273 | 231 | 8.5934 | 19.4484 | stalled |
| r250_n4_ct10/sample_008 | P-Flow | 185 | 166 | 9.6538 | 19.2281 | stalled |
| r250_n4_ct10/sample_009 | P-Flow | 172 | 140 | 9.0341 | 19.5127 | stalled |
| r250_n4_ct10/sample_000 | DLR D-Flow | 1000 | 963 | 21.9554 | 20.3101 | step_limit |
| r250_n4_ct10/sample_001 | DLR D-Flow | 782 | 755 | 11.3269 | 19.7518 | stalled |
| r250_n4_ct10/sample_002 | DLR D-Flow | 747 | 719 | 53.8351 | 28.9652 | stalled |
| r250_n4_ct10/sample_003 | DLR D-Flow | 570 | 538 | 20.2099 | 19.9531 | stalled |
| r250_n4_ct10/sample_004 | DLR D-Flow | 1000 | 970 | 59.2304 | 34.6358 | step_limit |
| r250_n4_ct10/sample_005 | DLR D-Flow | 1000 | 957 | 14.2871 | 19.9031 | step_limit |
| r250_n4_ct10/sample_006 | DLR D-Flow | 782 | 745 | 43.0796 | 25.1044 | stalled |
| r250_n4_ct10/sample_007 | DLR D-Flow | 1000 | 960 | 70.6821 | 37.4249 | step_limit |
| r250_n4_ct10/sample_008 | DLR D-Flow | 1000 | 968 | 57.1844 | 30.9721 | step_limit |
| r250_n4_ct10/sample_009 | DLR D-Flow | 1000 | 968 | 41.2372 | 24.5895 | step_limit |
| r250_n1_mri4/sample_000 | D-Flow | 537 | 535 | 27.2971 | 9.8379 | stalled |
| r250_n1_mri4/sample_001 | D-Flow | 484 | 482 | 24.4966 | 10.9321 | stalled |
| r250_n1_mri4/sample_002 | D-Flow | 536 | 534 | 24.0511 | 10.6711 | stalled |
| r250_n1_mri4/sample_003 | D-Flow | 573 | 570 | 25.5303 | 9.8460 | stalled |
| r250_n1_mri4/sample_004 | D-Flow | 536 | 534 | 23.9280 | 9.1238 | stalled |
| r250_n1_mri4/sample_005 | D-Flow | 1000 | 1000 | 25.4441 | 10.0914 | step_limit |
| r250_n1_mri4/sample_006 | D-Flow | 659 | 656 | 26.2665 | 10.7050 | stalled |
| r250_n1_mri4/sample_007 | D-Flow | 29 | 27 | 99.6314 | 99.6073 | stalled |
| r250_n1_mri4/sample_008 | D-Flow | 1000 | 1000 | 5.6101 | 18.9550 | step_limit |
| r250_n1_mri4/sample_009 | D-Flow | 1000 | 1000 | 5.9787 | 16.8051 | step_limit |
| r250_n1_mri4/sample_010 | D-Flow | 627 | 625 | 21.8592 | 10.8002 | stalled |
| r250_n1_mri4/sample_011 | D-Flow | 1000 | 1000 | 24.1050 | 10.7669 | step_limit |
| r250_n1_mri4/sample_012 | D-Flow | 18 | 13 | 99.5923 | 99.5538 | stalled |
| r250_n1_mri4/sample_013 | D-Flow | 1000 | 1000 | 6.0497 | 15.7174 | step_limit |
| r250_n1_mri4/sample_014 | D-Flow | 520 | 517 | 97.9125 | 97.4022 | stalled |
| r250_n1_mri4/sample_015 | D-Flow | 557 | 555 | 26.1926 | 10.6924 | stalled |
| r250_n1_mri4/sample_016 | D-Flow | 527 | 525 | 26.4498 | 10.4207 | stalled |
| r250_n1_mri4/sample_017 | D-Flow | 29 | 27 | 99.4919 | 99.4720 | stalled |
| r250_n1_mri4/sample_018 | D-Flow | 97 | 90 | 99.5417 | 99.5166 | stalled |
| r250_n1_mri4/sample_019 | D-Flow | 1000 | 1000 | 25.1480 | 9.7445 | step_limit |
| r250_n1_mri4/sample_020 | D-Flow | 557 | 554 | 25.5473 | 9.1107 | stalled |
| r250_n1_mri4/sample_021 | D-Flow | 15 | 13 | 99.5007 | 99.4838 | stalled |
| r250_n1_mri4/sample_022 | D-Flow | 587 | 585 | 24.1131 | 10.6690 | stalled |
| r250_n1_mri4/sample_023 | D-Flow | 81 | 77 | 99.5849 | 99.5755 | stalled |
| r250_n1_mri4/sample_024 | D-Flow | 911 | 907 | 24.8029 | 10.4728 | stalled |
| r250_n1_mri4/sample_025 | D-Flow | 1000 | 1000 | 7.5018 | 15.1003 | step_limit |
| r250_n1_mri4/sample_026 | D-Flow | 1000 | 1000 | 26.1913 | 10.1003 | step_limit |
| r250_n1_mri4/sample_027 | D-Flow | 142 | 139 | 99.6523 | 99.6379 | stalled |
| r250_n1_mri4/sample_028 | D-Flow | 1000 | 1000 | 6.5706 | 15.9287 | step_limit |
| r250_n1_mri4/sample_029 | D-Flow | 18 | 13 | 99.5092 | 99.3691 | stalled |
| r250_n1_mri4/sample_000 | P-Flow | 168 | 163 | 26.7528 | 9.8379 | stalled |
| r250_n1_mri4/sample_001 | P-Flow | 165 | 160 | 25.4168 | 10.9321 | stalled |
| r250_n1_mri4/sample_002 | P-Flow | 57 | 48 | 1.8263 | 19.6458 | stalled |
| r250_n1_mri4/sample_003 | P-Flow | 34 | 26 | 1.8384 | 19.6202 | stalled |
| r250_n1_mri4/sample_004 | P-Flow | 182 | 175 | 24.9531 | 9.1238 | stalled |
| r250_n1_mri4/sample_005 | P-Flow | 12 | 8 | 2.0508 | 19.6470 | stalled |
| r250_n1_mri4/sample_006 | P-Flow | 194 | 186 | 25.9139 | 10.7050 | stalled |
| r250_n1_mri4/sample_007 | P-Flow | 159 | 155 | 29.1085 | 9.5334 | stalled |
| r250_n1_mri4/sample_008 | P-Flow | 226 | 213 | 28.6615 | 9.4075 | stalled |
| r250_n1_mri4/sample_009 | P-Flow | 45 | 38 | 1.9564 | 19.6681 | stalled |
| r250_n1_mri4/sample_010 | P-Flow | 183 | 177 | 23.8615 | 10.8002 | stalled |
| r250_n1_mri4/sample_011 | P-Flow | 176 | 168 | 25.7239 | 10.7659 | stalled |
| r250_n1_mri4/sample_012 | P-Flow | 201 | 187 | 27.8750 | 10.2050 | stalled |
| r250_n1_mri4/sample_013 | P-Flow | 190 | 181 | 23.6630 | 10.6192 | stalled |
| r250_n1_mri4/sample_014 | P-Flow | 175 | 167 | 27.4029 | 9.7659 | stalled |
| r250_n1_mri4/sample_015 | P-Flow | 190 | 183 | 26.0402 | 10.6924 | stalled |
| r250_n1_mri4/sample_016 | P-Flow | 189 | 185 | 26.0432 | 10.4207 | stalled |
| r250_n1_mri4/sample_017 | P-Flow | 188 | 180 | 28.0015 | 9.3692 | stalled |
| r250_n1_mri4/sample_018 | P-Flow | 181 | 176 | 29.2443 | 8.8165 | stalled |
| r250_n1_mri4/sample_019 | P-Flow | 177 | 169 | 24.6566 | 9.7433 | stalled |
| r250_n1_mri4/sample_020 | P-Flow | 35 | 29 | 1.8372 | 19.6644 | stalled |
| r250_n1_mri4/sample_021 | P-Flow | 26 | 20 | 1.8242 | 19.6781 | stalled |
| r250_n1_mri4/sample_022 | P-Flow | 165 | 159 | 26.7086 | 10.6690 | stalled |
| r250_n1_mri4/sample_023 | P-Flow | 189 | 176 | 29.1052 | 9.2078 | stalled |
| r250_n1_mri4/sample_024 | P-Flow | 189 | 179 | 25.8543 | 10.4728 | stalled |
| r250_n1_mri4/sample_025 | P-Flow | 19 | 15 | 1.8951 | 19.6549 | stalled |
| r250_n1_mri4/sample_026 | P-Flow | 165 | 162 | 27.3751 | 10.1003 | stalled |
| r250_n1_mri4/sample_027 | P-Flow | 180 | 174 | 28.4893 | 10.5043 | stalled |
| r250_n1_mri4/sample_028 | P-Flow | 193 | 182 | 27.5785 | 9.8474 | stalled |
| r250_n1_mri4/sample_029 | P-Flow | 178 | 172 | 24.5021 | 9.9511 | stalled |
| r250_n1_mri4/sample_000 | DLR D-Flow | 611 | 574 | 1.4261 | 19.5688 | stalled |
| r250_n1_mri4/sample_001 | DLR D-Flow | 356 | 340 | 1.4289 | 19.6211 | stalled |
| r250_n1_mri4/sample_002 | DLR D-Flow | 335 | 322 | 1.3175 | 19.6234 | stalled |
| r250_n1_mri4/sample_003 | DLR D-Flow | 44 | 42 | 2.1446 | 19.6436 | stalled |
| r250_n1_mri4/sample_004 | DLR D-Flow | 182 | 174 | 1.2073 | 19.6067 | stalled |
| r250_n1_mri4/sample_005 | DLR D-Flow | 151 | 146 | 1.5055 | 19.6144 | stalled |
| r250_n1_mri4/sample_006 | DLR D-Flow | 72 | 68 | 1.1150 | 19.6419 | stalled |
| r250_n1_mri4/sample_007 | DLR D-Flow | 22 | 18 | 99.4765 | 99.4461 | stalled |
| r250_n1_mri4/sample_008 | DLR D-Flow | 547 | 520 | 1.8708 | 19.6260 | stalled |
| r250_n1_mri4/sample_009 | DLR D-Flow | 354 | 339 | 1.4880 | 19.6159 | stalled |
| r250_n1_mri4/sample_010 | DLR D-Flow | 197 | 189 | 2.6358 | 19.7224 | stalled |
| r250_n1_mri4/sample_011 | DLR D-Flow | 836 | 806 | 1.3369 | 19.6632 | stalled |
| r250_n1_mri4/sample_012 | DLR D-Flow | 17 | 14 | 99.5921 | 99.5536 | stalled |
| r250_n1_mri4/sample_013 | DLR D-Flow | 435 | 419 | 1.3554 | 19.6056 | stalled |
| r250_n1_mri4/sample_014 | DLR D-Flow | 39 | 36 | 99.8747 | 99.8489 | stalled |
| r250_n1_mri4/sample_015 | DLR D-Flow | 172 | 163 | 1.2150 | 19.6214 | stalled |
| r250_n1_mri4/sample_016 | DLR D-Flow | 723 | 691 | 1.2312 | 19.6249 | stalled |
| r250_n1_mri4/sample_017 | DLR D-Flow | 20 | 17 | 99.5250 | 99.5084 | stalled |
| r250_n1_mri4/sample_018 | DLR D-Flow | 53 | 49 | 99.6111 | 99.5904 | stalled |
| r250_n1_mri4/sample_019 | DLR D-Flow | 256 | 250 | 1.4902 | 19.6342 | stalled |
| r250_n1_mri4/sample_020 | DLR D-Flow | 713 | 688 | 1.3412 | 19.6254 | stalled |
| r250_n1_mri4/sample_021 | DLR D-Flow | 13 | 11 | 97.8061 | 97.7051 | stalled |
| r250_n1_mri4/sample_022 | DLR D-Flow | 386 | 372 | 1.3365 | 19.6130 | stalled |
| r250_n1_mri4/sample_023 | DLR D-Flow | 23 | 20 | 99.2783 | 99.2682 | stalled |
| r250_n1_mri4/sample_024 | DLR D-Flow | 286 | 278 | 1.2639 | 19.6060 | stalled |
| r250_n1_mri4/sample_025 | DLR D-Flow | 380 | 366 | 1.4631 | 19.6297 | stalled |
| r250_n1_mri4/sample_026 | DLR D-Flow | 307 | 295 | 1.3742 | 19.6377 | stalled |
| r250_n1_mri4/sample_027 | DLR D-Flow | 18 | 16 | 99.6100 | 99.5978 | stalled |
| r250_n1_mri4/sample_028 | DLR D-Flow | 57 | 54 | 2.1730 | 19.7020 | stalled |
| r250_n1_mri4/sample_029 | DLR D-Flow | 14 | 12 | 99.4976 | 99.3545 | stalled |
| r250_n2_mri4/sample_000 | D-Flow | 621 | 619 | 24.8481 | 10.1890 | stalled |
| r250_n2_mri4/sample_001 | D-Flow | 819 | 816 | 24.1159 | 9.7897 | stalled |
| r250_n2_mri4/sample_002 | D-Flow | 345 | 340 | 44.4878 | 46.6471 | stalled |
| r250_n2_mri4/sample_003 | D-Flow | 1000 | 1000 | 3.3266 | 19.3013 | step_limit |
| r250_n2_mri4/sample_004 | D-Flow | 1000 | 1000 | 26.1189 | 10.4055 | step_limit |
| r250_n2_mri4/sample_005 | D-Flow | 1000 | 1000 | 5.6467 | 19.2958 | step_limit |
| r250_n2_mri4/sample_006 | D-Flow | 1000 | 1000 | 25.9387 | 10.6047 | step_limit |
| r250_n2_mri4/sample_007 | D-Flow | 568 | 566 | 21.9603 | 10.6390 | stalled |
| r250_n2_mri4/sample_008 | D-Flow | 685 | 682 | 22.8181 | 8.7354 | stalled |
| r250_n2_mri4/sample_009 | D-Flow | 876 | 874 | 27.2065 | 9.2074 | stalled |
| r250_n2_mri4/sample_000 | P-Flow | 178 | 173 | 25.4174 | 10.1890 | stalled |
| r250_n2_mri4/sample_001 | P-Flow | 172 | 167 | 26.3444 | 9.7897 | stalled |
| r250_n2_mri4/sample_002 | P-Flow | 166 | 162 | 26.9800 | 9.9038 | stalled |
| r250_n2_mri4/sample_003 | P-Flow | 179 | 171 | 26.5332 | 10.3511 | stalled |
| r250_n2_mri4/sample_004 | P-Flow | 174 | 170 | 26.1208 | 10.4055 | stalled |
| r250_n2_mri4/sample_005 | P-Flow | 182 | 179 | 23.2166 | 10.0574 | stalled |
| r250_n2_mri4/sample_006 | P-Flow | 179 | 172 | 26.8063 | 10.1451 | stalled |
| r250_n2_mri4/sample_007 | P-Flow | 159 | 154 | 25.6096 | 10.6390 | stalled |
| r250_n2_mri4/sample_008 | P-Flow | 186 | 177 | 24.2239 | 8.7354 | stalled |
| r250_n2_mri4/sample_009 | P-Flow | 198 | 190 | 27.7548 | 9.2074 | stalled |
| r250_n2_mri4/sample_000 | DLR D-Flow | 153 | 149 | 5.3161 | 20.1583 | stalled |
| r250_n2_mri4/sample_001 | DLR D-Flow | 554 | 535 | 1.7803 | 19.6545 | stalled |
| r250_n2_mri4/sample_002 | DLR D-Flow | 316 | 310 | 44.6888 | 46.8803 | stalled |
| r250_n2_mri4/sample_003 | DLR D-Flow | 1000 | 958 | 2.1232 | 19.6772 | step_limit |
| r250_n2_mri4/sample_004 | DLR D-Flow | 183 | 175 | 4.0814 | 19.8830 | stalled |
| r250_n2_mri4/sample_005 | DLR D-Flow | 243 | 235 | 83.4664 | 82.6766 | stalled |
| r250_n2_mri4/sample_006 | DLR D-Flow | 653 | 636 | 1.7966 | 19.6895 | stalled |
| r250_n2_mri4/sample_007 | DLR D-Flow | 37 | 33 | 71.7844 | 70.0133 | stalled |
| r250_n2_mri4/sample_008 | DLR D-Flow | 1000 | 967 | 1.7387 | 19.6613 | step_limit |
| r250_n2_mri4/sample_009 | DLR D-Flow | 431 | 415 | 1.3773 | 19.6248 | stalled |
| r250_n4_mri4/sample_000 | D-Flow | 1000 | 1000 | 2.5555 | 19.5176 | step_limit |
| r250_n4_mri4/sample_001 | D-Flow | 1000 | 1000 | 3.1003 | 19.2313 | step_limit |
| r250_n4_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.5041 | 19.5979 | step_limit |
| r250_n4_mri4/sample_003 | D-Flow | 986 | 982 | 23.8612 | 10.4588 | stalled |
| r250_n4_mri4/sample_004 | D-Flow | 1000 | 1000 | 11.4042 | 18.9088 | step_limit |
| r250_n4_mri4/sample_005 | D-Flow | 680 | 676 | 22.6673 | 11.3302 | stalled |
| r250_n4_mri4/sample_006 | D-Flow | 858 | 856 | 25.8813 | 10.3669 | stalled |
| r250_n4_mri4/sample_007 | D-Flow | 1000 | 1000 | 26.5161 | 10.0228 | step_limit |
| r250_n4_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.8541 | 19.5872 | step_limit |
| r250_n4_mri4/sample_009 | D-Flow | 787 | 785 | 25.7109 | 10.6859 | stalled |
| r250_n4_mri4/sample_000 | P-Flow | 242 | 235 | 24.3038 | 10.1670 | stalled |
| r250_n4_mri4/sample_001 | P-Flow | 190 | 187 | 25.9545 | 10.4574 | stalled |
| r250_n4_mri4/sample_002 | P-Flow | 178 | 175 | 25.1697 | 11.0549 | stalled |
| r250_n4_mri4/sample_003 | P-Flow | 174 | 169 | 24.3697 | 10.4588 | stalled |
| r250_n4_mri4/sample_004 | P-Flow | 179 | 175 | 26.2715 | 10.4237 | stalled |
| r250_n4_mri4/sample_005 | P-Flow | 179 | 174 | 22.9361 | 11.3302 | stalled |
| r250_n4_mri4/sample_006 | P-Flow | 196 | 191 | 25.9852 | 10.3669 | stalled |
| r250_n4_mri4/sample_007 | P-Flow | 179 | 175 | 26.7395 | 10.0138 | stalled |
| r250_n4_mri4/sample_008 | P-Flow | 246 | 228 | 25.9901 | 10.6981 | stalled |
| r250_n4_mri4/sample_009 | P-Flow | 181 | 178 | 26.1232 | 10.6859 | stalled |
| r250_n4_mri4/sample_000 | DLR D-Flow | 1000 | 968 | 47.1640 | 46.3205 | step_limit |
| r250_n4_mri4/sample_001 | DLR D-Flow | 605 | 575 | 33.6452 | 35.0722 | stalled |
| r250_n4_mri4/sample_002 | DLR D-Flow | 1000 | 959 | 49.8874 | 50.2261 | step_limit |
| r250_n4_mri4/sample_003 | DLR D-Flow | 1000 | 955 | 14.1609 | 22.9142 | step_limit |
| r250_n4_mri4/sample_004 | DLR D-Flow | 1000 | 965 | 19.4176 | 25.7391 | step_limit |
| r250_n4_mri4/sample_005 | DLR D-Flow | 986 | 945 | 42.8247 | 45.0706 | stalled |
| r250_n4_mri4/sample_006 | DLR D-Flow | 1000 | 960 | 50.9313 | 49.7859 | step_limit |
| r250_n4_mri4/sample_007 | DLR D-Flow | 205 | 200 | 17.4433 | 24.2029 | stalled |
| r250_n4_mri4/sample_008 | DLR D-Flow | 829 | 803 | 73.7976 | 72.8998 | stalled |
| r250_n4_mri4/sample_009 | DLR D-Flow | 1000 | 969 | 42.0490 | 43.4575 | step_limit |

Terminal method runs: 1954/3240. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.
