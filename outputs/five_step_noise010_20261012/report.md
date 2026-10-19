# Five-step Euler pilot

10% relative measurement noise; CT10 and Cartesian MRI4; 30 single-circle test samples at each resolution. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank8, analytical alpha=1-t, no trace probes/enrichment and previous V.

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

Terminal method runs: 1080/1080. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including image/residual panels and error versus optimizer attempts.
