# Five-step Euler pilot

0% relative measurement noise; CT10 and Cartesian MRI4; 30 test samples per acquisition at each resolution; separated shapes: 1–4 circles/triangles. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 8, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_shapes_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.7279 | 0.0462 | step_limit |
| r64_shapes_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.5355 | 0.0512 | step_limit |
| r64_shapes_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.6889 | 0.0580 | step_limit |
| r64_shapes_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.7579 | 0.0560 | step_limit |
| r64_shapes_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.3049 | 0.0225 | step_limit |
| r64_shapes_ct10/sample_005 | D-Flow | 1000 | 1000 | 1.5954 | 0.0333 | step_limit |
| r64_shapes_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.6618 | 0.0517 | step_limit |
| r64_shapes_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.6925 | 0.1705 | step_limit |
| r64_shapes_ct10/sample_008 | D-Flow | 1000 | 1000 | 2.3932 | 0.0465 | step_limit |
| r64_shapes_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.9508 | 0.0844 | step_limit |
| r64_shapes_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.2556 | 0.0437 | step_limit |
| r64_shapes_ct10/sample_000 | P-Flow | 1000 | 991 | 4.6331 | 0.6917 | step_limit |
| r64_shapes_ct10/sample_001 | P-Flow | 279 | 231 | 5.1225 | 1.6351 | stalled |
| r64_shapes_ct10/sample_002 | P-Flow | 9 | 7 | 100.2009 | 98.6354 | stalled |
| r64_shapes_ct10/sample_003 | P-Flow | 1000 | 814 | 6.8053 | 2.2904 | step_limit |
| r64_shapes_ct10/sample_004 | P-Flow | 1000 | 817 | 9.8452 | 2.9682 | step_limit |
| r64_shapes_ct10/sample_005 | P-Flow | 417 | 340 | 4.0601 | 0.9921 | stalled |
| r64_shapes_ct10/sample_006 | P-Flow | 1000 | 961 | 4.2923 | 0.4668 | step_limit |
| r64_shapes_ct10/sample_007 | P-Flow | 51 | 41 | 54.5456 | 28.1188 | stalled |
| r64_shapes_ct10/sample_008 | P-Flow | 5 | 3 | 100.0326 | 99.0201 | stalled |
| r64_shapes_ct10/sample_009 | P-Flow | 18 | 16 | 82.0445 | 51.2712 | stalled |
| r64_shapes_ct10/sample_010 | P-Flow | 92 | 86 | 2.2873 | 0.6092 | stalled |
| r64_shapes_ct10/sample_000 | DLR D-Flow | 739 | 692 | 78.8474 | 68.9183 | stalled |
| r64_shapes_ct10/sample_001 | DLR D-Flow | 228 | 220 | 102.8367 | 96.8816 | stalled |
| r64_shapes_ct10/sample_002 | DLR D-Flow | 60 | 56 | 100.8987 | 97.7778 | stalled |
| r64_shapes_ct10/sample_003 | DLR D-Flow | 697 | 674 | 103.1740 | 94.9935 | stalled |
| r64_shapes_ct10/sample_004 | DLR D-Flow | 221 | 212 | 30.2782 | 18.4231 | stalled |
| r64_shapes_ct10/sample_005 | DLR D-Flow | 1000 | 981 | 102.6544 | 70.0049 | step_limit |
| r64_shapes_ct10/sample_006 | DLR D-Flow | 1000 | 964 | 108.5081 | 93.5329 | step_limit |
| r64_shapes_ct10/sample_007 | DLR D-Flow | 759 | 737 | 78.1420 | 52.9361 | stalled |
| r64_shapes_ct10/sample_008 | DLR D-Flow | 1000 | 954 | 100.5704 | 95.8739 | step_limit |
| r64_shapes_ct10/sample_009 | DLR D-Flow | 1000 | 974 | 72.9598 | 47.1471 | step_limit |
| r64_shapes_ct10/sample_010 | DLR D-Flow | 282 | 272 | 63.1994 | 53.7074 | stalled |
| r64_shapes_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.4149 | 0.0432 | step_limit |
| r64_shapes_mri4/sample_001 | D-Flow | 1000 | 1000 | 100.0664 | 100.0790 | step_limit |
| r64_shapes_mri4/sample_002 | D-Flow | 989 | 987 | 2.6615 | 0.0051 | stalled |
| r64_shapes_mri4/sample_003 | D-Flow | 842 | 840 | 1.8157 | 0.0057 | stalled |
| r64_shapes_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.2436 | 0.0207 | step_limit |
| r64_shapes_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.1774 | 0.0303 | step_limit |
| r64_shapes_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.2225 | 0.0907 | step_limit |
| r64_shapes_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.5514 | 0.3784 | step_limit |
| r64_shapes_mri4/sample_008 | D-Flow | 1000 | 999 | 2.1424 | 0.1584 | step_limit |
| r64_shapes_mri4/sample_009 | D-Flow | 1000 | 1000 | 2.0984 | 0.2020 | step_limit |
| r64_shapes_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.0968 | 0.0455 | step_limit |
| r64_shapes_mri4/sample_000 | P-Flow | 94 | 84 | 3.5853 | 1.8255 | stalled |
| r64_shapes_mri4/sample_001 | P-Flow | 77 | 68 | 3.3672 | 1.7566 | stalled |
| r64_shapes_mri4/sample_002 | P-Flow | 105 | 91 | 5.9641 | 2.3325 | stalled |
| r64_shapes_mri4/sample_003 | P-Flow | 124 | 104 | 2.6960 | 1.4515 | stalled |
| r64_shapes_mri4/sample_004 | P-Flow | 824 | 706 | 3.6889 | 0.7536 | stalled |
| r64_shapes_mri4/sample_005 | P-Flow | 18 | 15 | 79.7986 | 76.2018 | stalled |
| r64_shapes_mri4/sample_006 | P-Flow | 126 | 117 | 2.8600 | 1.2557 | stalled |
| r64_shapes_mri4/sample_007 | P-Flow | 176 | 153 | 3.3056 | 1.3283 | stalled |
| r64_shapes_mri4/sample_008 | P-Flow | 4 | 2 | 99.9722 | 99.9398 | stalled |
| r64_shapes_mri4/sample_009 | P-Flow | 98 | 88 | 3.6444 | 1.2070 | stalled |
| r64_shapes_mri4/sample_010 | P-Flow | 44 | 38 | 2.0798 | 1.3244 | stalled |
| r64_shapes_mri4/sample_000 | DLR D-Flow | 450 | 428 | 76.3618 | 74.8839 | stalled |
| r64_shapes_mri4/sample_001 | DLR D-Flow | 284 | 275 | 99.8533 | 99.5994 | stalled |
| r64_shapes_mri4/sample_002 | DLR D-Flow | 187 | 181 | 101.7113 | 98.0991 | stalled |
| r64_shapes_mri4/sample_003 | DLR D-Flow | 381 | 370 | 99.8358 | 99.6940 | stalled |
| r64_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 963 | 73.9069 | 72.9103 | step_limit |
| r64_shapes_mri4/sample_005 | DLR D-Flow | 140 | 133 | 99.8090 | 99.6376 | stalled |
| r64_shapes_mri4/sample_006 | DLR D-Flow | 285 | 270 | 100.3488 | 99.2442 | stalled |
| r64_shapes_mri4/sample_007 | DLR D-Flow | 632 | 613 | 100.6852 | 98.9287 | stalled |
| r64_shapes_mri4/sample_008 | DLR D-Flow | 297 | 291 | 99.7637 | 99.6284 | stalled |
| r64_shapes_mri4/sample_009 | DLR D-Flow | 59 | 54 | 83.9104 | 84.2522 | stalled |
| r64_shapes_mri4/sample_010 | DLR D-Flow | 392 | 368 | 61.1091 | 59.1649 | stalled |
| r96_shapes_ct10/sample_000 | D-Flow | 1000 | 1000 | 1.8871 | 0.0998 | step_limit |
| r96_shapes_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.8252 | 0.0666 | step_limit |
| r96_shapes_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.8036 | 0.0709 | step_limit |
| r96_shapes_ct10/sample_003 | D-Flow | 1000 | 1000 | 2.1390 | 0.0418 | step_limit |
| r96_shapes_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.2166 | 0.0225 | step_limit |
| r96_shapes_ct10/sample_005 | D-Flow | 1000 | 1000 | 1.5210 | 0.0974 | step_limit |
| r96_shapes_ct10/sample_006 | D-Flow | 1000 | 1000 | 8.4901 | 2.6138 | step_limit |
| r96_shapes_ct10/sample_007 | D-Flow | 1000 | 1000 | 24.5613 | 8.4389 | step_limit |
| r96_shapes_ct10/sample_008 | D-Flow | 1000 | 1000 | 2.6476 | 0.0511 | step_limit |
| r96_shapes_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.7728 | 0.0330 | step_limit |
| r96_shapes_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.6606 | 0.3070 | step_limit |
| r96_shapes_ct10/sample_000 | P-Flow | 1000 | 942 | 4.4115 | 0.4620 | step_limit |
| r96_shapes_ct10/sample_001 | P-Flow | 8 | 5 | 124.2786 | 100.9524 | stalled |
| r96_shapes_ct10/sample_002 | P-Flow | 31 | 26 | 64.1145 | 37.0613 | stalled |
| r96_shapes_ct10/sample_003 | P-Flow | 1000 | 798 | 7.4152 | 2.5148 | step_limit |
| r96_shapes_ct10/sample_004 | P-Flow | 1000 | 940 | 2.3708 | 0.4716 | step_limit |
| r96_shapes_ct10/sample_005 | P-Flow | 1000 | 814 | 15.8905 | 5.9248 | step_limit |
| r96_shapes_ct10/sample_006 | P-Flow | 12 | 9 | 103.2490 | 96.6992 | stalled |
| r96_shapes_ct10/sample_007 | P-Flow | 44 | 38 | 42.4280 | 20.3879 | stalled |
| r96_shapes_ct10/sample_008 | P-Flow | 44 | 36 | 7.3918 | 2.0416 | stalled |
| r96_shapes_ct10/sample_009 | P-Flow | 49 | 42 | 36.7271 | 13.7460 | stalled |
| r96_shapes_ct10/sample_010 | P-Flow | 238 | 207 | 2.8338 | 0.7605 | stalled |
| r96_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 968 | 84.8165 | 73.5801 | step_limit |
| r96_shapes_ct10/sample_001 | DLR D-Flow | 45 | 41 | 102.3571 | 95.3335 | stalled |
| r96_shapes_ct10/sample_002 | DLR D-Flow | 284 | 277 | 100.4289 | 94.2847 | stalled |
| r96_shapes_ct10/sample_003 | DLR D-Flow | 421 | 407 | 100.3618 | 95.2616 | stalled |
| r96_shapes_ct10/sample_004 | DLR D-Flow | 105 | 101 | 77.9043 | 60.2001 | stalled |
| r96_shapes_ct10/sample_005 | DLR D-Flow | 1000 | 966 | 67.2109 | 43.8887 | step_limit |
| r96_shapes_ct10/sample_006 | DLR D-Flow | 1000 | 977 | 106.7433 | 92.2502 | step_limit |
| r96_shapes_ct10/sample_007 | DLR D-Flow | 1000 | 974 | 60.9295 | 33.4996 | step_limit |
| r96_shapes_ct10/sample_008 | DLR D-Flow | 597 | 569 | 101.1764 | 93.2691 | stalled |
| r96_shapes_ct10/sample_009 | DLR D-Flow | 1000 | 980 | 99.1017 | 79.1098 | step_limit |
| r96_shapes_ct10/sample_010 | DLR D-Flow | 793 | 744 | 66.5140 | 42.2436 | stalled |
| r96_shapes_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.7818 | 0.2445 | step_limit |
| r96_shapes_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.5076 | 0.1189 | step_limit |
| r96_shapes_mri4/sample_002 | D-Flow | 1000 | 1000 | 2.7584 | 0.0044 | step_limit |
| r96_shapes_mri4/sample_003 | D-Flow | 1000 | 1000 | 2.0056 | 0.2763 | step_limit |
| r96_shapes_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.0286 | 0.0822 | step_limit |
| r96_shapes_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.2080 | 0.4286 | step_limit |
| r96_shapes_mri4/sample_006 | D-Flow | 320 | 318 | 111.6204 | 88.3635 | stalled |
| r96_shapes_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.4477 | 0.3221 | step_limit |
| r96_shapes_mri4/sample_008 | D-Flow | 1000 | 1000 | 2.1351 | 0.0040 | step_limit |
| r96_shapes_mri4/sample_009 | D-Flow | 1000 | 1000 | 1.3954 | 0.1753 | step_limit |
| r96_shapes_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.0737 | 0.2610 | step_limit |
| r96_shapes_mri4/sample_000 | P-Flow | 67 | 63 | 2.9843 | 1.5507 | stalled |
| r96_shapes_mri4/sample_001 | P-Flow | 142 | 122 | 3.0120 | 1.3826 | stalled |
| r96_shapes_mri4/sample_002 | P-Flow | 136 | 116 | 9.4722 | 6.3125 | stalled |
| r96_shapes_mri4/sample_003 | P-Flow | 53 | 46 | 2.9157 | 1.6281 | stalled |
| r96_shapes_mri4/sample_004 | P-Flow | 82 | 70 | 2.1357 | 1.2122 | stalled |
| r96_shapes_mri4/sample_005 | P-Flow | 56 | 48 | 2.6829 | 1.5346 | stalled |
| r96_shapes_mri4/sample_006 | P-Flow | 13 | 11 | 117.7905 | 82.8362 | stalled |
| r96_shapes_mri4/sample_007 | P-Flow | 61 | 52 | 2.4211 | 1.2754 | stalled |
| r96_shapes_mri4/sample_008 | P-Flow | 143 | 127 | 4.9679 | 2.2442 | stalled |
| r96_shapes_mri4/sample_009 | P-Flow | 79 | 72 | 2.8477 | 1.4382 | stalled |
| r96_shapes_mri4/sample_010 | P-Flow | 79 | 68 | 1.9387 | 1.1018 | stalled |
| r96_shapes_mri4/sample_000 | DLR D-Flow | 225 | 216 | 82.2297 | 81.9932 | stalled |
| r96_shapes_mri4/sample_001 | DLR D-Flow | 439 | 429 | 99.5716 | 99.2864 | stalled |
| r96_shapes_mri4/sample_002 | DLR D-Flow | 73 | 68 | 99.6326 | 99.4040 | stalled |
| r96_shapes_mri4/sample_003 | DLR D-Flow | 1000 | 970 | 99.4967 | 99.2978 | step_limit |
| r96_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 972 | 97.8279 | 94.9682 | step_limit |
| r96_shapes_mri4/sample_005 | DLR D-Flow | 1000 | 980 | 70.3816 | 70.0553 | step_limit |
| r96_shapes_mri4/sample_006 | DLR D-Flow | 1000 | 969 | 100.7105 | 98.1433 | step_limit |
| r96_shapes_mri4/sample_007 | DLR D-Flow | 602 | 584 | 70.5114 | 68.4543 | stalled |
| r96_shapes_mri4/sample_008 | DLR D-Flow | 523 | 502 | 7.8629 | 4.3744 | stalled |
| r96_shapes_mri4/sample_009 | DLR D-Flow | 1000 | 972 | 99.3559 | 98.9783 | step_limit |
| r96_shapes_mri4/sample_010 | DLR D-Flow | 349 | 335 | 53.6046 | 52.2132 | stalled |
| r128_shapes_ct10/sample_000 | D-Flow | 17 | 15 | 99.8204 | 95.8639 | stalled |
| r128_shapes_ct10/sample_001 | D-Flow | 1000 | 1000 | 1.7902 | 0.1357 | step_limit |
| r128_shapes_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.4559 | 0.0475 | step_limit |
| r128_shapes_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.7614 | 0.0456 | step_limit |
| r128_shapes_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.1422 | 0.0944 | step_limit |
| r128_shapes_ct10/sample_005 | D-Flow | 1000 | 1000 | 66.1657 | 43.3124 | step_limit |
| r128_shapes_ct10/sample_006 | D-Flow | 1000 | 1000 | 1.5229 | 0.2063 | step_limit |
| r128_shapes_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.9303 | 0.2947 | step_limit |
| r128_shapes_ct10/sample_008 | D-Flow | 1000 | 1000 | 30.4043 | 11.4656 | step_limit |
| r128_shapes_ct10/sample_009 | D-Flow | 1000 | 1000 | 1.4944 | 0.0791 | step_limit |
| r128_shapes_ct10/sample_010 | D-Flow | 1000 | 1000 | 1.0732 | 0.0377 | step_limit |
| r128_shapes_ct10/sample_011 | D-Flow | 1000 | 1000 | 2.5596 | 0.0391 | step_limit |
| r128_shapes_ct10/sample_012 | D-Flow | 1000 | 1000 | 96.0709 | 77.1734 | step_limit |
| r128_shapes_ct10/sample_000 | P-Flow | 1000 | 975 | 4.0241 | 0.3465 | step_limit |
| r128_shapes_ct10/sample_001 | P-Flow | 93 | 78 | 2.8748 | 0.7164 | stalled |
| r128_shapes_ct10/sample_002 | P-Flow | 1000 | 975 | 5.0218 | 1.3105 | step_limit |
| r128_shapes_ct10/sample_003 | P-Flow | 1000 | 809 | 3.6804 | 1.4046 | step_limit |
| r128_shapes_ct10/sample_004 | P-Flow | 39 | 34 | 20.3017 | 7.4495 | stalled |
| r128_shapes_ct10/sample_005 | P-Flow | 1000 | 924 | 3.1844 | 0.6458 | step_limit |
| r128_shapes_ct10/sample_006 | P-Flow | 1000 | 950 | 2.5801 | 0.2664 | step_limit |
| r128_shapes_ct10/sample_007 | P-Flow | 1000 | 977 | 2.5596 | 0.4372 | step_limit |
| r128_shapes_ct10/sample_008 | P-Flow | 1000 | 985 | 3.5915 | 0.8727 | step_limit |
| r128_shapes_ct10/sample_009 | P-Flow | 129 | 112 | 7.4360 | 1.4840 | stalled |
| r128_shapes_ct10/sample_010 | P-Flow | 34 | 29 | 6.1319 | 2.1878 | stalled |
| r128_shapes_ct10/sample_011 | P-Flow | 1000 | 977 | 4.6935 | 0.9860 | step_limit |
| r128_shapes_ct10/sample_012 | P-Flow | 81 | 70 | 5.9339 | 1.7571 | stalled |
| r128_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 962 | 99.5985 | 92.6690 | step_limit |
| r128_shapes_ct10/sample_001 | DLR D-Flow | 1000 | 980 | 99.7100 | 94.2123 | step_limit |
| r128_shapes_ct10/sample_002 | DLR D-Flow | 1000 | 968 | 104.3587 | 90.5157 | step_limit |
| r128_shapes_ct10/sample_003 | DLR D-Flow | 243 | 230 | 104.8848 | 93.5074 | stalled |
| r128_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 970 | 79.6018 | 54.5463 | step_limit |
| r128_shapes_ct10/sample_005 | DLR D-Flow | 348 | 331 | 101.8980 | 96.4495 | stalled |
| r128_shapes_ct10/sample_006 | DLR D-Flow | 1000 | 955 | 104.7453 | 83.1087 | step_limit |
| r128_shapes_ct10/sample_007 | DLR D-Flow | 321 | 315 | 108.8492 | 88.6275 | stalled |
| r128_shapes_ct10/sample_008 | DLR D-Flow | 140 | 133 | 102.9138 | 95.9468 | stalled |
| r128_shapes_ct10/sample_009 | DLR D-Flow | 1000 | 976 | 109.7426 | 80.4702 | step_limit |
| r128_shapes_ct10/sample_010 | DLR D-Flow | 413 | 390 | 5.6459 | 2.4185 | stalled |
| r128_shapes_ct10/sample_011 | DLR D-Flow | 471 | 451 | 5.9333 | 1.8642 | stalled |
| r128_shapes_ct10/sample_012 | DLR D-Flow | 126 | 121 | 100.7029 | 95.6683 | stalled |
| r128_shapes_mri4/sample_000 | D-Flow | 1000 | 1000 | 1.1389 | 0.3312 | step_limit |
| r128_shapes_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.5155 | 0.6310 | step_limit |
| r128_shapes_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.8013 | 0.2934 | step_limit |
| r128_shapes_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.8654 | 1.1692 | step_limit |
| r128_shapes_mri4/sample_004 | D-Flow | 1000 | 1000 | 0.9459 | 0.2473 | step_limit |
| r128_shapes_mri4/sample_005 | D-Flow | 1000 | 1000 | 1.4786 | 0.6216 | step_limit |
| r128_shapes_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.2751 | 0.5850 | step_limit |
| r128_shapes_mri4/sample_007 | D-Flow | 1000 | 1000 | 64.7172 | 62.8336 | step_limit |
| r128_shapes_mri4/sample_008 | D-Flow | 1000 | 1000 | 2.9587 | 1.6206 | step_limit |
| r128_shapes_mri4/sample_009 | D-Flow | 1000 | 1000 | 78.4642 | 77.0712 | step_limit |
| r128_shapes_mri4/sample_010 | D-Flow | 1000 | 1000 | 0.8999 | 0.2697 | step_limit |
| r128_shapes_mri4/sample_011 | D-Flow | 1000 | 1000 | 2.1633 | 0.0412 | step_limit |
| r128_shapes_mri4/sample_012 | D-Flow | 1000 | 1000 | 2.2468 | 0.1086 | step_limit |
| r128_shapes_mri4/sample_000 | P-Flow | 153 | 134 | 2.3749 | 1.2138 | stalled |
| r128_shapes_mri4/sample_001 | P-Flow | 72 | 60 | 2.3209 | 1.3382 | stalled |
| r128_shapes_mri4/sample_002 | P-Flow | 11 | 9 | 99.9414 | 99.0143 | stalled |
| r128_shapes_mri4/sample_003 | P-Flow | 35 | 30 | 2.2205 | 1.5143 | stalled |
| r128_shapes_mri4/sample_004 | P-Flow | 91 | 80 | 2.0219 | 0.9919 | stalled |
| r128_shapes_mri4/sample_005 | P-Flow | 110 | 93 | 2.2706 | 1.1073 | stalled |
| r128_shapes_mri4/sample_006 | P-Flow | 102 | 87 | 2.0711 | 1.0575 | stalled |
| r128_shapes_mri4/sample_007 | P-Flow | 114 | 101 | 1.8903 | 0.9728 | stalled |
| r128_shapes_mri4/sample_008 | P-Flow | 97 | 87 | 3.2865 | 1.6024 | stalled |
| r128_shapes_mri4/sample_009 | P-Flow | 57 | 51 | 2.4481 | 1.3844 | stalled |
| r128_shapes_mri4/sample_010 | P-Flow | 85 | 74 | 1.6386 | 0.9179 | stalled |
| r128_shapes_mri4/sample_011 | P-Flow | 197 | 178 | 3.9844 | 1.0940 | stalled |
| r128_shapes_mri4/sample_012 | P-Flow | 11 | 9 | 99.8464 | 99.8113 | stalled |
| r128_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 967 | 99.3421 | 98.9728 | stalled |
| r128_shapes_mri4/sample_001 | DLR D-Flow | 30 | 28 | 99.9842 | 99.9507 | stalled |
| r128_shapes_mri4/sample_002 | DLR D-Flow | 59 | 53 | 100.0172 | 98.8679 | stalled |
| r128_shapes_mri4/sample_003 | DLR D-Flow | 410 | 399 | 99.5989 | 99.0272 | stalled |
| r128_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 957 | 73.4243 | 72.1408 | step_limit |
| r128_shapes_mri4/sample_005 | DLR D-Flow | 369 | 350 | 99.2656 | 98.9927 | stalled |
| r128_shapes_mri4/sample_006 | DLR D-Flow | 937 | 907 | 95.3878 | 94.9278 | stalled |
| r128_shapes_mri4/sample_007 | DLR D-Flow | 265 | 250 | 99.3900 | 99.1131 | stalled |
| r128_shapes_mri4/sample_008 | DLR D-Flow | 175 | 166 | 99.7145 | 99.5014 | stalled |
| r128_shapes_mri4/sample_009 | DLR D-Flow | 1000 | 975 | 95.5773 | 93.5662 | step_limit |
| r128_shapes_mri4/sample_010 | DLR D-Flow | 535 | 511 | 8.9647 | 7.1175 | stalled |
| r128_shapes_mri4/sample_011 | DLR D-Flow | 126 | 114 | 99.7342 | 99.3032 | stalled |
| r128_shapes_mri4/sample_012 | DLR D-Flow | 239 | 230 | 99.7459 | 99.6941 | stalled |
| r160_shapes_ct10/sample_000 | D-Flow | 589 | 585 | 99.4865 | 86.3258 | stalled |
| r160_shapes_ct10/sample_001 | D-Flow | 36 | 34 | 100.3363 | 96.0579 | stalled |
| r160_shapes_ct10/sample_002 | D-Flow | 1000 | 1000 | 2.8407 | 0.2784 | step_limit |
| r160_shapes_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.8365 | 0.2901 | step_limit |
| r160_shapes_ct10/sample_004 | D-Flow | 1000 | 1000 | 1.2321 | 0.1314 | step_limit |
| r160_shapes_ct10/sample_005 | D-Flow | 1000 | 1000 | 6.0277 | 1.6917 | step_limit |
| r160_shapes_ct10/sample_006 | D-Flow | 1000 | 1000 | 42.5030 | 26.6042 | step_limit |
| r160_shapes_ct10/sample_007 | D-Flow | 1000 | 1000 | 1.8554 | 0.2544 | step_limit |
| r160_shapes_ct10/sample_008 | D-Flow | 1000 | 1000 | 2.4133 | 0.1006 | step_limit |
| r160_shapes_ct10/sample_009 | D-Flow | 1000 | 1000 | 2.2851 | 0.3535 | step_limit |
| r160_shapes_ct10/sample_010 | D-Flow | 1000 | 1000 | 57.1185 | 40.7699 | step_limit |
| r160_shapes_ct10/sample_011 | D-Flow | 1000 | 1000 | 2.4942 | 0.0462 | step_limit |
| r160_shapes_ct10/sample_012 | D-Flow | 1000 | 1000 | 2.5825 | 0.0685 | step_limit |
| r160_shapes_ct10/sample_000 | P-Flow | 37 | 31 | 57.7655 | 31.1946 | stalled |
| r160_shapes_ct10/sample_001 | P-Flow | 66 | 57 | 2.3904 | 0.5621 | stalled |
| r160_shapes_ct10/sample_002 | P-Flow | 1000 | 995 | 5.0782 | 1.0469 | step_limit |
| r160_shapes_ct10/sample_003 | P-Flow | 1000 | 993 | 2.2621 | 0.5290 | step_limit |
| r160_shapes_ct10/sample_004 | P-Flow | 1000 | 938 | 2.4418 | 0.4192 | step_limit |
| r160_shapes_ct10/sample_005 | P-Flow | 1000 | 965 | 3.1574 | 0.6418 | step_limit |
| r160_shapes_ct10/sample_006 | P-Flow | 1000 | 929 | 3.1185 | 0.6283 | step_limit |
| r160_shapes_ct10/sample_007 | P-Flow | 1000 | 969 | 2.6204 | 0.4146 | step_limit |
| r160_shapes_ct10/sample_008 | P-Flow | 14 | 11 | 71.2411 | 47.2074 | stalled |
| r160_shapes_ct10/sample_009 | P-Flow | 36 | 30 | 45.5208 | 15.4431 | stalled |
| r160_shapes_ct10/sample_010 | P-Flow | 45 | 41 | 2.0468 | 0.4011 | stalled |
| r160_shapes_ct10/sample_011 | P-Flow | 1000 | 995 | 4.5490 | 1.0204 | step_limit |
| r160_shapes_ct10/sample_012 | P-Flow | 22 | 18 | 47.8763 | 21.5539 | stalled |
| r160_shapes_ct10/sample_000 | DLR D-Flow | 46 | 42 | 101.5283 | 95.6274 | stalled |
| r160_shapes_ct10/sample_001 | DLR D-Flow | 16 | 14 | 100.3453 | 96.0641 | stalled |
| r160_shapes_ct10/sample_002 | DLR D-Flow | 116 | 109 | 99.9719 | 96.4349 | stalled |
| r160_shapes_ct10/sample_003 | DLR D-Flow | 77 | 74 | 101.9155 | 97.2759 | stalled |
| r160_shapes_ct10/sample_004 | DLR D-Flow | 445 | 434 | 102.8103 | 79.7956 | stalled |
| r160_shapes_ct10/sample_005 | DLR D-Flow | 255 | 239 | 100.6639 | 93.3724 | stalled |
| r160_shapes_ct10/sample_006 | DLR D-Flow | 1000 | 970 | 100.4869 | 74.6263 | step_limit |
| r160_shapes_ct10/sample_007 | DLR D-Flow | 596 | 571 | 113.9150 | 80.4231 | stalled |
| r160_shapes_ct10/sample_008 | DLR D-Flow | 207 | 197 | 99.7497 | 94.7008 | stalled |
| r160_shapes_ct10/sample_009 | DLR D-Flow | 817 | 790 | 92.7941 | 63.4713 | stalled |
| r160_shapes_ct10/sample_010 | DLR D-Flow | 152 | 145 | 113.9318 | 80.2485 | stalled |
| r160_shapes_ct10/sample_011 | DLR D-Flow | 183 | 172 | 5.0860 | 1.3572 | stalled |
| r160_shapes_ct10/sample_012 | DLR D-Flow | 254 | 241 | 99.8958 | 94.9771 | stalled |
| r160_shapes_mri4/sample_000 | D-Flow | 1000 | 1000 | 99.1598 | 97.9999 | step_limit |
| r160_shapes_mri4/sample_001 | D-Flow | 1000 | 1000 | 1.1755 | 0.2002 | step_limit |
| r160_shapes_mri4/sample_002 | D-Flow | 1000 | 1000 | 1.8984 | 0.3622 | step_limit |
| r160_shapes_mri4/sample_003 | D-Flow | 1000 | 1000 | 1.3502 | 0.4324 | step_limit |
| r160_shapes_mri4/sample_004 | D-Flow | 1000 | 1000 | 1.3295 | 0.7040 | step_limit |
| r160_shapes_mri4/sample_005 | D-Flow | 1000 | 1000 | 3.1107 | 2.0661 | step_limit |
| r160_shapes_mri4/sample_006 | D-Flow | 1000 | 1000 | 1.1115 | 0.4507 | step_limit |
| r160_shapes_mri4/sample_007 | D-Flow | 1000 | 1000 | 1.4716 | 0.7063 | step_limit |
| r160_shapes_mri4/sample_008 | D-Flow | 1000 | 1000 | 1.7556 | 0.1679 | step_limit |
| r160_shapes_mri4/sample_009 | D-Flow | 1000 | 1000 | 77.2119 | 73.8626 | step_limit |
| r160_shapes_mri4/sample_010 | D-Flow | 1000 | 1000 | 1.1275 | 0.5250 | step_limit |
| r160_shapes_mri4/sample_011 | D-Flow | 1000 | 1000 | 1.6794 | 0.0079 | step_limit |
| r160_shapes_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.8429 | 0.5015 | step_limit |
| r160_shapes_mri4/sample_000 | P-Flow | 93 | 85 | 2.2097 | 1.2843 | stalled |
| r160_shapes_mri4/sample_001 | P-Flow | 78 | 69 | 1.9507 | 1.1255 | stalled |
| r160_shapes_mri4/sample_002 | P-Flow | 69 | 59 | 3.0552 | 1.8687 | stalled |
| r160_shapes_mri4/sample_003 | P-Flow | 12 | 10 | 51.8289 | 50.0620 | stalled |
| r160_shapes_mri4/sample_004 | P-Flow | 75 | 64 | 1.6452 | 0.9938 | stalled |
| r160_shapes_mri4/sample_005 | P-Flow | 68 | 60 | 2.1584 | 1.1720 | stalled |
| r160_shapes_mri4/sample_006 | P-Flow | 89 | 77 | 2.2857 | 1.0843 | stalled |
| r160_shapes_mri4/sample_007 | P-Flow | 137 | 119 | 1.8635 | 0.9589 | stalled |
| r160_shapes_mri4/sample_008 | P-Flow | 190 | 166 | 2.8472 | 1.3325 | stalled |
| r160_shapes_mri4/sample_009 | P-Flow | 198 | 167 | 2.0036 | 0.9662 | stalled |
| r160_shapes_mri4/sample_010 | P-Flow | 44 | 37 | 1.6729 | 0.9905 | stalled |
| r160_shapes_mri4/sample_011 | P-Flow | 163 | 134 | 3.2462 | 1.8427 | stalled |
| r160_shapes_mri4/sample_012 | P-Flow | 45 | 38 | 5.1101 | 3.3362 | stalled |
| r160_shapes_mri4/sample_000 | DLR D-Flow | 216 | 205 | 99.6357 | 99.4545 | stalled |
| r160_shapes_mri4/sample_001 | DLR D-Flow | 356 | 342 | 99.4535 | 99.2828 | stalled |
| r160_shapes_mri4/sample_002 | DLR D-Flow | 38 | 32 | 100.2464 | 98.9568 | stalled |
| r160_shapes_mri4/sample_003 | DLR D-Flow | 965 | 936 | 23.7943 | 21.0395 | stalled |
| r160_shapes_mri4/sample_004 | DLR D-Flow | 590 | 571 | 73.1857 | 71.1171 | stalled |
| r160_shapes_mri4/sample_005 | DLR D-Flow | 280 | 272 | 99.2336 | 98.6951 | stalled |
| r160_shapes_mri4/sample_006 | DLR D-Flow | 978 | 941 | 55.1615 | 51.1504 | stalled |
| r160_shapes_mri4/sample_007 | DLR D-Flow | 1000 | 968 | 98.9408 | 97.4926 | step_limit |
| r160_shapes_mri4/sample_008 | DLR D-Flow | 175 | 166 | 99.8887 | 99.3759 | stalled |
| r160_shapes_mri4/sample_009 | DLR D-Flow | 643 | 622 | 99.5040 | 99.2322 | stalled |
| r160_shapes_mri4/sample_010 | DLR D-Flow | 511 | 486 | 52.9996 | 49.8468 | stalled |
| r160_shapes_mri4/sample_011 | DLR D-Flow | 167 | 158 | 4.7756 | 2.6762 | stalled |
| r160_shapes_mri4/sample_012 | DLR D-Flow | 321 | 304 | 99.7518 | 99.5517 | stalled |
| r200_shapes_ct10/sample_000 | D-Flow | 716 | 714 | 53.5683 | 41.9370 | stalled |
| r200_shapes_ct10/sample_001 | D-Flow | 36 | 34 | 100.5058 | 97.2173 | stalled |
| r200_shapes_ct10/sample_002 | D-Flow | 243 | 241 | 101.8077 | 94.0505 | stalled |
| r200_shapes_ct10/sample_003 | D-Flow | 1000 | 1000 | 96.8387 | 86.7634 | step_limit |
| r200_shapes_ct10/sample_004 | D-Flow | 1000 | 1000 | 17.7420 | 5.0585 | step_limit |
| r200_shapes_ct10/sample_005 | D-Flow | 1000 | 1000 | 103.4901 | 83.7613 | step_limit |
| r200_shapes_ct10/sample_006 | D-Flow | 1000 | 1000 | 41.2962 | 18.5860 | step_limit |
| r200_shapes_ct10/sample_007 | D-Flow | 1000 | 1000 | 46.5057 | 28.3850 | step_limit |
| r200_shapes_ct10/sample_008 | D-Flow | 1000 | 1000 | 84.6368 | 60.3903 | step_limit |
| r200_shapes_ct10/sample_009 | D-Flow | 466 | 464 | 62.8638 | 42.0342 | stalled |
| r200_shapes_ct10/sample_010 | D-Flow | 1000 | 1000 | 56.8788 | 40.6429 | step_limit |
| r200_shapes_ct10/sample_011 | D-Flow | 1000 | 1000 | 84.2736 | 61.8719 | step_limit |
| r200_shapes_ct10/sample_012 | D-Flow | 1000 | 1000 | 2.6128 | 0.0478 | step_limit |
| r200_shapes_ct10/sample_000 | P-Flow | 1000 | 989 | 3.3702 | 0.6810 | step_limit |
| r200_shapes_ct10/sample_001 | P-Flow | 48 | 40 | 2.5854 | 0.5054 | stalled |
| r200_shapes_ct10/sample_002 | P-Flow | 20 | 16 | 41.2604 | 17.8488 | stalled |
| r200_shapes_ct10/sample_003 | P-Flow | 35 | 29 | 40.2182 | 19.7047 | stalled |
| r200_shapes_ct10/sample_004 | P-Flow | 1000 | 964 | 1.9059 | 0.2992 | step_limit |
| r200_shapes_ct10/sample_005 | P-Flow | 1000 | 990 | 2.2463 | 0.4477 | step_limit |
| r200_shapes_ct10/sample_006 | P-Flow | 13 | 10 | 57.9722 | 37.8931 | stalled |
| r200_shapes_ct10/sample_007 | P-Flow | 45 | 39 | 53.1287 | 25.8592 | stalled |
| r200_shapes_ct10/sample_008 | P-Flow | 530 | 414 | 4.2982 | 0.9094 | stalled |
| r200_shapes_ct10/sample_009 | P-Flow | 90 | 76 | 37.3222 | 12.4387 | stalled |
| r200_shapes_ct10/sample_010 | P-Flow | 68 | 61 | 1.6765 | 0.3007 | stalled |
| r200_shapes_ct10/sample_011 | P-Flow | 19 | 16 | 53.0181 | 29.7190 | stalled |
| r200_shapes_ct10/sample_012 | P-Flow | 24 | 20 | 18.3867 | 6.8763 | stalled |
| r200_shapes_ct10/sample_000 | DLR D-Flow | 755 | 734 | 92.2966 | 62.9338 | stalled |
| r200_shapes_ct10/sample_001 | DLR D-Flow | 396 | 372 | 100.3014 | 96.6735 | stalled |
| r200_shapes_ct10/sample_002 | DLR D-Flow | 190 | 182 | 101.2903 | 95.7179 | stalled |
| r200_shapes_ct10/sample_003 | DLR D-Flow | 271 | 260 | 100.4612 | 95.8054 | stalled |
| r200_shapes_ct10/sample_004 | DLR D-Flow | 645 | 623 | 31.6637 | 18.5434 | stalled |
| r200_shapes_ct10/sample_005 | DLR D-Flow | 314 | 300 | 106.3546 | 94.8069 | stalled |
| r200_shapes_ct10/sample_006 | DLR D-Flow | 421 | 403 | 56.0231 | 36.4776 | stalled |
| r200_shapes_ct10/sample_007 | DLR D-Flow | 1000 | 971 | 57.1794 | 30.2956 | step_limit |
| r200_shapes_ct10/sample_008 | DLR D-Flow | 109 | 102 | 100.0128 | 95.7655 | stalled |
| r200_shapes_ct10/sample_009 | DLR D-Flow | 142 | 135 | 108.2739 | 85.3549 | stalled |
| r200_shapes_ct10/sample_010 | DLR D-Flow | 160 | 150 | 74.5747 | 47.8559 | stalled |
| r200_shapes_ct10/sample_011 | DLR D-Flow | 384 | 359 | 99.8784 | 95.8817 | stalled |
| r200_shapes_ct10/sample_012 | DLR D-Flow | 1000 | 972 | 53.5374 | 26.0175 | step_limit |
| r200_shapes_mri4/sample_000 | D-Flow | 1000 | 1000 | 38.4270 | 33.8271 | step_limit |
| r200_shapes_mri4/sample_001 | D-Flow | 1000 | 1000 | 99.4709 | 99.2205 | step_limit |
| r200_shapes_mri4/sample_002 | D-Flow | 1000 | 1000 | 90.7925 | 86.4389 | step_limit |
| r200_shapes_mri4/sample_003 | D-Flow | 1000 | 1000 | 78.6161 | 76.2994 | step_limit |
| r200_shapes_mri4/sample_004 | D-Flow | 1000 | 1000 | 72.9079 | 71.7013 | step_limit |
| r200_shapes_mri4/sample_005 | D-Flow | 1000 | 1000 | 98.4920 | 98.1643 | step_limit |
| r200_shapes_mri4/sample_006 | D-Flow | 808 | 806 | 98.2433 | 97.6436 | stalled |
| r200_shapes_mri4/sample_007 | D-Flow | 1000 | 999 | 97.5621 | 96.6759 | step_limit |
| r200_shapes_mri4/sample_008 | D-Flow | 1000 | 1000 | 7.9398 | 4.3188 | step_limit |
| r200_shapes_mri4/sample_009 | D-Flow | 1000 | 1000 | 82.7413 | 81.7194 | step_limit |
| r200_shapes_mri4/sample_010 | D-Flow | 1000 | 1000 | 50.1214 | 47.3435 | step_limit |
| r200_shapes_mri4/sample_011 | D-Flow | 1000 | 1000 | 18.5461 | 13.4650 | step_limit |
| r200_shapes_mri4/sample_012 | D-Flow | 1000 | 1000 | 1.7016 | 0.3424 | step_limit |
| r200_shapes_mri4/sample_000 | P-Flow | 102 | 90 | 2.7617 | 1.3345 | stalled |
| r200_shapes_mri4/sample_001 | P-Flow | 81 | 64 | 1.9938 | 1.1407 | stalled |
| r200_shapes_mri4/sample_002 | P-Flow | 75 | 64 | 2.8127 | 1.4941 | stalled |
| r200_shapes_mri4/sample_003 | P-Flow | 39 | 34 | 2.3967 | 1.3441 | stalled |
| r200_shapes_mri4/sample_004 | P-Flow | 146 | 125 | 1.4419 | 0.7717 | stalled |
| r200_shapes_mri4/sample_005 | P-Flow | 83 | 72 | 1.6762 | 0.8684 | stalled |
| r200_shapes_mri4/sample_006 | P-Flow | 93 | 81 | 1.6246 | 0.8511 | stalled |
| r200_shapes_mri4/sample_007 | P-Flow | 104 | 92 | 1.4922 | 0.7502 | stalled |
| r200_shapes_mri4/sample_008 | P-Flow | 67 | 57 | 3.3273 | 1.8448 | stalled |
| r200_shapes_mri4/sample_009 | P-Flow | 82 | 75 | 1.7877 | 0.9517 | stalled |
| r200_shapes_mri4/sample_010 | P-Flow | 63 | 55 | 1.3006 | 0.7832 | stalled |
| r200_shapes_mri4/sample_011 | P-Flow | 77 | 64 | 3.5664 | 1.8178 | stalled |
| r200_shapes_mri4/sample_012 | P-Flow | 115 | 96 | 2.8725 | 1.5568 | stalled |
| r200_shapes_mri4/sample_000 | DLR D-Flow | 149 | 143 | 99.6888 | 99.6233 | stalled |
| r200_shapes_mri4/sample_001 | DLR D-Flow | 1000 | 943 | 99.6028 | 99.5119 | step_limit |
| r200_shapes_mri4/sample_002 | DLR D-Flow | 1000 | 947 | 97.9975 | 96.5082 | step_limit |
| r200_shapes_mri4/sample_003 | DLR D-Flow | 120 | 115 | 99.6854 | 99.6066 | stalled |
| r200_shapes_mri4/sample_004 | DLR D-Flow | 38 | 34 | 99.5719 | 99.4339 | stalled |
| r200_shapes_mri4/sample_005 | DLR D-Flow | 1000 | 952 | 99.5883 | 99.5068 | step_limit |
| r200_shapes_mri4/sample_006 | DLR D-Flow | 448 | 424 | 99.4682 | 99.3720 | stalled |
| r200_shapes_mri4/sample_007 | DLR D-Flow | 423 | 405 | 99.4767 | 99.3570 | stalled |
| r200_shapes_mri4/sample_008 | DLR D-Flow | 1000 | 953 | 99.7435 | 99.5115 | step_limit |
| r200_shapes_mri4/sample_009 | DLR D-Flow | 581 | 547 | 99.2955 | 99.0193 | stalled |
| r200_shapes_mri4/sample_010 | DLR D-Flow | 1000 | 971 | 59.7829 | 57.2606 | step_limit |
| r200_shapes_mri4/sample_011 | DLR D-Flow | 789 | 751 | 99.7313 | 99.6179 | stalled |
| r200_shapes_mri4/sample_012 | DLR D-Flow | 1000 | 961 | 56.6723 | 50.3212 | step_limit |
| r250_shapes_ct10/sample_000 | D-Flow | 1000 | 1000 | 76.3238 | 63.7405 | step_limit |
| r250_shapes_ct10/sample_001 | D-Flow | 572 | 570 | 99.5672 | 94.2664 | stalled |
| r250_shapes_ct10/sample_002 | D-Flow | 1000 | 1000 | 102.3439 | 88.7009 | step_limit |
| r250_shapes_ct10/sample_003 | D-Flow | 1000 | 1000 | 1.7090 | 0.2058 | step_limit |
| r250_shapes_ct10/sample_004 | D-Flow | 1000 | 1000 | 12.3072 | 2.9227 | step_limit |
| r250_shapes_ct10/sample_005 | D-Flow | 1000 | 1000 | 72.6982 | 57.2873 | step_limit |
| r250_shapes_ct10/sample_006 | D-Flow | 1000 | 1000 | 93.7081 | 83.2036 | step_limit |
| r250_shapes_ct10/sample_007 | D-Flow | 1000 | 1000 | 34.3141 | 19.9262 | step_limit |
| r250_shapes_ct10/sample_008 | D-Flow | 1000 | 1000 | 95.2118 | 79.7735 | step_limit |
| r250_shapes_ct10/sample_009 | D-Flow | 1000 | 1000 | 37.7245 | 12.7104 | step_limit |
| r250_shapes_ct10/sample_000 | P-Flow | 24 | 20 | 40.6426 | 21.3881 | stalled |
| r250_shapes_ct10/sample_001 | P-Flow | 1000 | 997 | 2.1077 | 0.3872 | step_limit |
| r250_shapes_ct10/sample_002 | P-Flow | 24 | 20 | 64.0688 | 35.7753 | stalled |
| r250_shapes_ct10/sample_003 | P-Flow | 803 | 636 | 2.1990 | 0.4864 | stalled |
| r250_shapes_ct10/sample_004 | P-Flow | 1000 | 986 | 1.6080 | 0.2256 | step_limit |
| r250_shapes_ct10/sample_005 | P-Flow | 160 | 140 | 3.5862 | 0.7153 | stalled |
| r250_shapes_ct10/sample_006 | P-Flow | 965 | 955 | 2.2786 | 0.3618 | stalled |
| r250_shapes_ct10/sample_007 | P-Flow | 1000 | 970 | 2.0214 | 0.3135 | step_limit |
| r250_shapes_ct10/sample_008 | P-Flow | 59 | 48 | 65.1575 | 39.7203 | stalled |
| r250_shapes_ct10/sample_009 | P-Flow | 61 | 53 | 41.6189 | 15.0574 | stalled |
| r250_shapes_ct10/sample_010 | P-Flow | 50 | 42 | 1.8544 | 0.6494 | paused |
| r250_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 967 | 82.8371 | 65.9608 | step_limit |
| r250_shapes_ct10/sample_001 | DLR D-Flow | 45 | 41 | 99.9850 | 97.1080 | stalled |
| r250_shapes_ct10/sample_002 | DLR D-Flow | 482 | 464 | 100.0471 | 94.4883 | stalled |
| r250_shapes_ct10/sample_003 | DLR D-Flow | 82 | 78 | 111.5932 | 91.2059 | stalled |
| r250_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 994 | 46.6633 | 23.3825 | step_limit |
| r250_shapes_ct10/sample_005 | DLR D-Flow | 332 | 323 | 106.9358 | 87.2989 | stalled |
| r250_shapes_ct10/sample_006 | DLR D-Flow | 328 | 307 | 101.3384 | 97.0292 | stalled |
| r250_shapes_ct10/sample_007 | DLR D-Flow | 99 | 94 | 96.3235 | 75.5623 | stalled |
| r250_shapes_ct10/sample_008 | DLR D-Flow | 321 | 301 | 99.7639 | 95.4824 | stalled |
| r250_shapes_ct10/sample_009 | DLR D-Flow | 52 | 45 | 110.3665 | 88.0114 | stalled |
| r250_shapes_mri4/sample_000 | D-Flow | 1000 | 1000 | 98.4597 | 98.0001 | step_limit |
| r250_shapes_mri4/sample_001 | D-Flow | 1000 | 1000 | 79.2150 | 77.3991 | step_limit |
| r250_shapes_mri4/sample_002 | D-Flow | 1000 | 1000 | 40.1550 | 33.5536 | step_limit |
| r250_shapes_mri4/sample_003 | D-Flow | 1000 | 1000 | 96.2802 | 94.9133 | step_limit |
| r250_shapes_mri4/sample_004 | D-Flow | 1000 | 1000 | 16.0625 | 12.8434 | step_limit |
| r250_shapes_mri4/sample_005 | D-Flow | 1000 | 1000 | 59.5944 | 58.4504 | step_limit |
| r250_shapes_mri4/sample_006 | D-Flow | 239 | 235 | 99.1952 | 98.6899 | stalled |
| r250_shapes_mri4/sample_007 | D-Flow | 657 | 654 | 82.3403 | 82.3582 | stalled |
| r250_shapes_mri4/sample_008 | D-Flow | 1000 | 1000 | 52.6445 | 34.8119 | step_limit |
| r250_shapes_mri4/sample_009 | D-Flow | 1000 | 1000 | 83.0145 | 82.6802 | step_limit |
| r250_shapes_mri4/sample_000 | P-Flow | 128 | 114 | 1.8316 | 0.9310 | stalled |
| r250_shapes_mri4/sample_001 | P-Flow | 70 | 60 | 1.7769 | 0.9460 | stalled |
| r250_shapes_mri4/sample_002 | P-Flow | 149 | 126 | 2.7504 | 1.5126 | stalled |
| r250_shapes_mri4/sample_003 | P-Flow | 82 | 72 | 1.7806 | 1.0029 | stalled |
| r250_shapes_mri4/sample_004 | P-Flow | 258 | 205 | 1.2400 | 0.6422 | stalled |
| r250_shapes_mri4/sample_005 | P-Flow | 125 | 105 | 1.6194 | 0.8462 | stalled |
| r250_shapes_mri4/sample_006 | P-Flow | 81 | 68 | 1.7407 | 0.9467 | stalled |
| r250_shapes_mri4/sample_007 | P-Flow | 86 | 72 | 1.3677 | 0.7075 | stalled |
| r250_shapes_mri4/sample_008 | P-Flow | 130 | 104 | 2.7085 | 1.4433 | stalled |
| r250_shapes_mri4/sample_009 | P-Flow | 104 | 88 | 1.6456 | 0.8012 | stalled |
| r250_shapes_mri4/sample_010 | P-Flow | 50 | 44 | 1.3481 | 0.8637 | paused |
| r250_shapes_mri4/sample_000 | DLR D-Flow | 202 | 188 | 99.2466 | 99.0341 | stalled |
| r250_shapes_mri4/sample_001 | DLR D-Flow | 660 | 627 | 99.4823 | 99.4310 | stalled |
| r250_shapes_mri4/sample_002 | DLR D-Flow | 523 | 504 | 99.6921 | 99.5981 | stalled |
| r250_shapes_mri4/sample_003 | DLR D-Flow | 10 | 8 | 99.6892 | 99.6293 | stalled |
| r250_shapes_mri4/sample_004 | DLR D-Flow | 137 | 130 | 88.5921 | 88.4106 | stalled |
| r250_shapes_mri4/sample_005 | DLR D-Flow | 1000 | 960 | 72.2414 | 71.8032 | step_limit |
| r250_shapes_mri4/sample_006 | DLR D-Flow | 22 | 18 | 99.6381 | 99.3843 | stalled |
| r250_shapes_mri4/sample_007 | DLR D-Flow | 866 | 830 | 82.9834 | 83.3521 | stalled |
| r250_shapes_mri4/sample_008 | DLR D-Flow | 289 | 275 | 99.7312 | 99.6468 | stalled |
| r250_shapes_mri4/sample_009 | DLR D-Flow | 568 | 544 | 98.7833 | 98.5144 | stalled |

Terminal method runs: 426/1080. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.
