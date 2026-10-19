# Five-step Euler pilot

0% relative measurement noise; CT10 and Cartesian MRI4; 30 test samples per acquisition at each resolution; separated shapes: 1–4 circles/triangles. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank 16, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |
| r64_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 966 | 29.0171 | 11.2666 | step_limit |
| r64_shapes_ct10/sample_001 | DLR D-Flow | 128 | 123 | 102.4328 | 97.3477 | stalled |
| r64_shapes_ct10/sample_002 | DLR D-Flow | 739 | 720 | 101.1973 | 95.0360 | stalled |
| r64_shapes_ct10/sample_003 | DLR D-Flow | 762 | 738 | 102.2819 | 94.2613 | stalled |
| r64_shapes_ct10/sample_004 | DLR D-Flow | 432 | 420 | 13.5674 | 3.8932 | stalled |
| r64_shapes_ct10/sample_005 | DLR D-Flow | 1000 | 974 | 109.1463 | 86.1091 | step_limit |
| r64_shapes_ct10/sample_006 | DLR D-Flow | 174 | 169 | 98.1950 | 92.1287 | stalled |
| r64_shapes_ct10/sample_007 | DLR D-Flow | 1000 | 981 | 78.2284 | 50.7964 | step_limit |
| r64_shapes_ct10/sample_008 | DLR D-Flow | 1000 | 970 | 100.5970 | 94.9120 | step_limit |
| r64_shapes_ct10/sample_009 | DLR D-Flow | 1000 | 962 | 61.6309 | 43.2010 | step_limit |
| r64_shapes_ct10/sample_010 | DLR D-Flow | 1000 | 965 | 64.6097 | 52.2606 | step_limit |
| r64_shapes_ct10/sample_011 | DLR D-Flow | 79 | 75 | 100.7917 | 98.3951 | stalled |
| r64_shapes_ct10/sample_012 | DLR D-Flow | 1000 | 940 | 101.4205 | 95.6194 | step_limit |
| r64_shapes_ct10/sample_013 | DLR D-Flow | 1000 | 980 | 77.8995 | 70.0944 | step_limit |
| r64_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 976 | 50.6677 | 48.9173 | step_limit |
| r64_shapes_mri4/sample_001 | DLR D-Flow | 384 | 375 | 99.7801 | 99.4836 | stalled |
| r64_shapes_mri4/sample_002 | DLR D-Flow | 187 | 176 | 101.8063 | 98.0203 | stalled |
| r64_shapes_mri4/sample_003 | DLR D-Flow | 794 | 763 | 100.9376 | 98.6394 | stalled |
| r64_shapes_mri4/sample_004 | DLR D-Flow | 438 | 426 | 6.1493 | 3.5943 | stalled |
| r64_shapes_mri4/sample_005 | DLR D-Flow | 1000 | 973 | 72.2726 | 71.3955 | step_limit |
| r64_shapes_mri4/sample_006 | DLR D-Flow | 640 | 622 | 100.3416 | 98.7632 | stalled |
| r64_shapes_mri4/sample_007 | DLR D-Flow | 699 | 677 | 100.9467 | 98.7125 | stalled |
| r64_shapes_mri4/sample_008 | DLR D-Flow | 1000 | 971 | 99.6925 | 99.5109 | step_limit |
| r64_shapes_mri4/sample_009 | DLR D-Flow | 1000 | 973 | 59.5852 | 57.3224 | step_limit |
| r64_shapes_mri4/sample_010 | DLR D-Flow | 165 | 156 | 61.3642 | 59.4362 | stalled |
| r64_shapes_mri4/sample_011 | DLR D-Flow | 990 | 964 | 99.7484 | 99.5942 | stalled |
| r64_shapes_mri4/sample_012 | DLR D-Flow | 759 | 740 | 99.7398 | 99.4387 | stalled |
| r64_shapes_mri4/sample_013 | DLR D-Flow | 348 | 333 | 73.1518 | 75.2563 | stalled |
| r96_shapes_ct10/sample_000 | DLR D-Flow | 1000 | 983 | 83.3446 | 72.7747 | step_limit |
| r96_shapes_ct10/sample_001 | DLR D-Flow | 192 | 185 | 102.3263 | 95.1127 | stalled |
| r96_shapes_ct10/sample_002 | DLR D-Flow | 137 | 131 | 100.4389 | 93.9618 | stalled |
| r96_shapes_ct10/sample_003 | DLR D-Flow | 711 | 696 | 104.8339 | 91.2545 | stalled |
| r96_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 980 | 79.1190 | 55.4744 | step_limit |
| r96_shapes_ct10/sample_005 | DLR D-Flow | 793 | 755 | 102.4457 | 94.4974 | stalled |
| r96_shapes_ct10/sample_006 | DLR D-Flow | 1000 | 978 | 105.2497 | 91.9888 | step_limit |
| r96_shapes_ct10/sample_007 | DLR D-Flow | 1000 | 981 | 70.6472 | 42.5369 | step_limit |
| r96_shapes_ct10/sample_008 | DLR D-Flow | 51 | 48 | 100.6517 | 93.7321 | stalled |
| r96_shapes_ct10/sample_009 | DLR D-Flow | 1000 | 965 | 95.8252 | 81.2676 | step_limit |
| r96_shapes_ct10/sample_010 | DLR D-Flow | 1000 | 963 | 57.8637 | 41.2741 | step_limit |
| r96_shapes_ct10/sample_011 | DLR D-Flow | 51 | 48 | 100.0167 | 95.0984 | stalled |
| r96_shapes_ct10/sample_012 | DLR D-Flow | 283 | 270 | 99.8453 | 95.0597 | stalled |
| r96_shapes_ct10/sample_013 | DLR D-Flow | 1000 | 968 | 42.2653 | 29.6640 | step_limit |
| r96_shapes_mri4/sample_000 | DLR D-Flow | 68 | 63 | 99.8342 | 99.0357 | stalled |
| r96_shapes_mri4/sample_001 | DLR D-Flow | 302 | 295 | 99.8498 | 99.0519 | stalled |
| r96_shapes_mri4/sample_002 | DLR D-Flow | 289 | 269 | 99.9453 | 99.0748 | stalled |
| r96_shapes_mri4/sample_003 | DLR D-Flow | 360 | 350 | 99.5807 | 99.2498 | stalled |
| r96_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 972 | 70.6358 | 68.2128 | step_limit |
| r96_shapes_mri4/sample_005 | DLR D-Flow | 1000 | 973 | 50.5900 | 49.4231 | step_limit |
| r96_shapes_mri4/sample_006 | DLR D-Flow | 1000 | 968 | 100.4879 | 97.6184 | step_limit |
| r96_shapes_mri4/sample_007 | DLR D-Flow | 940 | 916 | 70.6315 | 68.3072 | stalled |
| r96_shapes_mri4/sample_008 | DLR D-Flow | 269 | 258 | 4.3639 | 2.3678 | stalled |
| r96_shapes_mri4/sample_009 | DLR D-Flow | 1000 | 978 | 81.1261 | 80.4267 | step_limit |
| r96_shapes_mri4/sample_010 | DLR D-Flow | 1000 | 976 | 80.2187 | 78.6886 | step_limit |
| r96_shapes_mri4/sample_011 | DLR D-Flow | 179 | 167 | 99.7968 | 99.4748 | stalled |
| r96_shapes_mri4/sample_012 | DLR D-Flow | 205 | 200 | 100.0598 | 99.3473 | stalled |
| r96_shapes_mri4/sample_013 | DLR D-Flow | 1000 | 980 | 85.4390 | 82.3359 | step_limit |
| r128_shapes_ct10/sample_000 | DLR D-Flow | 209 | 197 | 100.0027 | 91.0735 | stalled |
| r128_shapes_ct10/sample_001 | DLR D-Flow | 39 | 36 | 99.9955 | 99.6344 | stalled |
| r128_shapes_ct10/sample_002 | DLR D-Flow | 1000 | 952 | 105.7076 | 89.5804 | step_limit |
| r128_shapes_ct10/sample_003 | DLR D-Flow | 1000 | 966 | 5.4322 | 2.7745 | step_limit |
| r128_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 971 | 87.3557 | 62.2395 | step_limit |
| r128_shapes_ct10/sample_005 | DLR D-Flow | 579 | 564 | 101.4933 | 91.7651 | stalled |
| r128_shapes_ct10/sample_006 | DLR D-Flow | 1000 | 980 | 41.8225 | 26.2431 | step_limit |
| r128_shapes_ct10/sample_007 | DLR D-Flow | 1000 | 975 | 83.4671 | 52.2901 | step_limit |
| r128_shapes_ct10/sample_008 | DLR D-Flow | 393 | 371 | 102.8504 | 95.6840 | stalled |
| r128_shapes_ct10/sample_009 | DLR D-Flow | 1000 | 965 | 108.8773 | 92.4824 | step_limit |
| r128_shapes_ct10/sample_010 | DLR D-Flow | 1000 | 975 | 1.6889 | 0.3239 | step_limit |
| r128_shapes_ct10/sample_011 | DLR D-Flow | 1000 | 987 | 4.2327 | 1.1738 | step_limit |
| r128_shapes_ct10/sample_012 | DLR D-Flow | 58 | 52 | 100.7239 | 95.6578 | stalled |
| r128_shapes_ct10/sample_013 | DLR D-Flow | 229 | 223 | 100.1988 | 94.7052 | stalled |
| r128_shapes_mri4/sample_000 | DLR D-Flow | 523 | 511 | 99.1408 | 98.6403 | stalled |
| r128_shapes_mri4/sample_001 | DLR D-Flow | 63 | 59 | 99.8962 | 99.1836 | stalled |
| r128_shapes_mri4/sample_002 | DLR D-Flow | 1000 | 959 | 99.8556 | 97.6635 | step_limit |
| r128_shapes_mri4/sample_003 | DLR D-Flow | 160 | 150 | 99.6275 | 99.0238 | stalled |
| r128_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 964 | 73.4837 | 72.2634 | step_limit |
| r128_shapes_mri4/sample_005 | DLR D-Flow | 803 | 782 | 81.0286 | 80.6810 | stalled |
| r128_shapes_mri4/sample_006 | DLR D-Flow | 1000 | 984 | 92.2767 | 92.4359 | step_limit |
| r128_shapes_mri4/sample_007 | DLR D-Flow | 188 | 175 | 99.6666 | 99.3037 | stalled |
| r128_shapes_mri4/sample_008 | DLR D-Flow | 920 | 872 | 100.2947 | 99.0674 | stalled |
| r128_shapes_mri4/sample_009 | DLR D-Flow | 257 | 248 | 99.8709 | 99.8019 | stalled |
| r128_shapes_mri4/sample_010 | DLR D-Flow | 1000 | 988 | 81.4976 | 80.5483 | step_limit |
| r128_shapes_mri4/sample_011 | DLR D-Flow | 38 | 34 | 99.6989 | 99.2513 | stalled |
| r128_shapes_mri4/sample_012 | DLR D-Flow | 219 | 211 | 99.7668 | 99.6585 | stalled |
| r128_shapes_mri4/sample_013 | DLR D-Flow | 293 | 280 | 100.0453 | 99.0929 | stalled |
| r160_shapes_ct10/sample_000 | DLR D-Flow | 125 | 118 | 101.4907 | 94.5393 | stalled |
| r160_shapes_ct10/sample_001 | DLR D-Flow | 16 | 14 | 100.3390 | 96.0586 | stalled |
| r160_shapes_ct10/sample_002 | DLR D-Flow | 467 | 444 | 100.0423 | 94.2351 | stalled |
| r160_shapes_ct10/sample_003 | DLR D-Flow | 155 | 149 | 100.9294 | 94.6700 | stalled |
| r160_shapes_ct10/sample_004 | DLR D-Flow | 1000 | 979 | 43.2094 | 20.3054 | step_limit |
| r160_shapes_ct10/sample_005 | DLR D-Flow | 182 | 177 | 102.1733 | 90.2318 | stalled |
| r160_shapes_ct10/sample_006 | DLR D-Flow | 436 | 418 | 101.2152 | 93.9748 | stalled |
| r160_shapes_ct10/sample_007 | DLR D-Flow | 1000 | 975 | 82.3019 | 48.2876 | step_limit |
| r160_shapes_ct10/sample_008 | DLR D-Flow | 121 | 115 | 99.7934 | 94.6864 | stalled |
| r160_shapes_ct10/sample_009 | DLR D-Flow | 28 | 24 | 104.9624 | 94.8301 | stalled |
| r160_shapes_ct10/sample_010 | DLR D-Flow | 721 | 697 | 58.6510 | 41.2685 | stalled |
| r160_shapes_ct10/sample_011 | DLR D-Flow | 575 | 537 | 4.6317 | 0.9986 | stalled |
| r160_shapes_ct10/sample_012 | DLR D-Flow | 467 | 447 | 99.8735 | 94.9649 | stalled |
| r160_shapes_ct10/sample_013 | DLR D-Flow | 189 | 181 | 103.5583 | 94.4224 | stalled |
| r160_shapes_mri4/sample_000 | DLR D-Flow | 328 | 320 | 99.0664 | 98.4950 | stalled |
| r160_shapes_mri4/sample_001 | DLR D-Flow | 220 | 212 | 1.9835 | 1.1448 | stalled |
| r160_shapes_mri4/sample_002 | DLR D-Flow | 68 | 64 | 100.1736 | 98.9157 | stalled |
| r160_shapes_mri4/sample_003 | DLR D-Flow | 171 | 166 | 99.1298 | 98.6355 | stalled |
| r160_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 974 | 73.3685 | 71.2903 | step_limit |
| r160_shapes_mri4/sample_005 | DLR D-Flow | 294 | 286 | 99.1147 | 98.5575 | stalled |
| r160_shapes_mri4/sample_006 | DLR D-Flow | 1000 | 959 | 89.7920 | 86.1674 | step_limit |
| r160_shapes_mri4/sample_007 | DLR D-Flow | 533 | 519 | 70.5607 | 69.5797 | stalled |
| r160_shapes_mri4/sample_008 | DLR D-Flow | 275 | 265 | 99.8960 | 99.3667 | stalled |
| r160_shapes_mri4/sample_009 | DLR D-Flow | 155 | 146 | 99.3187 | 98.9868 | stalled |
| r160_shapes_mri4/sample_010 | DLR D-Flow | 1000 | 975 | 53.0121 | 49.7889 | step_limit |
| r160_shapes_mri4/sample_011 | DLR D-Flow | 69 | 62 | 5.0161 | 2.8889 | stalled |
| r160_shapes_mri4/sample_012 | DLR D-Flow | 149 | 139 | 99.8016 | 99.5517 | stalled |
| r160_shapes_mri4/sample_013 | DLR D-Flow | 119 | 115 | 99.5292 | 99.3791 | paused |
| r200_shapes_ct10/sample_000 | DLR D-Flow | 137 | 128 | 103.1401 | 84.0985 | stalled |
| r200_shapes_ct10/sample_001 | DLR D-Flow | 190 | 182 | 100.0095 | 95.5536 | stalled |
| r200_shapes_ct10/sample_002 | DLR D-Flow | 79 | 71 | 101.2412 | 95.5142 | stalled |
| r200_shapes_ct10/sample_003 | DLR D-Flow | 666 | 631 | 100.3815 | 95.0178 | stalled |
| r200_shapes_ct10/sample_004 | DLR D-Flow | 183 | 176 | 108.0088 | 77.3944 | stalled |
| r200_shapes_ct10/sample_005 | DLR D-Flow | 82 | 77 | 103.2754 | 93.6140 | stalled |
| r200_shapes_ct10/sample_006 | DLR D-Flow | 565 | 546 | 116.7701 | 88.6766 | stalled |
| r200_shapes_ct10/sample_007 | DLR D-Flow | 256 | 249 | 73.8626 | 37.8109 | stalled |
| r200_shapes_ct10/sample_008 | DLR D-Flow | 1000 | 951 | 99.9111 | 94.6564 | step_limit |
| r200_shapes_ct10/sample_009 | DLR D-Flow | 654 | 639 | 93.8593 | 63.5322 | stalled |
| r200_shapes_ct10/sample_010 | DLR D-Flow | 1000 | 978 | 71.7725 | 43.8149 | step_limit |
| r200_shapes_ct10/sample_011 | DLR D-Flow | 627 | 593 | 99.8476 | 96.0298 | stalled |
| r200_shapes_ct10/sample_012 | DLR D-Flow | 1000 | 967 | 5.4412 | 1.3996 | step_limit |
| r200_shapes_mri4/sample_000 | DLR D-Flow | 1000 | 957 | 50.8727 | 50.0261 | step_limit |
| r200_shapes_mri4/sample_001 | DLR D-Flow | 1000 | 955 | 99.4605 | 99.3078 | step_limit |
| r200_shapes_mri4/sample_002 | DLR D-Flow | 1000 | 971 | 3.1782 | 1.6865 | step_limit |
| r200_shapes_mri4/sample_003 | DLR D-Flow | 1000 | 950 | 99.6036 | 99.4710 | step_limit |
| r200_shapes_mri4/sample_004 | DLR D-Flow | 622 | 595 | 99.4481 | 99.2412 | stalled |
| r200_shapes_mri4/sample_005 | DLR D-Flow | 227 | 215 | 99.5697 | 99.4885 | stalled |
| r200_shapes_mri4/sample_006 | DLR D-Flow | 215 | 203 | 99.2670 | 99.1324 | stalled |
| r200_shapes_mri4/sample_007 | DLR D-Flow | 172 | 164 | 99.2899 | 99.1258 | stalled |
| r200_shapes_mri4/sample_008 | DLR D-Flow | 1000 | 953 | 99.6960 | 99.4687 | step_limit |
| r200_shapes_mri4/sample_009 | DLR D-Flow | 103 | 96 | 99.6095 | 99.4585 | stalled |
| r200_shapes_mri4/sample_010 | DLR D-Flow | 710 | 696 | 80.4478 | 76.7210 | stalled |
| r200_shapes_mri4/sample_011 | DLR D-Flow | 480 | 458 | 99.7341 | 99.6229 | stalled |
| r200_shapes_mri4/sample_012 | DLR D-Flow | 1000 | 953 | 4.8589 | 2.8511 | step_limit |
| r250_shapes_ct10/sample_000 | DLR D-Flow | 410 | 397 | 99.2497 | 77.7917 | stalled |
| r250_shapes_ct10/sample_001 | DLR D-Flow | 100 | 93 | 99.8598 | 96.4948 | stalled |
| r250_shapes_ct10/sample_002 | DLR D-Flow | 341 | 324 | 100.2270 | 96.6833 | stalled |
| r250_shapes_ct10/sample_003 | DLR D-Flow | 24 | 20 | 101.6151 | 95.9906 | stalled |
| r250_shapes_ct10/sample_004 | DLR D-Flow | 367 | 357 | 32.3162 | 18.6269 | stalled |
| r250_shapes_ct10/sample_005 | DLR D-Flow | 112 | 103 | 107.0333 | 91.1981 | stalled |
| r250_shapes_ct10/sample_006 | DLR D-Flow | 682 | 664 | 101.5877 | 90.1545 | stalled |
| r250_shapes_ct10/sample_007 | DLR D-Flow | 461 | 439 | 98.6354 | 70.4236 | stalled |
| r250_shapes_ct10/sample_008 | DLR D-Flow | 610 | 589 | 99.9217 | 94.6531 | stalled |
| r250_shapes_ct10/sample_009 | DLR D-Flow | 40 | 36 | 107.9239 | 85.8235 | stalled |
| r250_shapes_ct10/sample_010 | DLR D-Flow | 1000 | 986 | 58.4736 | 41.1747 | step_limit |
| r250_shapes_ct10/sample_011 | DLR D-Flow | 389 | 373 | 100.4983 | 95.0688 | stalled |
| r250_shapes_ct10/sample_012 | DLR D-Flow | 28 | 24 | 100.6414 | 95.2657 | stalled |
| r250_shapes_mri4/sample_000 | DLR D-Flow | 109 | 102 | 99.2712 | 99.0645 | stalled |
| r250_shapes_mri4/sample_001 | DLR D-Flow | 306 | 292 | 99.4668 | 99.3807 | stalled |
| r250_shapes_mri4/sample_002 | DLR D-Flow | 462 | 434 | 99.6650 | 99.5319 | stalled |
| r250_shapes_mri4/sample_003 | DLR D-Flow | 127 | 120 | 99.5612 | 99.4605 | stalled |
| r250_shapes_mri4/sample_004 | DLR D-Flow | 1000 | 961 | 66.3410 | 66.4395 | step_limit |
| r250_shapes_mri4/sample_005 | DLR D-Flow | 342 | 331 | 72.1002 | 71.7024 | stalled |
| r250_shapes_mri4/sample_006 | DLR D-Flow | 799 | 773 | 98.8427 | 98.4005 | stalled |
| r250_shapes_mri4/sample_007 | DLR D-Flow | 836 | 808 | 45.2494 | 43.4168 | stalled |
| r250_shapes_mri4/sample_008 | DLR D-Flow | 149 | 140 | 99.7191 | 99.6239 | stalled |
| r250_shapes_mri4/sample_009 | DLR D-Flow | 1000 | 964 | 99.0102 | 98.8530 | step_limit |
| r250_shapes_mri4/sample_010 | DLR D-Flow | 1000 | 948 | 78.5859 | 74.9463 | step_limit |
| r250_shapes_mri4/sample_011 | DLR D-Flow | 389 | 375 | 99.8332 | 99.6290 | stalled |
| r250_shapes_mri4/sample_012 | DLR D-Flow | 354 | 339 | 99.7123 | 99.5970 | stalled |

Terminal method runs: 163/360. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including images and clean-reference measurements.

Original D/P controls and rank-8 results: /home/samzheng/d-flow/outputs/separated_shapes_noiseless_20261016. Completed rank pairs: 124.

| Paired case | Rank 8 image error % | Rank 16 image error % |
| --- | ---: | ---: |
| r64_shapes_ct10/sample_000 | 78.8474 | 29.0171 |
| r64_shapes_ct10/sample_001 | 102.8367 | 102.4328 |
| r64_shapes_ct10/sample_002 | 100.8987 | 101.1973 |
| r64_shapes_ct10/sample_003 | 103.1740 | 102.2819 |
| r64_shapes_ct10/sample_004 | 30.2782 | 13.5674 |
| r64_shapes_ct10/sample_005 | 102.6544 | 109.1463 |
| r64_shapes_ct10/sample_006 | 108.5081 | 98.1950 |
| r64_shapes_ct10/sample_007 | 78.1420 | 78.2284 |
| r64_shapes_ct10/sample_008 | 100.5704 | 100.5970 |
| r64_shapes_ct10/sample_009 | 72.9598 | 61.6309 |
| r64_shapes_ct10/sample_010 | 63.1994 | 64.6097 |
| r64_shapes_mri4/sample_000 | 76.3618 | 50.6677 |
| r64_shapes_mri4/sample_001 | 99.8533 | 99.7801 |
| r64_shapes_mri4/sample_002 | 101.7113 | 101.8063 |
| r64_shapes_mri4/sample_003 | 99.8358 | 100.9376 |
| r64_shapes_mri4/sample_004 | 73.9069 | 6.1493 |
| r64_shapes_mri4/sample_005 | 99.8090 | 72.2726 |
| r64_shapes_mri4/sample_006 | 100.3488 | 100.3416 |
| r64_shapes_mri4/sample_007 | 100.6852 | 100.9467 |
| r64_shapes_mri4/sample_008 | 99.7637 | 99.6925 |
| r64_shapes_mri4/sample_009 | 83.9104 | 59.5852 |
| r64_shapes_mri4/sample_010 | 61.1091 | 61.3642 |
| r96_shapes_ct10/sample_000 | 84.8165 | 83.3446 |
| r96_shapes_ct10/sample_001 | 102.3571 | 102.3263 |
| r96_shapes_ct10/sample_002 | 100.4289 | 100.4389 |
| r96_shapes_ct10/sample_003 | 100.3618 | 104.8339 |
| r96_shapes_ct10/sample_004 | 77.9043 | 79.1190 |
| r96_shapes_ct10/sample_005 | 67.2109 | 102.4457 |
| r96_shapes_ct10/sample_006 | 106.7433 | 105.2497 |
| r96_shapes_ct10/sample_007 | 60.9295 | 70.6472 |
| r96_shapes_ct10/sample_008 | 101.1764 | 100.6517 |
| r96_shapes_ct10/sample_009 | 99.1017 | 95.8252 |
| r96_shapes_ct10/sample_010 | 66.5140 | 57.8637 |
| r96_shapes_mri4/sample_000 | 82.2297 | 99.8342 |
| r96_shapes_mri4/sample_001 | 99.5716 | 99.8498 |
| r96_shapes_mri4/sample_002 | 99.6326 | 99.9453 |
| r96_shapes_mri4/sample_003 | 99.4967 | 99.5807 |
| r96_shapes_mri4/sample_004 | 97.8279 | 70.6358 |
| r96_shapes_mri4/sample_005 | 70.3816 | 50.5900 |
| r96_shapes_mri4/sample_006 | 100.7105 | 100.4879 |
| r96_shapes_mri4/sample_007 | 70.5114 | 70.6315 |
| r96_shapes_mri4/sample_008 | 7.8629 | 4.3639 |
| r96_shapes_mri4/sample_009 | 99.3559 | 81.1261 |
| r96_shapes_mri4/sample_010 | 53.6046 | 80.2187 |
| r128_shapes_ct10/sample_000 | 99.5985 | 100.0027 |
| r128_shapes_ct10/sample_001 | 99.7100 | 99.9955 |
| r128_shapes_ct10/sample_002 | 104.3587 | 105.7076 |
| r128_shapes_ct10/sample_003 | 104.8848 | 5.4322 |
| r128_shapes_ct10/sample_004 | 79.6018 | 87.3557 |
| r128_shapes_ct10/sample_005 | 101.8980 | 101.4933 |
| r128_shapes_ct10/sample_006 | 104.7453 | 41.8225 |
| r128_shapes_ct10/sample_007 | 108.8492 | 83.4671 |
| r128_shapes_ct10/sample_008 | 102.9138 | 102.8504 |
| r128_shapes_ct10/sample_009 | 109.7426 | 108.8773 |
| r128_shapes_mri4/sample_000 | 99.3421 | 99.1408 |
| r128_shapes_mri4/sample_001 | 99.9842 | 99.8962 |
| r128_shapes_mri4/sample_002 | 100.0172 | 99.8556 |
| r128_shapes_mri4/sample_003 | 99.5989 | 99.6275 |
| r128_shapes_mri4/sample_004 | 73.4243 | 73.4837 |
| r128_shapes_mri4/sample_005 | 99.2656 | 81.0286 |
| r128_shapes_mri4/sample_006 | 95.3878 | 92.2767 |
| r128_shapes_mri4/sample_007 | 99.3900 | 99.6666 |
| r128_shapes_mri4/sample_008 | 99.7145 | 100.2947 |
| r128_shapes_mri4/sample_009 | 95.5773 | 99.8709 |
| r160_shapes_ct10/sample_000 | 101.5283 | 101.4907 |
| r160_shapes_ct10/sample_001 | 100.3453 | 100.3390 |
| r160_shapes_ct10/sample_002 | 99.9719 | 100.0423 |
| r160_shapes_ct10/sample_003 | 101.9155 | 100.9294 |
| r160_shapes_ct10/sample_004 | 102.8103 | 43.2094 |
| r160_shapes_ct10/sample_005 | 100.6639 | 102.1733 |
| r160_shapes_ct10/sample_006 | 100.4869 | 101.2152 |
| r160_shapes_ct10/sample_007 | 113.9150 | 82.3019 |
| r160_shapes_ct10/sample_008 | 99.7497 | 99.7934 |
| r160_shapes_ct10/sample_009 | 92.7941 | 104.9624 |
| r160_shapes_mri4/sample_000 | 99.6357 | 99.0664 |
| r160_shapes_mri4/sample_001 | 99.4535 | 1.9835 |
| r160_shapes_mri4/sample_002 | 100.2464 | 100.1736 |
| r160_shapes_mri4/sample_003 | 23.7943 | 99.1298 |
| r160_shapes_mri4/sample_004 | 73.1857 | 73.3685 |
| r160_shapes_mri4/sample_005 | 99.2336 | 99.1147 |
| r160_shapes_mri4/sample_006 | 55.1615 | 89.7920 |
| r160_shapes_mri4/sample_007 | 98.9408 | 70.5607 |
| r160_shapes_mri4/sample_008 | 99.8887 | 99.8960 |
| r160_shapes_mri4/sample_009 | 99.5040 | 99.3187 |
| r200_shapes_ct10/sample_000 | 92.2966 | 103.1401 |
| r200_shapes_ct10/sample_001 | 100.3014 | 100.0095 |
| r200_shapes_ct10/sample_002 | 101.2903 | 101.2412 |
| r200_shapes_ct10/sample_003 | 100.4612 | 100.3815 |
| r200_shapes_ct10/sample_004 | 31.6637 | 108.0088 |
| r200_shapes_ct10/sample_005 | 106.3546 | 103.2754 |
| r200_shapes_ct10/sample_006 | 56.0231 | 116.7701 |
| r200_shapes_ct10/sample_007 | 57.1794 | 73.8626 |
| r200_shapes_ct10/sample_008 | 100.0128 | 99.9111 |
| r200_shapes_ct10/sample_009 | 108.2739 | 93.8593 |
| r200_shapes_mri4/sample_000 | 99.6888 | 50.8727 |
| r200_shapes_mri4/sample_001 | 99.6028 | 99.4605 |
| r200_shapes_mri4/sample_002 | 97.9975 | 3.1782 |
| r200_shapes_mri4/sample_003 | 99.6854 | 99.6036 |
| r200_shapes_mri4/sample_004 | 99.5719 | 99.4481 |
| r200_shapes_mri4/sample_005 | 99.5883 | 99.5697 |
| r200_shapes_mri4/sample_006 | 99.4682 | 99.2670 |
| r200_shapes_mri4/sample_007 | 99.4767 | 99.2899 |
| r200_shapes_mri4/sample_008 | 99.7435 | 99.6960 |
| r200_shapes_mri4/sample_009 | 99.2955 | 99.6095 |
| r250_shapes_ct10/sample_000 | 82.8371 | 99.2497 |
| r250_shapes_ct10/sample_001 | 99.9850 | 99.8598 |
| r250_shapes_ct10/sample_002 | 100.0471 | 100.2270 |
| r250_shapes_ct10/sample_003 | 111.5932 | 101.6151 |
| r250_shapes_ct10/sample_004 | 46.6633 | 32.3162 |
| r250_shapes_ct10/sample_005 | 106.9358 | 107.0333 |
| r250_shapes_ct10/sample_006 | 101.3384 | 101.5877 |
| r250_shapes_ct10/sample_007 | 96.3235 | 98.6354 |
| r250_shapes_ct10/sample_008 | 99.7639 | 99.9217 |
| r250_shapes_ct10/sample_009 | 110.3665 | 107.9239 |
| r250_shapes_mri4/sample_000 | 99.2466 | 99.2712 |
| r250_shapes_mri4/sample_001 | 99.4823 | 99.4668 |
| r250_shapes_mri4/sample_002 | 99.6921 | 99.6650 |
| r250_shapes_mri4/sample_003 | 99.6892 | 99.5612 |
| r250_shapes_mri4/sample_004 | 88.5921 | 66.3410 |
| r250_shapes_mri4/sample_005 | 72.2414 | 72.1002 |
| r250_shapes_mri4/sample_006 | 99.6381 | 98.8427 |
| r250_shapes_mri4/sample_007 | 82.9834 | 45.2494 |
| r250_shapes_mri4/sample_008 | 99.7312 | 99.7191 |
| r250_shapes_mri4/sample_009 | 98.7833 | 99.0102 |
