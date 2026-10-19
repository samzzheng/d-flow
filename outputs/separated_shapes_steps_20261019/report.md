# Separated-shapes Euler-step pilot

Selected failure cases plus two successful controls; original epoch-200 larger models. Noiseless CT10/MRI4x, paired sources and masks, unchanged LBFGS/Armijo, up to 1000 attempts. Rank 32, analytical alpha=1-t, previous-V reuse. These selected cases cannot establish dataset-wide reliability. Elapsed times include GPU sharing and are not isolated speed benchmarks.

| Steps | Case | Method | Status | Attempts | Image error % | Measurement error % | Seconds |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 10 | r250_shapes_mri4/sample_000 | D-Flow | step_limit | 1000 | 76.266 | 75.001 | 242.5 |
| 10 | r250_shapes_mri4/sample_000 | P-Flow | stalled | 169 | 1.344 | 0.673 | 204.1 |
| 10 | r250_shapes_mri4/sample_000 | DLR D-Flow | stalled | 42 | 99.902 | 99.877 | 84.6 |
| 10 | r64_shapes_mri4/sample_001 | D-Flow | step_limit | 1000 | 36.088 | 29.177 | 144.0 |
| 10 | r64_shapes_mri4/sample_001 | P-Flow | stalled | 35 | 2.340 | 1.832 | 19.3 |
| 10 | r64_shapes_mri4/sample_001 | DLR D-Flow | stalled | 228 | 99.877 | 99.806 | 111.8 |
| 20 | r250_shapes_mri4/sample_000 | D-Flow | stalled | 81 | 107.068 | 106.168 | 144.6 |
| 20 | r250_shapes_mri4/sample_000 | P-Flow | stalled | 189 | 1.005 | 0.445 | 395.4 |
| 20 | r250_shapes_mri4/sample_000 | DLR D-Flow | stalled | 38 | 108.722 | 107.781 | 181.3 |
| 20 | r64_shapes_mri4/sample_001 | D-Flow | stalled | 151 | 99.861 | 99.801 | 68.4 |
| 20 | r64_shapes_mri4/sample_001 | P-Flow | stalled | 42 | 1.606 | 1.042 | 51.4 |
| 20 | r64_shapes_mri4/sample_001 | DLR D-Flow | stalled | 55 | 100.031 | 100.036 | 50.7 |
| 10 | r128_shapes_ct10/sample_000 | D-Flow | stalled | 214 | 100.815 | 94.922 | 45.7 |
| 10 | r128_shapes_ct10/sample_000 | P-Flow | stalled | 45 | 54.651 | 28.664 | 34.4 |
| 10 | r128_shapes_ct10/sample_000 | DLR D-Flow | stalled | 17 | 111.584 | 100.023 | 22.2 |
| 10 | r96_shapes_mri4/sample_001 | D-Flow | stalled | 212 | 99.599 | 99.454 | 55.2 |
| 10 | r96_shapes_mri4/sample_001 | P-Flow | stalled | 50 | 1.664 | 0.862 | 38.7 |
| 10 | r96_shapes_mri4/sample_001 | DLR D-Flow | stalled | 61 | 99.689 | 99.600 | 65.9 |
| 20 | r128_shapes_ct10/sample_000 | D-Flow | stalled | 25 | 108.738 | 100.620 | 59.7 |
| 20 | r128_shapes_ct10/sample_000 | P-Flow | stalled | 45 | 60.479 | 31.347 | 82.1 |
| 20 | r128_shapes_ct10/sample_000 | DLR D-Flow | stalled | 12 | 109.740 | 98.950 | 35.2 |
| 20 | r96_shapes_mri4/sample_001 | D-Flow | stalled | 46 | 99.666 | 99.576 | 43.3 |
| 20 | r96_shapes_mri4/sample_001 | P-Flow | stalled | 8 | 99.704 | 99.629 | 16.1 |
| 20 | r96_shapes_mri4/sample_001 | DLR D-Flow | stalled | 31 | 99.685 | 99.603 | 60.9 |
| 10 | r160_shapes_ct10/sample_000 | D-Flow | stalled | 386 | 100.608 | 93.300 | 93.1 |
| 10 | r160_shapes_ct10/sample_000 | P-Flow | stalled | 24 | 93.278 | 78.733 | 22.4 |
| 10 | r160_shapes_ct10/sample_000 | DLR D-Flow | stalled | 10 | 110.397 | 99.337 | 20.0 |
| 10 | r200_shapes_ct10/sample_001 | D-Flow | stalled | 282 | 103.629 | 96.490 | 87.0 |
| 10 | r200_shapes_ct10/sample_001 | P-Flow | stalled | 47 | 1.935 | 0.423 | 50.6 |
| 10 | r200_shapes_ct10/sample_001 | DLR D-Flow | stalled | 32 | 103.130 | 97.394 | 52.0 |
| 20 | r160_shapes_ct10/sample_000 | D-Flow | stalled | 14 | 99.838 | 97.657 | 34.9 |
| 20 | r160_shapes_ct10/sample_000 | P-Flow | stalled | 33 | 101.156 | 73.456 | 58.9 |
| 20 | r160_shapes_ct10/sample_000 | DLR D-Flow | stalled | 17 | 101.996 | 97.502 | 56.5 |
| 20 | r200_shapes_ct10/sample_001 | D-Flow | stalled | 166 | 117.211 | 105.300 | 153.5 |
| 20 | r200_shapes_ct10/sample_001 | P-Flow | stalled | 33 | 8.236 | 3.134 | 66.2 |
| 20 | r200_shapes_ct10/sample_001 | DLR D-Flow | stalled | 33 | 118.698 | 106.887 | 82.0 |
| 10 | r128_shapes_mri4/sample_002 | D-Flow | stalled | 111 | 99.958 | 99.250 | 37.2 |
| 10 | r128_shapes_mri4/sample_002 | P-Flow | stalled | 17 | 63.394 | 54.911 | 14.8 |
| 10 | r128_shapes_mri4/sample_002 | DLR D-Flow | stalled | 87 | 99.934 | 98.897 | 75.7 |
| 10 | r64_shapes_ct10/sample_000 | D-Flow | stalled | 424 | 77.804 | 66.161 | 60.0 |
| 10 | r64_shapes_ct10/sample_000 | P-Flow | stalled | 26 | 43.715 | 20.901 | 14.1 |
| 10 | r64_shapes_ct10/sample_000 | DLR D-Flow | stalled | 406 | 85.152 | 64.380 | 210.2 |
| 20 | r128_shapes_mri4/sample_002 | D-Flow | stalled | 118 | 99.762 | 99.673 | 124.3 |
| 20 | r128_shapes_mri4/sample_002 | P-Flow | stalled | 22 | 58.563 | 47.471 | 43.9 |
| 20 | r128_shapes_mri4/sample_002 | DLR D-Flow | stalled | 33 | 127.446 | 126.871 | 78.8 |
| 20 | r64_shapes_ct10/sample_000 | D-Flow | stalled | 346 | 89.575 | 74.587 | 118.5 |
| 20 | r64_shapes_ct10/sample_000 | P-Flow | stalled | 28 | 55.534 | 29.012 | 24.8 |
| 20 | r64_shapes_ct10/sample_000 | DLR D-Flow | stalled | 31 | 95.621 | 79.934 | 47.3 |
