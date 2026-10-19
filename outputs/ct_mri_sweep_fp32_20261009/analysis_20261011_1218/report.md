# Completed noiseless sweep audit: 2026-10-11T12:19:29.490612-04:00

Frozen 504 paired terminal cases; 16 acquisitions have all 30 cases.
Only matched terminal triples are included. Partial acquisitions may be completion-biased.
Success below is an exploratory <=5% final relative image L2 threshold; severe error is >20%.
A stalled run is not necessarily inaccurate, and a 100-attempt run is not necessarily converged.

| Acquisition | n | Median image error %: D / P / DLR | Success <=5%: D / P / DLR | Severe >20%: D / P / DLR |
| --- | --- | --- | --- | --- |
| r128_n1_ct10 | 30 | 2.63 / 4.86 / 3.23 | 16 / 15 / 17 | 11 / 10 / 13 |
| r128_n1_mri4 | 30 | 10.46 / 0.90 / 1.85 | 15 / 26 / 18 | 14 / 3 / 12 |
| r128_n2_ct10 | 30 | 38.17 / 28.21 / 49.11 | 8 / 9 / 7 | 19 / 18 / 23 |
| r128_n2_mri4 | 30 | 45.52 / 0.88 / 50.27 | 13 / 23 / 9 | 16 / 7 / 19 |
| r128_n3_ct10 | 30 | 71.26 / 26.13 / 79.62 | 6 / 10 / 0 | 21 / 17 / 28 |
| r128_n3_mri4 | 30 | 64.55 / 0.97 / 80.44 | 5 / 29 / 0 | 24 / 1 / 28 |
| r250_n1_ct10 | 30 | 0.99 / 5.27 / 1.23 | 22 / 15 / 23 | 7 / 11 / 6 |
| r250_n1_mri4 | 30 | 1.03 / 0.81 / 1.08 | 22 / 30 / 21 | 8 / 0 / 7 |
| r250_n2_ct10 | 30 | 21.27 / 19.21 / 14.96 | 13 / 12 / 11 | 15 / 14 / 13 |
| r250_n2_mri4 | 30 | 31.22 / 0.74 / 59.51 | 12 / 27 / 8 | 17 / 2 / 19 |
| r250_n3_ct10 | 24 (partial) | 63.12 / 9.47 / 46.02 | 7 / 11 / 0 | 15 / 6 / 20 |
| r64_n1_ct10 | 30 | 1.14 / 10.52 / 1.42 | 19 / 14 / 19 | 9 / 12 / 10 |
| r64_n1_mri4 | 30 | 1.20 / 1.06 / 1.28 | 19 / 24 / 19 | 10 / 6 / 10 |
| r64_n2_ct10 | 30 | 51.11 / 33.56 / 34.48 | 10 / 10 / 13 | 18 / 19 / 16 |
| r64_n2_mri4 | 30 | 21.68 / 1.02 / 35.53 | 15 / 22 / 14 | 15 / 8 / 16 |
| r64_n3_ct10 | 30 | 21.08 / 25.51 / 66.70 | 12 / 11 / 1 | 15 / 16 / 22 |
| r64_n3_mri4 | 30 | 2.94 / 1.04 / 39.69 | 16 / 21 / 2 | 12 / 9 / 18 |

## Stopping and initialization (complete acquisitions only)

- D-Flow: 231/480 severe; 151 severe stalls; 80 severe step limits, of which 55 still reduced the objective by >1% over the last 20 attempts. Among severe stalls, median attempts 32.
- P-Flow: 153/480 severe; 146 severe stalls; 7 severe step limits, of which 3 still reduced the objective by >1% over the last 20 attempts. Among severe stalls, median attempts 18.
- DLR D-Flow: 260/480 severe; 188 severe stalls; 72 severe step limits, of which 51 still reduced the objective by >1% over the last 20 attempts. Among severe stalls, median attempts 31.

## Paired <=5% success comparisons

A-only / B-only are discordant paired cases. Intervals are paired percentile bootstrap (10,000 resamples).
P-values are exploratory exact discordance tests, unadjusted for inspecting multiple regimes/thresholds.

| Acquisition | A vs B | A-only / B-only | Success difference pp (95% interval) | p |
| --- | --- | --- | --- | --- |
| r128_n1_ct10 | DLR D-Flow vs D-Flow | 2 / 1 | 3.3 (-6.7, 16.7) | 1 |
| r128_n1_ct10 | DLR D-Flow vs P-Flow | 6 / 4 | 6.7 (-13.3, 26.7) | 0.7539 |
| r128_n1_ct10 | P-Flow vs D-Flow | 4 / 5 | -3.3 (-23.3, 16.7) | 1 |
| r128_n1_mri4 | DLR D-Flow vs D-Flow | 3 / 0 | 10.0 (0.0, 23.3) | 0.25 |
| r128_n1_mri4 | DLR D-Flow vs P-Flow | 1 / 9 | -26.7 (-43.3, -10.0) | 0.02148 |
| r128_n1_mri4 | P-Flow vs D-Flow | 11 / 0 | 36.7 (20.0, 53.3) | 0.0009766 |
| r128_n2_ct10 | DLR D-Flow vs D-Flow | 3 / 4 | -3.3 (-20.0, 13.3) | 1 |
| r128_n2_ct10 | DLR D-Flow vs P-Flow | 4 / 6 | -6.7 (-26.7, 13.3) | 0.7539 |
| r128_n2_ct10 | P-Flow vs D-Flow | 5 / 4 | 3.3 (-16.7, 23.3) | 1 |
| r128_n2_mri4 | DLR D-Flow vs D-Flow | 0 / 4 | -13.3 (-26.7, -3.3) | 0.125 |
| r128_n2_mri4 | DLR D-Flow vs P-Flow | 0 / 14 | -46.7 (-63.3, -30.0) | 0.0001221 |
| r128_n2_mri4 | P-Flow vs D-Flow | 11 / 1 | 33.3 (13.3, 53.3) | 0.006348 |
| r128_n3_ct10 | DLR D-Flow vs D-Flow | 0 / 6 | -20.0 (-36.7, -6.7) | 0.03125 |
| r128_n3_ct10 | DLR D-Flow vs P-Flow | 0 / 10 | -33.3 (-50.0, -16.7) | 0.001953 |
| r128_n3_ct10 | P-Flow vs D-Flow | 8 / 4 | 13.3 (-10.0, 36.7) | 0.3877 |
| r128_n3_mri4 | DLR D-Flow vs D-Flow | 0 / 5 | -16.7 (-30.0, -3.3) | 0.0625 |
| r128_n3_mri4 | DLR D-Flow vs P-Flow | 0 / 29 | -96.7 (-100.0, -90.0) | 3.725e-09 |
| r128_n3_mri4 | P-Flow vs D-Flow | 24 / 0 | 80.0 (63.3, 93.3) | 1.192e-07 |
| r250_n1_ct10 | DLR D-Flow vs D-Flow | 2 / 1 | 3.3 (-6.7, 16.7) | 1 |
| r250_n1_ct10 | DLR D-Flow vs P-Flow | 10 / 2 | 26.7 (6.7, 46.7) | 0.03857 |
| r250_n1_ct10 | P-Flow vs D-Flow | 2 / 9 | -23.3 (-43.3, -3.3) | 0.06543 |
| r250_n1_mri4 | DLR D-Flow vs D-Flow | 1 / 2 | -3.3 (-13.3, 6.7) | 1 |
| r250_n1_mri4 | DLR D-Flow vs P-Flow | 0 / 9 | -30.0 (-46.7, -13.3) | 0.003906 |
| r250_n1_mri4 | P-Flow vs D-Flow | 8 / 0 | 26.7 (13.3, 43.3) | 0.007812 |
| r250_n2_ct10 | DLR D-Flow vs D-Flow | 3 / 5 | -6.7 (-23.3, 10.0) | 0.7266 |
| r250_n2_ct10 | DLR D-Flow vs P-Flow | 6 / 7 | -3.3 (-26.7, 20.0) | 1 |
| r250_n2_ct10 | P-Flow vs D-Flow | 5 / 6 | -3.3 (-26.7, 20.0) | 1 |
| r250_n2_mri4 | DLR D-Flow vs D-Flow | 2 / 6 | -13.3 (-30.0, 3.3) | 0.2891 |
| r250_n2_mri4 | DLR D-Flow vs P-Flow | 0 / 19 | -63.3 (-80.0, -46.7) | 3.815e-06 |
| r250_n2_mri4 | P-Flow vs D-Flow | 15 / 0 | 50.0 (33.3, 66.7) | 6.104e-05 |
| r64_n1_ct10 | DLR D-Flow vs D-Flow | 1 / 1 | 0.0 (-10.0, 10.0) | 1 |
| r64_n1_ct10 | DLR D-Flow vs P-Flow | 7 / 2 | 16.7 (0.0, 36.7) | 0.1797 |
| r64_n1_ct10 | P-Flow vs D-Flow | 2 / 7 | -16.7 (-36.7, 0.0) | 0.1797 |
| r64_n1_mri4 | DLR D-Flow vs D-Flow | 1 / 1 | 0.0 (-10.0, 10.0) | 1 |
| r64_n1_mri4 | DLR D-Flow vs P-Flow | 1 / 6 | -16.7 (-33.3, 0.0) | 0.125 |
| r64_n1_mri4 | P-Flow vs D-Flow | 7 / 2 | 16.7 (-3.3, 36.7) | 0.1797 |
| r64_n2_ct10 | DLR D-Flow vs D-Flow | 3 / 0 | 10.0 (0.0, 23.3) | 0.25 |
| r64_n2_ct10 | DLR D-Flow vs P-Flow | 9 / 6 | 10.0 (-13.3, 33.3) | 0.6072 |
| r64_n2_ct10 | P-Flow vs D-Flow | 6 / 6 | 0.0 (-23.3, 23.3) | 1 |
| r64_n2_mri4 | DLR D-Flow vs D-Flow | 1 / 2 | -3.3 (-16.7, 6.7) | 1 |
| r64_n2_mri4 | DLR D-Flow vs P-Flow | 0 / 8 | -26.7 (-43.3, -10.0) | 0.007812 |
| r64_n2_mri4 | P-Flow vs D-Flow | 9 / 2 | 23.3 (3.3, 43.3) | 0.06543 |
| r64_n3_ct10 | DLR D-Flow vs D-Flow | 0 / 11 | -36.7 (-53.3, -20.0) | 0.0009766 |
| r64_n3_ct10 | DLR D-Flow vs P-Flow | 0 / 10 | -33.3 (-50.0, -16.7) | 0.001953 |
| r64_n3_ct10 | P-Flow vs D-Flow | 6 / 7 | -3.3 (-26.7, 20.0) | 1 |
| r64_n3_mri4 | DLR D-Flow vs D-Flow | 0 / 14 | -46.7 (-63.3, -30.0) | 0.0001221 |
| r64_n3_mri4 | DLR D-Flow vs P-Flow | 0 / 19 | -63.3 (-80.0, -46.7) | 3.815e-06 |
| r64_n3_mri4 | P-Flow vs D-Flow | 8 / 3 | 16.7 (-3.3, 36.7) | 0.2266 |

Source manifests, full frozen histories, threshold sensitivity (2/5/10/20%), and per-case data are saved alongside this report.
Best-image values in cases.csv use retrospective ground truth selection, not an available stopping rule.
No additional reconstructions were run by this script.
