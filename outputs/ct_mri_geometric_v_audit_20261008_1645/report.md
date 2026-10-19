# Completed noiseless sweep audit: 2026-10-08T16:45:57.951476-04:00

Frozen 197 paired terminal cases; 5 acquisitions have all 30 cases.
Only matched terminal triples are included. Partial acquisitions may be completion-biased.
Success below is an exploratory <=5% final relative image L2 threshold; severe error is >20%.
A stalled run is not necessarily inaccurate, and a 100-attempt run is not necessarily converged.

| Acquisition | n | Median image error %: D / P / DLR | Success <=5%: D / P / DLR | Severe >20%: D / P / DLR |
| --- | --- | --- | --- | --- |
| r128_n1_ct10 | 30 | 3.90 / 3.25 / 3.50 | 16 / 16 / 17 | 14 / 9 / 13 |
| r128_n1_mri4 | 30 | 2.58 / 0.85 / 10.40 | 16 / 26 / 15 | 14 / 4 / 13 |
| r128_n2_ct10 | 30 | 48.44 / 31.37 / 39.51 | 11 / 10 / 9 | 18 / 19 / 17 |
| r128_n2_mri4 | 30 | 64.00 / 0.85 / 66.88 | 11 / 23 / 9 | 19 / 7 / 21 |
| r128_n3_ct10 | 26 (partial) | 71.96 / 26.59 / 79.57 | 4 / 10 / 0 | 18 / 15 / 24 |
| r250_n1_ct10 | 30 | 1.33 / 5.03 / 1.58 | 20 / 15 / 20 | 8 / 10 / 8 |
| r250_n1_mri4 | 21 (partial) | 1.19 / 0.75 / 1.17 | 11 / 21 / 13 | 8 / 0 / 6 |

## Stopping and initialization (complete acquisitions only)

- D-Flow: 73/150 severe; 64 severe stalls; 9 severe step limits, of which 8 still reduced the objective by >1% over the last 20 attempts. Among severe stalls, median attempts 25.
- P-Flow: 49/150 severe; 49 severe stalls; 0 severe step limits, of which 0 still reduced the objective by >1% over the last 20 attempts. Among severe stalls, median attempts 15.
- DLR D-Flow: 72/150 severe; 70 severe stalls; 2 severe step limits, of which 2 still reduced the objective by >1% over the last 20 attempts. Among severe stalls, median attempts 21.

## Paired <=5% success comparisons

A-only / B-only are discordant paired cases. Intervals are paired percentile bootstrap (10,000 resamples).
P-values are exploratory exact discordance tests, unadjusted for inspecting multiple regimes/thresholds.

| Acquisition | A vs B | A-only / B-only | Success difference pp (95% interval) | p |
| --- | --- | --- | --- | --- |
| r128_n1_ct10 | DLR D-Flow vs D-Flow | 2 / 1 | 3.3 (-6.7, 13.3) | 1 |
| r128_n1_ct10 | DLR D-Flow vs P-Flow | 6 / 5 | 3.3 (-20.0, 26.7) | 1 |
| r128_n1_ct10 | P-Flow vs D-Flow | 6 / 6 | 0.0 (-23.3, 23.3) | 1 |
| r128_n1_mri4 | DLR D-Flow vs D-Flow | 1 / 2 | -3.3 (-13.3, 6.7) | 1 |
| r128_n1_mri4 | DLR D-Flow vs P-Flow | 0 / 11 | -36.7 (-53.3, -20.0) | 0.0009766 |
| r128_n1_mri4 | P-Flow vs D-Flow | 11 / 1 | 33.3 (13.3, 53.3) | 0.006348 |
| r128_n2_ct10 | DLR D-Flow vs D-Flow | 4 / 6 | -6.7 (-26.7, 13.3) | 0.7539 |
| r128_n2_ct10 | DLR D-Flow vs P-Flow | 5 / 6 | -3.3 (-26.7, 16.7) | 1 |
| r128_n2_ct10 | P-Flow vs D-Flow | 5 / 6 | -3.3 (-26.7, 16.7) | 1 |
| r128_n2_mri4 | DLR D-Flow vs D-Flow | 2 / 4 | -6.7 (-23.3, 10.0) | 0.6875 |
| r128_n2_mri4 | DLR D-Flow vs P-Flow | 1 / 15 | -46.7 (-66.7, -26.7) | 0.0005188 |
| r128_n2_mri4 | P-Flow vs D-Flow | 12 / 0 | 40.0 (23.3, 56.7) | 0.0004883 |
| r250_n1_ct10 | DLR D-Flow vs D-Flow | 1 / 1 | 0.0 (-10.0, 10.0) | 1 |
| r250_n1_ct10 | DLR D-Flow vs P-Flow | 8 / 3 | 16.7 (-3.3, 36.7) | 0.2266 |
| r250_n1_ct10 | P-Flow vs D-Flow | 3 / 8 | -16.7 (-36.7, 3.3) | 0.2266 |

Source manifests, full frozen histories, threshold sensitivity (2/5/10/20%), and per-case data are saved alongside this report.
Best-image values in cases.csv use retrospective ground truth selection, not an available stopping rule.
No additional reconstructions were run by this script.
