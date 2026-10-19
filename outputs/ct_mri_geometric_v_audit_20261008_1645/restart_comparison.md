# Restart comparison: 2026-10-08T16:45:57.951476-04:00

All 197 old/new pairs have matching protocols except manifest hash and bitwise-equal truths, sources, clean/observed measurements and masks.
Current sweep uses geometric halving for every method and previous-V sketch reuse for DLR. These changes are not isolated in the DLR comparison.

| Acquisition | n | Method | <=5% old / new | Rescues / regressions | Median image % old / new |
| --- | ---: | --- | ---: | ---: | ---: |
| r128_n1_ct10 | 30 | D-Flow | 16 / 16 | 2 / 2 | 4.43 / 3.90 |
| r128_n1_ct10 | 30 | P-Flow | 11 / 16 | 5 / 0 | 13.96 / 3.25 |
| r128_n1_ct10 | 30 | DLR D-Flow | 18 / 17 | 1 / 2 | 2.05 / 3.50 |
| r128_n1_mri4 | 30 | D-Flow | 15 / 16 | 1 / 0 | 11.88 / 2.58 |
| r128_n1_mri4 | 30 | P-Flow | 27 / 26 | 0 / 1 | 0.86 / 0.85 |
| r128_n1_mri4 | 30 | DLR D-Flow | 14 / 15 | 3 / 2 | 25.61 / 10.40 |
| r128_n2_ct10 | 30 | D-Flow | 6 / 11 | 5 / 0 | 49.86 / 48.44 |
| r128_n2_ct10 | 30 | P-Flow | 5 / 10 | 5 / 0 | 31.01 / 31.37 |
| r128_n2_ct10 | 30 | DLR D-Flow | 8 / 9 | 3 / 2 | 57.40 / 39.51 |
| r128_n2_mri4 | 30 | D-Flow | 10 / 11 | 1 / 0 | 64.64 / 64.00 |
| r128_n2_mri4 | 30 | P-Flow | 23 / 23 | 0 / 0 | 0.88 / 0.85 |
| r128_n2_mri4 | 30 | DLR D-Flow | 7 / 9 | 4 / 2 | 85.14 / 66.88 |
| r128_n3_ct10 (partial) | 26 | D-Flow | 4 / 4 | 2 / 2 | 82.69 / 71.96 |
| r128_n3_ct10 (partial) | 26 | P-Flow | 10 / 10 | 2 / 2 | 13.35 / 26.59 |
| r128_n3_ct10 (partial) | 26 | DLR D-Flow | 0 / 0 | 0 / 0 | 78.01 / 79.57 |
| r250_n1_ct10 | 30 | D-Flow | 21 / 20 | 1 / 2 | 1.34 / 1.33 |
| r250_n1_ct10 | 30 | P-Flow | 16 / 15 | 1 / 2 | 4.39 / 5.03 |
| r250_n1_ct10 | 30 | DLR D-Flow | 20 / 20 | 1 / 1 | 1.29 / 1.58 |
| r250_n1_mri4 (partial) | 21 | D-Flow | 11 / 11 | 1 / 1 | 1.25 / 1.19 |
| r250_n1_mri4 (partial) | 21 | P-Flow | 19 / 21 | 2 / 0 | 0.76 / 0.75 |
| r250_n1_mri4 (partial) | 21 | DLR D-Flow | 12 / 13 | 2 / 1 | 1.90 / 1.17 |
| all_complete_acquisitions | 150 | D-Flow | 68 / 74 | 10 / 4 | 11.21 / 7.17 |
| all_complete_acquisitions | 150 | P-Flow | 82 / 90 | 11 / 3 | 2.39 / 1.64 |
| all_complete_acquisitions | 150 | DLR D-Flow | 67 / 70 | 12 / 9 | 26.35 / 18.32 |

## Changes across the same complete acquisitions

- D-Flow: success change 4.0 percentage points, paired bootstrap 95% interval [-0.7, 9.3], exploratory discordance p=0.1796.
- P-Flow: success change 5.3 percentage points, paired bootstrap 95% interval [0.7, 10.7], exploratory discordance p=0.05737.
- DLR D-Flow: success change 2.0 percentage points, paired bootstrap 95% interval [-4.0, 8.0], exploratory discordance p=0.6636.

## DLR failure pattern

{
  "dlr_severe": 72,
  "also_exact_severe": 63,
  "proxy_success_among_dlr_severe": 37,
  "severe_stalls": 70,
  "median_stall_attempt": 21.0,
  "severe_stalls_last_search_at_trial_budget": 66,
  "minimum_severe_measurement_residual_pct": 8.294086158275604,
  "severe_ever_image_success5": 0
}

Intervals and tests are exploratory. The Holm adjustment in JSON covers the 15 pairwise method comparisons at the 5% threshold within complete acquisitions, not every analysis inspected.
No extra reconstructions or GPU work were performed. A 30-trial failure establishes budget exhaustion, not that further trials would succeed.
