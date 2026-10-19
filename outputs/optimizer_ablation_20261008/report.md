# LBFGS history and backtracking pilot

Selected 64x64 one-circle CPU cases, noiseless measurements, 100 Euler steps, rank 8.
Same sources, masks, fixed random sketches and first-attempt-only scaling; no V reuse.
Up to 200 attempts; two failures or sustained objective plateau may stop earlier.
Filtered: history 10 and s.T@y > max(1e-10, 0.01*||s||*||y||). Geometric: halve rejected rates.
All variants keep lr=1, Armijo c1=1e-4, 30 trials, and displacement cap 0.64.

| Case | baseline | short | filtered | geometric | combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| r64_n1_ct10/sample_003 | 74.030% (stalled, 30) | 74.032% (stalled, 29) | 74.026% (stalled, 36) | 74.031% (stalled, 27) | 74.050% (stalled, 27) |
| r64_n1_ct10/sample_008 | 30.868% (stalled, 51) | 31.200% (stalled, 46) | 33.555% (stalled, 44) | 5.156% (paused, 89) | 37.811% (stalled, 47) |
| r64_n1_ct10/sample_022 | 1.375% (stalled, 83) | 12.731% (paused, 97) | 25.083% (paused, 96) | 1.306% (stalled, 79) | 25.557% (paused, 93) |
| r64_n1_ct10/sample_028 | 1.185% (stalled, 80) | 1.182% (paused, 67) | 1.171% (paused, 66) | 1.176% (paused, 67) | 1.206% (paused, 64) |
| r64_n1_mri4/sample_000 | 24.909% (stalled, 48) | 26.575% (paused, 39) | 27.183% (stalled, 34) | 27.086% (paused, 29) | 26.875% (paused, 26) |
| r64_n1_mri4/sample_003 | 49.926% (paused, 22) | 55.491% (paused, 18) | 64.956% (paused, 13) | 97.018% (paused, 9) | 137.895% (paused, 5) |
| r64_n1_mri4/sample_018 | pending | pending | pending | pending | pending |
| r64_n1_mri4/sample_026 | pending | pending | pending | pending | pending |

Completed matched cases: 1/8.

| Variant | Mean image % | <=5% image | Lower image vs baseline | Mean residual % | Stalls | Filter rejections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 74.030 | 0/1 | 0/1 | 49.971 | 1 | 0 |
| short | 74.032 | 0/1 | 0/1 | 49.971 | 1 | 0 |
| filtered | 74.026 | 0/1 | 1/1 | 49.959 | 1 | 4 |
| geometric | 74.031 | 0/1 | 0/1 | 49.947 | 1 | 0 |
| combined | 74.050 | 0/1 | 0/1 | 49.911 | 1 | 8 |

A stall does not certify convergence. These selected single-seed CPU cases do not establish GPU or population gains.
Geometric halving changes the explored trial grid and forward-evaluation count, while keeping gradient products unchanged.

## At 100 attempts (earlier terminal values carried forward)

| Case | baseline | short | filtered | geometric | combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| r64_n1_ct10/sample_003 | 74.030% | 74.032% | 74.026% | 74.031% | 74.050% |
