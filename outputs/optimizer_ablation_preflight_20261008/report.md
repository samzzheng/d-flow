# LBFGS history and backtracking pilot

Selected 64x64 one-circle CPU cases, noiseless measurements, 100 Euler steps, rank 8.
Same sources, masks, fixed random sketches and first-attempt-only scaling; no V reuse.
Up to 2 attempts; two failures or sustained objective plateau may stop earlier.
Filtered: history 10 and s.T@y > max(1e-10, 0.01*||s||*||y||). Geometric: halve rejected rates.
All variants keep lr=1, Armijo c1=1e-4, 30 trials, and displacement cap 0.64.

| Case | baseline | combined |
| --- | ---: | ---: |
| r64_n1_ct10/sample_003 | 140.316% (step_limit, 2) | 140.316% (step_limit, 2) |

Completed matched cases: 1/1.

| Variant | Mean image % | <=5% image | Lower image vs baseline | Mean residual % | Stalls | Filter rejections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 140.316 | 0/1 | 0/1 | 119.241 | 0 | 0 |
| combined | 140.316 | 0/1 | 0/1 | 119.241 | 0 | 0 |

A stall does not certify convergence. These selected single-seed CPU cases do not establish GPU or population gains.
Geometric halving changes the explored trial grid and forward-evaluation count, while keeping gradient products unchanged.
