# Completed five-step noisy reconstruction studies

Final saved iterates; no ground-truth-based checkpoint selection. Single-circle epoch200 models, CT10/MRI4, five Euler steps, FP32, shared LBFGS/geometric Armijo, up to1,000 attempts. DLR:rank8, analytical alpha=1-t, previous-V sketch initialization. All methods completed without execution failures; a stalled line search is a terminal optimizer outcome, not convergence.

Inputs/model/operator/seed protocols match across methods and the overlapping noise-study cases. All saved objectives are nonincreasing. Every case has PNG/PDF reconstruction figures. All four noise studies cover the same six resolutions, with paired samples and seeds.

## 1% noise (1080 method runs)

Relative image L2 error in percent: median / mean / count above20% (30 samples per cell).

| Problem | Resolution | D-Flow | P-Flow | DLR D-Flow |
| --- | --- | --- | --- | --- |
| CT | 64 | 3.97 / 7.70 / 1 | 3.43 / 5.27 / 1 | 2.18 / 9.72 / 2 |
| CT | 96 | 3.42 / 17.93 / 4 | 3.18 / 3.72 / 0 | 2.34 / 29.28 / 7 |
| CT | 128 | 4.18 / 8.16 / 1 | 3.48 / 4.51 / 0 | 2.10 / 12.62 / 4 |
| CT | 160 | 4.43 / 16.72 / 4 | 2.56 / 3.36 / 0 | 2.18 / 21.98 / 6 |
| CT | 200 | 4.76 / 12.07 / 2 | 4.65 / 15.81 / 5 | 1.49 / 14.29 / 4 |
| CT | 250 | 5.89 / 14.70 / 3 | 2.35 / 6.66 / 1 | 1.70 / 19.99 / 5 |
| MRI | 64 | 1.46 / 1.47 / 0 | 2.18 / 2.31 / 0 | 2.03 / 8.29 / 2 |
| MRI | 96 | 1.36 / 14.93 / 4 | 2.34 / 18.21 / 4 | 1.81 / 30.97 / 8 |
| MRI | 128 | 1.16 / 8.10 / 2 | 2.17 / 4.21 / 1 | 2.22 / 26.79 / 9 |
| MRI | 160 | 1.14 / 11.70 / 3 | 2.33 / 4.87 / 1 | 4.45 / 35.90 / 10 |
| MRI | 200 | 1.06 / 7.92 / 2 | 1.60 / 1.81 / 0 | 1.63 / 31.85 / 12 |
| MRI | 250 | 1.06 / 33.79 / 10 | 1.55 / 1.69 / 0 | 1.44 / 31.17 / 9 |

| Problem | Method | Median image % | Mean image % | ≤5% | >20% | Stalled | Median attempts | Residual / noise norm | By100 image % | Worse image after100 despite lower objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CT | D-Flow | 4.43 | 12.88 | 108/180 | 15/180 | 20/180 | 1000 | 0.239 | 1.91 | 148/169 |
| CT | P-Flow | 2.85 | 6.56 | 128/180 | 7/180 | 153/180 | 96 | 1.148 | 2.89 | 32/86 |
| CT | DLR D-Flow | 1.91 | 17.98 | 138/180 | 28/180 | 166/180 | 289.5 | 1.053 | 3.40 | 18/158 |
| MRI | D-Flow | 1.25 | 12.99 | 158/180 | 21/180 | 20/180 | 1000 | 0.739 | 1.55 | 14/168 |
| MRI | P-Flow | 2.09 | 5.51 | 172/180 | 6/180 | 180/180 | 117 | 1.201 | 1.54 | 88/101 |
| MRI | DLR D-Flow | 2.04 | 27.50 | 122/180 | 50/180 | 169/180 | 314 | 1.642 | 3.94 | 2/141 |

Wins on paired samples: CT {'D-Flow': 22, 'DLR D-Flow': 104, 'P-Flow': 54}; MRI {'D-Flow': 159, 'DLR D-Flow': 1, 'P-Flow': 20}

## 5% noise (1080 method runs)

Relative image L2 error in percent: median / mean / count above20% (30 samples per cell).

| Problem | Resolution | D-Flow | P-Flow | DLR D-Flow |
| --- | --- | --- | --- | --- |
| CT | 64 | 32.77 / 34.31 / 25 | 4.13 / 12.54 / 8 | 2.81 / 10.37 / 3 |
| CT | 96 | 28.30 / 36.23 / 21 | 3.64 / 7.44 / 2 | 3.58 / 37.03 / 9 |
| CT | 128 | 25.00 / 24.12 / 19 | 4.32 / 9.77 / 8 | 2.45 / 11.74 / 3 |
| CT | 160 | 27.75 / 33.91 / 21 | 3.60 / 8.69 / 7 | 2.19 / 12.25 / 3 |
| CT | 200 | 29.21 / 31.35 / 22 | 9.25 / 20.85 / 11 | 2.31 / 18.05 / 6 |
| CT | 250 | 29.19 / 32.10 / 21 | 3.27 / 12.67 / 9 | 1.91 / 19.54 / 5 |
| MRI | 64 | 4.75 / 8.20 / 1 | 7.11 / 6.24 / 0 | 2.04 / 8.38 / 2 |
| MRI | 96 | 4.88 / 24.79 / 6 | 5.28 / 19.95 / 4 | 2.08 / 34.36 / 9 |
| MRI | 128 | 5.32 / 8.51 / 1 | 6.99 / 7.02 / 1 | 2.64 / 21.45 / 7 |
| MRI | 160 | 4.81 / 10.26 / 2 | 7.43 / 9.40 / 1 | 2.22 / 31.52 / 9 |
| MRI | 200 | 4.64 / 13.48 / 3 | 7.35 / 6.89 / 0 | 2.25 / 27.85 / 9 |
| MRI | 250 | 6.18 / 36.08 / 10 | 7.38 / 5.80 / 0 | 1.49 / 27.69 / 8 |

| Problem | Method | Median image % | Mean image % | ≤5% | >20% | Stalled | Median attempts | Residual / noise norm | By100 image % | Worse image after100 despite lower objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CT | D-Flow | 28.74 | 32.00 | 1/180 | 129/180 | 86/180 | 1000 | 0.070 | 4.56 | 155/172 |
| CT | P-Flow | 4.09 | 11.99 | 104/180 | 45/180 | 139/180 | 116 | 1.005 | 3.67 | 61/93 |
| CT | DLR D-Flow | 2.44 | 18.16 | 137/180 | 29/180 | 174/180 | 289 | 1.001 | 3.67 | 38/150 |
| MRI | D-Flow | 4.88 | 16.89 | 95/180 | 23/180 | 37/180 | 1000 | 0.707 | 1.91 | 130/166 |
| MRI | P-Flow | 7.24 | 9.22 | 50/180 | 6/180 | 180/180 | 217.5 | 0.535 | 2.59 | 121/125 |
| MRI | DLR D-Flow | 2.07 | 25.21 | 124/180 | 44/180 | 166/180 | 264 | 1.034 | 3.97 | 12/149 |

Wins on paired samples: CT {'DLR D-Flow': 126, 'P-Flow': 50, 'D-Flow': 4}; MRI {'D-Flow': 32, 'DLR D-Flow': 105, 'P-Flow': 43}

## 10% noise (1080 method runs)

Relative image L2 error in percent: median / mean / count above20% (30 samples per cell).

| Problem | Resolution | D-Flow | P-Flow | DLR D-Flow |
| --- | --- | --- | --- | --- |
| CT | 64 | 57.45 / 53.70 / 24 | 6.33 / 24.13 / 12 | 4.38 / 15.40 / 3 |
| CT | 96 | 56.04 / 63.49 / 27 | 4.15 / 17.19 / 11 | 3.84 / 27.62 / 7 |
| CT | 128 | 51.24 / 49.31 / 23 | 4.40 / 19.08 / 10 | 4.38 / 12.51 / 2 |
| CT | 160 | 60.62 / 60.66 / 24 | 8.45 / 23.21 / 13 | 4.44 / 27.18 / 8 |
| CT | 200 | 69.01 / 66.39 / 25 | 11.88 / 28.10 / 13 | 2.81 / 20.04 / 6 |
| CT | 250 | 67.33 / 62.57 / 24 | 3.74 / 19.14 / 10 | 2.95 / 20.97 / 5 |
| MRI | 64 | 7.14 / 11.95 / 1 | 12.96 / 11.13 / 0 | 2.30 / 10.81 / 4 |
| MRI | 96 | 12.30 / 23.02 / 4 | 12.99 / 23.18 / 4 | 2.02 / 34.74 / 9 |
| MRI | 128 | 10.33 / 12.90 / 1 | 13.33 / 11.10 / 1 | 2.16 / 19.04 / 6 |
| MRI | 160 | 11.21 / 21.02 / 4 | 14.00 / 15.46 / 1 | 2.37 / 38.33 / 11 |
| MRI | 200 | 10.11 / 12.62 / 1 | 13.76 / 15.33 / 1 | 2.24 / 27.20 / 10 |
| MRI | 250 | 12.75 / 37.38 / 9 | 13.80 / 11.28 / 0 | 1.46 / 31.05 / 9 |

| Problem | Method | Median image % | Mean image % | ≤5% | >20% | Stalled | Median attempts | Residual / noise norm | By100 image % | Worse image after100 despite lower objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CT | D-Flow | 59.45 | 59.35 | 0/180 | 147/180 | 134/180 | 670 | 0.064 | 7.21 | 157/173 |
| CT | P-Flow | 5.65 | 21.81 | 82/180 | 69/180 | 151/180 | 135.5 | 1.000 | 4.65 | 73/96 |
| CT | DLR D-Flow | 3.78 | 20.62 | 113/180 | 31/180 | 173/180 | 257.5 | 0.999 | 4.27 | 70/151 |
| MRI | D-Flow | 11.36 | 19.81 | 7/180 | 20/180 | 91/180 | 991.5 | 0.599 | 2.34 | 140/171 |
| MRI | P-Flow | 13.52 | 14.58 | 44/180 | 7/180 | 180/180 | 178 | 0.531 | 4.47 | 129/129 |
| MRI | DLR D-Flow | 2.14 | 26.86 | 119/180 | 49/180 | 169/180 | 246 | 1.007 | 3.24 | 16/137 |

Wins on paired samples: CT {'DLR D-Flow': 115, 'P-Flow': 61, 'D-Flow': 4}; MRI {'D-Flow': 31, 'DLR D-Flow': 114, 'P-Flow': 35}

## 20% noise (1080 method runs)

Relative image L2 error in percent: median / mean / count above20% (30 samples per cell).

| Problem | Resolution | D-Flow | P-Flow | DLR D-Flow |
| --- | --- | --- | --- | --- |
| CT | 64 | 120.99 / 111.47 / 25 | 11.59 / 54.03 / 14 | 5.32 / 13.36 / 3 |
| CT | 96 | 114.15 / 111.40 / 25 | 9.14 / 44.39 / 13 | 6.90 / 40.77 / 10 |
| CT | 128 | 111.73 / 107.98 / 21 | 8.29 / 38.68 / 12 | 6.89 / 16.25 / 4 |
| CT | 160 | 118.42 / 114.88 / 30 | 12.21 / 43.39 / 14 | 6.78 / 22.57 / 5 |
| CT | 200 | 157.81 / 150.58 / 25 | 9.75 / 39.58 / 14 | 4.70 / 19.83 / 6 |
| CT | 250 | 116.39 / 99.48 / 21 | 37.78 / 43.99 / 15 | 4.68 / 22.19 / 5 |
| MRI | 64 | 23.34 / 22.37 / 18 | 24.33 / 20.39 / 24 | 2.82 / 10.51 / 3 |
| MRI | 96 | 23.99 / 28.35 / 22 | 24.91 / 30.96 / 20 | 2.26 / 36.47 / 10 |
| MRI | 128 | 23.09 / 22.20 / 19 | 26.58 / 19.71 / 21 | 2.45 / 16.34 / 5 |
| MRI | 160 | 23.15 / 30.56 / 20 | 26.31 / 26.54 / 28 | 2.56 / 33.54 / 10 |
| MRI | 200 | 23.82 / 30.34 / 22 | 25.31 / 20.58 / 23 | 1.71 / 27.89 / 9 |
| MRI | 250 | 25.54 / 44.25 / 25 | 25.88 / 20.87 / 23 | 1.49 / 30.87 / 9 |

| Problem | Method | Median image % | Mean image % | ≤5% | >20% | Stalled | Median attempts | Residual / noise norm | By100 image % | Worse image after100 despite lower objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CT | D-Flow | 117.99 | 115.97 | 0/180 | 147/180 | 149/180 | 583.5 | 0.066 | 10.05 | 162/175 |
| CT | P-Flow | 10.39 | 44.01 | 41/180 | 82/180 | 158/180 | 228 | 0.997 | 6.79 | 84/101 |
| CT | DLR D-Flow | 5.94 | 22.50 | 72/180 | 33/180 | 171/180 | 257 | 0.999 | 5.86 | 76/154 |
| MRI | D-Flow | 24.02 | 29.68 | 2/180 | 126/180 | 136/180 | 576 | 0.551 | 3.46 | 139/168 |
| MRI | P-Flow | 25.40 | 23.18 | 41/180 | 139/180 | 180/180 | 170.5 | 0.530 | 6.52 | 133/134 |
| MRI | DLR D-Flow | 2.40 | 25.94 | 125/180 | 46/180 | 171/180 | 245 | 1.002 | 3.49 | 18/141 |

Wins on paired samples: CT {'DLR D-Flow': 116, 'P-Flow': 59, 'D-Flow': 5}; MRI {'P-Flow': 39, 'DLR D-Flow': 122, 'D-Flow': 19}

By100 uses the checkpoint at attempt100, or the terminal checkpoint for runs that stopped earlier. Post100 deterioration counts only runs with more than100 attempts. Best-image errors in JSON/CSV are retrospective ground-truth diagnostics, not deployable stopping rules. Thresholds5%/20% describe accuracy and severe errors; they do not define optimizer failure. Group medians do not identify paired wins.
