# Noise regularization: paired trajectory audit

Read-only CPU analysis of 4,320 completed method histories: 1,440 paired cases, single-circle models, six resolutions, CT10/MRI4, 30 samples per configuration, five Euler steps, shared LBFGS/Armijo settings, rank-8 DLR. Each table entry pools the same 180 cases per problem/noise. Protocol hashes, seeds, operators, and monotone saved objectives were checked. All numbers below are relative image L2 percentages unless labeled otherwise.

## Final reconstruction versus retrospective best

Best checkpoints are selected using ground truth and are an oracle diagnostic, not an available stopping criterion. A low retrospective error shows that the trajectory found an accurate reconstruction before its final iterate.

| Problem | Noise | D-Flow final / best | P-Flow final / best | DLR final / best |
| --- | --- | --- | --- | --- |
| CT | 1% | 4.43 / 1.51 | 2.85 / 2.28 | 1.91 / 1.89 |
| CT | 5% | 28.74 / 1.87 | 4.09 / 2.70 | 2.44 / 2.23 |
| CT | 10% | 59.45 / 2.31 | 5.65 / 3.19 | 3.78 / 2.66 |
| CT | 20% | 117.99 / 3.14 | 10.39 / 4.60 | 5.94 / 3.77 |
| MRI | 1% | 1.25 / 1.17 | 2.09 / 1.53 | 2.04 / 2.03 |
| MRI | 5% | 4.88 / 1.51 | 7.24 / 1.71 | 2.07 / 2.07 |
| MRI | 10% | 11.36 / 1.59 | 13.52 / 1.91 | 2.14 / 2.10 |
| MRI | 20% | 24.02 / 1.78 | 25.40 / 2.11 | 2.40 / 2.23 |

## Matched attempts and residuals

D-Flow/DLR are compared at the last attempt completed by both methods in each case. This avoids granting D-Flow its later additional attempts, although the common attempt number varies by sample and the methods still use different directions and accepted steps. Residual ratios use final observed residual divided by the known noise norm. Near one is consistent with leaving measurement noise unfitted; it is not proof that image content is correct.

| Problem | Noise | Median common attempt | D-Flow common error | DLR common error | DLR wins / 180 at common attempt | Final residual/noise: D / P / DLR |
| --- | --- | --- | --- | --- | --- | --- |
| CT | 1% | 271.5 | 3.06 | 1.91 | 125 | 0.239 / 1.148 / 1.053 |
| CT | 5% | 283 | 17.74 | 2.45 | 146 | 0.070 / 1.005 / 1.001 |
| CT | 10% | 255.5 | 44.41 | 3.78 | 148 | 0.064 / 1.000 / 0.999 |
| CT | 20% | 245.5 | 71.97 | 5.91 | 153 | 0.066 / 0.997 / 0.999 |
| MRI | 1% | 310.5 | 1.30 | 2.09 | 20 | 0.739 / 1.201 / 1.642 |
| MRI | 5% | 245 | 2.91 | 2.12 | 110 | 0.707 / 0.535 / 1.034 |
| MRI | 10% | 246 | 5.94 | 2.14 | 129 | 0.599 / 0.531 / 1.007 |
| MRI | 20% | 242 | 17.39 | 2.48 | 137 | 0.551 / 0.530 / 1.002 |

## Distinguishing late degradation from failure to reach an accurate reconstruction

Here a large final error means >20%; a previously accurate checkpoint means <=10%. These are descriptive thresholds, not objective convergence or certified optimizer failure.

| Problem | Noise | Method | Final error >20% | Of these, previously <=10% | Of these, never <=10% |
| --- | --- | --- | --- | --- | --- |
| CT | 5% | D-Flow | 129 | 113 | 16 |
| CT | 5% | P-Flow | 45 | 39 | 6 |
| CT | 5% | DLR D-Flow | 29 | 0 | 29 |
| CT | 20% | D-Flow | 147 | 133 | 14 |
| CT | 20% | P-Flow | 82 | 68 | 14 |
| CT | 20% | DLR D-Flow | 33 | 1 | 32 |
| MRI | 5% | D-Flow | 23 | 0 | 23 |
| MRI | 5% | P-Flow | 6 | 0 | 6 |
| MRI | 5% | DLR D-Flow | 44 | 0 | 44 |
| MRI | 20% | D-Flow | 126 | 104 | 22 |
| MRI | 20% | P-Flow | 139 | 130 | 9 |
| MRI | 20% | DLR D-Flow | 46 | 0 | 46 |

DLR errors above 20% persist at every noise level for 22 CT cases and 35 MRI cases. This suggests a separate basin/subspace issue in addition to late noise fitting; error alone does not identify optimizer failure.

## Discrepancy stopping replay

For each saved history, choose the first checkpoint with observed residual/noise norm <=tau; if none exists, keep the final checkpoint. All 180 cases per problem/noise remain included. This uses the known experimental noise norm and no ground-truth image criterion. It is a retrospective replay, not a newly optimized trajectory. Values are median image errors at tau=1.00 / tau=1.05.

| Problem | Noise | D-Flow | P-Flow | DLR D-Flow |
| --- | --- | --- | --- | --- |
| CT | 1% | 1.63 / 1.80 | 2.39 / 2.39 | 1.91 / 2.01 |
| CT | 5% | 2.21 / 4.32 | 2.99 / 4.28 | 2.49 / 5.05 |
| CT | 10% | 2.89 / 7.55 | 3.68 / 8.09 | 3.37 / 9.06 |
| CT | 20% | 4.31 / 16.15 | 5.78 / 15.48 | 5.23 / 17.96 |
| MRI | 1% | 1.22 / 1.20 | 1.67 / 1.64 | 2.04 / 2.04 |
| MRI | 5% | 1.61 / 2.24 | 1.85 / 2.15 | 2.07 / 2.47 |
| MRI | 10% | 1.72 / 3.68 | 2.05 / 3.61 | 2.14 / 4.36 |
| MRI | 20% | 1.95 / 7.47 | 2.34 / 7.17 | 2.44 / 8.27 |

At 20% noise, D-Flow reaches tau=1.00 in 166/180 CT and 157/180 MRI cases. With final-checkpoint fallback for the rest, its medians become 4.31% and 1.95%, compared with final 117.99% and 24.02%. This is close to or below final DLR errors 5.94% and 2.40%. The tau=1.05 medians 16.15% and 7.47% show substantial threshold sensitivity; noise-floor stopping must be specified and tested rather than assumed to be universally optimal.

## Interpretation and next checks

- The strongest observed regularization signature is resistance to late noise fitting: D-Flow frequently visits an accurate image and then decreases the observed loss while worsening the image. At 20% noise DLR final beats D-Flow's retrospective best in only 9/180 CT and 30/180 MRI cases.
- The DLR advantage survives matched D-Flow/DLR attempt prefixes at high noise, so different final stopping counts alone are insufficient to explain it. With a prefix common to all three methods, P-Flow often limits the comparison to an earlier optimization stage; those results are preserved separately in summary.json.
- A specified discrepancy stopping rule recovers much of D-Flow's median accuracy. The final-iterate comparison therefore does not prove a unique attainable-accuracy advantage for DLR. The observed benefit is consistent with an implicit limitation on noise fitting under the existing optimizer and stopping policy.
- Histories cannot establish that the retained Jacobian subspace removes noise. Compare exact and approximate gradients at the same source, separating the clean signal and added noise terms, and measure the restricted forward sensitivity A J V as well as J. Candidate paired cases are in diagnostic_candidates.json.

Artifacts: runs.csv contains per-run matched-prefix and first-crossing metrics; summary.json contains each resolution as well as pooled results; trajectory_summary.png/pdf show medians; noise20_stopping_replay.png/pdf show both discrepancy thresholds by resolution at 20% noise. No new reconstruction jobs were started.
