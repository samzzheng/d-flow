# KSL core fidelity and coarse-step stability

The notes in Sections6/8 assign S1=R_L^T after the L-step QR. The existing fast profile instead uses the KSL bases and then projects an unsplit Euler residual target. These are different numerical updates. These fixed-source checks isolate the actual L-step core; they are not new reconstruction trials.

All comparisons use the same saved source, native five-step endpoint, terminal loss gradient, initialization seed and reused right basis within each case. Entries below are dimensionless relative gradient errors:1 means100%.

| Case | Rank | Fast recompression | Direct recompression | Notes Euler KSL | Stable frozen S/L, actual core | Frozen increment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MRI1 | 32 | 1.00197 | Not tested | 1.00099 | 1.00126 | 1.00203 |
| MRI1 | 128 | 0.990237 | Not tested | 0.988417 | 0.991668 | 0.989017 |
| CT7 | 32 | 0.408494 | 0.40849 | 4.1477e+06 | 16.3232 | 0.201876 |
| CT7 | 128 | 0.306875 | 0.306871 | 2.28406e+06 | 12.9132 | 0.0517699 |

The literal one-Euler-per-stage KSL core is unstable at the CT7 source. Stable frozen S/L integration removes the million-fold amplification but still gives a poor gradient for the discrete five-step objective. The explicit matrix reference test verifies that the implementation follows each of the notes' three stages; a separate fine-RK4 check verifies the stable S/L solution. Eleven CPU tests pass.

Direct evaluation of the fast recompression gives almost the same CT7 gradient error as its compact algebra, so it does not materially repair this case. Neither literal core restoration nor stable S/L alone is promoted as an accuracy correction.

The frozen-increment alternative applies the same K/S/L projection splitting to F held at the beginning of the step. Its S and L terms cancel algebraically to L1=(Y0+h F0)^T U1. This consistently approximates the discrete Euler sensitivity target, avoids the unstable intermediate core, and uses the same rank-column JVP/VJP counts. Its reconstruction benefit still needs testing; its MRI1 fixed-source result remains poor.

Paired shared-budget reconstruction controls now combine this frozen-increment update with the exact cached source pools for250CT7,250MRI2 and200CT1. Rank128, optimizer settings and1000 total attempts remain unchanged. They change the factor integrator, not the measurement loss or projection criterion.

The existing fast profile, the notes' literal Euler KSL, the frozen-increment retraction, and the goal-oriented projection are now distinguished in experiment labels. Production defaults remain unchanged.
