# Goal-oriented terminal projection

This experimental variant preserves rank-r factors and the original DLR prefix. It changes the final projection criterion to retain the current terminal loss-gradient action. It is not the original unmodified KSL update, and no production default has changed.

At the penultimate Euler point, write the DLR approximation as

\[
\widehat J=hI+USV^\top.
\]

Here A denotes the velocity Jacobian at that state. The next discrete sensitivity target is

\[
T=(I+hA)\widehat J.
\]

The ordinary terminal projection can discard directions needed by \(T^\top g_x\). Instead, obtain an orthonormal basis \(Q\) of width r containing \(g_x\), replacing one column of the ordinary K-step basis. Then compute

\[
T^\top Q=V_1R,\qquad U_1=Q,\qquad S_1=R^\top.
\]

Because \(QQ^\top g_x=g_x\),

\[
(U_1S_1V_1^\top)^\top g_x
=V_1RQ^\top g_x
=T^\top QQ^\top g_x
=T^\top g_x.
\]

The final matrix still has rank at most r; it need not approximate T well on other vectors. The approximation now depends on the measurement loss. It removes the final projection error for this particular action, not the accumulated prefix error.

The two derivative blocks have the same width as the existing final step. An additional QR is required. The prototype also canonicalizes observed prefix factors and recomputes prefix primal states, so unchanged derivative-column counts do not establish unchanged wall time.

## Fixed-source checks

| Case, saved right basis | Rank | Original gradient error | Goal-oriented gradient error | Difference from one-step exact suffix / exact-gradient norm |
| --- | ---: | ---: | ---: | ---: |
| MRI1, conditioned source | 32 | 100.197% | 17.834% | 0.000246% |
| MRI1, conditioned source | 128 | 99.024% | 17.756% | 0.000199% |
| CT0, conditioned source | 32 | 54.691% | 53.578% | 0.002328% |
| CT0, conditioned source | 128 | 53.409% | 52.564% | 0.001445% |

Eight CPU tests pass. Dense controls verify the gradient identity at ranks1/4/16, rank preservation, orthonormal factors, and that the rank-deficient result is not secretly the full target matrix. Native full-map products independently validate the real-model reference gradients. The CT discrepancies are comparable to the independently measured derivative arithmetic variability.

## Reconstruction status at 04:48

- The separate one-exact-suffix-step hybrid, rank32 MRI4, completes1000 attempts at1.695% image error, versus94.224% for the ordinary rank32 control under the same cap and source. This is a selected-case hybrid result, not a population or pure-DLR claim.
- The goal-oriented rank32 MRI1 conditioned reconstruction is still running, at2.074% after340 attempts; ordinary rank32 finished at12.306%. Do not treat the partial number as final.
- Goal-oriented rank32 CT0 finishes at36.452%, versus37.118% previously. This does not resolve the CT failure.
- Frozen-increment rank128 CT7 finishes at20.241%, versus34.280% with the original update. Better fixed-source gradients did not yield a fully accurate reconstruction.

The broader correction remains incomplete. In particular, a useful terminal projection does not repair all earlier subspace errors, poor initial basins, or inverse-problem ambiguity.

[Completed paired-control verification](completed_correction_controls.json) checks bitwise-equal initial sources/truths/observations, checkpoint hashes, equal optimizer/grid/rank/stopping settings, nonincreasing objectives, and image metrics recomputed with float64 reductions. It covers the completed MRI4 hybrid, CT7 increment, and CT0 goal-projection comparisons.
