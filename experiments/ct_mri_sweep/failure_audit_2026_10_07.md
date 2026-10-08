# Interim noiseless CT/MRI failure audit — 2026-10-07

The clearest observed regime is P-Flow's advantage on 128² MRI. Fast DLR and
exact D-Flow often fail on the same cases. Most severe errors are early
line-search stalls, not exhaustion of the 100-attempt budget. Low rank alone
does not explain them: at two failed checkpoints, fresh CPU rank-8 gradients
are already within 1.22% and 0.60% of the CPU exact gradient.

Evidence is frozen at 20:30 EDT, rather than continually changing as workers
finish. The running GPU sweep was not modified or interrupted. All new
diagnostics ran on CPU, with two threads per process.

- [Frozen histories and protocol](../../outputs/ct_mri_failure_audit_20261007/snapshot.json)
- [Full regime table and paired statistics](../../outputs/ct_mri_failure_audit_20261007/report.md)
- [Per-case CSV](../../outputs/ct_mri_failure_audit_20261007/cases.csv)
- [Paired final-error scatter](../../outputs/ct_mri_failure_audit_20261007/paired_errors.png)
- [CPU saved-source diagnostics](../../outputs/ct_mri_failure_audit_20261007/shadow_diagnostics.json)
- [Smaller-displacement checks](../../outputs/ct_mri_failure_audit_20261007/fine_steps/shadow_diagnostics.json)
- [CPU factor ablations](../../outputs/ct_mri_failure_audit_20261007/factor_diagnostics.json)

## Protocol and scope

244 paired terminal cases were available. Seven acquisitions had all 30 test
cases, giving 210 complete-acquisition cases used for the main conclusions.
The remaining 29 cases of 128²/two-circle MRI and five of 250²/one-circle MRI
are tabulated separately; completion order can bias those partial subsets.
Three-circle results were not ready for this snapshot.

These are **noiseless** observations, epoch-200 non-attention models, 100 Euler
steps, and at most 100 optimizer attempts. All methods use unconstrained
LBFGS/Armijo, history 100, source displacement cap 0.01 sqrt(d), and the same
source and acquisition within each case. DLR uses rank 8, analytical alpha
1-t, zero enrichment, initialization oversampling 8, and fresh factors per
gradient. Terminal alpha is exactly zero. P-Flow here denotes the identity
gradient proxy with the common optimizer, not a reproduction of every
optimizer choice in its original paper.

The exploratory success threshold is final relative image L2 error <=5%;
severe error means >20%. These are reporting thresholds, not stopping rules.
Threshold sensitivity at 2/5/10/20% is saved in summary.json. None of these
thresholds makes a stalled run a certified local optimum.

| Complete acquisition | Successful D-Flow / 30 | Successful P-Flow / 30 | Successful DLR / 30 |
| --- | ---: | ---: | ---: |
| 64², one circle, CT10 | 19 | 12 | 18 |
| 128², one circle, CT10 | 16 | 11 | 18 |
| 250², one circle, CT10 | 21 | 16 | 20 |
| 64², two circles, CT10 | 9 | 9 | 12 |
| 128², two circles, CT10 | 6 | 5 | 8 |
| 64², one circle, MRI4 | 19 | 21 | 14 |
| 128², one circle, MRI4 | 15 | 27 | 14 |

DLR has encouraging CT counts, especially relative to the proxy, but its
small advantages over exact D-Flow are unresolved with 30 cases per group.
For 128²/one-circle MRI, P-Flow succeeds alone relative to DLR on 14 cases;
DLR succeeds alone on one. The paired success difference is 43.3 percentage
points in P-Flow's favor, with a paired bootstrap 95% interval of 23.3–63.3
points and an exploratory exact discordance p=0.00098. The direction persists
at all four reporting thresholds. These statistics are unadjusted for the
regimes and thresholds inspected, not a preregistered confirmatory test.

There is no injected measurement noise to explain the variation in these
runs. Random sources, DLR sketches, MRI masks, test-image sampling, and trained
models remain possible sources of variability. One source/sketch realization
per image does not identify their separate effects. Cross-resolution models
also use different dataset seeds (782/1422/2642 for one-circle 64/128/250),
so the apparent resolution trend is not a controlled discretization study.

## What the failures show

Across the 210 cases, DLR has 98 severe image errors: **95 stalled early**,
with median terminal attempt 31; only three reached the step limit. All
three still reduced their objective by >1% over the final 20 attempts.
Exact D-Flow has 93 severe errors (88 stalls), and P-Flow has 88 (all stalls).
Merely increasing max_updates will not revive a terminal stalled run.

84 of DLR's 98 severe cases are also severe for exact D-Flow. There are eight
cases where DLR has >20% error and exact has <=5%, and eight with the reverse
outcome. Thus dramatic individual reversals exist, but much of the failure
set is shared. None of the severe cases has a final measurement residual
below 5%, and none previously achieved <=5% image error. This is not evidence
of fitting the noiseless measurements well while producing the wrong image,
or of late overfitting after an accurate reconstruction.

Worse initial generated-image error is associated with worse final DLR
error: within complete groups the Spearman correlations range from 0.24 to
0.61. This is compatible with sensitivity to starting location, but does not
separate difficult truths from unlucky sources. Two-circle CT is harder for
all three methods; this does not establish that rank 8 is insufficient.

Every severe stall's last line search evaluated fewer than its 30 permitted
candidates (DLR: 10–28). The implementation terminates a search when the
float32 candidate equals the current source, and clears LBFGS memory after
failure. Therefore simply allowing more backtracks or adding the already
implemented memory reset does not address these cases. The absence of an
accepted representable step can come from a bad direction, sharp local
curvature, or numerical error; the histories alone cannot choose among them.

For example, 64²/one-circle CT sample 28 ends at 112.03% DLR image error,
versus 1.40% for exact D-Flow. Its last accepted source displacement is
8.05e-8, followed by two failures, despite a saved gradient norm of 8,020.
Its reconstruction has a misplaced, undersized circle. This is not a small
image-quality difference around a common accurate solution.

## Fixed-source diagnostics

Five saved sources were replayed using native CPU float32 and CPU float64
state/convolutions. The model's sinusoidal time embedding remains float32;
a local input hook converts its output for the float64 linear layer.
These are CPU shadow objectives, not bitwise replicas of CUDA. The same
observations and CT grids/MRI masks are preserved. All trial displacements
respect the original cap.

Four selected inaccurate checkpoints show CPU64 endpoint differences of
0.62–5.70% relative to their saved GPU endpoint; the accurate control differs
by 0.0128%. CPU32 replay also differs substantially at those failures.
At the failed exact-D-Flow checkpoint CT sample 8, the saved CUDA gradient
norm is 2,859, while CPU64 gives 142, despite cosine 0.997. This points to
backend/precision sensitivity of the local flow map, not uniquely a DLR issue.

At CT sample 28 the CPU64 negative saved-gradient direction has true slope
-10,822. Steps of length 6.4e-5 increase the objective, but 6.4e-7 and smaller
decrease it. At 6.4e-11 the finite-displacement slope agrees with the analytic
slope within 0.01%. A genuinely descending direction can require an extremely
small displacement here. This does not prove the GPU objective has the same
local behavior, or that higher precision will yield a good reconstruction.

The current manifest permits convolution TF32 while disabling matrix-multiply
TF32. Permission does not establish which kernels actually used it. A direct
same-GPU TF32-on/off replay is the necessary next check; disabling it is not
yet a demonstrated speed-neutral fix. PyTorch documents both CPU/GPU
differences and TF32's reduced input precision in its
[numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

Separately, factors were recomputed at three fixed CPU32 sources with the
same native endpoint gradient. The GPU seed integer does not imply an
identical CPU/GPU random sketch; comparisons below are within CPU.

| Saved DLR source | Final GPU image error | CPU rank-8 gradient error | CPU rank-16 gradient error |
| --- | ---: | ---: | ---: |
| 64² CT, one circle, sample 22 | 35.12% | 1.22% | 0.56% |
| 64² MRI, one circle, sample 26 | 124.49% | 0.60% | 0.30% |
| 64² CT, one circle, sample 3 | 1.21% | 13.88% | 7.62% |

Errors are relative to the CPU exact source gradient, not reconstruction
errors. The accurate control has a very small exact gradient, making its
relative gradient error less informative without the absolute norm. These
checks do not establish a universal relationship between gradient error and
image error. Orthogonal subspace/coefficient error decomposition was verified
for all 12 factor variants; maximum discrepancy is below 1e-4 after division
by the squared exact-gradient norm. Factor/native CPU endpoint differences
were below 0.008%.

Replacing one of the 16 initialization sketch columns with the saved source
gradient changes rank-8 gradient errors to 1.72%, 0.60%, and 12.55%. Changing
the sketch seed gives 1.59%, 0.53%, and 12.77%. Neither is a consistent cure.
These are fixed-point ablations, not continued reconstruction trials.

## Accuracy candidates at approximately unchanged iteration cost

There is **no demonstrated cost-free accuracy improvement yet**. The strongest
candidates keep rank, factor steps, alpha, and model-product counts fixed:

1. Filter poorly scaled LBFGS curvature pairs and compare history 10/20/100.
   Existing acceptance only requires a positive absolute dot product above
   1e-10. A scale-aware filter and shorter memory cost vector arithmetic, not
   new model evaluations. They may prevent bad directions earlier in a run;
   resetting only after a failure is already present. Approximate gradients
   can corrupt curvature differences, but this is a hypothesis about these
   runs, not a demonstrated cause. See
   [Shi et al., Noise-Tolerant Quasi-Newton](https://arxiv.org/html/2010.04352v2).
2. Compare geometric Armijo backtracking with the current interpolation that
   uses the approximate slope, and reuse a sensible previous accepted trial
   scale after restarts. Preserve the first-attempt-only scaling and cap.
   This adds no Jacobian products but can alter forward-evaluation counts.
3. Seed part of the existing width-16 initialization sketch with previous
   right factors, retaining random columns. The public `seed_basis` option
   already supports this. Always recompute factors; do not reuse stale
   factors as the gradient. Prior-gradient seeding above was mixed, so this
   remains an experiment, not a recommendation to enable it globally.

Higher rank, extra basis enrichment, exact-gradient rescue, and increased
precision can be useful diagnostic controls, but have potential execution
cost. An identity/P-Flow blend changes the gradient estimator and terminal
baseline premise; it is not being substituted into DLR in this audit.

## Proposed tests, in priority order

1. **Replay failed searches on the same GPU.** Use saved sources for severe
   failures and accurate controls. Compare TF32 permitted/disabled,
   graph/eager DLR and the native Euler objective, then a higher-precision
   CPU reference. Record actual candidate displacements/changed coordinates,
   finite differences, true g_exact dot direction, and objective decrease.
   Negative slope with precision-dependent acceptance implicates numerics;
   nonnegative slope implicates the chosen estimator/preconditioner. Inspect
   backend equivalence at stalled sources, not only initial sources.
2. **Separate source luck from sketch luck.** For fixed truths/masks use five
   paired source seeds for all methods, and three DLR sketch seeds at each
   source. In a separate MRI arm vary masks with source held fixed. Start
   with representative failures/successes for diagnosis, then confirm on an
   unselected held-out set. Cluster uncertainty by image, not by treating
   each sketch as an independent test image.
3. **Resume identical checkpoints with optimizer ablations.** Compare shared
   LBFGS settings against shorter/filtered history, GD/Adam, and geometric
   backtracking. Include both retained and cleared history controls. Keep
   the same cap, measurement, source and gradient estimator; tune separately
   on validation data. Report rescue rates and final residual/image error,
   rather than only whether one step decreases the objective.
4. **Separate subspace error from coefficient error.** At matched initial,
   pre-stall and terminal sources compare exact gradient, DLR gradient,
   V V^T g_exact, and a certified rank-r reference. Test ranks 4/8/16 plus
   fixed-width sketch reuse. A good subspace with poor coefficients directs
   attention to factor evolution; missing exact-gradient energy directs
   attention to rank/basis discovery. Rank-r SVD optimality for the Jacobian
   does not ensure optimality for a particular measurement gradient.
5. **Test acquisition and model conditioning separately.** Follow the strong
   128² MRI signal with multiple masks and CT10/CT20 controls. Complete the
   two/three-circle groups before declaring a circle-count cutoff. For a
   resolution claim, render identical continuous phantoms across resolutions
   and repeat training seeds; current datasets and models change together.

The immediate development priority is reliable descent through the learned
flow map. The current evidence does not justify discarding low-rank evolution
or claiming its apparent CT improvements are already statistically established.
