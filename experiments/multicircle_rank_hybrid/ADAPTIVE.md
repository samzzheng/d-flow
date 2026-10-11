# Gradient-guided rank adaptation pilot

This first adaptive variant changes rank **between optimization updates**.
Each 100-step Euler/factor evolution keeps its selected rank. It is not an
implementation of variable-rank integration within a single flow trajectory.
Production algorithms, existing rank measurements, and alternating runs stay
unchanged.

Use the rank ladder4/8/16/32/64, starting at8. Every five attempts, and after
one failed line search, compare gradients at the same source using the same
canonical endpoint, observation, seed, and frozen previous right basis. Rank
probes do not advance that basis. Starting one rank below the current rank,
require relative gradient agreement of half the tolerance to shrink; otherwise
compare the current rank with the next larger rank and grow until agreement
passes. Test tolerances5% and1%. Record reaching64 without a passing check as
unvalidated; pairwise agreement is not an exact-gradient error bound and can
miss approximation bias shared by both ranks.

On a rank change clear LBFGS pairs and the cached gradient. Keep the global
attempt count and failure counters. Both closures within each optimizer step
use the same rank, so no curvature pair mixes rank estimators. Retain geometric
Armijo, first-attempt-only scaling, the0.01 sqrt(d) displacement cap, and two
consecutive failures as the stopping rule for all four variants.

Fresh fixed8/fixed32 controls and adaptive5%/adaptive1% runs use identical
paired inputs, analytical alpha=1-t, trace_probes=0, zero enrichment, initial
oversampling8, previous-V reuse, and FP32 without TF32. The batch-graph backend
uses bounded6-column JVPs and4-column VJPs, with whole-step graphs disabled to
limit memory across many ranks. All four variants use this same backend.
The pilot targets100 attempts for each of eight cases:64px, two/three circles,
CT10/MRI4, samples0–1. It does not choose a production rank or promise a speed
benefit; refinement checks add factor evaluations and rank changes can reduce
usable LBFGS history.

One GPU0 worker, one CPU thread, and a4,000MiB PyTorch allocation cap leave
space alongside the existing rank and hybrid workers. Existing controllers
are not stopped or edited. Larger-resolution adaptation is deferred until
this bounded pilot establishes correctness and accuracy behavior.

```sh
.venv/bin/python -m experiments.multicircle_rank_hybrid.test_adaptive
.venv/bin/python -m experiments.multicircle_rank_hybrid.adaptive_queue \
  --output outputs/rank_adaptive_dlr_20261011/production
```

Each case saves inputs, optimizer/source states, hashes, rank-comparison
records, evaluation counts, and figures with final-checkpoint errors. The live
report includes unfinished rows explicitly. Tests cover shrink hysteresis,
rank-cap reporting, the shared-bias limitation, rank-change curvature resets,
and bitwise resumed/uninterrupted histories. The GPU smoke checks graph/eager
gradient agreement at every ladder rank and analytical alpha throughout all
100 flow steps.
