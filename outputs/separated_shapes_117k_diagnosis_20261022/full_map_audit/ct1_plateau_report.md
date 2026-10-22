# Accepted tiny steps at the later CT plateau

The unchanged prefix3 CT continuation reaches26.9338% image error at attempt240, versus26.9978% at the audited attempt190: only0.0640 percentage points after50 updates. Its objective decreases5.8416e-6 over that interval. The complete matching baseline prefix is archived for the queued angle-safeguard comparison.

At saved attempt250, the behavior has changed from the earlier angle cancellation. The direction cosine against the approximate negative gradient is about0.99946, so a0.05 angle safeguard would not activate here. Its norm is only8.6560e-8. The inverse-Hessian initial scale from the latest curvature pair is approximately0.000306; three pairs remain in history.

Native GPU differentiation exactly reproduces the saved endpoint and objective. The true gradient norm is8.2920e-4 versus approximate2.8259e-4, and gradient relative error is91.24%. Thus a near-stationary approximate gradient cannot be interpreted as true convergence. Nevertheless, the current L-BFGS direction is a genuine descent direction: unit-direction derivative -3.4487e-4, confirmed by central differences.

| Same direction, different starting length | Accepted displacement | Objective decrease | Trial evaluations |
|---|---:|---:|---:|
| Saved L-BFGS proposal | 8.6560e-8 | 2.9852e-11 | 1 |
| Start at the existing cap2, then geometric halving | 0.125 | 1.9893e-5 | 5 |

The latter is about666000 times the objective decrease, at four additional objective evaluations. This changes neither gradient nor Jacobian rank. It is one fixed-state line search, not a completed reconstruction and not a cost-free improvement.

The existing optimizer accepts the tiny first trial, cannot expand it within that search, and therefore does not activate its failure-triggered scale recovery. Later saved states retain the same three curvature pairs; the accumulated s dot y across several accepted updates is far below the absolute1e-10 pair-admission threshold. This is evidence of an undersized-step feedback loop at this state, not proof that changing the curvature threshold alone would fix it.

The previously queued angle control starts earlier, before this plateau, and should still run unchanged. A separate bounded step-expansion control would be needed to test the later scale mechanism without conflating the two interventions. No production optimizer setting is changed.

[Verification](ct1_plateau_verification.json) · [Saved proposal](ct1_plateau_length0.json) · [Cap-started proposal](ct1_plateau_length2.json) · [CPU cross-check](ct1_plateau_direction_cpu.json) · [Paired50-update baseline](../direction_recovery_ct1_cos005/baseline_first050.json).

## Queued starting-length control

`CapStartedLBFGS` now supplies the separate experimental control. It preserves the first-attempt scaling and normalizes later valid L-BFGS directions so the initial proposed displacement equals the existing cap. It keeps the ray, curvature history, geometric halving, trial budget, strict objective decrease and gradient-evaluation policy. It can require more objective evaluations; this is not a free speedup.

A branch from the saved attempt250 snapshot is queued after the angle control, onGPU1. Its state and three curvature pairs are preserved; its complete history prefix matches the ongoing unchanged prefix3 baseline. It has638 selected-source attempts left within the original888+112=1000 total budget. No angle safeguard is applied in this arm. Queue PID1560792 was verified live.

Four CPU tests cover the angle guarantees plus preserved first-attempt behavior, the unchanged L-BFGS ray, cap compliance and recovery from an artificially small curvature scale. The restored real checkpoint also proposes displacement2 as intended. The idle angle queue was restarted after reviewing the added implementation branch and updating its recorded hashes; active GPU workers were left running.

[Starting-length queue](../cap_started_recovery_ct1/queue.json) · [State restoration check](../cap_started_recovery_ct1/initial_state_verification.json).
