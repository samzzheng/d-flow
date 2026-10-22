# Early proxy initialization control — 2026-10-22

This pilot is queued behind the spectral-weighting study on the same sample. It has not produced reconstruction results. The coordinator waits for the preceding process identity to exit and requires its queue to report completion before allocating GPU1 memory.

The completed late-fallback tests rejected every P-Flow proposal at the failed DLR endpoints. P-Flow can nevertheless reconstruct from the original source in this MRI case. This motivates testing an earlier, single switch before DLR reaches the nearly blank region.

Use separated-shapes MRI250 sample2, noiseless MRI4x, the same original cached source and mask, five Euler steps, rank32, analytical alpha, zero enrichment, unconstrained sources, deterministic cuDNN and FP32 without TF32. Compare:

1. Five DLR/L-BFGS attempts, reset, then DLR.
2. Five P-Flow/L-BFGS attempts, the same reset, then DLR.

The prefix uses the established L-BFGS/geometric Armijo optimizer, rather than the plain objective-only proxy search used in the late fallback pilot. Both phases retain the actual-objective decrease safeguard and displacement cap2.5. The total maximum remains1000 attempts including the prefix; ordinary rejection stopping remains active. No ground-truth metric chooses the prefix length, source or checkpoint. If a prefix terminates before five attempts, only its actual attempts count against the remaining budget.

At the single switch, instantiate a new optimizer, clear gradient/objective caches, discard the prefix's factor seed and initialize DLR from the same original40-column random sketch in both arms. Thus no curvature pair, gradient or previous-V subspace crosses the estimator switch. Recompute the factors at the new source. The DLR-prefix control separates the effect of the proxy path from the effect of this reset. Previous-V reuse continues normally within each DLR phase.

This is not a restart of the canceled periodic alternating experiment. It tests one data-dependent initialization stage. Any success must be compared with the prefix image itself and the existing pure P-Flow result to establish what returning to DLR contributes. It does not establish a correction merely by reaching a low measurement objective.

The runner saves both phase checkpoints, phase settings, source/model/sketch hashes, optimizer states, a source hash at the switch, evaluation counts and combined histories. It asserts identical objective values at the handoff and subtracts prefix attempts from the DLR budget. Syntax checks pass; GPU execution and saved-array verification are pending. Raw tensors remain local and are excluded from Git.

[Launch](launch.json) · [Live status](status.json) · [Preceding spectral study](../spectral_mri2/report.md)
