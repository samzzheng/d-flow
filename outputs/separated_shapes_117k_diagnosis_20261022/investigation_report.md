# DLR failure investigation: verified findings and remaining work

The failures have several causes. Two implementation defects have been corrected and independently validated. Further controls show that correct execution of the low-rank algorithm can still lose useful gradient directions or make numerically unresolved updates. A generally reliable reconstruction correction has not yet been established.

All reconstruction evidence below comes from selected noiseless separated-shapes cases with the trained117k models and five Euler steps. It is not a population success-rate estimate, and it does not establish how these interventions affect noise regularization.

## Corrections already in the implementation

- **Time conditioning:** hand-derived model products now use the trained time-embedding width, rather than assuming it equals the convolution channel width.
- **Forward-map consistency:** shared linearization now uses native normalization statistics. The previously different internally evolved image could make the DLR factors approximate a different map from the objective evaluator. Native statistics retain graph replay and accelerated derivative products.
- **Independent validation:** compare against native model autodiff, not merely another hand-derived path. All six trained resolutions have identical tested endpoints; maximum local JVP/VJP discrepancy is1.16e-7 and full-factor gradient discrepancy3.86e-5. This verifies backend equivalence at the tested states, not that the low-rank approximation equals the full Jacobian.

[All-resolution validation](backend_all_resolutions_report.md) · [Failed-source validation](full_map_audit/backend_seed_report.md)

## What remains after those fixes

| Mechanism | Evidence | Current implication |
| --- | --- | --- |
| Excessive adaptive truncation | Raising the minimum rank4 to32 changes250CT7 image error97.082% to3.428%; MRI2 remains failed. | A rank floor can rescue a case; no universal default is justified. |
| Early projection loss | At failed200CT1, estimated Jacobian action error reaches34.62% at flow step3 while rank is still128. Delayed initialization reduces exact-gradient error97.94% to23.43%. | Initialization/projection history matters, not just terminal rank. A paired50-update reconstruction improves27.5825% to27.3367%, a modest gain rather than a rescue. |
| Precision-limited line evaluations | At a failed MRI source, the same saved direction has no decreasing tested FP32 trial but a1.744e-6 decrease in FP64. | Higher precision can restore progress even when a gradient is well aligned. |
| Missing task-relevant directions | A rank32 terminal SVD reference passes its residual and refinement checks at a failed MRI point but still discards most of the exact gradient. In another case, retaining one informed initialization direction changes image error12.306% to2.982%. | A good low-rank matrix approximation need not preserve the measurement-loss gradient. The informed seed uses a cached exact gradient and is not a free default. |
| Starting region | Matched source-pool selection rescues selected MRI cases that fail from the original source; some cases also fail under exact D-Flow. | Not every failure can be assigned to DLR approximation. |
| Near-data-fit alternatives | One CT control reaches0.005691% measurement error but26.097% image error. | Lower measurement loss alone does not ensure the correct image for that case. |

[Rank32 reference checks](full_map_audit/mri1_svd_refined12.json) · [Rank-floor controls](adaptive_floor32_report.md) · [Prefix-loss controls](full_map_audit/prefix_defect_report.md) · [Precision controls](full_map_audit/adaptive_pool_precision_report.md) · [Retained direction](full_map_audit/retained_seed_report.md) · [Near-data-fit CT alternative](full_map_audit/task_terminal_report.md)

## Why matrix rank alone is insufficient

For an exact singular decomposition, the full gradient and its truncated counterpart satisfy

\[
g=J^\top g_x=\sum_i\sigma_i(u_i^\top g_x)v_i,
\qquad
\|g-J_r^\top g_x\|^2=\sum_{i>r}\sigma_i^2|u_i^\top g_x|^2.
\]

A cutoff based only on singular values does not account for how the current terminal gradient aligns with their left vectors. The DLR factors also accumulate approximation error during evolution. They therefore have both subspace error and coefficient error beyond a terminal truncated-SVD reference. Inspecting only their retained core cannot reveal directions that are already absent.

This provides a plausible tradeoff: filtering directions can regularize a noisy problem, while discarding useful noiseless corrections can prevent accurate reconstruction. These controls demonstrate the latter failure; they are not a new causal test of the earlier noisy-measurement advantage.

## Adaptive rank implementation limitation

With zero enrichment, each factor update has only the current number of candidate columns. The core cutoff can shrink rank but cannot regrow discarded rank within that flow pass. Each new gradient evaluation initializes again using the maximum-rank sketch. Thus the current policy is adaptive compression, not an error-controlled growth scheme. A higher floor avoids some irreversible truncation but cannot recover directions already missed by initialization. This follows from the candidate widths and cutoff in `dlr_dflow/implementation/enriched.py`.

The paired oversampling0/8 diagnostic found no stability advantage from removing oversampling. Both gradient estimates followed the strongly varying exact gradient, with approximately1.3% error in its change. [Fixed-source sketch control](full_map_audit/ct1_sketch_smoothness_report.md).

## L-BFGS directional amplification

At a saved CT prefix3 state, gradient error is only1.34% but L-BFGS predicts51.5 times the true first-order decrease along its proposed direction. Native GPU derivatives and finite differences confirm this. An experimental minimum descent cosine0.05 improves the same-state accepted objective decrease1.18e-9 to5.21e-6 with the same10 trials, without extra model derivatives. This candidate has geometric regression tests; a preserved-state paired reconstruction is queued after the current same-sample controls. It changes only the direction cosine to0.05 and keeps the original budget. No completed reconstruction result yet. [Direction audit and safeguard](full_map_audit/lbfgs_direction_report.md).

## Accepted tiny-step plateau

Later in the same CT continuation, the direction aligns well with the approximate gradient, but its norm is only8.66e-8. Native GPU checks confirm actual descent; starting geometric Armijo at the existing cap instead accepts0.125, increasing the one-step objective decrease2.99e-11 to1.99e-5 at five versus one trial evaluations. Approximation error has meanwhile risen to91.24%. Thus gradient accuracy, direction angle and step length are separate issues. The failed-search safeguard does not address accepted tiny steps. [Plateau audit](full_map_audit/ct1_plateau_report.md).

## Current bounded recovery tests

- The three-object MRI4 retained-seed FP64 continuation completed the original1000-attempt budget: image error94.224% to73.720%. This is substantial numerical progress but remains an inaccurate reconstruction.
- MRI2 adaptive64–128 recovery: the first50 native FP64 updates reached11.367% image error, versus FP3211.659%. It is continuing within the original total1000-attempt budget, with optimizer state preserved. Validated FP64 graph replay replaces the native derivative backend; no factor equations change. Use the live summary for its latest partial result.
- CT200 sample1: paired fixed-rank128, FP64 prefix1/prefix3 continuations share one source and original budget. Prefix1 completed50 additional updates at27.583%, essentially unchanged. Prefix3 also completed50 additional updates, ending at27.3367% image error and12.8943% residual. Both arms are now resumed sequentially to their original1000-total-attempt cap (888 selected-source attempts plus112 prior discarded-source attempts), preserving optimizer histories. Prefix3 has advanced past its saved171-attempt checkpoint and its entire history prefix matches. Prefix1 is queued after it. The initializer and later KSL update are existing code paths; no task-oriented projection or enrichment is added.

[Completed MRI4 verification](native_precision_recovery_mri4/verification.json) · [Live MRI2 recovery](native_precision_recovery_mri2_floor64/summary.json) · [FP64 backend validation](full_map_audit/fp64_backend_report.md) · [CT pairing verification](prefix_recovery_200ct1/protocol_verification.json) · [Completed CT segments](prefix_recovery_200ct1/segment_050_verification.json)

No experimental initialization, precision or rank policy has been promoted to a production default. The original sweep remains held by its source-hash guard; results from modified implementations must use a separate manifest. The next decision is whether the paired reconstruction controls support a correction beyond the verified backend fixes.
