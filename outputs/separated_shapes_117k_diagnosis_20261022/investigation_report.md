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

At a saved CT prefix3 state, gradient error is only1.34% but L-BFGS predicts51.5 times the true first-order decrease along its proposed direction. Native GPU derivatives and finite differences confirm this. An experimental minimum descent cosine0.05 improves the same-state accepted objective decrease1.18e-9 to5.21e-6 with the same10 trials, without extra model derivatives. This candidate has geometric regression tests; a preserved-state paired reconstruction is running while the unchanged baseline is temporarily suspended in memory. It changes only the direction cosine to0.05 and keeps the original budget. The first actual update matches the fixed-state diagnostic, but the first50 paired updates show no image advantage:26.9338% unchanged versus26.9388% safeguarded, with118 versus598 line-search evaluations. The full-budget result is pending; the safeguard is not justified for adoption. [Direction audit and safeguard](full_map_audit/lbfgs_direction_report.md).

## Accepted tiny-step plateau

Later in the same CT continuation, the direction aligns well with the approximate gradient, but its norm is only8.66e-8. Native GPU checks confirm actual descent; starting geometric Armijo at the existing cap instead accepts0.125, increasing the one-step objective decrease2.99e-11 to1.99e-5 at five versus one trial evaluations. Approximation error has meanwhile risen to91.24%. Thus gradient accuracy, direction angle and step length are separate issues. The failed-search safeguard does not address accepted tiny steps. A separate cap-started L-BFGS continuation from the saved attempt250 state is now queued after the angle control; it retains the ray and history and does not combine the two changes. [Plateau audit](full_map_audit/ct1_plateau_report.md).

## Higher-start, gradual rank compression

A separate paired pilot now tests rank256 with floor32 and relative core cutoff0.001 on250CT7 and250MRI4. The three controls are unrestricted truncation, retain rank256 only at initialization, and retain256 then discard at most half the current rank per flow step. Every gradient evaluation starts with the same high-rank sketch mechanism; previous-V reuse and the optimizer remain unchanged. No production defaults change.

At the shared initial source, unrestricted and initialization-retained controls already have the same rank paths. Gradual compression only changes the last retained rank: CT32 to69, MRI82 to126. Gradient errors are essentially unchanged (CT35.87–35.89%, MRI63.64–63.65%). Thus retaining more terminal directions alone provides no initial fixed-point rescue in these cases; subsequent reconstruction behavior remains open. Live-checkpoint verification confirms source/model/operator/seed pairing, saved image and complex residual metrics, objective monotonicity, displacement caps and rank bounds. CPU verification accumulates norms in FP64 to avoid reduction differences with saved CUDA FP32 metrics.

[Progressive-rank pilot](progressive_rank/report.md) · [Latest checkpoint verification](progressive_rank/live_verification.json)

## Current bounded recovery tests

- The three-object MRI4 retained-seed FP64 continuation completed the original1000-attempt budget: image error94.224% to73.720%. This is substantial numerical progress but remains an inaccurate reconstruction.
- MRI2 adaptive64–128 recovery is complete: FP64 reaches6.598% image error and4.205% measurement residual at the original1000-total-attempt budget, versus FP3211.659% and7.848% after rejected searches. The original50-update history hashes remain intact. This shows restored numerical progress followed by additional optimization, not equal accepted-update counts or complete elimination of reconstruction error. Validated FP64 graph replay replaced native products after the first50 updates without changing factor equations. [Completed precision recovery](native_precision_recovery_mri2_floor64/report.md).
- CT200 sample1: paired fixed-rank128, FP64 prefix1/prefix3 continuations share one source and original budget. Prefix1 completed50 additional updates at27.583%, essentially unchanged. Prefix3 also completed50 additional updates, ending at27.3367% image error and12.8943% residual. Both arms are now resumed sequentially to their original1000-total-attempt cap (888 selected-source attempts plus112 prior discarded-source attempts), preserving optimizer histories. Prefix3 has advanced past its saved171-attempt checkpoint and its entire history prefix matches. The baseline process is temporarily suspended in memory while the angle and cap-started controls run sequentially; automatic resumption then continues this baseline and its queued prefix1 arm. The initializer and later KSL update are existing code paths; no task-oriented projection or enrichment is added.

[Completed MRI4 verification](native_precision_recovery_mri4/verification.json) · [Completed MRI2 verification](native_precision_recovery_mri2_floor64/verification.json) · [FP64 backend validation](full_map_audit/fp64_backend_report.md) · [CT pairing verification](prefix_recovery_200ct1/protocol_verification.json) · [Completed CT segments](prefix_recovery_200ct1/segment_050_verification.json)

No experimental initialization, precision or rank policy has been promoted to a production default. The original sweep remains held by its source-hash guard; results from modified implementations must use a separate manifest. The next decision is whether the paired reconstruction controls support a correction beyond the verified backend fixes.

## Dataset difficulty versus optimization failure

An independent CPU audit verifies 64 completed paired D-Flow/P-Flow cases against saved geometry and original complex measurements. At 250² MRI, P-Flow reaches below 10% image error on all 30 cases, while D-Flow exceeds 50% on 19. These failures are nearly blank: median reconstruction norm is 0.0211 times the truth norm. Useful images are therefore representable by the same discrete generator; capacity alone cannot explain the paired failures. Count stratification does not show a simple increase in difficulty with more objects. Legacy DLR results are excluded because its backend was subsequently corrected.

Five-step unconditional quality also degrades at high resolution: the detector finds no valid component in 71/128 sources at 250², versus 5/128 with 100 steps. This is a contributing trajectory concern, not proof that more steps alone fix inversion.

At the final corrected FP64 MRI2 source, raising the adaptive floor from 64 to 128 barely changes reused-basis gradient error (30.8869% to 30.8774%). Missing-subspace error dominates; terminal rank alone is insufficient.

[Geometry and paired-method audit](dataset_difficulty/report.md) · [Final MRI gradient audit](full_map_audit/mri2_final_gradient_report.md)

## Narrow descent regions at collapsed endpoints

Six selected MRI250 endpoints (four D-Flow, two corrected DLR) have nonzero exact gradients but very narrow sampled descent regions. For corrected DLR samples0/1, the largest tested decreasing displacements along the normalized negative exact gradient are1e-6/1e-7; larger tested steps increase the objective. Central differences at1e-8 agree with native derivatives to at most3.8e-6 of the gradient norm across all tested directions. This is a diagnostic, not a GD reconstruction.

Straight paths to paired successful P-Flow sources all show a small initial objective rise before reaching low-error images. These particular paths expose local barriers, not a proof that no descending path exists. Promoted FP64 maps change failed endpoint images by up to0.451% relative to the original nearly blank image. The evidence supports both subspace loss and difficult local objective geometry; larger rank alone cannot address both. No relaxed line-search or production optimizer change is justified by this diagnostic.

[Source-path report and figure](dataset_difficulty/source_bridge_report.md).

## Precision sensitivity before DLR truncation

Native FP32 gradients at six selected collapsed endpoints differ from FP64 by5.7–309.8%, versus0.019–0.120% at five successful paired P-Flow endpoints. At corrected DLR samples0/1 the differences are74.5%/56.6%. Both objectives use FP64 scalar reductions; the discrepancy comes from evaluating and differentiating different floating-point realizations of the map. One failed exact D-Flow source has negative FP32/FP64 gradient cosine.

Higher precision only in GroupNorm, Euler-state accumulation, or both is inconsistent and sometimes reverses the gradient relative to FP64. Do not promote these partial-precision options. A paired rank32 FP32/FP64 reconstruction from the original Gaussian source is running on MRI250 sample0, with unchanged optimizer and1000-attempt cap. This tests prevention of collapse rather than recovery afterward. Both derivative backends are independently checked at initialization; experimental precision remains isolated.

[Precision report](dataset_difficulty/endpoint_precision_report.md) · [From-start control](precision_from_start_mri0/launch.json).

## Rank during optimization is not permanently decreasing

The verified high-start checkpoints at14:30 reach CT250 sample7 image error5.324% after780 attempts and MRI250 sample4 error1.518% after790. Both runs use near-cap terminal ranks during major improvements; the latest CT rank record then drops to40 while MRI remains253. These are still partial unrestricted-truncation controls, with initialization-retained and gradual variants pending. The rank criterion acts on the approximated core, not a certified exact Jacobian.

Within a single zero-enrichment flow pass, rank can only shrink. Across optimizer iterations, the source changes and each refresh initializes at the maximum rank, so terminal rank can rise again. The observed rank trajectories argue against assuming that a small terminal rank at initialization remains sufficient throughout optimization. They do not establish a causal benefit for the gradual policy before its paired controls run.

[Latest convergence and terminal-rank curves](progressive_rank/latest_progress.md).

## First completed high-start control and plateau secant check

CT250 sample7 unrestricted256 completes at5.194% image error and0.929% measurement residual after836 attempts with the existing rejection safeguard. The earlier maximum128/floor32 control achieved3.428%. Its cached data/source/model/optimizer are paired, but its native-primal/Triton derivative actions and full-step graphs differ from the current native-action, bounded-product execution. This is a historical reference, not a strict rank-cap-only comparison; the accuracy difference cannot be assigned solely to rank. Initialization retained256 in all824 unique recorded evaluations of the completed run: forcing that initial rank changes no ranks along this observed trajectory. Its separate control also probes execution variability; do not attribute a difference to rank retention automatically. Other policies remain pending or active.

At the same CT200 plateau source, appending the discarded positive aggregate secant improves one-step objective decrease2.99e-11 to5.32e-6. Cap-starting the unchanged direction improves it further to1.99e-5. Keep the existing cap-start control queued rather than adding a new curvature-policy variant.

[Curvature diagnostic](full_map_audit/ct1_aggregate_secant_report.md).

The completed CT7 panel verifies cached truth/observation/source pairing and shows all four reconstructed shapes. D-Flow reaches1.495%, P-Flow104.025%, and the corrected adaptive32–256 DLR5.194% image error. This is a completed-method comparison on one sample, not a population ranking. [Paired panel](progressive_rank/r250_shapes_ct10/sample_007/unrestricted256/paired_methods.png) · [Historical comparison scope](progressive_rank/ct7_historical_comparison.json).

## Completed MRI high-start control and numerical sensitivity

MRI250 sample4 unrestricted32–256 completes1000 attempts at1.4921% image error and0.7720% measurement residual. Independent saved-array verification checks source/model/operator pairing, original history preservation, caps, alpha, ranks and complex residual metrics. Its paired D-Flow/P-Flow image errors are0.99%/1.53%; the shared comparison panel is inspected. This is one successful selected case, not a general cure.

The initial-floor-only control is mathematically equivalent on the recorded unrestricted trajectories: initialization already retains256 in every refresh. The first accepted objectives differ by roughly4.08e-8 relatively for MRI, yet at attempt20 the objective difference reaches30.9%. Both later pass through similar31% image-error plateaus. The floor-only arm remains partial; do not interpret divergent trajectories as evidence for a changed rank policy. [Trajectory sensitivity](progressive_rank/trajectory_sensitivity.md).

The requested fixed1024/2048 audits also completed on MRI250 sample1. At the original source, relative gradient errors decline63.04/46.38/38.06/35.79% at ranks32/256/1024/2048; at the failed endpoint45.59/29.24/23.52/21.84%. These reference the FP32 map and do not resolve failed-endpoint precision sensitivity. Both high ranks fit on GPU1. Paired fresh-source reconstructions are running sequentially; no high-rank accuracy result yet. [High-rank audit](high_rank/report.md).

## Rank1024 does not prevent collapse on the selected MRI failure

The fresh-source MRI250 sample1 rank1024 control completes under the unchanged rejected-search rule after13 attempts at100.0238% image error, essentially identical to paired rank32 (100.0241%,16 attempts). Paired D-Flow and P-Flow reach1.219% and2.375%. The higher rank improved initial gradient error63.04% to38.06% but did not prevent nearly blank output: reconstructed norm is2.31% of truth, and the objective is close to that of the zero image. This rules out rank1024 as a sufficient correction on this case, not all possible benefits of higher rank. Original-source pairing, nested sketch, optimizer settings, rank, caps, metrics and objective histories are independently verified. Rank2048 remains active. [Completed rank comparison](high_rank/report.md).

## Failed-search interval at the rank1024 endpoint

The paired exact D-Flow trajectory also becomes nearly blank at attempt2; it continues small accepted steps and recovers substantially around180. At the actual stopped rank1024 source, saved-gradient cosine is0.97744 with native FP32 and0.92684 with native FP64. The saved basis retains96.20% of the FP64 exact-gradient squared norm. Its final history-free search begins at displacement0.62483 and only halves; an independently tested displacement2.5 along the same negative saved gradient passes Armijo and decreases the original FP32 objective by0.16478. Thus the rejected finite search did not establish absence of a useful larger step. This is an objective decrease at a fixed source, not a recovered reconstruction; image error remains approximately100%. The existing cap-start policy has an active queued CT reconstruction control; no global optimizer default is promoted from a one-step diagnostic. [Endpoint audit](high_rank/endpoint_report.md).

At the user request, fixed rank5096 is queued on the same MRI case after2048 and sufficient GPU1 memory, preserving the original random sketch columns and all non-rank algorithm settings. [Queue](high_rank/rank5096_queue.json).

## P-Flow fallback from failed DLR endpoints

Three selected endpoints from two samples were tested with paired reset-DLR, one safeguarded proxy then DLR, and proxy-only continuations. All used 250², noiseless measurements and five Euler steps. The proxy search starts at the displacement cap and halves, accepting only repeated actual-objective decrease beyond a declared roundoff margin. All proxy attempts were rejected; image errors remain 99.7683% for MRI sample2, 105.3665% for Gaussian-initialized CT sample0, and 37.1184% for measurement-initialized CT sample0. Resetting alone improves the Gaussian CT error by only 0.0046 percentage points. This tests the requested fallback at already-failed sources, not early or periodic estimator switching. No default change is supported; an accepted proxy followed by DLR recovery remains unobserved. [Protocol, verification and figures](proxy_recovery_report.md).

## Rank2048 completes without recovery

The same MRI250 sample1 fails at rank2048 after18 attempts, with 100.0237% image error versus D-Flow1.2187% and P-Flow2.3747%. Independent saved-array verification passes pairing, settings, caps, budget and complex metrics. A separate nonlinear CPU full-rank check recovers the exact discrete Euler Jacobian within4.36e-15 across16 cases, which supports the factor algebra in that limit but says nothing about adequacy of rank2048 in62500 dimensions. Increasing rank alone remains insufficient on this selected failure; finite search intervals and precision remain active reconstruction controls. [Updated high-rank report](high_rank/report.md).

## Exact gradients do not rescue the two tested late endpoints

Paired native D-Flow continuations from the same failed MRI250 sample2 and conditioned CT250 sample0 sources also reject two searches, preserving99.7683% and37.1184% image error. Reset-DLR controls agree. A cap-started DLR reset also fails on both. These results distinguish late rescue from the successful D/P runs from other starting sources. At the MRI endpoint, promoted FP64 finds only9.21e-8 objective decrease at displacement1.49e-7; FP32 finds none across the tested cap-halving grid. The identical-source CT FP64 audit finds only1.97e-9. Neither tiny decrease demonstrates reconstruction recovery. Approximation errors remain29.1%/54.7% against native FP32 gradients, but eliminating them at these endpoints is insufficient; local precision and earlier trajectory selection matter. [Exact continuation and precision report](exact_recovery_report.md), [cap-started controls](cap_started_pilot.md).
