# DLR failure investigation: fixed-source evidence

Snapshot: 2026-10-22, 05:20 local time. The two-pass MRI0 factor-refinement control completes2.629% versus6.639% at the same rank128 and source. Three shared-budget source-selection controls with one goal-oriented terminal projection are now running on GPU1. No general accuracy correction has been established.

## Verified findings

- The manual normalization path could evolve a different forward image and even reverse the approximate gradient. Native normalization now gives the same endpoint as the objective. Independent native-autodiff validation replaces the old comparison between two hand-coded derivative paths. This correction alone did not resolve the failed reconstructions.
- On selected 250-square CT0/MRI1/MRI0 cases, rank32 DLR with measurement-informed initialization ended at 37.12/12.31/9.34% image error. Matched random mixing ended at 109.49/99.94/100.00%. This is selected-case evidence for initialization, not a population result.
- With the identical cached measurement-informed source, rank128 DLR on MRI1 finished at **2.405%**, versus rank32's **12.306%**. CT0 did not improve: rank128 finished at 39.22%, versus rank32's 37.12%. Larger rank is not a universal correction.

## Why the rank32 MRI1 reconstruction stopped improving

All diagnostics below hold the model, source, endpoint, observation, and terminal measurement gradient fixed at the completed 12.306%-error reconstruction. The exact source-gradient norm is 0.029815. Independent products through the complete discrete five-step map agree with native backpropagation to about 1.4e-6 relative error.

| Approximation | Rank | Reuse saved V | Gradient relative error | Exact-gradient component outside V | Cosine with exact gradient |
| --- | ---: | --- | ---: | ---: | ---: |
| DLR | 32 | No | 101.56% | 95.89% | -0.0926 |
| DLR | 32 | Yes | 100.20% | 96.46% | -0.0148 |
| DLR | 128 | No | 100.21% | 95.03% | 0.0562 |
| DLR | 128 | Yes | 99.02% | 95.32% | 0.1407 |
| DLR with terminal-gradient sketch column | 32 | Yes | 96.72% | 93.95% | 0.2555 |
| DLR with terminal-gradient sketch column | 128 | Yes | 95.24% | 92.59% | 0.3053 |

The current right subspace excludes most of the exact gradient at this source. Correcting only the coefficients within that subspace cannot recover the excluded component. These measurements establish a local limitation, not the cause of every failed trajectory.

The terminal-gradient seed replaces one existing random sketch column. It retains the same sketch width, rank, analytical alpha=1-t, KSL evolution, and objective. No additional Jacobian-product columns are introduced. The experimental evaluator currently computes the terminal measurement gradient a second time; this could be removed if reconstruction results justify adoption.

## Controls that did not provide a correction

- **More accurate S/L integration:** exact frozen S/L substeps, with the existing Euler K-step and Euler-target recompression, passed a fine-RK4 matrix check but did not materially improve the real gradient. Rank32 with reused V still had 100.14% gradient error. This remains an isolated experiment, not production behavior.
- **Scalar identity correction:** even fitting the best scalar using the exact gradient only reduces reused rank32 error from 100.20% to 97.77%; the fitted scalar is negative. An omitted positive identity baseline does not explain this case well. Production alpha remains unchanged.
- **Terminal randomized SVD:** three power iterations yield rank32/rank128 gradient errors of 95.36/91.88%, but the references are **uncertified**: maximum singular-triplet residual is 0.00817 times the leading singular value, and the last gradient changes are 4.96/10.84% of the exact-gradient norm. These fail the earlier 1e-3 and 1% criteria. Do not call these the true truncated-SVD gradients or use them to rule out integration error.

## Reconstruction follow-up

The exact-suffix diagnostic motivates a rank-preserving goal-oriented terminal projection: include the known terminal loss gradient in the last left basis and factor the transpose Euler target on that basis. This exactly retains the target's action on that covector in real arithmetic while keeping rank r; it changes the last projection criterion and is not the unmodified KSL update. Rank32 MRI1 fixed-source gradient error drops100.197% to17.834%; CT0 changes54.691% to53.578%. The CT reconstruction completes36.452%, versus37.118% previously; the paired MRI1 reconstruction completes1.607%, versus12.306% with ordinary rank32. Its Gaussian-source control still fails at100.018%. Four goal-oriented projections giveCT0 26.097% image error with only0.005691% measurement error, andCT7 34.805% image error; this preserves rank but requires additional native adjoints and changes the projection criterion. The separate rank32 one-step exact-suffix hybrid MRI4 completes1.695%, and the rank128 frozen-increment CT7 control completes20.241%. These selected-case improvements do not establish a general correction. [Derivation, validation, and status](full_map_audit/task_terminal_report.md).

An observer-based prefix diagnostic localizes approximation loss along the unchanged five-step generator. At failedMRI4, reused rank32 DLR has69.99% gradient error; using an exact native adjoint for the last Euler step reduces it to40.08%, and using four exact suffix steps reduces it to1.55%. Rank128 gives65.10/32.49/0.966%. The all-exact suffix agrees with independent backpropagation. The one-step-suffix rank32 MRI4 reconstruction completes1.695% under explicit hybrid metadata, versus94.224% for the ordinary rank32 paired control. This is **not** terminal rank-r DLR: it applies the shifted-low-rank prefix Jacobian to an exact suffix pullback, so the final implicit Jacobian need not have rank r. Seven tests pass, including a dense low-rank chain check and verified early termination of prefix factor evolution. See [the suffix diagnostic](full_map_audit/suffix_report.md).

A further fixed-source control separates the frozen discrete Euler target from subspace-iteration accuracy. On250CT7 at rank128 with the saved right basis, gradient error falls from30.69% with the original update to5.18% with the frozen-increment update; a second bilateral subspace iteration gives5.70%. On conditioned250MRI0 the corresponding errors are96.16/95.82/88.47%. The latter still has88.01% subspace error. The dense full-rank Euler identity, nonincreasing target projection error under refinement, and independent real-model product checks pass. Full CT7 increment and paired MRI0 original/refined reconstruction controls are running. This remains experimental: two passes increase derivative-column work from648 to1160 per direction at rank128/five steps. See [the refinement diagnostic table](full_map_audit/refinement_report.md).

The terminal-gradient sketch change was compared with an identical cold-optimizer restart at the saved failed sources. Both arms reuse the saved V basis and receive the same additional optimizer budget. The unchanged MRI1 restart makes no progress (12.306%); the seeded restart finishes at11.018%. CT0 and MRI0 make no progress with either restart, remaining37.118% and9.336%. This is not sufficient evidence to promote the sketch change.

The four-case rank-adaptive pilot is complete. MRI2 fails at every tested rank choice; CT20 adaptive gives3.742%, versus fixed32's3.946% and fixed128's3.999%. MRI4 adaptive finishes at1.743%, versus fixed32's94.224% and fixed128's31.235%; however, fixed128 using the canonical adaptive-code path also reaches1.597%, so changing rank is not required for that rescue. CT7 adaptive worsens to97.082%, versus46.905% for fixed32 and34.280% for fixed128. Additional adaptive runs on CT0/MRI1/MRI0 Gaussian starts finish at101.799/100.024/99.767%. Adaptation is not a universal correction, and close differences need repeated trials.

## Basis reuse and L-BFGS consistency

At the failed rank128 MRI4 endpoint, hold the source, model, measurements, and flow grid fixed. Repeatedly returning the terminal right basis to the next initialization changes the estimated gradient by **8.12–19.00% of its previous norm** per refresh. Repeating the fixed-seed unseeded sketch changes it by only **0.00253%**. Every endpoint is identical to the native Euler endpoint. Evidence: `full_map_audit/mri4_fixed_source_basis_drift_verified.json`.

Write the approximate gradient as $\hat g(z;V_{\mathrm{seed}})$, where the second argument is the initialization basis, not the terminal factor. A curvature difference with changing seeds contains two effects:

$$
\begin{aligned}
\hat g(z_{k+1};V_k)-\hat g(z_k;V_{k-1})
={}&[\hat g(z_{k+1};V_k)-\hat g(z_k;V_k)]\\
 &+[\hat g(z_k;V_k)-\hat g(z_k;V_{k-1})].
\end{aligned}
$$

The second term changes the gradient without moving the source. Thus L-BFGS can interpret approximation drift as objective curvature. A fixed-seed sketch removes this particular state dependence in exact arithmetic; it does not make the approximate field an exact or necessarily conservative gradient.

Disabling reuse in the paired rank128 MRI4 reconstruction finishes at **2.102%** after1000 attempts, versus31.235% after370 attempts with reuse (both had a1000-attempt cap and the same stopping safeguards). This supports the mechanism as a contributor. It does not prove a universal fix: fresh-sketch MRI1 and CT0 still finish at100.024% and91.052%. The rank32 MRI4 control finishes at73.666%; rank128 CT20 finishes at3.492%, versus3.999% with reuse. These close CT differences require repeats. On the identical cached restart pool for MRI2, disabling reuse worsens DLR from3.155% to14.723%, so it is not promoted as a default.

Bounded recovery after a failed search also does not rescue these endpoints. Three alternative fixed-seed sketches, with cleared L-BFGS curvature and unchanged current source, leave MRI4 at31.235%, MRI2 at14.723%, and CT0 at91.045% (previously91.052%). Every attempt counts toward the original1000-attempt cap, including150 discarded initialization probes for MRI2. Totals are383/232/44 respectively; the unchanged two-failed-search safeguard terminates each trial. Saved histories, objective nonincrease, cumulative counters, and budget accounting pass checks. This is a negative control, not a production change. Evidence: `sketch_recovery_mri4/`, `sketch_recovery_mri2/`, and `sketch_recovery_ct0/`.

## Coarse generator and source-selection controls

The saved117k unconditional evaluations show substantial five-step sampling collapse. At resolutions64/96/128/160/200/250, the numbers with no detected foreground are4/14/4/39/65/71 out of128; at100 steps they are0/0/0/1/5/5. Detection uses8-connected components above0.5 with at least4 pixels; this is not geometric certification. See `sampling_quality_summary.json`.

On32 paired250-square sources, uniform5 produces24 empty samples, uniform10/20 each2, and the proposed nonuniform5 grids31/32. Uniform5 reproduces the saved images to5.37e-6 relative error. Nevertheless, a ten-step Gaussian-start MRI1 test still fails for D-Flow and DLR while P-Flow reaches1.780%. Sampling refinement alone is not an inverse correction.

With the same cached measurement-informed MRI1 source, DLR gives12.306% at5 steps/rank32,2.405% at5 steps/rank128, and **2.354% at20 steps/rank32**. The latter two use648 JVP columns and648 VJP columns per factor refresh under the zero-enrichment profile, including eight startup oversampling columns; wall time is not established by this count. Changing the step count also changes the discrete generator, so this does not isolate factor integration error. D-Flow/P-Flow at20 steps finish0.943/8.547% on that source. The corresponding CT control finishes at60.271% for DLR, versus37.118% at5 steps/rank32 and39.215% at5 steps/rank128; D-Flow/P-Flow at20 steps finish54.627/212.706%. This is further evidence against a universal refinement default.

A separate restart test gives all three methods identical cached pools of four sources, probes each briefly, and selects by measurement objective only. All probes count toward the shared1000-attempt budget. Completed reused-basis DLR results are3.155% on250MRI2 and8.298% on250CT7. The fixed-sketch200CT1 restart finishes at30.332%, versus99.781% for its native rank32 Gaussian control; this comparison changes both initialization and rank. Full method results and budget/source checks are in [the restart report](multistart_report/report.md). These selected cases do not establish population performance.

## CT precision and optimizer conditioning

A saved-direction audit tests both signs of the actual cached optimizer gradient, without refreshing the DLR basis. At conditioned MRI1 (12.306% image error), its cosine with the exact gradient is-0.01792; neither sign yields an accepted FP32 decrease over30 halved displacements. The exact negative gradient accepts the full2.5 displacement and lowers the loss by0.070996, versus a linear prediction of0.074537. This isolates a poor DLR direction at this source. At fixed-sketch restarted MRI2 (14.723% error), the DLR cosine is0.91085, yet even the exact direction only accepts displacements around6e-7, with loss changes around7e-7. Fresh-sketch CT0 (91.052% error) has cosine0.91362 and similarly tiny accepted changes. Replacing approximate Armijo slopes by exact directional derivatives does not produce a useful new accepted DLR step in these controls. Do not promote a sign flip or slope-only correction. Evidence: `full_map_audit/{mri1,mri2,ct0}_saved_direction.json`.

At the conditioned CT0 endpoint, reused rank32 DLR has cosine0.837 with the exact gradient. Increasing rank to512 only improves it to0.857. Both the exact and DLR negative-gradient directions fail FP32 Armijo tests over30 displacement sizes, from the cap2.5 down to about5e-9. At small displacements the intended decrease is below the forward objective's floating-point variation.

Sixteen repeated native evaluations at separate saved CT0/MRI4 endpoints give bitwise-identical forward images and identical objective values. Their maximum relative exact-gradient changes are2.89e-5 and3.28e-6, respectively. These are far below the measured DLR approximation errors, although tiny differences can still change long optimization trajectories. This distinguishes gradient repeatability from finite-displacement roundoff sensitivity; it does not prove deterministic CT adjoints. Evidence: `full_map_audit/{ct0,mri4}_native_repeats.json`.

A bounded FP64 control preserves the checkpoint values, recorded CT grid, and observed data while promoting their arithmetic. It finds a decrease of only1.97e-9 at source displacement1.19e-6. Thus a useful descent direction can be too numerically delicate for the FP32 search to realize at this source. The tiny decrease does **not** demonstrate a precision-only reconstruction cure; full FP64 reconstruction is not being promoted.

An isolated optimizer control retries the older half of the L-BFGS history after a failed search, before falling back to no history. It keeps the same gradient estimators, objective, strict Armijo acceptance, displacement cap, and total attempt limit. Every rejected search still counts; halving guarantees finite history recovery. All three methods receive the same cached conditioned CT0 source and optimizer change. Completed image errors are D-Flow6.315%, P-Flow62.881%, DLR38.978%: no DLR rescue. D-Flow performed zero history retries yet differs from its earlier same-source runs, confirming that matching source hashes alone does not eliminate numerical trajectory variability. These selected CT reruns do not support fine causal accuracy comparisons without repeats or more deterministic execution. Evidence: `history_fallback_ct0/`.

## Reproducibility

A further initialization screen integrates backward Euler from a scalar least-squares-scaled measurement adjoint, then evaluates every candidate with the unchanged five-step forward generator. Backward Euler is not the exact inverse of that discrete map. Candidate counts are5/20/100 backward steps plus the original source; selection uses only measurement loss. Matched random image controls preserve the adjoint image's mean and centered norm. All14 cached selections over seven cases pass source-hash, objective-selection, and unchanged-data checks. All three MRI measurement cases select20 backward steps, with initial image errors58.867/61.233/26.276%; all four CT measurement cases retain the original source. Completed paired MRI1 results are D4.965/P9.985/DLR10.467%; MRI2 results are D6.309/P15.830/DLR14.912%. DLR is worse than the earlier simple-mixture results2.405/3.155%; this initializer is not promoted. The matched-random MRI2 control is still running. Evidence: `reverse_initialization/`, `reverse_mri1/`, and `reverse_mri2/`.

- Fixed-source code: `experiments/separated_shapes_117k/gradient_audit.py`.
- Independent product, frozen-S/L, and optimizer recovery checks: `tests/test_discrete_flow_products.py` (four tests pass, including frozen-increment KSL and the full-rank Euler identity).
- Raw evidence: `full_map_audit/` in this directory.
- Paired restart controls: `terminal_seed_restart/`, `terminal_seed_restart_ct0/`, `terminal_seed_restart_mri0/`.
- Adaptive pilot: [rank paths and current saved metrics](adaptive_pilot/report.md).

The production sweep remains held at its source-review guard. Corrected experiments are separate from the original results. The broader accuracy objective is not yet complete.

## Combining source selection and goal-oriented projection

The isolated multistart runner now supports an explicit goal-projection option, defaulting to the original update. New rank128 controls use exactly the cached pools and settings of250CT7,250MRI2 and200CT1. All four candidate probes count toward1000 attempts; selection uses measurement loss only, and continuation preserves optimizer history and the right basis. The200CT1 comparison retains its existing no-reuse setting. Completed D/P controls are retained from the identical pools. This tests a combined correction without changing rank, source pools or optimizer budget. Status and checks are in [the combined report](multistart_goal_report/report.md); [protocol verification](multistart_goal_report/paired_protocols.json) establishes equality of settings other than the explicitly recorded gradient variant.

## Backend coverage and a rejected source correction, 05:34

The corrected accelerated backend passes independent native-model autodiff checks for all six117k models at64/96/128/160/200/250. Each audit uses one seeded source, all five Euler states, three JVP/VJP test columns, and rank32 fresh/seeded factor-gradient actions. Maximum local relative error is1.16e-7; maximum full-factor gradient discrepancy is3.86e-5. Endpoint discrepancies are zero in these checks. This validates execution equivalence at the tested sources, not DLR approximation accuracy. The diagnostic CLI now defaults to native normalization instead of silently selecting legacy manual statistics at larger resolutions. [Full audit](backend_all_resolutions_report.md).

Failed optimized sources sometimes have a large negative mean, but subtracting25/50/100% of that mean at eight fixed saved reconstructions worsens both measurement objective and image error in every check. This rejects direct post-hoc centering as a correction here; no source constraint is adopted. [Counterfactual report](full_map_audit/source_centering_report.md).
