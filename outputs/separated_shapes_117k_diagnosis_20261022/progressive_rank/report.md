# High-start progressive rank pilot

Status: reconstruction controls running on GPU 1; no accuracy conclusion yet.

Two selected noiseless cases: 250x250 CT10 sample 7 and MRI4x sample 4, using the trained 117k separated-shapes model. All controls share cached source, truth, observations, model, seeds, previous-V reuse, five Euler steps and the established LBFGS/geometric Armijo settings (up to 1000 attempts with existing failed-search stopping). FP32, TF32 disabled; corrected native normalization statistics, primal and derivative actions.

Each gradient starts with a rank-256 sketch plus the existing eight oversampling columns. The relative singular-value cutoff is 0.001 and the floor is 32. Compare unrestricted truncation, retain-256-at-initialization only, and retain-256 then discard at most half the current rank per flow step. Rank cannot regrow within a pass; the next gradient starts high again. This criterion measures the approximated residual core, not certified true Jacobian or gradient error. Production defaults are unchanged.

## Initial fixed-source diagnostics

| Case | Policy | Ranks after each flow step | Gradient relative error | Gradient cosine |
|---|---|---|---|---|
| r250_shapes_ct10/sample_007 | unrestricted256 | [256, 256, 254, 138, 32] | 35.8723% | 0.933512 |
| r250_shapes_ct10/sample_007 | retain_initial256 | [256, 256, 254, 138, 32] | 35.8723% | 0.933512 |
| r250_shapes_ct10/sample_007 | gradual256 | [256, 256, 254, 138, 69] | 35.8853% | 0.933461 |
| r250_shapes_mri4/sample_004 | unrestricted256 | [256, 256, 255, 251, 82] | 63.6481% | 0.773435 |
| r250_shapes_mri4/sample_004 | retain_initial256 | [256, 256, 255, 251, 82] | 63.6481% | 0.773437 |
| r250_shapes_mri4/sample_004 | gradual256 | [256, 256, 255, 251, 126] | 63.6430% | 0.773481 |

These initial gradients are very similar; this does not establish a reconstruction advantage. Both endpoints used to evaluate the terminal losses match the canonical Euler endpoint. Three CPU tests pass: rank bounds and unchanged primal trajectory, full-rank agreement with a dense Euler Jacobian, and restoration of the original implementation after an error. The rank bound test also verifies equivalence to the existing no-truncation adaptive path when retention is 100%.

Each sample runs its three variants sequentially. Rank paths, optimizer states, metrics and model/code/source hashes are saved alongside each run. Saved histories and summaries provide the current state; this report is a launch snapshot.

## Graph-cache recovery

Both initial workers terminated with CUDA out-of-memory errors at their 20% per-process memory limits. The GPU itself still had free memory. Variable adaptive ranks created product graphs for many remainder-column widths; private graph pools accounted for approximately7.56GiB in each failed process. Saved optimizer checkpoints were preserved.

The experiment now uses an eight-entry least-recently-used product-graph cache. Eviction synchronizes outstanding work and releases only an old product graph; input widths, products, shared primal, factor equations and optimizer remain unchanged. An actual GPU1 test varies widths and states, forces eviction, checks retained output ownership, and compares with the unbounded implementation. Maximum relative product difference is2.61e-7 in FP32, below the2e-6 test threshold. Independent captures are not guaranteed bitwise identical, so this is numerical equivalence rather than a claim of identical floating-point trajectories. The three dense CPU rank-policy tests also pass.

Both workers resumed from their latest saved checkpoints with unchanged protocols, histories and attempt budgets. The reviewed script/backend hashes are recorded in each case's `resume_cache_review.json`; `before_cache_resume.pt` preserves the pre-resume state for history verification. The initial diagnostic table above is the original launch snapshot; repeated diagnostic gradient-error/cosine values agree to displayed precision apart from rounding in the last digits. No production backend or default changed.
