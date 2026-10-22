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
