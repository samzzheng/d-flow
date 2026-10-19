# Separated-shapes Jacobian rank pilot

Sample 0, noiseless CT10/MRI4×; native five-step Euler Jacobian at initial and final D-Flow sources. FP32, TF32 off. Energy ranks use squared singular values plus independently estimated omitted energy; they are not algebraic ranks. Unresolved cutoffs remain unresolved. Rank-truncated gradients are independent SVD references, not the evolved DLR approximation.

| Case / source | 99% rank | 99.9% rank | 99.99% rank | Rank-8 energy (%) | Rank-8 gradient error (%) | Gradient reference converged | Spectrum status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| r128_ct10_final | 1 | 1 | 1 | 99.999964 | 43.79 | True | resolved_estimate |
| r128_ct10_initial | unresolved | unresolved | unresolved | 53.741095 | 3.01 | True | unresolved_at_budget |
| r128_mri4_final | 15 | unresolved | unresolved | 93.413468 | 27.89 | True | unresolved_at_budget |
| r128_mri4_initial | unresolved | unresolved | unresolved | 53.741095 | 18.63 | True | unresolved_at_budget |
| r64_ct10_final | 48 | unresolved | unresolved | 83.263334 | 70.39 | True | unresolved_at_budget |
| r64_ct10_initial | 1 | 2 | 4 | 99.997270 | 0.18 | True | resolved_estimate |
| r64_mri4_final | 24 | unresolved | unresolved | 84.413464 | 30.51 | True | unresolved_at_budget |
| r64_mri4_initial | 1 | 2 | 4 | 99.997270 | 0.17 | True | resolved_estimate |

Only one paired sample per resolution: this is a diagnostic pilot, not a dataset-wide rank estimate. No intermediate optimization sources were saved by the original sweep. All sources used here are snapshotted alongside the results.

The overall spectral status requires the stricter 99.99% cutoff to resolve; a 99% rank estimate may still be available. Tail sampling uncertainty and individual triplet residuals are in spectrum.json. Large relative gradient errors near convergence can correspond to tiny absolute differences; exact gradient norms are saved in gradient.json. Neither spectral truncation error nor a single final-source diagnostic proves the cause of a reconstruction failure.
