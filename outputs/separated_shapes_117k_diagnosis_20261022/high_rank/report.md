# Fixed-rank 1024/2048 pilot

Noiseless separated-shapes MRI4x, 250 squared, sample 1; same model, five Euler steps, analytical alpha, zero enrichment. Each source is fixed across ranks, with nested random Rademacher sketch columns. The failed source is from the corrected baseline; the initial source is the original paired Gaussian source.

Gradient errors below are relative to native autodiff of the FP32 five-step map, not image errors or certified FP64 Jacobian errors.

| Rank | Initial gradient error | Failed-source gradient error | Initial cosine | Failed cosine | Initial cold refresh | Peak allocated |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 63.04% | 45.59% | 0.7888 | 0.8946 | 1.77 s | 0.93 GiB |
| 256 | 46.38% | 29.24% | 0.8872 | 0.9565 | 6.93 s | 1.75 GiB |
| 1024 | 38.06% | 23.52% | 0.9252 | 0.9720 | 25.84 s | 5.16 GiB |
| 2048 | 35.79% | 21.84% | 0.9341 | 0.9759 | 55.82 s | 9.79 GiB |

Both high ranks fit within a 30% per-process memory cap on GPU 1. Times are single cold refreshes under concurrent GPU load, include graph capture and gradient application, and are not steady-state speed comparisons. Memory includes the diagnostic sketch and reference arrays, not a full optimizer run.

Increasing rank improves these fixed-source gradients, with diminishing returns above 1024. At rank 2048, the omitted-subspace errors remain 30.95% / 20.98% (initial / failed). These observations do not establish reconstruction improvement or a universal sufficient rank.

Validation: terminal alpha exactly zero; all endpoints bitwise equal to the native Euler reference; finite gradients; requested fixed rank retained; orthogonal gradient-error decomposition checked. Rank32 products independently repeated with native autodiff (relative backend discrepancies 5.34e-06 / 0.000512). Higher-rank native repetition has not been performed. The first failed-source calibration exceeded 0.001 and terminated; a fresh-process repeat passed. The original failure is retained in first_validation_failure.json; floating-point sensitivity remains unresolved. Earlier precision diagnostics show the failed-source FP32 exact gradient itself can be sensitive to precision.

Paired reconstructions at ranks 32, 1024 and 2048 are launched sequentially for this sample on GPU 1. Original source, model, measurements, optimizer, previous-V reuse, 1000-attempt limit and early-rejection safeguards are shared. The first sketch uses the same nested random columns as this audit; no measurement-informed initialization. The rank32 control completed after16 attempts at100.024% image error and100.011% measurement residual under the original rejected-search stopping rule. Rank1024 is active; rank2048 is queued next. There is no completed high-rank reconstruction result yet.

[Initial audit](mri1_initial.json) · [Failed-source audit](mri1_failed.json) · [Reconstruction launch](reconstruction_launch.json)
