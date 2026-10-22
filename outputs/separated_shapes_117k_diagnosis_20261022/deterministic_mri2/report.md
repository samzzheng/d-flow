# Deterministic execution control — 2026-10-22

Deterministic cuDNN makes the two tested MRI reconstruction trajectories reproduce exactly, but does not recover the image. This is a numerical repeatability improvement, not an accuracy correction.

All runs use the same original source, model, MRI mask, observation and 40-column initialization sketch for separated-shapes MRI250 sample2: noiseless, five Euler steps, rank32, analytical alpha, zero enrichment/probing, FP32 with TF32 disabled. Previous-V reuse, L-BFGS/geometric Armijo, displacement cap2.5 and the original1000-attempt/two-rejection safeguard are unchanged. Variants ran sequentially on shared GPU1. This initialization sketch differs from the2056-column sketch used in the separate high-rank study; only within-study comparisons are paired here.

| Execution | Attempts completed | Image relative L2 (%) | Measurement residual (%) |
|---|---:|---:|---:|
| Default cuDNN | 96 | 99.688210 | 99.561919 |
| Deterministic A | 145 | 99.689384 | 99.558357 |
| Deterministic B | 145 | 99.689384 | 99.558357 |

Each run ends under the existing rejection safeguard. A/B match all146 saved non-timing history records, including the initial record, and their final sources are bitwise equal. More attempts did not yield meaningful accuracy improvement. Existing paired D/P reference images appear in the panels; those reference runs are not deterministic-execution controls.

At a frozen source and sketch, four recomputed DLR gradients varied by up to1.78e-5 relatively with default cuDNN. All four deterministic repetitions were bitwise equal. The canonical terminal image was cached and held fixed within each calibration; endpoint equality in that diagnostic is therefore by construction. The first gradient changes by1.10e-5 between execution modes. Calibration evaluations are recorded separately from optimizer attempts. Two separate deterministic processes then reproduce the complete reconstruction, a stronger check than fixed-source repetition alone.

Saved-array verification checks source/model/operator/seeds/sketch pairing, identical configuration except the execution flag, original budgets/caps, monotone objectives, and image/complex residual metrics. CPU fallback tests and script compilation pass. The shared four-row comparison panel was inspected. Raw tensors remain local and are excluded from Git.

This is one MRI sample, not proof of repeatability for every model/operator. CT-adjoint nondeterminism remains a separate issue. Production defaults are unchanged; deterministic execution can support subsequent isolated MRI comparisons.

- [Calibration verification](calibration_verification.json)
- [Reconstruction verification](reconstruction_verification.json)
- [Completed queue and commands](reconstruction_queue.json)
- [Paired comparison panel](reconstruction/deterministic_a/r250_shapes_mri4/sample_002/rank32/paired_methods.png)
