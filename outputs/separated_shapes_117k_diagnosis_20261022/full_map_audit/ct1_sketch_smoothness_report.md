# Paired initialization sketch stability control

At the completed prefix3 CT200 sample1 source, compare fixed rank128 initialization with zero versus eight oversampling columns. Both use the same normalized Gaussian136-column sketch, taking nested prefixes, FP64, five Euler steps and three-step residual initialization. Perturb the source along the same saved negative normalized gradient. These are fixed-point diagnostics, not new reconstruction trials.

| Oversampling | Source displacement | Exact gradient change / initial norm | Approximate gradient change / initial exact norm | Change in approximation error / exact gradient change |
|---|---:|---:|---:|---:|
| 0 | 1e-05 | 7.733% | 7.673% | 1.357% |
| 0 | 0.001 | 249.021% | 247.094% | 1.356% |
| 8 | 1e-05 | 7.733% | 7.680% | 1.299% |
| 8 | 0.001 | 249.021% | 247.316% | 1.298% |

Removing oversampling does not improve this measured stability: both estimators closely track the exact gradient changes, and oversampling8 has slightly smaller errors. At displacement1e-5, even the exact gradient changes7.73%; at1e-3 it changes249.02%. This indicates strong directional sensitivity of the actual discrete objective at this source, rather than evidence of extra roughness introduced by the sketch truncation. It is one source and one direction, not a global smoothness claim. No oversampling change is promoted.

All six native/accelerated comparisons pass with identical endpoints. Full sketch hashes and paired source hashes match. Exact gradients agree to3.58e-16 relative; the initial bitwise equality assertion failed at roundoff, so the verification records measured differences rather than claiming bitwise equality. [Verification](ct1_sketch_smoothness_verification.json).
