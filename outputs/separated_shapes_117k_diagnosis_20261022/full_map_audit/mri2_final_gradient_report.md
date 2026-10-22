# Final MRI2 gradient audit

At the completed FP64 recovery source (6.598% image error), compare adaptive floors 64 and 128 with the same maximum rank 128, five Euler steps, analytical alpha, and zero enrichment. This is a fixed-source diagnostic, not a reconstruction comparison.

| Rank floor | Basis | Gradient error | Missing-subspace error | Coefficient error | Cosine |
|---:|---|---:|---:|---:|---:|
| 64 | Fresh | 33.8077% | 32.8572% | 7.9603% | 0.942109 |
| 64 | Reused V | 30.8869% | 30.0038% | 7.3330% | 0.951105 |
| 128 | Fresh | 33.8000% | 32.3698% | 9.7283% | 0.942137 |
| 128 | Reused V | 30.8774% | 29.6385% | 8.6584% | 0.951136 |

Errors are normalized by the exact gradient norm. The squared total error equals squared subspace error plus squared coefficient error. Source hash, objective, unchanged endpoint, native derivative comparison, and adjoint consistency all pass. The exact gradient norm is 1.4244, so the endpoint is not stationary.

With reused V, doubling the floor changes gradient error only from 30.8869% to 30.8774%. Retaining more terminal directions does not repair the dominant missing-subspace error here. This supports examining initialization and subspace evolution rather than assuming the terminal rank is the sole problem. Reuse modestly improves alignment relative to the fresh sketch in this case; it is not a general guarantee.

[Floor 64 evidence](mri2_final_fp64.json) · [Floor 128 evidence](mri2_final_fp64_floor128.json)
