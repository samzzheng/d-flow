# Rank-adaptive DLR pilot

64px, two/three circles, CT10/MRI4, samples0–1. 100 Euler steps, FP32 without TF32, noiseless observations, up to100 optimizer attempts. Pure fixed8/fixed32 controls use the same batched-graph backend as adaptation.

Rank is fixed during each factor evolution and optimizer step. Every five attempts and after a failed search, compare same-source gradients along4/8/16/32/64. The two tolerances are5% and1%; shrinking requires twice as close agreement. Clear LBFGS pairs and cached gradient only when the rank changes. Agreement is a heuristic; shared approximation bias can fool it. Rank64 is marked unvalidated if no comparison satisfies tolerance.

| Case | Variant | Attempts | Final image error % | Measurement residual % | Rank min–max | Checks at unresolved cap | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r64_n2_ct10/sample_000 | fixed8 | 100 | 2.990 | 0.945 | 8–8 | 0 | step_limit |
| r64_n2_ct10/sample_000 | fixed32 | 100 | 1.198 | 0.284 | 32–32 | 0 | step_limit |
| r64_n2_ct10/sample_000 | adaptive05 | 100 | 1.192 | 0.323 | 8–64 | 1 | step_limit |
| r64_n2_ct10/sample_000 | adaptive01 | 100 | 1.111 | 0.234 | 16–64 | 16 | step_limit |
| r64_n2_ct10/sample_001 | fixed8 | 100 | 3.638 | 1.159 | 8–8 | 0 | step_limit |
| r64_n2_ct10/sample_001 | fixed32 | 100 | 24.622 | 8.454 | 32–32 | 0 | step_limit |
| r64_n2_ct10/sample_001 | adaptive05 | 100 | 3.562 | 1.350 | 8–64 | 1 | step_limit |
| r64_n2_ct10/sample_001 | adaptive01 | 100 | 42.380 | 20.321 | 32–64 | 9 | step_limit |
| r64_n2_mri4/sample_000 | fixed8 | 100 | 1.760 | 1.440 | 8–8 | 0 | step_limit |
| r64_n2_mri4/sample_000 | fixed32 | 100 | 52.128 | 48.232 | 32–32 | 0 | step_limit |
| r64_n2_mri4/sample_000 | adaptive05 | 21 | 92.333 | 84.111 | 8–32 | 0 | stalled |
| r64_n2_mri4/sample_000 | adaptive01 | 29 | 89.917 | 82.091 | 16–64 | 2 | stalled |
| r64_n2_mri4/sample_001 | fixed8 | 100 | 1.128 | 0.817 | 8–8 | 0 | step_limit |
| r64_n2_mri4/sample_001 | fixed32 | 100 | 7.124 | 5.394 | 32–32 | 0 | step_limit |
| r64_n2_mri4/sample_001 | adaptive05 | 100 | 3.388 | 2.857 | 8–32 | 0 | step_limit |
| r64_n2_mri4/sample_001 | adaptive01 | 100 | 4.496 | 4.000 | 64–64 | 20 | step_limit |
| r64_n3_ct10/sample_000 | fixed8 | 100 | 61.976 | 34.868 | 8–8 | 0 | step_limit |
| r64_n3_ct10/sample_000 | fixed32 | 100 | 65.788 | 35.700 | 32–32 | 0 | step_limit |
| r64_n3_ct10/sample_000 | adaptive05 | 100 | 42.994 | 20.931 | 8–32 | 0 | step_limit |
| r64_n3_ct10/sample_000 | adaptive01 | 100 | 13.828 | 4.942 | 32–64 | 7 | step_limit |
| r64_n3_ct10/sample_001 | fixed8 | 65 | 3.937 | 1.857 | 8–8 | 0 | stalled |
| r64_n3_ct10/sample_001 | fixed32 | 100 | 0.994 | 0.173 | 32–32 | 0 | step_limit |
| r64_n3_ct10/sample_001 | adaptive05 | 100 | 1.220 | 0.233 | 16–64 | 1 | step_limit |
| r64_n3_ct10/sample_001 | adaptive01 | 100 | 1.041 | 0.180 | 64–64 | 20 | step_limit |
| r64_n3_mri4/sample_000 | fixed8 | 100 | 50.832 | 46.948 | 8–8 | 0 | step_limit |
| r64_n3_mri4/sample_000 | fixed32 | 100 | 14.211 | 10.429 | 32–32 | 0 | step_limit |
| r64_n3_mri4/sample_000 | adaptive05 | 21 | 86.365 | 81.245 | 16–16 | 0 | stalled |
| r64_n3_mri4/sample_000 | adaptive01 | 100 | 2.277 | 1.647 | 32–64 | 18 | step_limit |
| r64_n3_mri4/sample_001 | fixed8 | 68 | 3.772 | 3.417 | 8–8 | 0 | stalled |
| r64_n3_mri4/sample_001 | fixed32 | 100 | 1.003 | 0.539 | 32–32 | 0 | step_limit |
| r64_n3_mri4/sample_001 | adaptive05 | 100 | 1.243 | 0.845 | 16–32 | 0 | step_limit |
| r64_n3_mri4/sample_001 | adaptive01 | 100 | 1.072 | 0.576 | 64–64 | 20 | step_limit |

Completed four-way paired cases: 8/8. Running rows are not final comparisons.

Errors use final saved checkpoints, not best-image selection. Histories record rank changes, curvature use, caps, failed searches, and factor-evaluation counts. No speed advantage or exact-gradient certification is implied by this pilot.
