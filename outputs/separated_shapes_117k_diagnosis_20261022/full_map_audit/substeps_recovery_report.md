# Factor substeps and recovery-scale controls

Updated 2026-10-22 06:40 local time. These bounded GPU1 experiments are complete. No production default changed.

## Factor substeps

The forward map stays at five Euler steps. For each factor interval, hold F0=A(alpha I+Y0)+I fixed and resolve the projected matrix equation dY/ds=P_Y F0 with four KSL substeps. At full rank this still gives Y0+hF0; it does not replace the forward Euler map by a finer flow. At low rank it resolves a different approximation to that same frozen increment. This costs four rank-column JVP/VJP blocks per evolved factor interval, versus one.

CPU tests compare against an independent dense projected-ODE RK4 reference, check convergence under subdivision, and verify the full-rank Euler identity. The unchanged one-substep limit matches the existing frozen-increment update.

| Saved source | Rank | One-substep gradient error | Four-substep gradient error | Four-substep subspace error |
| --- | ---: | ---: | ---: | ---: |
| ct7 | 32 | 20.188% | 12.771% | 11.680% |
| ct7 | 128 | 5.177% | 6.191% | 4.892% |
| mri1 | 32 | 100.203% | 99.367% | 97.840% |
| mri1 | 128 | 98.902% | 98.246% | 96.700% |

Rows use the identical saved right basis, source, endpoint and terminal gradient within each case. CT rank32 improves, CT rank128 slightly worsens, and MRI still loses almost all of the gradient. No full reconstruction sweep is justified by this control.

## Bounded recovery scale

At the completed200CT1 increment endpoint (25.790% image error), the stored gradient has cosine0.1552 with the exact gradient. The rank128 right subspace excludes97.981% of the exact gradient norm. Exact descent accepts the full displacement2.0 and lowers the objective by0.0017085. The same cached DLR direction accepts displacement0.125, but the history-free optimizer proposes only0.00018935 at learning rate1. A halving-only search cannot enlarge that proposal.

The isolated recovery starts the existing history-free direction at the displacement cap only after a rejected search. Ordinary first-attempt scaling, cached gradients, right bases, Armijo checks, actual-displacement caps, failure counters and the original shared1000-attempt budget are preserved. Each arm resumes the same saved optimizer and source; no fresh initialization or additional attempt budget is granted.

| Case | Recovery | Image error | Measurement residual | Total attempts including prior starts | Scaled recoveries |
| --- | --- | ---: | ---: | ---: | ---: |
| 200ct1 | unchanged | 25.790% | 13.221% | 246 | 0 |
| 200ct1 | scaled | 25.896% | 12.893% | 257 | 3 |
| mri2 | unchanged | 3.179% | 1.649% | 621 | 0 |
| mri2 | scaled | 3.179% | 1.649% | 621 | 1 |

All13 CPU tests pass, including an FP32 case where the raw fallback step is not representable but the scaled direction produces a valid strictly decreasing capped step. Completed GPU outputs pass independent saved-array metric checks, history-prefix checks, monotonic measurement objectives and total-budget accounting.

CT recovery lowers measurement loss4.90% but slightly worsens image error; MRI makes no progress. Thus an early termination can miss available descent without explaining the reconstruction failure. Do not promote recovery scaling as an accuracy fix. Useful-gradient exclusion remains unresolved, and these noiseless controls do not establish noise-filtering behavior.

[CT endpoint audit](ct1_increment_endpoint.json) · [CT factor substeps](ct7_increment_substeps4.json) · [MRI factor substeps](mri1_increment_substeps4.json)

Reconstruction panels (other methods are explicitly unavailable in these endpoint controls): [CT unchanged](../scale_recovery_200ct1/unchanged/comparison.png), [CT scaled](../scale_recovery_200ct1/scaled/comparison.png), [MRI unchanged](../scale_recovery_mri2/unchanged/comparison.png), [MRI scaled](../scale_recovery_mri2/scaled/comparison.png).
