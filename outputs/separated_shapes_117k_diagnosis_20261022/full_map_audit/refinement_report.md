# Euler residual refinement at fixed failed sources

All comparisons use the same source, endpoint, terminal measurement gradient and model. The independent native JVP/VJP checks pass. These are diagnostics, not reconstruction outcomes.

| Case | Rank | Update | Gradient error | Subspace error | Coefficient error | Cosine |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ct7 | 32 | dynamic | 40.85% | 35.61% | 20.02% | 0.9128 |
| ct7 | 128 | dynamic | 30.69% | 28.61% | 11.10% | 0.9518 |
| ct7 | 32 | increment_ksl | 20.19% | 13.40% | 15.10% | 0.9797 |
| ct7 | 128 | increment_ksl | 5.18% | 4.98% | 1.40% | 0.9987 |
| ct7 | 32 | refined_increment | 15.56% | 14.48% | 5.70% | 0.9878 |
| ct7 | 128 | refined_increment | 5.70% | 5.51% | 1.46% | 0.9984 |
| mri0 | 32 | dynamic | 100.60% | 96.89% | 27.09% | -0.0058 |
| mri0 | 128 | dynamic | 96.16% | 90.95% | 31.24% | 0.2778 |
| mri0 | 32 | increment_ksl | 100.51% | 97.51% | 24.38% | 0.0040 |
| mri0 | 128 | increment_ksl | 95.82% | 91.07% | 29.78% | 0.2908 |
| mri0 | 32 | refined_increment | 97.34% | 97.29% | 3.38% | 0.2296 |
| mri0 | 128 | refined_increment | 88.47% | 88.01% | 9.04% | 0.4663 |

The increment update approximates the frozen discrete Euler residual target. Refinement applies a second bilateral subspace iteration to that same target. Neither is labeled exact integration of continuous KSL. The refinement doubles rank-column JVP/VJP work on each evolved factor step; it adds no rank or enrichment columns.

For rank128 at five flow steps with eight startup oversampling columns, the original/increment paths use 648 JVP and 648 VJP columns per factor refresh. Refinement uses 1160 of each. This count does not establish wall time.

CT7 gains mostly from the consistent Euler target. MRI0 gains from refinement but still excludes most of the exact gradient, so a general correction is not established.
