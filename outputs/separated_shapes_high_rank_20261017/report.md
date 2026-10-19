# Noiseless separated-shapes rank sweep

Five paired samples (0–4) per resolution and inverse problem. CT10 / Cartesian MRI4×; five Euler steps; unchanged LBFGS and geometric Armijo, up to 1,000 attempts with the existing safeguards. Analytical alpha=1-t, FP32, no added regularization. Rank-specific source directories and reused runs are recorded in plan.json.

Medians below use only samples completed at every rank in that row. Pending runs are not counted as failures. Image and measurement metrics use final checkpoints. Rank selection on these five samples is exploratory and needs confirmation on other samples. A winner at the largest tested rank does not establish a sufficient rank.

| Resolution / problem | Common samples | Rank 8 image error % | Rank 16 image error % | Rank 32 image error % | Rank 64 image error % | Rank 128 image error % | Rank 256 image error % | Rank 512 image error % | D-Flow % | P-Flow % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64² / ct10 | 5/5 | 100.899 | 101.197 | 101.014 | 3.748 | 2.604 | 2.083 | 2.512 | 1.728 | 6.805 |
| 64² / mri4 | 5/5 | 99.836 | 99.780 | 99.773 | 50.711 | 2.342 | 3.506 | 5.290 | 1.816 | 3.585 |
| 96² / ct10 | 5/5 | 100.362 | 100.439 | 83.540 | 72.490 | 83.094 | 3.402 | 3.599 | 1.887 | 7.415 |
| 96² / mri4 | 5/5 | 99.497 | 99.834 | 99.832 | 82.012 | 99.786 | 71.227 | 3.176 | 1.782 | 2.984 |
| 128² / ct10 | 5/5 | 99.710 | 99.996 | 84.229 | 99.822 | 3.604 | 77.400 | 78.079 | 1.790 | 4.024 |
| 128² / mri4 | 5/5 | 99.599 | 99.627 | 99.662 | 99.704 | 67.228 | 99.661 | 99.065 | 1.516 | 2.321 |
| 160² / ct10 | 5/5 | 101.528 | 100.339 | 100.340 | 100.929 | 100.269 | 100.645 | 100.550 | 2.841 | 2.442 |
| 160² / mri4 | 5/5 | 99.454 | 99.066 | 99.231 | 68.276 | 8.000 | 66.338 | 3.152 | 1.350 | 2.210 |
| 200² / ct10 | 5/5 | 100.301 | 101.241 | 100.948 | 100.628 | 99.823 | 99.787 | 100.218 | 96.839 | 3.370 |
| 200² / mri4 | 5/5 | 99.603 | 99.448 | 99.480 | 99.468 | 99.470 | 99.416 | 99.438 | 78.616 | 2.397 |
| 250² / ct10 | 5/5 | 99.985 | 99.860 | 99.573 | 100.147 | 99.579 | 99.555 | 99.916 | 76.324 | 2.199 |
| 250² / mri4 | 5/5 | 99.482 | 99.467 | 99.404 | 99.399 | 99.467 | 99.455 | 99.415 | 79.215 | 1.781 |

| Case | Completed rank 8 | Completed rank 16 | Completed rank 32 | Completed rank 64 | Completed rank 128 | Completed rank 256 | Completed rank 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r64_shapes_ct10 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r64_shapes_mri4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r96_shapes_ct10 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r96_shapes_mri4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r128_shapes_ct10 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r128_shapes_mri4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r160_shapes_ct10 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r160_shapes_mri4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r200_shapes_ct10 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r200_shapes_mri4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r250_shapes_ct10 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| r250_shapes_mri4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

Per-sample image errors, measurement residuals, attempt counts and termination reasons are saved in summary.json. Standard four-column reconstruction figures are saved inside each rank's production directory; unavailable controls are marked explicitly.
