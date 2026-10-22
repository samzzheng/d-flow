# Corrected backend audit across trained resolutions

All six trained 117k separated-shapes models pass. Each uses one seeded Gaussian source, five Euler steps and FP32 with TF32 disabled on GPU1. At each flow step, three random JVP/VJP columns are compared with independent native-model PyTorch autodiff. Rank32 full-factor gradient actions are also compared, first with a fresh initialization sketch and then with the identical cached right basis supplied to both backends.

| Resolution | Maximum local JVP relative error | Maximum local VJP relative error | Fresh-factor gradient relative error | Seeded-factor gradient relative error | Endpoint discrepancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64² | 1.055e-07 | 1.048e-07 | 2.219e-05 | 3.057e-05 | 0.0e+00 |
| 96² | 9.559e-08 | 1.057e-07 | 9.222e-06 | 8.297e-06 | 0.0e+00 |
| 128² | 1.131e-07 | 1.157e-07 | 3.852e-05 | 2.763e-05 | 0.0e+00 |
| 160² | 9.043e-08 | 9.871e-08 | 1.556e-05 | 5.155e-06 | 0.0e+00 |
| 200² | 9.105e-08 | 1.109e-07 | 5.795e-06 | 4.037e-06 | 0.0e+00 |
| 250² | 9.327e-08 | 1.069e-07 | 6.637e-06 | 7.391e-06 | 0.0e+00 |

Terminal alpha is exactly zero in every factor comparison. The maximum normalized local adjoint-consistency error is below1.5e-9. These are execution-equivalence checks at the tested sources, not proof that DLR approximates the exact full-map gradient accurately or that every reconstruction will converge.

The diagnostic launcher now defaults to native normalization; the explicitly selected legacy fast backend remains available only to reproduce earlier manual-statistics controls. The production default was already corrected.

[Raw checks and source/checkpoint hashes](backend_all_resolutions.json)
