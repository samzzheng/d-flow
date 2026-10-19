# Sparse-CT circle benchmark: Forward Euler versus RK4

## Test status

- Complete at 64×64, 96×96, and 128×128: five samples per solver.
- Incomplete at 160×160: samples 0–2 finished for both solvers, sample 3 finished only for Forward Euler, and sample 4 did not finish.
- The tables use only solver-matched samples: 18 pairs in total.
- Each measurement contains noise satisfying `||noise||₂ / ||clean sinogram||₂ = 0.01`.
- Both solvers used the same initial source, five flow steps, and the same LBFGS settings.

## Reconstruction quality

Lower image relative ℓ₂ error is better; higher SSIM is better.

| Resolution | Matched samples | Mean image rel. ℓ₂, Euler | Mean image rel. ℓ₂, RK4 | Median image rel. ℓ₂, Euler | Median image rel. ℓ₂, RK4 | Mean SSIM, Euler | Mean SSIM, RK4 | RK4 image-error wins |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64×64 | 5 | 0.0902 | 0.0895 | 0.0230 | 0.0151 | 0.859 | 0.784 | 2/5 |
| 96×96 | 5 | 0.0251 | 0.1030 | 0.00894 | 0.1045 | 0.983 | 0.611 | 1/5 |
| 128×128 | 5 | 0.0525 | 0.0933 | 0.0531 | 0.0975 | 0.966 | 0.575 | 0/5 |
| 160×160 | 3 | 0.0256 | 0.0475 | 0.00471 | 0.0476 | 0.990 | 0.727 | 1/3 |
| **All matched** | **18** | **0.0509** | **0.0873** | **0.0113** | **0.0805** | **0.945** | **0.668** | **4/18** |

## Measurement convergence and cost

The target is a noisy-measurement relative ℓ₂ error of at most 0.01. Lower error, fewer iterations, and less time are better.

| Resolution | Mean measurement rel. ℓ₂, Euler | Mean measurement rel. ℓ₂, RK4 | Reached target, Euler | Reached target, RK4 | Mean outer steps, Euler | Mean outer steps, RK4 | Mean time, Euler (s) | Mean time, RK4 (s) | RK4/Euler time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64×64 | 0.0421 | 0.0401 | 3/5 | 3/5 | 37.8 | 33.2 | 55.9 | 194.1 | 3.47× |
| 96×96 | 0.0138 | 0.0333 | 4/5 | 0/5 | 33.8 | 50.0 | 70.7 | 411.9 | 5.83× |
| 128×128 | 0.0206 | 0.0279 | 1/5 | 0/5 | 45.0 | 50.0 | 145.0 | 686.1 | 4.73× |
| 160×160 | 0.0134 | 0.0149 | 2/3 | 1/3 | 30.0 | 39.7 | 146.4 | 803.3 | 5.49× |
| **All matched** | **0.0235** | **0.0306** | **10/18** | **4/18** | **37.4** | **43.6** | **99.8** | **492.8** | **4.94×** |

## Outcome

- With the current five-step flow discretization and shared LBFGS configuration, Forward Euler is the better end-to-end D-Flow solver.
- RK4's mean image relative ℓ₂ error is 0.0873 versus 0.0509 for Euler, and its median is 0.0805 versus 0.0113.
- RK4 is substantially slower because every RK4 flow step evaluates the velocity model four times.
- This does not demonstrate that RK4 is intrinsically less accurate as an ODE integrator. It shows that the current RK4-plus-LBFGS configuration produces worse inverse reconstructions. The optimizer trajectory and optimized source differ between solvers.
- Several RK4 runs show repeated LBFGS rollbacks, especially 64×64 sample 3, 96×96 samples 0 and 2, and 160×160 sample 0. This is evidence that the shared LBFGS learning rate is less stable for the RK4 objective.
