# Equal-velocity-evaluation Euler versus RK4

## Test configuration

- Four resolutions: 64×64, 96×96, 128×128, and 160×160
- Five matched samples per resolution; 20 matched pairs total
- Forward Euler: 20 steps and 20 velocity evaluations per flow solve
- RK4: 5 steps and 20 velocity evaluations per flow solve
- Exactly 1% relative ℓ₂ noise added to each sinogram
- LBFGS learning rate 0.1; 50 outer iterations; no noise-level early stopping
- CPU execution

## Reconstruction quality

| Resolution | Mean image rel. ℓ₂, Euler | Mean image rel. ℓ₂, RK4 | Median image rel. ℓ₂, Euler | Median image rel. ℓ₂, RK4 | Mean SSIM, Euler | Mean SSIM, RK4 | RK4 rel.-ℓ₂ wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64×64 | 0.1383 | **0.1168** | 0.1927 | **0.0636** | **0.847** | 0.656 | 3/5 |
| 96×96 | 0.0889 | **0.0661** | 0.0979 | **0.0498** | **0.844** | 0.713 | 4/5 |
| 128×128 | 0.0876 | **0.0831** | **0.0943** | 0.0981 | **0.967** | 0.696 | 2/5 |
| 160×160 | **0.0665** | 0.0682 | **0.0734** | 0.0793 | **0.979** | 0.741 | 1/5 |
| **Overall** | 0.0953 | **0.0835** | 0.0952 | **0.0715** | **0.909** | 0.702 | **10/20** |

RK4 reduced the overall mean image relative ℓ₂ error by 12.4% and the median by 24.9%. However, it won exactly half of the paired comparisons. Euler had higher SSIM in 18 of 20 pairs.

## Measurement fit, integration reference, and cost

| Metric | Euler | RK4 | RK4 paired wins |
|---|---:|---:|---:|
| Mean noisy-measurement rel. ℓ₂ | 0.0326 | **0.0285** | 11/20 |
| Median noisy-measurement rel. ℓ₂ | 0.0286 | **0.0209** | 11/20 |
| Mean clean-measurement rel. ℓ₂ | 0.0313 | **0.0291** | 10/20 |
| Mean error against fine RK4 flow reference | 0.0813 | **0.0700** | 10/20 |
| Final noisy residual at or below 1% | **3/20** | 1/20 | — |
| Mean CPU time per reconstruction | 603.4 s | **589.0 s** | 14/20 |
| Mean LBFGS rollbacks | **3.40** | 7.45 | 4/20 |

With equal velocity-evaluation budgets, runtime is effectively comparable. RK4's mean runtime was 2.4% lower, which is not evidence of a meaningful solver-complexity advantage and may reflect system variability.

## Conclusion

- RK4 is not clearly better than Euler for this noisy inverse problem.
- RK4 has modestly better aggregate relative ℓ₂ and measurement-fit values, but its image relative-ℓ₂ win rate is only 10/20.
- Euler is decisively better by SSIM and visually produces cleaner backgrounds. Coarse Euler appears to provide useful implicit regularization through numerical damping.
- RK4 often produces low-amplitude background speckle. This can reduce total pixel-energy and measurement errors while harming structural fidelity.
- RK4's relative-ℓ₂ advantage is strongly tied to measurement fit: the paired differences have correlation 0.983. SSIM differences are uncorrelated with measurement-fit differences (correlation approximately -0.006).
- A fixed-source comparison against a fine ODE reference is required to demonstrate RK4's integration-order advantage independently of inverse optimization and noise fitting.
