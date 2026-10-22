# Explicit seed retention during residual initialization

The paired rank32 MRI1 continuation improves from **12.306% to 2.982%** image error. Both the unchanged and matched-random controls remain12.306%. This is a selected-case result with one oracle exact-gradient seed, not a general default or a free speed improvement.

## Change and mathematical reason

A vector supplied to the randomized sketch is not necessarily retained after its SVD truncation. Let $B=h(A_0+I)$ be the first Euler residual and let $w$ be a normalized seed. Reserve one of the existing r right-basis directions for $w$, choose the other r-1 directions from the previous approximation restricted to $w^\perp$, and recompute the left factors from $BV$. The initial approximation is

$$Y_h=BV V^\top,\qquad w\in\operatorname{span}(V),\qquad Y_h w=Bw.$$

Thus the first residual action on the seed is preserved. This changes only initialization; later fast KSL factor updates remain unchanged and can rotate the seed out. Rank stays r. The initializer adds r JVP columns plus factorization work. The informed seed is one exact objective gradient at the failed parent, held fixed during all subsequent optimization; no exact-gradient optimizer steps are taken.

## Fixed-source diagnostics

At250-square MRI1 with rank32 and saved V, explicit retention improves gradient cosine0.6912 to0.8054 and relative gradient error73.823% to61.238%. Terminal captured squared-gradient norm increases49.539% to65.798%. Both variants use the same informed seed and ordinary later factor updates.

The200-square CT1 increment control does not show a comparable gain: rank128 relative gradient error99.023% to98.669%. Although initial capture is effectively100%, it falls to17.377% in the next factor step and9.879% at the endpoint. Initialization cannot by itself guarantee later directional retention. Capture slightly above100% (up to0.0023 percentage points) is FP32 orthogonality roundoff.

## Completed paired MRI reconstruction

Same failed parent at attempt196, rank32, five Euler steps, noiseless MRI4x, model, source, measurements, saved right basis, LBFGS/Armijo, and original1000-attempt cap. Curvature is reset equally in all arms. Exact and random seed tensors are bitwise identical to the preceding soft-seed experiment; the random seed has the same norm. Ground truth does not choose the direction or acceptance.

| Initialization | Image error | Measurement residual | Total attempts |
| --- | ---: | ---: | ---: |
| unchanged | 12.306% | 6.833% | 198 |
| gradient_seed | 2.982% | 1.725% | 468 |
| random_seed | 12.306% | 6.833% | 198 |

The preceding soft-seed continuation ended at9.040% image error; explicit retention of that same cached seed ends at2.982%. The new run does not compute another exact gradient: it reuses the previously computed seed, whose one-gradient cost remains part of the method. Attempt counts include the failed parent. Early stopping remains the established rejected-search safeguard.

Sixteen CPU tests pass, including dense first-residual action preservation, rank-one/full-rank cases and equivalence of the unmodified Rademacher startup control. Saved-array image and complex measurement metrics, source/seed hashes, original history prefixes, monotone objectives and original budgets pass independent checks. Panels were visually inspected.

The successful MRI1 truth contains only one shape. The separate three-object MRI4 paired continuation is also complete: unchanged/informed/random finish at94.224% image error and93.707% measurement residual, with203/201/201 total attempts. The informed direction does not rescue this case. Its source/seed identities, metrics, history and budget checks pass ([verification](../retained_seed_recovery_mri4/verification.json)). All bounded recovery workers have exited. No production defaults changed; the source-review hold remains in place.

The practical conclusion is that seed retention can remedy an initialization deficiency on one trajectory, but does not establish a general correction for diverse shapes. Spectrum-only rank adaptation also remains mixed; preserving useful objective directions and selecting rank are separate problems.

[Verification](../retained_seed_recovery_mri1/verification.json) · [Run configuration](../retained_seed_recovery_mri1/config.json) · [Completed metrics](../retained_seed_recovery_mri1/summary.json) · [MRI fixed-source audit](mri1_retained_seed.json) · [CT fixed-source audit](ct1_retained_seed.json)

[Improved MRI endpoint](../retained_seed_recovery_mri1/gradient_seed/comparison.png) · [Unchanged control](../retained_seed_recovery_mri1/unchanged/comparison.png) · [Random control](../retained_seed_recovery_mri1/random_seed/comparison.png)
