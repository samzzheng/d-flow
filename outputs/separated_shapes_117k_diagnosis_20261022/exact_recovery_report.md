# Exact-gradient continuation from failed DLR sources

Both selected failed DLR endpoints also reject their searches when continued with native exact D-Flow. Thus removing the DLR gradient approximation at these endpoints is insufficient under the current FP32 optimization protocol. This does not establish local optimality or explain every DLR failure.

## Paired reconstruction controls

Each comparison begins at an identical saved 250² source with the same 117k model, noiseless measurements, five Euler steps, unconstrained source, L-BFGS/Armijo parameters and original remaining1000-attempt budget. Both arms reset L-BFGS; no curvature pair crosses estimators. DLR preserves its parent rank policy and uses the previous V only as an initialization sketch seed. D-Flow differentiates the native discrete Euler map, without DLR factors or custom derivative products. TF32 is disabled.

| Endpoint | Parent image error | Reset DLR | Exact D-Flow continuation | Additional attempts |
|---|---:|---:|---:|---:|
| MRI4× sample2, DLR adaptive32–128 | 99.7683% | 99.7683% | 99.7683% | 2 per arm |
| CT10 sample0, measurement-informed source, DLR rank32 | 37.1184% | 37.1184% | 37.1184% | 2 per arm |

All four arms preserve their parent sources bitwise. They terminate under the original two-rejected-search safeguard, not because the remaining attempt budget is exhausted. This is different from running D-Flow from the original Gaussian source: the path leading to these endpoints has already been chosen by DLR.

At these same endpoints, the saved DLR gradients have relative errors29.07%/54.69% and cosines0.9568/0.8372 against the saved native FP32 exact gradients (MRI/CT). Their rank32 right bases capture92.01%/70.94% of the exact squared gradient norm. These are real approximation discrepancies, but replacing the gradient by the exact one still gives no accepted update. The discrepancies alone therefore do not explain the observed endpoint stagnation.

## Precision and local descent

A paired fixed-source MRI diagnostic repeats the native exact-gradient line search in FP32 and promoted FP64. It tries the displacement cap2.5 and29 successive halvings, without performing reconstruction updates.

| MRI endpoint diagnostic | FP32 | Promoted FP64 |
|---|---:|---:|
| Objective | 171.6301310041 | 171.6301310698 |
| Exact gradient norm | 2.18117 | 0.94678 |
| Independent JVP/VJP adjoint relative error | 3.54e-5 | 1.96e-13 |
| Largest tested objective decrease along negative exact gradient | 0 | 9.21e-8 |

The FP32/FP64 exact gradient cosine is0.97185. FP64 accepts displacements around1.49e-7; the best tested decrease is only5.37e-10 of the objective. The saved FP32 gradient also yields a tiny accepted decrease when evaluated in FP64, so its direction is not simply reversed. These promoted-arithmetic results concern a slightly different numerical map, not a certified exact-real gradient.

For CT, an existing FP64 audit uses the identical parent source; its promoted source hash was checked again. It finds a best decrease1.97e-9 at displacement1.19e-6, with adjoint discrepancy1.05e-13. Reuse of that evidence avoids a redundant GPU test. No FP64 reconstruction rescue is established by either tiny decrease.

**Working interpretation:** these endpoints combine approximate-gradient error with locally very small usable steps and numerical sensitivity. Native exact gradients do not automatically rescue them in FP32. Late estimator replacement, proxy fallback and starting at the cap have all failed on these two endpoints. This shifts attention toward preventing the problematic trajectory earlier; the ongoing original-source FP32/FP64 and rank-retention controls address that question. It does not justify relaxing acceptance to allow uphill steps or promoting a precision-only cure.

## Validation and artifacts

Saved-array verification passes source/model/operator pairing, identical starting objectives, original budgets, caps, monotone histories, unchanged rejected sources and original complex measurement metrics. The fallback CPU tests still pass after adding the isolated exact-control arm; the runner and verifier compile. No production defaults change.

- [Continuation verification](exact_recovery_verification.json).
- [Same-source gradient comparison](exact_recovery_gradient_comparison.json).
- [MRI FP32 audit](exact_recovery_mri2/endpoint_fp32.json) and [FP64 audit](exact_recovery_mri2/endpoint_fp64.json).
- [Reused CT FP64 audit](full_map_audit/ct0_line_search_fp64.json).
- [MRI continuation figure](exact_recovery_mri2/comparison.png) and [CT continuation figure](exact_recovery_conditioned_ct0/comparison.png).

P-Flow is explicitly unavailable in these two-arm figures. The earlier separate proxy-control figures remain in the proxy fallback report. Checkpoints stay local; configurations, source hashes, logs and compact reports are retained in Git.
