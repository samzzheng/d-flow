# Why separated shapes are harder: evidence audit

This audit separates dataset/generator difficulty from DLR-specific approximation error. It does not establish a single cause.

## Data complexity

The generator samples 1–4 objects, each independently a disk or general triangle. A disk has 3 continuous geometric parameters (center and radius); a triangle has 6 (center, three side lengths, orientation). A fixed composition therefore has 3Nc+6Nt geometric parameters, up to 24, plus discrete count/type choices. This is a geometric description, not a prediction of the learned Jacobian's numerical rank. One mixed-shape model handles all compositions, whereas the circle study used separate models for fixed circle counts.

## Five-step generator quality

Existing paired unconditional checks use 128 source samples per resolution. Blank means the existing raster component detector finds no valid component; this is a detector-based diagnostic, not proof of an identically zero image.

| Resolution | Blank at 5 Euler steps | Blank at 100 Euler steps |
|---|---:|---:|
| 64² | 4/128 | 0/128 |
| 96² | 14/128 | 0/128 |
| 128² | 4/128 | 0/128 |
| 160² | 39/128 | 1/128 |
| 200² | 65/128 | 5/128 |
| 250² | 71/128 | 5/128 |

This shows substantial distortion of the learned sampling map at five steps, particularly at high resolution. It can create poor starting images for every inverse method. It does not prove that increasing steps alone will fix inversion; earlier paired controls found mixed reconstruction outcomes.

## Completed paired inverse cases

Coverage: 64 paired cases; resolutions/counts: {200: 4, 250: 60}. CT uses 10 projection angles; MRI uses 4x Cartesian sampling. All are noiseless, five Euler steps, with the established LBFGS/geometric Armijo setup. The partial sweep is not a random or balanced sample.

| Resolution | Problem | Objects | Method | Cases | Median image error | Median residual | Image error > 50% |
|---|---|---:|---|---:|---:|---:|---:|
| 200² | CT | 1 | d_flow | 3 | 99.78% | 90.76% | 2/3 |
| 200² | CT | 1 | p_flow | 3 | 4.24% | 1.06% | 0/3 |
| 200² | CT | 3 | d_flow | 1 | 53.23% | 37.32% | 1/1 |
| 200² | CT | 3 | p_flow | 1 | 3.72% | 0.77% | 0/1 |
| 250² | CT | 1 | d_flow | 7 | 100.38% | 98.80% | 7/7 |
| 250² | CT | 1 | p_flow | 7 | 5.11% | 1.24% | 2/7 |
| 250² | CT | 2 | d_flow | 4 | 99.66% | 91.56% | 3/4 |
| 250² | CT | 2 | p_flow | 4 | 16.87% | 5.97% | 0/4 |
| 250² | CT | 3 | d_flow | 8 | 23.18% | 14.92% | 3/8 |
| 250² | CT | 3 | p_flow | 8 | 4.56% | 0.99% | 2/8 |
| 250² | CT | 4 | d_flow | 11 | 1.90% | 0.19% | 1/11 |
| 250² | CT | 4 | p_flow | 11 | 14.54% | 4.33% | 4/11 |
| 250² | MRI | 1 | d_flow | 7 | 2.66% | 1.34% | 3/7 |
| 250² | MRI | 1 | p_flow | 7 | 3.14% | 1.55% | 0/7 |
| 250² | MRI | 2 | d_flow | 4 | 99.92% | 99.88% | 3/4 |
| 250² | MRI | 2 | p_flow | 4 | 2.09% | 1.11% | 0/4 |
| 250² | MRI | 3 | d_flow | 8 | 99.86% | 99.83% | 6/8 |
| 250² | MRI | 3 | p_flow | 8 | 1.59% | 0.81% | 0/8 |
| 250² | MRI | 4 | d_flow | 11 | 99.49% | 99.28% | 7/11 |
| 250² | MRI | 4 | p_flow | 11 | 1.75% | 0.88% | 0/11 |

At 250² MRI, P-Flow finishes below 10% image error on 30/30 paired cases, while D-Flow exceeds 50% on 19/30. The failed D-Flow cases have median output norm 0.0211 times the true image norm. All true-object projection amplitudes are below 0.5 in 18/19 of these failures.

Successful P-Flow images establish representability for those specific cases under the same five-step generator. They rule out insufficient generator expressivity as a complete explanation of the paired D-Flow failures. They do not prove a global capacity claim or identify one optimizer mechanism. The common setup differs in source-gradient estimator, which also changes the LBFGS curvature history and line searches.

Only completed paired D-Flow/P-Flow runs. Legacy DLR results excluded because derivative-backend defects were subsequently corrected. Incomplete and nonrandom coverage; descriptive strata, not population estimates or causal tests.

Cached truths match separately rasterized saved geometry; observed measurements are noiseless; methods share exact saved input protocol; image/complex residual metrics and foreground/background error partition verified.

Per-case records include true object types, geometric parameter count, signal amplitude at each true object's location, and the exact foreground/background partition of squared image error. These help distinguish weak/missing true objects from spurious background signal without relying on visual impressions.

## What would discriminate the remaining causes

- Fit the full image using the same generator and paired sources. Failure even with full-image observations would implicate the generator or its optimization; success would isolate difficulty added by CT/MRI.
- Compare 5-step and 100-step full-image fits before changing the inverse problem. This separates coarse integration from measurement ambiguity.
- At the true geometry, measure how CT/MRI attenuate small shape-parameter changes. Compare measurement directions with image directions; full-image Jacobian rank alone does not diagnose acquisition identifiability.
- Compare exact and DLR gradients at identical sources, including missing-subspace and coefficient error. The ongoing high-start gradual-rank controls address one specific source of projection loss.

A small data residual with a wrong image can indicate a measurement ambiguity or a nearly invisible image difference. A large residual with a wrong image instead indicates unresolved data fitting; it does not by itself establish non-identifiability. Neither classification alone proves model capacity is insufficient.
