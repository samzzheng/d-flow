# Same-source gradient diagnostics

Original larger models, noiseless data, FP32/TF32 off. Exact gradients differentiate the actual Euler map. Both gradients use the identical native endpoint and terminal gradient. DLR uses a fresh seeded sketch, analytical alpha=1-t, zero enrichment and oversampling 8. This isolates factor accuracy; it does not reproduce the previous-V history of an entire optimization trajectory.

At the 64 MRI initial source, rank-32 gradient error decreases from 53.3% at five steps to 4.7% at twenty. The 128 CT source does not show the same monotonic improvement: its initial errors are 12.4%, 63.9%, 43.2%. Thus more steps can improve DLR's gradient approximation, but this is source-dependent. A relatively accurate gradient of a poor coarse-grid map can still lead to a failed reconstruction.

Subspace error is the distance from the exact gradient to its orthogonal projection into DLR's right basis. Coefficient error is the remaining error inside that basis. Their squared sum matches total squared error to within 6.3e-6 times the exact gradient's squared norm in this audit. Eager DLR endpoints match native endpoints exactly.

| Case | Source | Steps | Rank | Gradient error % | Cosine | Norm ratio | Subspace error % | Coefficient error % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r64_shapes_mri4/sample_001 | initial | 5 | 32 | 53.33 | 0.8460 | 0.8336 | 51.36 | 14.36 |
| r64_shapes_mri4/sample_001 | initial | 5 | 128 | 43.73 | 0.8994 | 0.8917 | 41.38 | 14.12 |
| r64_shapes_mri4/sample_001 | initial | 10 | 32 | 18.69 | 0.9825 | 0.9676 | 17.62 | 6.23 |
| r64_shapes_mri4/sample_001 | initial | 10 | 128 | 17.09 | 0.9853 | 0.9815 | 16.25 | 5.29 |
| r64_shapes_mri4/sample_001 | initial | 20 | 32 | 4.72 | 0.9990 | 0.9845 | 3.09 | 3.57 |
| r64_shapes_mri4/sample_001 | initial | 20 | 128 | 1.94 | 0.9998 | 0.9953 | 1.57 | 1.14 |
| r64_shapes_mri4/sample_001 | failed_dflow_final | 5 | 32 | 55.76 | 0.8309 | 0.7940 | 52.57 | 18.57 |
| r64_shapes_mri4/sample_001 | failed_dflow_final | 5 | 128 | 45.68 | 0.8899 | 0.8657 | 42.35 | 17.12 |
| r64_shapes_mri4/sample_001 | failed_dflow_final | 10 | 32 | 11.80 | 0.9931 | 0.9804 | 11.05 | 4.13 |
| r64_shapes_mri4/sample_001 | failed_dflow_final | 10 | 128 | 11.26 | 0.9937 | 0.9889 | 10.75 | 3.35 |
| r64_shapes_mri4/sample_001 | failed_dflow_final | 20 | 32 | 4.62 | 0.9990 | 0.9858 | 3.21 | 3.32 |
| r64_shapes_mri4/sample_001 | failed_dflow_final | 20 | 128 | 2.25 | 0.9998 | 0.9954 | 1.91 | 1.20 |
| r128_shapes_ct10/sample_000 | initial | 5 | 32 | 12.44 | 0.9933 | 0.9474 | 11.11 | 5.59 |
| r128_shapes_ct10/sample_000 | initial | 5 | 128 | 12.64 | 0.9922 | 0.9715 | 11.67 | 4.87 |
| r128_shapes_ct10/sample_000 | initial | 10 | 32 | 63.93 | 0.7691 | 0.7557 | 59.30 | 23.87 |
| r128_shapes_ct10/sample_000 | initial | 10 | 128 | 42.42 | 0.9056 | 0.8967 | 39.05 | 16.58 |
| r128_shapes_ct10/sample_000 | initial | 20 | 32 | 43.21 | 0.9024 | 0.8708 | 41.03 | 13.55 |
| r128_shapes_ct10/sample_000 | initial | 20 | 128 | 34.09 | 0.9403 | 0.9224 | 31.95 | 11.91 |
| r128_shapes_ct10/sample_000 | failed_dflow_final | 5 | 32 | 43.57 | 0.9002 | 0.8877 | 40.89 | 15.02 |
| r128_shapes_ct10/sample_000 | failed_dflow_final | 5 | 128 | 61.36 | 0.7906 | 0.7506 | 53.28 | 30.44 |
| r128_shapes_ct10/sample_000 | failed_dflow_final | 10 | 32 | 40.31 | 0.9151 | 0.9163 | 39.37 | 8.64 |
| r128_shapes_ct10/sample_000 | failed_dflow_final | 10 | 128 | 41.00 | 0.9121 | 0.9185 | 38.90 | 12.95 |
| r128_shapes_ct10/sample_000 | failed_dflow_final | 20 | 32 | 80.95 | 0.5884 | 0.5496 | 71.71 | 37.56 |
| r128_shapes_ct10/sample_000 | failed_dflow_final | 20 | 128 | 58.95 | 0.8089 | 0.7654 | 53.26 | 25.26 |

## Narrow descent region at the failed 128 CT source

The initial FP32 finite-difference check failed at this source on the five-step map. A refined epsilon sweep and FP64 replay resolve the discrepancy: the derivative is correct locally, but the linear neighborhood is very small. Along the normalized exact descent direction, a displacement of 1e-4 increases the objective, while 1e-5 decreases it by only about 3.6e-7. FP32 rounding is comparable to these small decreases. The FP64 check reaches relative derivative error below 1e-7. FP32 and FP64 gradients differ by 1.28% at five steps, versus 0.00061% at twenty steps at the same source.

This is evidence of sharp local curvature plus a precision-sensitive objective decrease in one failed case. It is not evidence that all failures are floating-point bugs, nor a recommendation to use FP64 in production. The audit casts the existing FP32 time embeddings to FP64 before the time projection and retains the same CT grid coordinates.

## Mathematical interpretation

For an Euler step, x_(k+1)=x_k+h*v(t_k,x_k), so its derivative is I+h*A_k, where A_k is the spatial derivative of the velocity. The exact endpoint Jacobian is the ordered product (I+h*A_(N-1)) ... (I+h*A_0). Changing N changes both these factors and the states where they are evaluated. It changes the inverse objective itself, not just the accuracy of a gradient of one fixed objective.

The training objective in train_fm_unet.py regresses x_1-x_0 along straight interpolants. For its ideal conditional-mean velocity, x+(1-t)*v(t,x)=E[x_1 | x_t=x]. The last uniform Euler step has h=1-t, so it evaluates this denoiser at t=1-h. Five steps therefore make the last denoising operation at t=0.8, versus t=0.99 for 100 steps. Conditional averaging can attenuate uncertain structures. For a learned, discretized trajectory this relation motivates a mechanism; it does not prove the observed failures are caused by conditional averaging.

Low-rank approximation cannot guarantee escape from an unfavorable coarse-map basin. P-Flow changes the source update direction by bypassing the endpoint Jacobian, so it can behave differently even with the same generator and objective. The paired reconstructions are required to test which configuration actually works.
