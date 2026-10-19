# Measurement-informed initialization

Five Euler steps, analytical alpha=1-t, zero enrichment, unchanged L-BFGS/Armijo and rank 32 unless stated. Only the starting source changes. Let q be A*y after subtracting its spatial mean and normalizing its norm to sqrt(d). Initialize z = sqrt(1-rho^2) z_random + rho q. This uses the observation and acquisition operator, not the ground-truth image. It changes the source distribution; it is not an unconditional standard-Gaussian draw. Corrected runs cache one conditioned source for all three methods. The stronger mix was selected using the small-model 64 MRI sample 1 result, so that case is tuning evidence. Timing has not been benchmarked in isolation.

| Model | Mix | Rank | Case | D-Flow % | P-Flow % | DLR % |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| original epoch 200 (original) | 0.5 | 32 | r128_shapes_ct10/sample_000 | unpaired: 1.990 | unpaired: 24.524 | unpaired: 34.977 |
| original epoch 200 (original) | 0.5 | 32 | r250_shapes_mri4/sample_000 | 1.105 | 8.173 | 10.510 (incomplete) |
| original epoch 200 (original) | 0.5 | 32 | r64_shapes_mri4/sample_001 | 2.007 | 3.512 | 2.741 |
| original epoch 200 (original) | 0.5 | 32 | r64_shapes_mri4/sample_007 | 1.500 | 3.562 | 4.023 |
| original epoch 200 (original) | 0.5 | 128 | r128_shapes_ct10/sample_000 | not run yet | not run yet | 69.155 (incomplete) |
| small epoch 1000 (original) | 0.25 | 32 | r64_shapes_mri4/sample_001 | 2.325 | 3.967 | 3.089 |
| small epoch 1000 (original) | 0.5 | 32 | r64_shapes_mri4/sample_001 | 2.386 | 4.244 | 2.739 |

A missing result is not a failure. Incomplete values are not final. Rank-128 runs execute DLR only. Values marked unpaired come from the original CT initializer, which recomputed the adjoint separately for each method and produced different source hashes; these are excluded from strictly paired comparisons and are being rerun with one cached source. The source mixing formula was reconstructed exactly on CPU. When an original reference exists, truths, observations, masks and random sources were verified bitwise, with matching model and acquisition metadata.

The completed 64 MRI sample 1 controls show a large improvement over random initialization: small-model DLR 99.79% to 2.74% and original-model DLR 99.77% to 2.74%. The earlier CT runs reached DLR32 34.98% and D-Flow 1.99%, but their source hashes differ; the cached-source repeat is required before interpreting this comparison as paired. A consistent accurate configuration across shape counts, resolutions and both operators has not been established.

[Amplitude diagnostic](amplitude_audit.md) explains why shrinking poorly placed shapes can decrease the measurement objective. That mechanism is plausible, not a proof that it causes all observed failures.
