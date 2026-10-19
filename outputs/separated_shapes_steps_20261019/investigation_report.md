# Separated-shapes collapse investigation

Five Euler steps affect the generator substantially, but increasing to 10 or 20 did not consistently resolve collapse. A 100-step exact D-Flow control also failed. No consistently accurate, fast DLR configuration has yet been established.

Eight deliberately selected cases across six resolutions; original epoch-200 models, noiseless CT10/MRI4x, paired seeds, L-BFGS history 100 and geometric Armijo, 0.01 sqrt(d) displacement cap, up to 1000 attempts. DLR rank 32 throughout this table, analytical alpha=1-t, zero enrichment, previous-V reuse. All 24 input bundles were checked bitwise against the five-step baseline, including truths, sources, observations and MRI masks; checkpoint hashes and acquisition metadata match. This selected-case pilot is not a population success-rate estimate.

| Case | Euler steps | D-Flow error % | P-Flow error % | DLR32 error % |
| --- | ---: | ---: | ---: | ---: |
| 250 mri4 sample 0 | 5 | 98.460 | 1.832 | 99.020 |
| 250 mri4 sample 0 | 10 | 76.266 | 1.344 | 99.902 |
| 250 mri4 sample 0 | 20 | 107.068 | 1.005 | 108.722 |
| 64 mri4 sample 1 | 5 | 100.066 | 3.367 | 99.773 |
| 64 mri4 sample 1 | 10 | 36.088 | 2.340 | 99.877 |
| 64 mri4 sample 1 | 20 | 99.861 | 1.606 | 100.031 |
| 128 ct10 sample 0 | 5 | 99.820 | 4.024 | 100.115 |
| 128 ct10 sample 0 | 10 | 100.815 | 54.651 | 111.584 |
| 128 ct10 sample 0 | 20 | 108.738 | 60.479 | 109.740 |
| 96 mri4 sample 1 | 5 | 1.508 | 3.012 | 99.912 |
| 96 mri4 sample 1 | 10 | 99.599 | 1.664 | 99.689 |
| 96 mri4 sample 1 | 20 | 99.666 | 99.704 | 99.685 |
| 160 ct10 sample 0 | 5 | 99.486 | 57.765 | 101.568 |
| 160 ct10 sample 0 | 10 | 100.608 | 93.278 | 110.397 |
| 160 ct10 sample 0 | 20 | 99.838 | 101.156 | 101.996 |
| 200 ct10 sample 1 | 5 | 100.506 | 2.585 | 101.505 |
| 200 ct10 sample 1 | 10 | 103.629 | 1.935 | 103.130 |
| 200 ct10 sample 1 | 20 | 117.211 | 8.236 | 118.698 |
| 64 ct10 sample 0 | 5 | 1.728 | 4.633 | 4.054 |
| 64 ct10 sample 0 | 10 | 77.804 | 43.715 | 85.152 |
| 64 ct10 sample 0 | 20 | 89.575 | 55.534 | 95.621 |
| 128 mri4 sample 2 | 5 | 1.801 | 99.941 | 100.063 |
| 128 mri4 sample 2 | 10 | 99.958 | 63.394 | 99.934 |
| 128 mri4 sample 2 | 20 | 99.762 | 58.563 | 127.446 |

## Additional controls

| Control | Case | D-Flow % | P-Flow % | DLR32 % |
| --- | --- | ---: | ---: | ---: |
| L-BFGS history 5 | r64_shapes_mri4/sample_001 | 100.066 | 3.417 | 99.735 |
| L-BFGS history 5 | r128_shapes_ct10/sample_000 | 90.951 | 3.879 | 99.742 |
| Tenfold smaller displacement cap | r250_shapes_mri4/sample_000 | 99.845 | 99.809 | 99.224 |
| Small model, epoch 1000 | r64_shapes_mri4/sample_001 | 2.513 | 3.844 | 99.792 |
| Terminal singular values capped at median | r64_shapes_mri4/sample_001 | not run | not run | 99.947 |
| 100 Euler steps, checkpointed exact gradient | r250_shapes_mri4/sample_000 | 108.475 | not run | not run |

### Small-model gradient-origin controls

Same 64 MRI sample 1, five Euler steps and rank 32. Exact gradients are computed for diagnostics; these runs are not speed measurements. Projected-exact is a diagnostic surrogate, not standard DLR.

| Gradient | Image error % | Attempts |
| --- | ---: | ---: |
| DLR with previous-V reuse | 99.807 | 78 |
| DLR with fresh sketch | 99.773 | 19 |
| Exact gradient projected onto DLR V | 99.818 | 59 |

Initially, the exact gradient loses 66.8% of its norm when projected onto DLR's right subspace; the coefficient error within the subspace is 8.6% of the exact gradient norm. Replacing those coefficients with exact ones does not rescue this case. This rules out coefficient error alone as a sufficient explanation under this optimizer; it does not prove that subspace truncation is the sole cause. The instrumented baseline also collapses, but its termination iteration differs from the earlier run (78 versus 93); no bitwise trajectory equivalence is claimed.

The small-model result changes both architecture and training duration. It shows that five steps can support an accurate reconstruction in this case, but does not establish general reliability or resolve DLR failure. The uncheckpointed 100-step run ran out of allocated memory before optimization; only its completed checkpointed rerun is reported above.

## Evidence and interpretation

- Fixed-source replays reproduce all saved five-step final images exactly. Several failed images are nearly blank. Refining the grid restores image amplitude at those same sources without necessarily recovering the target shapes.
- The exact Euler Jacobian is the ordered product of I+h*A_k. Changing the time grid changes both the generator and its inverse objective; it is not merely a more accurate gradient of the same objective.
- A successful P-Flow reconstruction on the same five-step generator proves that a good solution is representable for that sample. D-Flow/DLR failures on it therefore cannot be attributed solely to insufficient generator expressivity.
- Fixed-point gradient errors are source-dependent. At the initial 64 MRI source, DLR32 error falls from 53.3% to 4.7% when increasing from five to twenty steps. At 128 CT, errors are 12.4%, 63.9%, 43.2% for 5/10/20. Neither high rank nor more steps guarantees a useful optimization trajectory.
- At one failed 128 CT source, refined FP64 finite differences verify the derivative while revealing a very narrow descent neighborhood: displacement 1e-4 along normalized descent raises the loss, and 1e-5 lowers it by only about 3.6e-7. This supports sharp local curvature and precision sensitivity, not a universal derivative bug.
- The sampled rank-pilot truths cover only one- and three-object shapes. Broader validation must include two and four objects and independent sources before calling any configuration consistent.

Detailed evidence: [fixed-source audit](diagnostics/report.md), [gradient audit](gradients/report.md), [Euler reconstructions](report.md), [short-history controls](optimizer_controls/report.md). Further diagnostic controls separate coefficient error from subspace loss on the small-model case. Enrichment and GD are excluded at the user's request. GPU 0 is off limits for subsequent work.
