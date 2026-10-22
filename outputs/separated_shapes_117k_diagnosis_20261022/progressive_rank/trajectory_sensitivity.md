# Numerical trajectory sensitivity in the initial-rank control

Both controls use identical cached sources, model, measurements, seeds, optimizer and rank cutoff. The extra initialization floor is inactive whenever the unrestricted initializer already keeps256. It did so in every recorded refresh for both completed unrestricted arms. All subsequent steps use the same minimum32.

| Case | Attempt | Unrestricted image error | Initial-floor image error | Relative objective difference |
|---|---:|---:|---:|---:|
| r250_shapes_ct10/sample_007 | 1 | 105.1495% | 105.1495% | 2.74931e-06% |
| r250_shapes_ct10/sample_007 | 10 | 109.7261% | 109.7881% | 0.235115% |
| r250_shapes_ct10/sample_007 | 20 | 101.0120% | 101.4491% | 0.804806% |
| r250_shapes_ct10/sample_007 | 50 | 97.7736% | 98.0835% | 0.835875% |
| r250_shapes_ct10/sample_007 | 100 | 97.0172% | 93.9645% | 9.90363% |
| r250_shapes_ct10/sample_007 | 200 | 62.3337% | 47.1388% | 47.0512% |
| r250_shapes_mri4/sample_004 | 1 | 106.1728% | 106.1728% | 4.08358e-06% |
| r250_shapes_mri4/sample_004 | 10 | 96.8894% | 97.0237% | 0.203645% |
| r250_shapes_mri4/sample_004 | 20 | 81.4513% | 69.3072% | 30.9013% |
| r250_shapes_mri4/sample_004 | 50 | 53.2176% | 31.1585% | 64.2604% |
| r250_shapes_mri4/sample_004 | 100 | 31.1715% | 31.1601% | 0.239338% |

The first accepted objectives differ by only2.75e-8 (CT) and4.08e-8 (MRI) relatively. At attempt20, the MRI objectives differ by30.9%; at attempt50, image errors are53.22% and31.16%. By attempt100 both return to a similar31% image-error plateau. The completed unrestricted MRI arm eventually reaches1.492% at1000 attempts; the other arm is still running. Earlier improvement alone does not predict the final outcome.

The initial independent gradient diagnostics also differ slightly between these mathematically equivalent controls (approximately1e-6 in their reported relative errors). No source snapshots for those early accepted states are available, so this audit compares histories rather than claiming a measured source-perturbation growth rate. GPU product/QR/SVD rounding, graph-cache state and optimizer amplification have not been separately isolated. Do not claim a causal rank-policy advantage from these two trajectories.

[Saved comparisons](trajectory_sensitivity.json) · [Live completed/partial results](latest_progress.md)
