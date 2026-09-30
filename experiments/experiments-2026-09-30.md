# Experiments — 2026-09-30

Times are local America/New_York; durations in HH:MM.

| Time | Experiment | Motivation | What Tried / Evidence | Correctness | Timing | Merge decision |
| --- | --- | --- | --- | --- | --- | --- |
| 15:35 | First-order D-Flow/DLR pilot launched | Approximate gradients can interact with LBFGS; fresh-factor GD/Adam comparisons and shared-source gradient diagnostics can distinguish optimizer effects, subspace errors, and noise suppression. | Implemented isolated 64×64 non-attention epoch-100 study: 100 Euler steps, 15-angle sparse CT, 1% relative noise, free/spherical sources, analytical alpha=1−t with zero trace probes. CPU pilot started; results pending. | Six numerical tests pass: analytical alpha/no trace columns, exact discrete products/dense spectral reference, projected curves/caps, first-attempt scaling after failure, Radon finite differences, and orthogonal error decomposition. | Budget 04:00; two CPU threads; no GPU. | Commit 1a963f5; isolated runner only, no production changes. |
