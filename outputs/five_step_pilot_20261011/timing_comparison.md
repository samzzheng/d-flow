# Recorded five-step versus 100-step wall times

Paired sample0 at64/128/250, CT10 and MRI4, noiseless. Identical saved truths, sources, masks and observations; checkpoint hashes and numerical settings verified. One-vs-thirty-sample dataset hashes differ, so pairing was checked from saved tensors.

These are solver wall times including Armijo, metrics and intermediate checkpointing. They exclude model loading, backend validation, final figure writing and queue waits. GPU contention differed; optimizer trajectories and line-search counts also differ. This is not an isolated ODE benchmark or time-to-equal-accuracy comparison.

| Method | 5 steps: s/attempt | 100 steps: s/attempt | Observed ratio | 5 steps: total min | 100 steps: total min |
| --- | --- | --- | --- | --- | --- |
| D-Flow | 0.160 | 3.380 | 21.1 | 15.32 | 29.97 |
| P-Flow | 0.496 | 12.714 | 25.6 | 23.61 | 52.55 |
| DLR D-Flow | 0.334 | 7.423 | 22.2 | 16.85 | 49.36 |

Entire six-resolution five-step pilot, including smoke/overhead: 100.86 elapsed minutes.
