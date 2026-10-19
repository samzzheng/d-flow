# Sparse CT LBFGS speed and memory benchmark

## Protocol

- GPU: NVIDIA RTX 6000 Ada Generation (48 GiB)
- Problem: 15-view sparse CT, one test sample per resolution
- Resolutions: 64, 96, 128, 160, 200, 250 pixels
- Flow: 100 Euler steps
- Optimizer: LBFGS, learning rate 1.0, `max_iter=1`, history 100, no line search
- Timing: one untimed warm-up outer iteration and three timed outer iterations; table reports the arithmetic mean
- Closure evaluations: exactly one for every warm-up and timed outer iteration in both methods
- DLR: rank 8, 32 trace probes, initialization oversampling 8, enrichment probes 4, factor stride 1
- DLR acceleration: exact fused lane reduction, hand JVP/VJP, shared primal tape, persistent CUDA graphs, and exact resolution-specific attention/GroupNorm kernels
- Precision: float32; no reduced-precision or altered-enrichment shortcuts
- D-Flow uses ordinary uncheckpointed autograd through 128px. Exact per-flow-step activation checkpointing is enabled at 160px and above because the 160px uncheckpointed run exceeded 48 GiB.

## Results

| Resolution | D-Flow s/iter | DLR s/iter | DLR / D-Flow | D-Flow peak allocated GiB | DLR peak allocated GiB | D-Flow peak reserved GiB | DLR peak reserved GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.268 | 2.101 | 1.66x | 4.873 | 0.414 | 4.938 | 1.568 |
| 96 | 1.320 | 6.381 | 4.83x | 12.253 | 0.864 | 12.791 | 3.191 |
| 128 | 1.317 | 15.296 | 11.61x | 25.726 | 1.527 | 26.562 | 5.633 |
| 160 | 2.514 | 32.862 | 13.07x | 0.579 | 2.965 | 0.783 | 8.455 |
| 200 | 3.120 | 70.128 | 22.48x | 1.129 | 4.630 | 1.424 | 13.268 |
| 250 | 5.945 | 158.559 | 26.67x | 2.345 | 7.280 | 2.789 | 21.340 |

Peak allocated is live tensor memory. Peak reserved is the CUDA allocator pool, including memory retained by DLR's persistent CUDA graphs. The D-Flow memory drop at 160px is caused by activation checkpointing.

The JSON record contains configuration metadata, raw iteration timings, closure counts, median timings, setup and iteration peaks, persistent memory, terminal losses, and final source norms. The CSV contains the compact comparison table.
