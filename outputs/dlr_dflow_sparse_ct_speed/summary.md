# Sparse CT speed and memory benchmark

## Protocol

- GPU: NVIDIA RTX 6000 Ada Generation (48 GiB)
- Problem: 15-view sparse CT, one test sample per resolution
- Resolutions: 64, 96, 128, 160, 200, 250 pixels
- Flow: 100 Euler steps
- Optimizer: Adam, learning rate 0.05
- Timing: one untimed warm-up optimizer iteration and three timed iterations; table reports the arithmetic mean
- DLR: rank 8, 32 trace probes, initialization oversampling 8, enrichment probes 4, factor stride 1
- DLR acceleration: exact fused lane reduction, hand JVP/VJP, shared primal tape, persistent CUDA graphs, and exact resolution-specific attention/GroupNorm kernels
- Precision: float32; no reduced-precision or altered-enrichment shortcuts
- D-Flow uses ordinary uncheckpointed autograd through 128px. Exact per-flow-step activation checkpointing is enabled at 160px and above because the 160px uncheckpointed run exceeded 48 GiB.

## Results

| Resolution | D-Flow s/iter | DLR s/iter | DLR / D-Flow | D-Flow peak allocated GiB | DLR peak allocated GiB | D-Flow peak reserved GiB | DLR peak reserved GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.277 | 2.150 | 1.68x | 4.873 | 0.405 | 4.938 | 1.547 |
| 96 | 1.287 | 6.383 | 4.96x | 12.289 | 0.881 | 12.877 | 3.201 |
| 128 | 1.324 | 15.375 | 11.61x | 25.753 | 1.544 | 26.643 | 5.713 |
| 160 | 2.529 | 32.002 | 12.65x | 0.515 | 2.901 | 0.633 | 8.271 |
| 200 | 3.113 | 70.098 | 22.52x | 1.073 | 4.620 | 1.303 | 13.260 |
| 250 | 5.937 | 158.644 | 26.72x | 2.310 | 7.272 | 2.695 | 21.336 |

Peak allocated is live tensor memory. Peak reserved is the CUDA allocator pool, including memory retained by DLR's persistent CUDA graphs. The D-Flow memory drop at 160px is caused by activation checkpointing.

The JSON record contains configuration metadata, raw iteration timings, median timings, setup and iteration peaks, persistent memory, terminal losses, and final source norms. The CSV contains the compact comparison table.
