# Accelerated FP64 factor recovery

The native FP64 controls expose objective decreases that FP32 misses at selected failed sources. Their cost motivates testing the existing shared-linearization/CUDA-graph derivative backend in FP64, keeping the same DLR equations and native GroupNorm statistics.

| Fixed source | Factor configuration | Maximum accelerated/native gradient difference | Endpoint difference |
| --- | --- | ---: | ---: |
| 250² MRI sample 4, failed retained-seed endpoint | Fixed rank32, ordinary startup in this backend audit | 1.94e-13 | Bitwise identical |
| 250² CT sample20, completed higher-floor endpoint | Adaptive64–128 | 5.20e-14 | Bitwise identical |
| 250² MRI sample2, native recovery attempt138 | Adaptive64–128 | 3.98e-13 | Bitwise identical |

Each source is checked with fresh and reused right bases. The comparison is to independent native model autodiff, not another hand-derived backend. Terminal alpha is exactly zero. It validates the backend at these sources; it does not establish that the DLR gradient equals the full gradient. For example, the reused-basis MRI2 DLR gradient still has34.47% relative error against the exact gradient at attempt138.

The CT control also measures two warmed repetitions of complete factor refresh plus gradient action per backend. Accelerated times are15.47–15.63 seconds; native times30.66–30.78 seconds. This is about2x in this shared-GPU measurement, excludes forward-objective/line-search work and initial graph capture, and is not a general FP64-versus-FP32 timing claim. Repeated gradient actions are checked for agreement. Five existing CPU shared-linearization tests also pass, covering FP32/FP64 and independent JVP/VJP comparisons.

The MRI2 continuation uses graph replay only after its first50 native FP64 updates and after validation at that exact checkpoint. It preserves optimizer state and the original remaining budget. The configuration records the backend transition. No production default changes.

[Fixed-rank MRI audit](mri4_fp64_backend.json) · [Adaptive CT audit and timings](ct20_fp64_adaptive_backend.json) · [Continuation-checkpoint MRI audit](mri2_fp64_adaptive_backend.json) · [Recovery report](adaptive_pool_precision_report.md)
