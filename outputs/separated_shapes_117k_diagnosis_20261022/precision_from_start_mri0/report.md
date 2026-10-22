# Precision from the original source

Paired MRI250 sample0, noiseless, rank32, five Euler steps, analytical alpha, zero enrichment, previous-V reuse and unchanged LBFGS/geometric Armijo with1000-attempt cap. FP32 and FP64 run sequentially onGPU1. Original source values, observations, model weights and masks are exactly paired; Rademacher seeds are unchanged. No restart from a failed optimizer state.

Initial accelerated/native DLR gradient discrepancies: FP321.02e-5, FP641.76e-14, with identical tested endpoints. Initial canonical source hashes match.

FP32 completed after64 attempts at99.844% image error and99.792% residual, following two rejected searches. FP64 has passed initialization validation and is running; its accuracy is not yet known. A successful FP64 continuation would support precision prevention in this selected case, not a population guarantee.

[Launch](launch.json) · [FP32 initialization](float32/initial_validation.json) · [FP64 initialization](float64/initial_validation.json)
