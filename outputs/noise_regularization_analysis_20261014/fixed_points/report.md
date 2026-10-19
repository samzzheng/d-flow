# Fixed-source clean/noisy gradient diagnostics

Illustrative 64² single-circle cases; five Euler steps, rank 8, analytical alpha = 1-t. CPU only.

R = ||approximate noisy gradient - exact clean gradient|| / ||exact noisy gradient - exact clean gradient||.
R < 1 supports the clean-gradient hypothesis at that fixed point. Noise ratio measures only the noise-induced gradient, excluding clean-gradient bias.

| Case | Source | Basis seed / backend | Image error (%) | DLR R | Projected R | DLR noise norm / exact noise norm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| r64_n1_ct10 / 5% / 15 | initial | fresh | 190.228 | 299.5799 | 264.7183 | 0.4729 |
| r64_n1_ct10 / 5% / 15 | D-Flow final | fresh | 104.063 | 0.5553 | 0.6665 | 0.2151 |
| r64_n1_ct10 / 5% / 15 | DLR final | fresh | 5.053 | 1.0888 | 1.0826 | 0.7279 |
| r64_n1_ct10 / 5% / 15 | DLR final | saved terminal V, new evolution | 5.053 | 1.0955 | 1.0926 | 0.8060 |
| r64_n1_ct10 / 5% / 15 | D-Flow final | fresh / native AD control | 104.063 | 0.5785 | 0.6661 | 0.2152 |
| r64_n1_mri4 / 20% / 14 | initial | fresh | 113.509 | 33.1919 | 30.3877 | 0.3546 |
| r64_n1_mri4 / 20% / 14 | D-Flow final | fresh | 22.671 | 1.0000 | 1.0000 | 0.8461 |
| r64_n1_mri4 / 20% / 14 | DLR final | fresh | 2.943 | 1.0008 | 1.0007 | 0.6142 |
| r64_n1_mri4 / 20% / 14 | DLR final | saved terminal V, new evolution | 2.943 | 1.0005 | 1.0001 | 0.6187 |
| r64_n1_mri4 / 20% / 14 | D-Flow final | fresh / native AD control | 22.671 | 1.0000 | 1.0000 | 0.8461 |

The noise-only gradient is attenuated in these examples, but R is not consistently below one. A smaller noise component does not by itself imply a more accurate total gradient: discarded clean-gradient components contribute bias.

The saved-V calculation evaluates a new factor evolution at the saved source. It does not recover the gradient used by the final production optimizer step.

The following check compares the actual infinitesimal image update J Bᵀ to its symmetric surrogate B Bᵀ, where B = U S Vᵀ. A small discrepancy supports that approximation locally, not throughout the reconstruction.

| Case | Source | Initialization / backend | ||JV-US|| / ||US|| | ||JBᵀ-BBᵀ|| / ||BBᵀ|| | Factor-primal endpoint discrepancy |
| --- | --- | --- | ---: | ---: | ---: |
| ct 5% | initial | fresh / CPU hand products, shared primal | 0.9303 | 0.5393 | 5.09e-07 |
| ct 5% | D-Flow final | fresh / CPU hand products, shared primal | 1.6742 | 0.0825 | 0.00171 |
| ct 5% | DLR final | fresh / CPU hand products, shared primal | 0.2912 | 0.0524 | 3.74e-06 |
| ct 5% | DLR final | saved terminal V / CPU hand products, shared primal | 0.1696 | 0.0091 | 3.74e-06 |
| ct 5% | D-Flow final | fresh / native AD, canonical primal control | 1.6714 | 0.0831 | 0 |
| mri 20% | initial | fresh / CPU hand products, shared primal | 0.7982 | 0.1998 | 4.12e-07 |
| mri 20% | D-Flow final | fresh / CPU hand products, shared primal | 0.4187 | 0.4154 | 2.62e-07 |
| mri 20% | DLR final | fresh / CPU hand products, shared primal | 0.3276 | 0.0367 | 2.79e-06 |
| mri 20% | DLR final | saved terminal V / CPU hand products, shared primal | 0.2177 | 0.0038 | 2.79e-06 |
| mri 20% | D-Flow final | fresh / native AD, canonical primal control | 0.4187 | 0.4154 | 0 |

- CPU recomputation of saved sources; production CUDA kernels are not replayed.
- CPU randomized sketches are new draws from the saved seed, not bitwise reproductions of CUDA random draws.
- Factor bases are fixed when varying noise at each source.
- Ten fresh noise draws are fixed-point diagnostics, not reconstruction trials.
- Only saved terminal sources are available; no intermediate source trajectories.
- DLR core singular values are not a certified spectrum of the full Jacobian.
- All gradients use the canonical Euler endpoint; any internal factor-primal drift is recorded, not substituted into the loss.
