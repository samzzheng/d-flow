# DLR failure investigation: fixed-source evidence

Snapshot: 2026-10-22, 01:50 local time. Reconstruction controls are still running.

## Verified findings

- The manual normalization path could evolve a different forward image and even reverse the approximate gradient. Native normalization now gives the same endpoint as the objective. Independent native-autodiff validation replaces the old comparison between two hand-coded derivative paths. This correction alone did not resolve the failed reconstructions.
- On selected 250-square CT0/MRI1/MRI0 cases, rank32 DLR with measurement-informed initialization ended at 37.12/12.31/9.34% image error. Matched random mixing ended at 109.49/99.94/100.00%. This is selected-case evidence for initialization, not a population result.
- With the identical cached measurement-informed source, rank128 DLR on MRI1 finished at **2.405%**, versus rank32's **12.306%**. CT0 did not improve: rank128 finished at 39.22%, versus rank32's 37.12%. Larger rank is not a universal correction.

## Why the rank32 MRI1 reconstruction stopped improving

All diagnostics below hold the model, source, endpoint, observation, and terminal measurement gradient fixed at the completed 12.306%-error reconstruction. The exact source-gradient norm is 0.029815. Independent products through the complete discrete five-step map agree with native backpropagation to about 1.4e-6 relative error.

| Approximation | Rank | Reuse saved V | Gradient relative error | Exact-gradient component outside V | Cosine with exact gradient |
| --- | ---: | --- | ---: | ---: | ---: |
| DLR | 32 | No | 101.56% | 95.89% | -0.0926 |
| DLR | 32 | Yes | 100.20% | 96.46% | -0.0148 |
| DLR | 128 | No | 100.21% | 95.03% | 0.0562 |
| DLR | 128 | Yes | 99.02% | 95.32% | 0.1407 |
| DLR with terminal-gradient sketch column | 32 | Yes | 96.72% | 93.95% | 0.2555 |
| DLR with terminal-gradient sketch column | 128 | Yes | 95.24% | 92.59% | 0.3053 |

The current right subspace excludes most of the exact gradient at this source. Correcting only the coefficients within that subspace cannot recover the excluded component. These measurements establish a local limitation, not the cause of every failed trajectory.

The terminal-gradient seed replaces one existing random sketch column. It retains the same sketch width, rank, analytical alpha=1-t, KSL evolution, and objective. No additional Jacobian-product columns are introduced. The experimental evaluator currently computes the terminal measurement gradient a second time; this could be removed if reconstruction results justify adoption.

## Controls that did not provide a correction

- **More accurate S/L integration:** exact frozen S/L substeps, with the existing Euler K-step and Euler-target recompression, passed a fine-RK4 matrix check but did not materially improve the real gradient. Rank32 with reused V still had 100.14% gradient error. This remains an isolated experiment, not production behavior.
- **Scalar identity correction:** even fitting the best scalar using the exact gradient only reduces reused rank32 error from 100.20% to 97.77%; the fitted scalar is negative. An omitted positive identity baseline does not explain this case well. Production alpha remains unchanged.
- **Terminal randomized SVD:** three power iterations yield rank32/rank128 gradient errors of 95.36/91.88%, but the references are **uncertified**: maximum singular-triplet residual is 0.00817 times the leading singular value, and the last gradient changes are 4.96/10.84% of the exact-gradient norm. These fail the earlier 1e-3 and 1% criteria. Do not call these the true truncated-SVD gradients or use them to rule out integration error.

## Reconstruction follow-up

The terminal-gradient sketch change is being compared with an identical cold-optimizer restart at the saved failed sources. Both arms reuse the saved V basis and receive the same additional optimizer budget. The unchanged MRI1 restart makes no progress (12.306%); the seeded restart is progressing but is not complete. CT0 and MRI0 controls are also running.

The separate rank-adaptive pilot remains active. Its first MRI case fails at every tested rank choice; its completed CT20 adaptive result is 3.742%, versus fixed32's 3.946%. A single close CT comparison does not establish an adaptive advantage, especially given GPU floating-point sensitivity.

## Reproducibility

- Fixed-source code: `experiments/separated_shapes_117k/gradient_audit.py`.
- Independent product and frozen-S/L checks: `tests/test_discrete_flow_products.py` (two tests pass).
- Raw evidence: `full_map_audit/` in this directory.
- Paired restart controls: `terminal_seed_restart/`, `terminal_seed_restart_ct0/`, `terminal_seed_restart_mri0/`.
- Adaptive pilot: [rank paths and current saved metrics](adaptive_pilot/report.md).

The production sweep remains held at its source-review guard. Corrected experiments are separate from the original results. The broader accuracy objective is not yet complete.
