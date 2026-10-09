# DLR failure investigation — 2026-10-08

The completed diagnostics identify a reproducible numerical-consistency problem at two selected failed sources. They do **not** establish that low rank is intrinsically inaccurate, that TF32 explains every failed reconstruction, or that disabling TF32 improves final population accuracy. All four CPU continuations have finished: spectral damping rescued the selected MRI case, but neither variant recovered the selected CT case. These are two selected-source recovery tests, not population evidence.

## Protocol and scope

Frozen data: `outputs/ct_mri_geometric_v_audit_20261008_1645/snapshot.json`. Diagnostic outputs: `outputs/dlr_failure_tests_20261008/`. Non-attention epoch-200 circles models, 128×128, 100 Euler steps, noiseless CT10/MRI4, rank 8, analytical alpha=1−t ending exactly at zero, zero enrichment, initialization oversampling 8. The sweep uses previous-V initialization and geometric Armijo L-BFGS with history 100 and displacement cap 1.28.

The CPU rank study uses three failed sources and one accurate control. GPU replays use failed CT n3/sample008 and MRI n2/sample003, plus accurate MRI n2/sample019. These are deliberately selected diagnostic cases, not independent or representative reconstruction trials. All tests preserve the saved source, truth, acquisition and model checkpoint.

No production code or sweep settings changed. All 30 monitored source hashes still match the live manifest. GPU0 remained unused. Eight bounded GPU1 replays took 17–121 seconds each, with less than 269 MiB of allocated PyTorch tensors per process, excluding CUDA context/reserved memory. The queue controller's admission loop was temporarily stopped during each replay and resumed in `finally`; existing sweep workers continued. Longer work runs on CPU.

## 1. History length and more backtracking are insufficient explanations

In the frozen 128px cohort, 83 of 88 severe DLR failures (>20% image error) stalled. Of those 83, 39 last accepted displacements were below 1e−5, and 18 below 1e−7. The six individually inspected saved states already have empty L-BFGS histories after two consecutive failed searches. The second failure therefore already tested the negative DLR gradient: another history reset alone cannot change that direction.

At trial 30, selected failed searches change only 2–5 of 16,384 float32 source coordinates. Explicit GPU replays extended halving to 60 trials but stopped when the candidate became identical to the source. Neither exact nor ordinary DLR gradients found an accepted candidate at either severe source under the original TF32 settings. This concerns the tested halving sequence, not every possible rate or direction. The optimizer already has an unchanged-source guard.

Saved right bases remain approximately orthonormal: maximum double-precision Frobenius error below 5.6e−5 in the audited 128px states. There is no obvious loss of basis orthogonality explaining the failures.

## 2. Precision and fast-primal consistency are concrete failure mechanisms

The objective uses native `euler_flow`; fast factors evolve through the shared manual/fused primal. These implement the same mathematical Euler method but do not promise identical floating-point arithmetic. At sensitive optimized sources the difference becomes substantial:

| Saved source | Factor/objective endpoint difference, TF32 on | TF32 off | Refreshed DLR gradient error, TF32 on | TF32 off |
| --- | ---: | ---: | ---: | ---: |
| CT n3/008, failed | 39.273% | 0.000733% | 81.16% | 2.516% |
| MRI n2/003, failed | 2.843% | 0.007524% | 727.35% | 2.371% |
| MRI n2/019, accurate control | 0.01044% | 0.00001195% | 38.83% | 30.35% |

Endpoint difference means `||x_factor−x_canonical||/||x_canonical||` at the same source and precision. Gradient error means `||g_DLR−g_exact||/||g_exact||`, using the canonical endpoint gradient and same-precision exact Euler adjoint. The accurate control has a small exact gradient, so its large relative gradient error is not comparable to the severe cases' absolute errors.

The TF32-on canonical objective and image replay the saved CUDA results exactly. At CT008 the DLR and exact gradients are nearly parallel (cosine 0.99970), but both tested searches fail. At MRI003 they are nearly opposite (cosine −0.98233), and both searches still fail. Disabling TF32 and recomputing factors/gradients restores agreement and local descent:

| Source | TF32-on exact / rank-8 search | TF32-off exact / rank-8 search | Rank-8 objective change within TF32-off |
| --- | --- | --- | ---: |
| CT n3/008 | Neither accepts a tested halving candidate | Both accept trial 7 | 8.77315 → 7.79367 |
| MRI n2/003 | Neither accepts a tested halving candidate | Both accept trial 8 | 1838.89730 → 1832.63547 |

Disabling TF32 changes the numerical map at the fixed saved source: the canonical image changes by 40.53% for CT008 and 20.30% for MRI003, versus only 0.0219% for the accurate control. CPU32 and CPU64 agree much more closely and corroborate the TF32-off behavior. CPU64 finite differences agree with their directional-derivative predictions at small displacements.

TF32 uses reduced input mantissa precision for convolution operations; PyTorch documents that it can alter results and can be disabled when full float32 precision is needed ([numerical accuracy documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#tensorfloat-32-tf32-on-nvidia-ampere-and-later-devices)). The data support amplification of arithmetic differences through a sensitive flow, rather than a demonstrated algebraic, indexing or aliasing bug. Autodiff follows the smooth mathematical operations; it does not make the rounded numerical objective smooth.

**This is local-descent recovery, not reconstruction recovery.** The accepted TF32-off objectives above are still higher than the saved TF32-on values (7.74796 and 1806.59927); those are different numerical baselines. CT's first TF32-off DLR step changes image error 87.178%→79.304%, while MRI changes 119.319%→119.447%. Neither is an accurate reconstruction.

Plot: `outputs/dlr_failure_tests_20261008/precision_line_search.png`. Full replay records and gradients are in `gpu/` and `gpu_verified/`; the latter contains repeated CT checks with endpoint diagnostics and the accurate control. Repeated CT factor gradients differ slightly numerically but give the same conclusion.

## 3. Higher rank improves approximation, but preserves the dominant amplification

All entries below use a common **native CPU32** source, endpoint and terminal gradient. They must not be interpreted as replaying the TF32 GPU map.

| Source | Rank 8 random | Rank 8 saved V | Rank 16 | Rank 32 | Rank 8, enrichment 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CT n3/008 | 5.629% | 2.877% | 1.667% | 0.753% | 5.726% |
| CT n3/018 | 5.773% | 3.853% | 1.139% | 0.635% | 5.842% |
| MRI n2/003 | 2.776% | 2.453% | 1.060% | 0.746% | 2.978% |
| MRI n2/019, accurate | 41.628% | 30.134% | 7.435% | 3.944% | 43.004% |

Values are relative source-gradient errors. All 20 factor endpoints exactly equal their native CPU reference; alpha and the orthogonal gradient-error decomposition pass checks. Rank-8 cosines at the failed sources already exceed 0.9988, and all 20 gradients are descent directions for this reference.

The three failed sources' leading rank-8 singular values are approximately 2,141, 247 and 51,856. Their first right singular direction contains 99.998%, 98.399% and 99.9999% of approximate gradient energy, respectively. Exact-gradient energy is also dominated by those directions. Increasing rank retains this amplification. A low-rank approximation keeps the strongest singular directions; it does not automatically damp a direction with a very large singular value.

Saved-V seeding improves the diagnostic gradient in all four cases, but this reuses the saved **same-source terminal V** for an extra factor solve. It is not a controlled reconstruction test of previous-iterate reuse, nor an exact replay of the unavailable original seed basis. Enrichment 2 slightly worsens all four approximation errors; these tests offer no reason to enable it.

Plot/table: `factor_accuracy.png`, `factor_accuracy.csv`; full reference tensors, decompositions and spectra: `factors/`.

## 4. Improvements under test and recommended order

1. **Precision-consistent recovery:** keep the rank and DLR equations, but test TF32-off from initialization or as an explicit recovery phase. When switching, recompute the objective and gradient, clear L-BFGS pairs, and rebuild precision-dependent CUDA graphs. Reusing old cached objectives/gradients would mix numerical maps. The current tests establish accepted local steps, not a speed-neutral or final-accuracy benefit.
2. **Monitor factor/objective endpoint agreement:** both endpoints are already computed. Retaining and comparing them adds reductions but no model or Jacobian products. This can identify when a fast factor solve is no longer consistent with its objective. Endpoint agreement alone does not certify gradient accuracy. Initial-source backend checks are insufficient for later optimized sources.
3. **Damp dominant singular values in the update:** cap the singular values of the small terminal core at their lower median before applying the endpoint gradient. This preserves rank, factor evolution and model-product count, but deliberately changes gradient weighting. An 8×8 SVD is additional algebra; an overall speed/accuracy gain is not yet measured. Under TF32-on, it found an MRI objective decrease (1806.599→1799.570) when ordinary/exact directions failed, but image error worsened (118.283%→118.673%). It did not help the CT point.
4. **Rank 16 as a secondary accuracy check:** it improves gradient fidelity at these sources, at additional factor cost. There is no completed reconstruction evidence here that justifies promoting it over rank 8. Extra enrichment and a larger line-search budget are lower priority based on these tests.

Four CPU continuations compare unchanged rank-8 DLR against the lower-median cap on CT008 and MRI003. They start at identical saved GPU sources/V, recompute the native CPU32 map and fresh optimizer, and use up to 100 attempts with a two-failure stop. The cap uses `torch.median()` (lower middle at rank 8). Early statistical-median preflight output was archived separately and excluded. These are recovery tests from selected failures, not fresh-start population comparisons.

All four continuations completed by 19:30; results verified at 23:47. Driver PID109536 has exited. Per-attempt source/V/gradient/optimizer archives, `report.md`, `convergence.png` and standard reconstruction panels are in `outputs/dlr_failure_tests_20261008/continuations/`.

| Case | Variant | Final image error | Final clean measurement residual | Attempts / accepted updates | Stop |
| --- | --- | ---: | ---: | ---: | --- |
| CT n3/008 | Unchanged DLR | 79.295% | 41.838% | 100 / 100 | Attempt limit |
| CT n3/008 | Median cap | 78.887% | 41.440% | 14 / 12 | Two consecutive failed searches |
| MRI n2/003 | Unchanged DLR | 119.369% | 118.070% | 100 / 100 | Attempt limit |
| MRI n2/003 | Median cap | **1.377%** | **1.053%** | 100 / 95 | Attempt limit |

The paired native-CPU initial errors were 87.179% (CT) and 119.319% (MRI). The damped MRI trajectory reached below5% image error at attempt70 and below2% at76, after a long plateau around15%. Its retrospectively best image error was1.367% at83; the table reports the final checkpoint, not ground-truth-based checkpoint selection. The undamped trajectory remained above119% throughout. This demonstrates an image-accuracy rescue in this selected MRI continuation, beyond merely an objective decrease.

CT improved only marginally: at the common14-attempt prefix, unchanged/capped errors were79.303%/78.887%. The unchanged run then reached79.295% at100, while the capped run remained stopped. Neither is a successful reconstruction.

Both variants share the exact same source-checkpoint hash, settings, acquisition, model and initial native-CPU objective; only terminal singular-value weighting differs. All objective histories are nonincreasing. The comparison supports a damping benefit for this specific MRI case on the consistent CPU map, not a GPU TF32-on rescue or an average benefit over datasets. Rank, factor evolution and derivative-product counts per gradient remain unchanged; no speed-equivalence claim is established. Next evidence should be paired fresh-start and recovery trials across more MRI/CT samples before changing defaults.

## Correctness and reproduction

The recomputed exact adjoint matches full 100-step autograd to 4.44e−16 maximum absolute error on a nonlinear float64 toy flow with complex MRI measurements. Three central finite differences have relative errors at most 2.35e−7. Checkpoint SHA256, acquisition masks, saved histories, alpha endpoint, paired continuation initialization, core-cap algebra and update bounds were checked. All five new scripts compile. The main sweep's source hashes remain unchanged.

```bash
# CPU rank/reference study
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.ct_mri_sweep.failure_factor_tests
# CPU32/64 directional checks
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.ct_mri_sweep.failure_precision_tests
# Isolated GPU1 fixed-source replay; coordinate queue admission as described above.
.venv/bin/python -m experiments.ct_mri_sweep.failure_gpu_replay \
  --output outputs/dlr_failure_tests_20261008/gpu \
  --case r128_n2_mri4/sample_003 --tf32 off --factor-tests \
  --directions exact refreshed_v8 spectral_cap
# Refresh plots and inspect the completed continuation results.
.venv/bin/python -m experiments.ct_mri_sweep.failure_report
.venv/bin/python -m experiments.ct_mri_sweep.failure_continuation --report-only
```

Keep spectral damping experimental: one completed MRI recovery is promising, while the CT recovery failed. Prioritize paired multi-sample tests under consistent precision before changing the default algorithm.
