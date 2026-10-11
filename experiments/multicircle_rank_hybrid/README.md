# Multi-circle Jacobian ranks and alternating P/DLR pilot

This isolated study leaves the production CT/MRI sweep unchanged. All workers
use GPU 0, FP32 with TF32 disabled, epoch-200 non-attention models, and **100
Euler steps**. The controller permits at most one rank worker and one hybrid
worker, each with two CPU threads. Reserved memory stays below 44,000 MiB.

## Independent ranks

Cover all available combinations of resolutions 64, 96, 128, 160, 200, 250
and circle counts 1–4. At launch, the 200px n3/n4 epoch-200 checkpoints are
unavailable; record these cells explicitly and check their presence again at
the end. Measure three paired Gaussian initial sources per model. Additionally
measure available completed D-Flow and DLR final sources from sample 0 of the
current 64/128/250 CT10/MRI4 sweep. Optimized sources are a separate cohort.

The matrix is the **Euler-map Jacobian**, not the image array or dataset PCA.
Native per-step automatic differentiation along the saved native trajectory
applies its exact mathematical JVP/VJP without storing a dense Jacobian. CUDA
graphs accelerate replay without using manual DLR derivatives. Each source
checks these products against independent full-map AD and adjoint consistency.

Randomized subspace iteration uses widths 48, 96, 192, 256 and up to three
refinements at each width. A thin SVD produces approximate singular triplets.
Twenty-four independent Gaussian products estimate energy outside the computed
left basis. Total energy includes that tail; it is not normalized only by the
singular values computed so far. Report energy ranks at 99%, 99.9%, and 99.99%
(relative Frobenius errors 10%, 3.16%, and 1%). Record tail standard error,
two-standard-error rank sensitivity, singular-threshold counts, triplet
residuals, and rank stability. A resolved estimate requires stable ranks, a
resolved 99.99% tail, and relevant left triplet residuals below 1e-3 of sigma1.
The other triplet equation holds by the small SVD construction. This is
numerical evidence, not a rigorous proof of globally leading triplets. Mark
unresolved results explicitly rather than assigning rank 256.

Energy rank can be small because of one huge singular value even when weaker
directions matter to reconstruction. No production rank is selected by this
pilot. This measures terminal spectra; intermediate-time residual ranks and
measurement-weighted truncation errors remain separate questions.

## Alternating reconstructions

Fresh paired reconstructions on samples 0–2, resolutions 64/128/250, circles
1–3, CT10 and MRI4: 54 cases, four variants (216 runs). Wait for the production
sweep to create any missing paired inputs rather than fabricate a different
case. Run pure P-Flow, pure rank-8 DLR, 1P:1DLR, and 3P:1DLR. Each variant has
up to 100 **attempts**, noiseless observations, the same initial source, and
the existing LBFGS/geometric Armijo settings. Reuse previous V, analytical
alpha=1-t, no trace probes, and zero enrichment. Rank 8 remains the preexisting
control while independent ranks are measured; it is not a new rank choice.

Hold the estimator fixed throughout each optimizer step, including its two
gradient endpoints. Clear curvature pairs and cached gradients when switching;
retain the global attempt count so first-attempt scaling is not repeated.
Consequently 1:1 uses no accumulated curvature, whereas 3:1 can retain history
within its consecutive P steps. The hybrids therefore change both direction
selection and usable history. This pilot does not isolate those two effects;
history-reset pure controls would be a follow-up ablation.

Failed searches still advance the schedule. Stop pure methods after two
consecutive failures; stop hybrids after two entire failed cycles (four/eight
consecutive failures). This explicitly differs from the pure stopping policy
to allow both estimators a chance. Actual accepted-step ratios are reported by
histories. Every accepted step strictly decreases the canonical objective and
obeys the 0.01 sqrt(d) displacement cap. Surrogate gradients supply Armijo's
slope estimate; this is not a true-gradient Wolfe test.

Save inputs, hashes, optimizer states, source checkpoints, gradients' estimator
labels, accepted displacements, and histories. Compare final checkpoints and
common iteration prefixes; do not select results using ground truth. Each case
gets error-versus-attempt curves and shared-renderer reconstruction panels.
The DLR column is explicitly labeled by variant in each figure title; D-Flow
is marked unavailable because it is outside this four-variant rerun.

## Run

```sh
.venv/bin/python -m experiments.multicircle_rank_hybrid.test_study
.venv/bin/python -m experiments.multicircle_rank_hybrid.queue \
  --output outputs/multicircle_rank_hybrid_20261011
.venv/bin/python -m experiments.multicircle_rank_hybrid.report \
  outputs/multicircle_rank_hybrid_20261011
```

The queue updates `queue_status.json`, `rank_report.md`, `ranks.csv`, and
`hybrid_report.md` as tasks finish. SIGTERM checkpoints hybrid workers at an
iteration boundary; interrupted spectral sources restart independently.
