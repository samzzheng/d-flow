# Separated-shapes measurement-loss pilot

Two selected failure cases: 250² CT10 sample 0 and MRI4× sample 1. Noiseless; existing 117k model, five Euler steps, rank32 DLR, analytical alpha=1-t, no trace probing/enrichment. Original cached sources, truths, observations, acquisition masks and DLR seeds are paired. All three methods are tested. This is a mechanism pilot, not a representative accuracy estimate.

| Variant | Objective and schedule |
| --- | --- |
| l2 | Ordinary squared measurement residual, up to1000 attempts |
| coarse_to_fine | 0.1 ordinary L2 +0.9 filtered L2; Gaussian sigma=0.04N,0.02N,0.01N,0 |
| intensity | Ordinary L2 plus9 times the squared DC/projection-total component |
| l2_resets | Ordinary L2 with the same stage limits and optimizer resets as coarse-to-fine |

For CT, the Gaussian convolution acts separately on each projection's detector axis with zero extension and a normalized kernel truncated at four sigma. For MRI, multiply the acquired complex residual by exp(-2*pi²*sigma²*|f|²), with centered FFT frequencies in cycles/pixel. The squared loss thus uses the square of this multiplier; no magnitude-only data fitting. At sigma=0 the implementation returns exactly ordinary L2.

For CT intensity weighting, C sums each projection divided by sqrt(N), making each row of C unit norm. MRI C extracts the acquired complex DC coefficient. Lambda9 gives10 times the ordinary weight to these components. No images are clipped and no extra data are introduced.

The first three continuation stages allow100 attempts each; the final ordinary-L2 stage receives all remaining attempts from the1000 total. Two consecutive rejected searches advance to the next stage, rather than terminating continuation before it reaches ordinary L2. L2-resets uses the identical rule. L-BFGS and objective/gradient caches are reset at every stage; the previous right basis is retained. Actual accepted update counts may differ. Stage transitions are fixed by attempts/rejections, never by ground truth.

All methods retain lr1, history100, geometric Armijo, max30 trials, c1=1e-4, and displacement cap0.01*sqrt(d)=2.5. Native model/DLR products, deterministic cuDNN and FP32 with TF32 disabled are shared across arms. CT GPU adjoints are not assumed bitwise deterministic. GPU1 only, one worker per sample. The production algorithm and its default loss are unchanged.

`ct/` and `mri/` contain automatically refreshed reports, image-error curves, all-method panels and original-L2 metrics. The `phi` history field is the active weighted objective and must not be compared between losses/stages. `original_l2` and relative measurement residuals provide a common scale. Final checkpoints, not ground-truth-selected best checkpoints, supply the table and figures. Missing methods remain marked unavailable until completed.

Validation: CPU complex gradcheck and exact full-Euler finite differences; bitwise original-L2 evaluator compatibility on a small nonlinear flow;24 four-attempt GPU smoke arms, including all stage handoffs. Independent saved-array checks cover source hashes, budgets, caps, per-stage monotonicity, weighted objective values and complex residual/image metrics. Smoke results are not convergence evidence.

Run a case:

```sh
MPLCONFIGDIR=/tmp/dflow-mpl UV_CACHE_DIR=/tmp/dflow-uv-cache uv run --no-sync --python .venv/bin/python python -m experiments.separated_shapes_117k.measurement_loss_study --case r250_shapes_mri4/sample_001 --out outputs/separated_shapes_loss_pilot_20261022/mri
```

Verify completed arms:

```sh
MPLCONFIGDIR=/tmp/dflow-mpl UV_CACHE_DIR=/tmp/dflow-uv-cache uv run --no-sync --python .venv/bin/python python -m experiments.separated_shapes_117k.verify_measurement_loss_study outputs/separated_shapes_loss_pilot_20261022/mri
```

Transport is deferred until these simpler controls show whether changing measurement weighting improves the selected failures. A CT transport experiment would need an explicit treatment of negative predictions and unequal projection mass; it is not interchangeable with complex MRI fitting.

Related motivation, not a guarantee for this generator: [Sobolev discrepancy norms](https://arxiv.org/html/2109.06255v2) and [Wasserstein signal fitting](https://arxiv.org/html/1311.4581v1).
