# P-Flow pilot

This folder implements the **supplied pseudocode**, not a verified reproduction
of the paper's original code or hyperparameters. It leaves DLR and existing
benchmark scripts unchanged.

Given the source `z`, integrate `x = G(z)` with Euler under `no_grad`, compute
`g = grad_x (0.5 * ||H(x) - y||²)`, then update

```
z_trial = z - learning_rate * g
z_next  = sqrt(d) * z_trial / ||z_trial||
```

The sensitivity proxy is the identity. No Jacobian factors, model backward
pass, or line search are used by P-Flow. Projection is per batch member.
`p_flow.py` exposes `proxy_gradient`, `project_source`, and a complete `step`.
The operator is caller supplied; blur belongs only to the experiment runner.

```python
from p_flow import step

for _ in range(outer_steps):
    source, (endpoint, loss, gradient) = step(
        model, source, forward_operator, measurement,
        learning_rate=0.1, n_steps=100,
    )
# endpoint above is the image BEFORE the last update; integrate the returned
# source once more when reporting its final reconstruction.
```

## Run

From the repository root:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m unittest p_flow.test_p_flow

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m p_flow.benchmark \
  --out p_flow/outputs/pilot_64_2026_09_29
```

The runner refuses to overwrite an existing output directory. Defaults:

- CPU, two threads, float32; no GPU jobs.
- Non-attention circles-64 model, epoch 100, **100 Euler steps**.
- Gaussian blur: 9×9 kernel, sigma 1.5 pixels, zero padding. Measurement noise
  standard deviation 0.05. Squared residual sum loss, without image clipping.
- One validation image: independently choose each method's constant learning
  rate from 0.001, 0.01, 0.1, 1 after five updates by final raw image MSE.
- Three held-out test images, 15 updates. Paired source seeds and noise across
  all methods. Initial Gaussian draws are normalized onto radius sqrt(d).
- P-Flow, exact uncheckpointed D-Flow, and fast DLR use the same projected
  gradient-descent update. DLR uses rank 8, four trace probes, no enrichment,
  terminal alpha zero, and factor refresh every five outer updates.
- These are fixed-rate updates as supplied, without LBFGS first-step scaling,
  a displacement cap, or Armijo. This is **not** a comparison against tuned
  LBFGS + Armijo D-Flow.

Each iteration's time includes generation, gradient calculation, and source
update. Extra diagnostics, accuracy reporting, and the final forward solve are
excluded. DLR timing includes its scheduled refreshes and their reuse. The
forward solve following the final update is always evaluated for final metrics.

`results.json` contains complete settings, model/data hashes, calibration
histories, paired test histories, raw/clipped PSNR and MSE, measurement MSE,
and shared-initial-source gradient diagnostics. `images.pt` contains ground
truth, observations, initial generations, and final reconstructions.

## Interpretation

Unprojected alignment `cos(J.T @ g, g)` alone does not establish descent after
shell projection. On the sphere, the first-order direction is the tangent
projection `P_z @ g`, where `P_z = I - z z.T / d`. The runner also measures the
cosine of projected gradients and the true derivative `-<J.T @ g, P_z @ g>`.
A negative derivative supports local descent; it does not guarantee that a
finite fixed step decreases the objective.

This is a small pilot on a single task and model, with short validation tuning.
Neither a global convergence claim nor a general reconstruction-quality ranking
is supported by it. See `results.md` for the measured pilot results.
