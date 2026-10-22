# Completed CT rank-retention control

All three paired policies have finished on CT10, 250², sample7: the same original source, model, observations, five Euler steps, rank range32–256, relative core cutoff0.001, no trace probing/enrichment, and L-BFGS/geometric Armijo. Each uses the original maximum1000-attempt budget and rejection safeguard.

| Policy | Final attempts | Image relative error | Observed residual | Image error at common attempt152 |
|---|---:|---:|---:|---:|
| Ordinary adaptive truncation | 836 | 5.194% | 0.929% | 71.417% |
| Force rank256 at initialization only | 593 | 34.038% | 19.922% | 69.360% |
| Additionally retain at least half the preceding rank at each flow step | 152 | 90.780% | 72.360% | 90.780% |

Gradual rank reduction is not a correction on this selected case. More retained directions need not produce a better optimization trajectory. These endpoints reflect different accepted-update counts under the same stopping policy, not equal completed optimization budgets.

The initialization-only floor is **inactive on the recorded ordinary trajectory**: initialization already retains256 at every recorded refresh. The two implementations are mathematically equivalent there. Their large final discrepancy therefore cannot be credited to different retained initial ranks. Floating-point variability and subsequent trajectory amplification remain confounders, so a single run cannot identify a causal policy advantage.

Independent saved-array verification checks cached source/model/operator/seeds, monotone objective, caps, complex metrics, rank bounds and retention rules, analytical alpha, and preserved histories across the earlier graph-cache recovery. The paired figures retain D/P results from the same original problem. MRI policy controls are still incomplete and remain labeled partial.

## Repeatability diagnostic

Three fixed CT sources were checked: the original source, the successful ordinary-adaptive endpoint, and the failed gradual endpoint. Each terminal gradient is the CT adjoint applied to one fixed residual. Eight native CUDA adjoints and model pullbacks were evaluated; the model pullback was also repeated with one identical terminal gradient.

| Source | Maximum relative change in CT terminal gradient | Maximum relative change in source gradient, varied terminal gradient | Maximum relative change with fixed terminal gradient | Fixed-terminal change with deterministic cuDNN |
|---|---:|---:|---:|---:|
| Initial | 3.49e-8 | 1.01e-6 | 7.55e-7 | 0 |
| Ordinary adaptive endpoint | 4.78e-8 | 7.98e-6 | 4.53e-6 | 0 |
| Gradual endpoint | 4.00e-8 | 4.29e-5 | 2.20e-6 | 0 |

Turning on `torch.backends.cudnn.deterministic` makes the fixed-terminal native pullback bitwise repeatable in these eight repetitions, with no change to the computed endpoint. This is a concrete repeatability improvement, not a reconstruction accuracy result. It does not remove CT-adjoint variability, certify DLR factor repeatability, or establish which numerical differences caused a particular reconstruction failure.

A CPU FP64 CT-adjoint reference preserves the actual CUDA grid values and fixed residual. Its terminal gradients differ from the CUDA FP32 results by3.68e-6–7.94e-6 relatively; propagation through the native model changes the source gradient by up to1.44e-4. This changes arithmetic, so it remains a diagnostic rather than an automatically substituted operator. No production settings have changed.

[Current verified results](live_verification.json) · [Trajectory/rank curves](latest_progress.png) · [Repeatability data](ct7_gradient_repeatability.json).

Completed paired panels: [ordinary](r250_shapes_ct10/sample_007/unrestricted256/paired_methods.png), [initialization floor](r250_shapes_ct10/sample_007/retain_initial256/paired_methods.png), [gradual](r250_shapes_ct10/sample_007/gradual256/paired_methods.png).
