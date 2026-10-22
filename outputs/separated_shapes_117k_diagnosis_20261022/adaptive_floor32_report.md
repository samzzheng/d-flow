# Adaptive rank floor: completed paired controls

Raising only the minimum adaptive rank from4 to32 rescues the selected CT7 failure, preserves the easier CT20 result, and does not rescue MRI2. All use maximum128, relative core cutoff0.001, five Euler steps, analytical alpha=1-t, zero enrichment/trace probes, native-primal FP32, the original paired inputs/seeds and LBFGS/Armijo settings. Each run has the same1000-attempt cap and rejected-search safeguard. The changed floor preserves more directions while retaining the existing factor update.

|250-square case|Original floor4 image error|Floor32 image error|Attempts: old / new|Residual: old / new|
|---|---:|---:|---:|---:|
|CT10 sample7|97.082%|3.428%|97 / 829|74.187% / 0.528%|
|CT10 sample20|3.742%|3.694%|343 / 261|0.572% / 0.557%|
|MRI4× sample2|100.040%|99.768%|12 / 240|100.006% / 99.659%|

These are last-source errors, not retrospectively selected best checkpoints. Different attempt counts reflect the unchanged safeguard. This is an improvement in the ability to continue useful optimization, not an equal-completed-iteration comparison. Three selected cases do not establish population performance or a noise-regularization benefit.

## Fixed-source mechanism

At the *original floor4 failure sources*, hold source, endpoint, terminal loss gradient, seeds and saved right basis fixed, and change only the rank floor:

|Source|Floor|Terminal rank|Gradient relative error|Cosine with exact gradient|Captured squared-gradient norm|
|---|---:|---:|---:|---:|---:|
|ct7|4|4|95.84%|0.285|8.15%|
|ct7|32|32|53.62%|0.845|73.86%|
|mri2|4|4|97.80%|0.241|10.09%|
|mri2|32|32|34.75%|0.938|88.64%|

The relative core singular-value cutoff discarded directions relevant to the objective gradient. A small residual-core singular value does not by itself bound that direction's contribution to the gradient. Raising the floor repairs much of the fixed-point gradient error in both cases, but only CT7 is rescued by the corresponding complete reconstruction. Gradient accuracy at one point is not a convergence guarantee; the source and subsequent trajectory still matter.

The CT7 floor32 fast/native-autodiff comparison passes: maximum gradient difference0.0501% of the DLR gradient norm, identical endpoint and exact returned terminal alpha0. This independently checks the changing-rank execution path. The remaining approximately54% error versus the full gradient is approximation error, not a comparably large derivative-backend discrepancy.

## Paired D-Flow / P-Flow comparison

The cached production D/P controls have bitwise paired sources, truths and measurements; model/operator/seeds and flow/optimizer/stopping settings match. Their terminal image metrics were recomputed from saved arrays. The DLR column uses the new adaptive32–128 controls.

|Case|D-Flow|P-Flow|DLR D-Flow|
|---|---:|---:|---:|
|CT10 sample7|1.495%|104.025%|3.428%|
|CT10 sample20|2.816%|29.753%|3.694%|
|MRI4× sample2|99.724%|3.597%|99.768%|

MRI2 also defeats D-Flow from this source, while P-Flow succeeds. Therefore representability by the same generator is not the sole explanation for this failure. A separate earlier fixed-rank source-pool control reached3.155% on MRI2; combining that source-selection protocol with the higher adaptive floor has not yet been tested.

[CT7 comparison](adaptive_floor32_ct7/paired_methods.png) · [CT20 comparison](adaptive_floor32_ct20/paired_methods.png) · [MRI2 comparison](adaptive_floor32_mri2/paired_methods.png)

[CT pairing/settings verification](adaptive_floor32_protocol_verification.json) · [D/P control verification](adaptive_floor32_method_controls.json) · [CT7 metrics](adaptive_floor32_ct7/completed_verification.json) · [CT20 metrics](adaptive_floor32_ct20/completed_verification.json) · [MRI2 pairing/metrics](adaptive_floor32_mri2/completed_verification.json) · [Independent adaptive backend check](full_map_audit/ct7_adaptive_floor32_native.json)

All three rank-floor reconstructions and their fixed-source diagnostics are complete. The separate native FP64 MRI4 continuation is still active. No production defaults or source-review holds were changed. The broader accuracy objective remains unresolved.
