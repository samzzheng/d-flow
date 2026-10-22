# Adaptive rank floor control

Raise only the minimum adaptive rank from4 to32; retain maximum128 and relative core cutoff0.001. Five Euler steps, noiseless CT10, same native-primal backend, paired initial source, measurements, model, seeds, original LBFGS/Armijo and1000-attempt limit. This tests whether the earlier terminal rank collapse to4 caused a regression without changing the existing factor-update rule.

Two separate-sample GPU1 runs are active:250-square CT sample7 (previous adaptive failure97.082%) and sample20 (previous success3.742%). There are no final results yet. Early trajectories are not evidence of a winner. The native FP64 MRI continuation runs on a different sample.

[Bitwise pairing and settings check](adaptive_floor32_protocol_verification.json) confirms the only settings difference is `rank_min: 4 -> 32`. The source-review hold and production defaults remain unchanged.
