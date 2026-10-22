# Terminal spectral weighting pilot — 2026-10-22

This is an isolated accuracy experiment, not a production correction. It tests whether reducing dominant terminal singular-value weights prevents the nearly blank reconstruction found in an original-source separated-shapes MRI case. An earlier circle MRI continuation improved under this weighting, but transfer to separated shapes is unproven.

MRI250 sample2 uses the same original cached source, model checkpoint, measurement mask, noiseless observation and rank32 initialization sketch in every arm. Five Euler steps, analytical alpha, zero enrichment, previous-V reuse, L-BFGS/geometric Armijo, displacement cap2.5, unconstrained sources,1000-attempt budget and the existing two-rejection safeguard remain fixed. Deterministic cuDNN is enabled in all arms. They run sequentially on GPU1; no concurrent variants share this sample.

## Intervention and controls

For the terminal general core, compute

\[
S=U_S\Sigma V_S^{\top},\qquad
c=\operatorname{median}(\sigma_i),\qquad
S_c=U_S\operatorname{diag}(\min(\sigma_i,c))V_S^{\top}.
\]

The median is PyTorch's lower middle value for an even rank. Compare:

1. **Unchanged:** \(g=VS^\top U^\top g_x\).
2. **Median cap:** \(g_c=VS_c^\top U^\top g_x\).
3. **Norm-matched cap:** \(g_c\|g\|/\|g_c\|\), with a zero-gradient guard.

The third arm separates direction reweighting from gradient-size changes, which can interact with L-BFGS and the initial step scale. All arms still evaluate the same actual objective and retain the original acceptance safeguard. No extra model derivative products are introduced by weighting; a small core SVD and two gradient actions are additional algebra. Factor evolution and the reused right basis are unchanged. This deliberately modifies the approximate gradient and is not an improved Jacobian approximation by definition.

If the factors were exact retained singular factors of the true Jacobian, the capped gradient would be a positive semidefinite preconditioning of the true gradient: each retained source coefficient is multiplied by \(\min(\sigma_i,c)/\sigma_i\), and omitted coefficients are set to zero. This yields a descent direction whenever the weighted gradient is nonzero. With approximate singular factors it does not guarantee alignment with the true gradient. Under the same exact-factor assumption, image-space local update weights change from \(\sigma_i^2\) to \(\sigma_i\min(\sigma_i,c)\); dominant directions become less dominant without amplifying any singular value. Neither statement guarantees reconstruction accuracy or escape from a failed region.

## Evidence and verification

The study is running. [Current verified metrics](verification.json) and [queue with exact commands](queue.json) distinguish partial results from completed runs. The new unchanged arm completes145 attempts at99.689384% image error and99.558357% residual. All146 non-timing records and its final source match the previous deterministic control exactly. The capped arm is partial at220 attempts with99.915711% image error and99.844520% residual; the norm-matched arm is pending. More attempts have not established an accuracy improvement.

At the common original source, the capped gradient has0.009433 times the original norm and cosine0.436743 with it. Thus this cap changes magnitude substantially as well as direction; the norm-matched control is necessary for interpretation. Its four repeated initial gradient evaluations are bitwise equal. Final results are not yet available for the modified arms.

Two CPU tests verify the spectral bound, factor-coordinate invariance, no mutation of the core, retained-subspace membership, nonexpansion of gradient norm, positive alignment with the original factor gradient, and norm matching. Compilation passes. The independent reporter checks source/sketch/model/operator pairing, recorded implementation hashes, original attempt budgets and displacement caps, monotone objectives, and image/complex-residual metrics from original arrays. Completed arms receive shared four-row paired D/P/DLR panels; existing D/P runs are reference results, not new executions of this control.

Raw tensors stay local and are excluded from Git. [Convergence curves](convergence.png) are partial until each saved status is terminal. The best image error is retrospective, not an objective-only selection rule.
