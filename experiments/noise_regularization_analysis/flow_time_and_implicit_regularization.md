# Flow-time Jacobian rank and implicit noise regularization

This follows the2026-10-14 trajectory audit. A discrepancy rule supplied with the true noise norm is a useful diagnostic benchmark, but it uses information the reconstruction method may not have. Resistance to noise fitting without specifying that norm is a separate practical advantage. The claims below distinguish that observed behavior from a guaranteed noise-level estimator or a theorem about the nonlinear LBFGS implementation.

## What is measured

The flow-time sensitivity is
\[
J_t=\frac{\partial x_t}{\partial z},\qquad J_0=I.
\]
It is not the velocity Jacobian or the derivative with respect to optimizer iteration. Native automatic differentiation computes products with the actual partial Euler map, preserving the sampler's time grid, step size, and multiplication order. The approximation is
\[
\widehat J_t=(1-t)I+U_tS_tV_t^\top.
\]
Before t=1 it includes a full-dimensional identity component. The residual factor rank8 is not the rank of the full approximation. Its full spectrum is available from a small block on span(U,V), plus d-dim(span(U,V)) singular values equal to |1-t|. This block construction was checked against a dense non-symmetric example.

For smooth well-posed continuous flows, the sensitivity solves dot J_t=D_xv(t,x_t)J_t and its determinant remains nonzero while the trace integral is finite. Thus contraction usually concerns **effective numerical rank**, not a literal loss of algebraic rank. Euler discretization is a separate map and need not inherit every invertibility property. See [Neural Ordinary Differential Equations, Section4/Theorem1](https://ar5iv.labs.arxiv.org/html/1806.07366).

The diagnostics report both stable rank
\[
r_{\rm stable}(J_t)=\frac{\|J_t\|_F^2}{\sigma_1(J_t)^2}
\]
and the number of singular values accounting for99% or99.9% of squared-singular-value energy. Stable rank is continuous, not an integer count. A very large leading singular value can reduce either relative rank even when absolute sensitivity increases.

Six five-step trajectories cover the initial, D-Flow-final, and DLR-final sources of two illustrative64px single-circle cases: CT10 at5% noise/sample15 and MRI4 at20% noise/sample14. A100-step companion uses the CT case's initial source and samples t=0,.1,.25,.5,.75,.9,.95,.99,1. It is a different discretized generator, not a100-step reconstruction rerun. Native randomized spectra use width32 for five-step cases and16 for the companion, with24 independent omitted-energy probes. Leading-norm power refinement is used where needed. Unresolved high-rank energy cutoffs remain explicitly unresolved.

DLR factors are captured using the existing callback; production code is unchanged. DLR-final sources use a new factor evolution seeded by saved terminal V, not an exact replay of the optimizer's last gradient. Independent native references avoid the hand-written factor-product backend. Internal fast/native endpoint differences are recorded.

## Measured evolution

For the100-step initial-source companion:

| Flow time | Native stable rank estimate | Native99.9% energy rank | Full DLR99.9% energy rank |
| --- | ---: | --- | ---: |
| 0 | 4096 | 4092, exact identity | 4092 |
| .10 | 1655.58 | unresolved | 4092 |
| .25 | 45.57 | unresolved | 4092 |
| .50 | 1.649 | unresolved | 4085 |
| .75 | 1.026 | unresolved | 3911 |
| .90 | 1.005 | unresolved | 1687 |
| .95 | 1.004 | 2, resolved estimate | 2 |
| .99 | 1.003 | 2, resolved estimate | 2 |
| 1 | 1.003 | 2, resolved estimate | 2 |

The early native99.9% cutoffs exceed what this bounded reference resolves; they must not be replaced by the DLR values. The two definitions answer different questions: the largest singular value can dominate enough to make stable rank close to one while many weak directions still contain more than0.1% of total energy. The identity has99.9%-energy rank4092 because retaining99.9%, rather than100%, permits discarding four of its4096 equal singular values.

The native100-step spectrum has resolved99%-energy rank1 at t=.90 and99.9%-energy rank2 by t=.95. The same initial source with five Euler steps instead ends with99.9%-energy rank3 and stable rank1.95. Discretization changes the generator as well as the sensitivity evolution.

The initial-source terminal matrix approximation error is76.72% for five steps and2.41% for100 steps. This comparison does not isolate factor-integration error because the true maps also differ. It does show that low effective rank alone does not establish that the fast factors accurately approximate the full matrix. The selected realized noise-gradient norm ratio is0.473 for five steps and0.993 for100, so strong noise attenuation is not universal merely because the surrogate residual has rank8.

## Comparing exact and approximate Jacobians

The primary global error is
\[
E_F(t)=\frac{\|J_t-\widehat J_t\|_F}{\|J_t\|_F}.
\]
Independent Gaussian products estimate this without forming a dense Jacobian. At t=1, a more precise low-rank identity is also used:
\[
\|J-USV^\top\|_F^2=\|J\|_F^2+\|S\|_F^2-2\langle US,JV\rangle_F.
\]
The inner product needs only eight JVPs; only the omitted part of the native Frobenius energy remains stochastically estimated. Numerical spectral convergence is reported separately from tail-probe uncertainty.

With orthonormal U,V, the terminal approximation error also decomposes as
\[
\|J-USV^\top\|_F^2
=\underbrace{\|J(I-VV^\top)\|_F^2}_{\text{omitted source directions}}
+\underbrace{\|JV-US\|_F^2}_{\text{error within retained source directions}}.
\]
The independent left-space loss ||(I-UU^T)J||_F measures how much native image-response energy is outside the DLR image basis. These quantities were measured using native JVPs/VJPs at the successful DLR endpoints:

| Selected case | Full relative matrix error | Relative left-space loss | Relative right-space loss | Relative within-right-space error |
| --- | ---: | ---: | ---: | ---: |
| CT5%, sample15 | 47.23% | 0.35% | 44.84% | 14.82% |
| MRI20%, sample14 | 78.98% | 0.18% | 77.84% | 13.39% |

Thus the dominant image response is well represented in these two successful cases, although the full source-to-image derivative is strongly restricted. This is evidence about Jacobian response energy, not proof that the directions are semantic circle parameters. The squared decomposition agrees to1.2e-6 in normalized squared norm across all seven terminal checks. Plot lines use the same probe estimator throughout time; hollow diamonds mark the more precise terminal estimates, avoiding an artificial trend from switching estimators.

Also report:

- Measurement-weighted error ||A(J-hat J)||_F/||AJ||_F. Errors in directions invisible to the acquisition have different consequences from observable errors.
- Pullback error ||hat J^T g_x-J^T g_x||/||J^T g_x|| and cosine similarity. These measure the action actually used by the optimizer; a small global matrix error does not guarantee a small task-gradient error.
- Leading singular-value gains and containment of the native top-three right singular vectors in the DLR residual right subspace. The latter is interpreted only when the reference triplets converge; intermediate-time identity directions make residual-subspace comparisons especially easy to misread.
- Noise-only gain ||hat J^T A*epsilon||/||J^T A*epsilon||, separately from bias on the clean gradient. This is not a clean-gradient accuracy score.

Intermediate-time gradient tests use a hypothetical loss on A(x_t). Only t=1 is the actual reconstruction loss. Raw tables and plots are in `outputs/noise_regularization_analysis_20261014/flow_time/`.

## A mechanism for a residual near the noise norm without knowing that norm

At t=1 the DLR gradient is
\[
g_{DLR}=VS^\top U^\top A^*r,\qquad r=A(x)-y.
\]
For orthonormal V and nonsingular S, it vanishes when
\[
U^\top A^*r=(AU)^\top r=0.
\]
The residual need only be orthogonal to the measurement directions represented by U. It can remain nonzero. Exact D-Flow with nonsingular J and the P-Flow proxy instead require A*r=0 for exact gradient stationarity, though practical line searches can stop sooner. DLR's current termination rule also uses line-search failures, so a terminal iterate is not automatically certified stationary.

To make the mechanism explicit, consider a **fixed local linear model** in which the truth is representable and permissible image corrections lie in span(U). Fitting noisy measurements solves
\[
\min_a\frac12\|AUa-\epsilon\|^2.
\]
With P_AU=(AU)(AU)^dagger, the least-squares residual is
\[
r=-(I-P_{AU})\epsilon.
\]
For isotropic zero-mean noise in m real measurement coordinates and q=rank(AU),
\[
\boxed{\mathbb E\|r\|^2=\sigma^2(m-q),\qquad
\mathbb E\|\epsilon\|^2=\sigma^2m.}
\]
Therefore the ratio of RMS norms is sqrt(1-q/m), close to one when q is small relative to the number of observations. The fit never needs sigma. This is a ratio of expected squared norms, not an identity for every realized norm or the expectation of their ratio. The same energy argument applies to rotationally invariant fixed-norm noise.

For64px CT10, m=640; q=8 gives0.9937. For MRI4 there are1024 acquired complex coefficients, hence m=2048 real coordinates; q=8 gives0.9980. These are illustrative restricted-model predictions, not fitted parameters or guarantees for the nonlinear algorithm.

This mechanism is useful only when the retained directions explain the clean signal and AU is sufficiently observable. A wrong subspace can leave clean signal in the residual; weakly observed retained directions can still amplify noise strongly. In the ideal restricted model, image-noise variance is sigma squared times ||U(AU)^dagger||_F squared. Small q alone does not bound this quantity.

## Does this mean the gradient is less noisy?

It can reduce the **noise-induced component** of a gradient while simultaneously biasing its clean component. At a fixed source, J has no measurement-noise label: noise enters through A*epsilon. The retained directions are useful only if they preserve signal disproportionately well. Calling every discarded singular direction a noise direction would be unjustified.

For fixed J, hat J and fresh isotropic noise,
\[
\mathbb E\|\widehat g_{noisy}-g_{clean}\|^2
=\|(\widehat J-J)^\top g_{x,clean}\|^2+\sigma^2\|A\widehat J\|_F^2.
\]
The first term is bias and the second is noise sensitivity. Lower noise sensitivity need not make the whole gradient closer to the clean gradient. The preceding fixed-source audit contains counterexamples to that stronger claim.

In our actual optimizer, locally delta x is approximately -eta J H_k hat J^T A*r, where H_k is the LBFGS inverse-Hessian action. It is not generally -eta hat J hat J^T A*r. The basis changes, and the optimizer uses history, so the fixed-subspace calculation explains a plausible local regularization mechanism rather than proving convergence on a fixed manifold. At each frozen step the noise-to-update map still has rank at most the DLR rank, but its singular-value gains matter and successive subspaces can explore more directions.

The analytical baseline (1-t)I preserves broad sensitivity early and removes that ambient component at the terminal gradient. This is consistent with a learned tangent-space restriction near the data, but geometric alignment with circle position/radius has not been established by singular values alone. A decisive control is to compare exact projected gradients VV^T g_exact, true truncated-SVD gradients, and the evolved DLR factors with the same optimizer and sources. That separates subspace restriction, coefficient/gain distortion, and optimizer interaction.

## Endpoint residual geometry across60 samples

An additional CPU check covers all30 single-circle64px samples for each inverse problem at20% noise. It recomputes warm-seeded terminal factors and projects the actual noise onto the measurement subspaces AU and AJV. The second uses native JVPs to measure the actual image tangent generated by the right basis, before LBFGS mixing.

| Problem | Median q for AJV | Predicted isotropic RMS ratio | Realized projected-noise ratio | Actual residual/noise | Cosine(actual residual, projected-noise residual) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CT10 | 8 | .9937 | .9937 | .9974 | .9977 |
| MRI4 | 8 | .9980 | .9983 | 1.0006 | .9975 |

All30 cases are retained in each median. The AU calculation gives similar residual cosines .9979 CT and .9976 MRI. This is consistent with leaving the noise outside the retained measurement subspace. It does not prove that the full nonlinear optimizer is an orthogonal projection: the saved basis depends on the noisy data, and the fixed-subspace expectation formula assumes a fixed basis independent of fresh noise.

Among the five MRI cases with image error above10%, median residual/noise is1.885 and the residual cosine falls to.524. These cases do not follow the successful-fit picture. Conversely, some large CT image errors still have residuals near the noise norm, illustrating that measurement agreement alone does not certify image accuracy. No noise-level stopping rule was added or used by the reconstructions.

## Reproduction

```bash
.venv/bin/python -m experiments.noise_regularization_analysis.flow_time --steps 5 --case ct
.venv/bin/python -m experiments.noise_regularization_analysis.flow_time --steps 5 --case mri
.venv/bin/python -m experiments.noise_regularization_analysis.flow_time --steps 100 --sources initial --case ct --width 16 --refinements 2
.venv/bin/python -m experiments.noise_regularization_analysis.flow_time --steps 5 --case ct --refine-existing
.venv/bin/python -m experiments.noise_regularization_analysis.flow_time --steps 5 --case mri --refine-existing
.venv/bin/python -m experiments.noise_regularization_analysis.flow_time --steps 100 --case ct --refine-existing
.venv/bin/python -m experiments.noise_regularization_analysis.subspace_residual
.venv/bin/python -m experiments.noise_regularization_analysis.flow_time_report
```

Run refinement only after the corresponding measurement process has finished. Native endpoint equality, adjoint consistency, analytical alpha, the small-block spectrum, and matrix-error identities are checked. Preliminary division-by-step-count references that failed endpoint equivalence at a highly sensitive source are archived separately and excluded from the report; the accepted prefix uses exactly the sampler's dt-times-velocity arithmetic.
