# DLR Gradient Flow

An isolated implementation of sensitivity transport from
[Section 11](../dlr_dflow/notes/dlr_dflow_notes-1789586570539.md), with pure-secant
and endpoint-correction variants for testing speed and accuracy. Production DLR
D-Flow and the running CT/MRI sweep are unchanged.

## Literal sensitivity transport

Let \(G\) be the actual \(N\)-step Euler map and \(J(z)=D_zG(z)\).
For a source displacement \(\delta\), `transport_factors` computes matrix actions
of \(W=D_zJ(z)[\delta]\), then evolves terminal factors by
\[
K=US+WV,\quad U_+\widehat S=\operatorname{QR}(K),\qquad
\widetilde S=\widehat S-U_+^\top WV,
\]
\[
L=V\widetilde S^\top+W^\top U_+,\qquad
V_+S_+^\top=\operatorname{QR}(L).
\]
The displacement includes the optimization-time step. These are frozen-variation
K/S/L updates: \(J(z+\delta)=J(z)+W+O(\|\delta\|^2)\), with additional error
from the old factors and the rank restriction. Zero displacement preserves the
represented matrix. No dense Jacobian or second-variation matrix is formed.

`variation_products(model, source, displacement, n_steps=100)` returns callable
actions of \(W\) and \(W^\top\), accepting matrices shaped `(dimension, lanes)`.
`transport_factors(..., factors, n_steps=100)` returns `(U, S, V)`.
The exact backend differentiates each native Euler update of the state and its
source-direction sensitivity. Local reverse differentiation also avoids native
SiLU's unsupported double-forward derivative. The `finite_difference` backend
instead uses \([J(z+\epsilon\delta)-J(z-\epsilon\delta)]/(2\epsilon)\);
this is an explicitly approximate derivative with truncation and cancellation
error. The experiment runner currently uses \(\epsilon=1\).

All terminal gradients are \(VS^\top U^\top g_x\). Initialization uses the
existing fast DLR solver with analytical \(\alpha_t=1-t\); \(\alpha_1=0\)
exactly, so there is no terminal identity contribution.

## Pure Jacobian-secant transport

`secant_transport_factors(model, old_source, new_source, factors, n_steps=100,
products=None)` returns the new Euler endpoint and factors. It uses
\[
\Delta=J(z_{\rm new})-J(z_{\rm old})
=\int_0^1 D_zJ(z_{\rm old}+s\delta)[\delta]\,ds
=W+O(\|\delta\|^2)
\]
in place of \(W\) in the same frozen K/S/L step. Four first-derivative passes
compute \(\Delta V\) and \(\Delta^\top U_+\). This agrees with literal Section 11
transport to first order, without Hessian products. It retains old factor error;
zero source movement preserves the factors exactly. At full rank the result is
\(U_{\rm old}S_{\rm old}V_{\rm old}^\top+\Delta\), not necessarily \(J(z_{\rm new})\).

## Corrected endpoint variant

`secant_factors(model, source, factors, n_steps=100, products=None)` returns
the new Euler endpoint and factors. It freezes the increment
\(\Delta=J(z_{\rm new})-U_{\rm old}S_{\rm old}V_{\rm old}^\top\).
Applying the same K/S/L algebra cancels the old core and left basis:
\[
K=J(z_{\rm new})V_{\rm old},\quad U_+=\operatorname{QR}(K)_Q,\qquad
L=J(z_{\rm new})^\top U_+,\quad V_+S_+^\top=\operatorname{QR}(L).
\]
This is one endpoint subspace iteration using two passes of first-derivative
products. It corrects existing factor error as well as source movement; it can
change the factors even at zero displacement. It is therefore distinct from
literal Section 11 transport and from the pure secant \(J(z_{\rm new})-J(z_{\rm old})\).
A supplied `GraphedAProducts` accelerates local products; its primal and
derivatives must agree with the canonical Euler objective.

## Paired experiments

The runner loads saved truths, initial sources, measurements, acquisition masks,
seeds, trained models, and settings from the FP32 sweep. Both convolution and
matrix-multiply TF32 are disabled. Observations must be noiseless. Objective
values always use the canonical forward map, including line-search candidates.
Both GD and L-BFGS use the inherited geometric Armijo search, first-attempt
scaling, and displacement cap \(0.01\sqrt d\). The default optimizer is L-BFGS.
Transport variants default to a fresh DLR solve every five gradient evaluations
(one fresh solve followed by four transported evaluations); `fresh` recomputes
every time. The factor basis may otherwise miss new sensitivity directions.

```sh
.venv/bin/python -m dlr_gradient_flow.experiment benchmark \
  --case r64_n1_ct10/sample_000 --device cuda:0 \
  --modes exact fresh transport secant_transport secant --repeats 1 \
  --out outputs/dlr_gradient_flow_benchmark

.venv/bin/python -m dlr_gradient_flow.experiment reconstruct \
  --case r64_n1_ct10/sample_000 --device cuda:0 \
  --modes exact proxy fresh secant_transport secant --optimizer lbfgs \
  --steps 100 --refresh-every 5 --out outputs/dlr_gradient_flow_reconstruction

.venv/bin/python -m unittest dlr_gradient_flow.test_transport \
  dlr_gradient_flow.test_secant_transport -v
```

Here `exact`, `proxy`, and `fresh` denote D-Flow, P-Flow, and freshly recomputed
DLR. Outputs include input/code hashes, settings, gradient diagnostics, histories,
sources, factors, and optimizer states. `--steps` limits optimizer attempts;
two consecutive failed searches can stop a run earlier. Thus `--steps 100` means
at most 100 attempts, with accepted updates reported separately. `report.py`
distinguishes saved sweep controls (`baseline`) from new runs (`rerun`) after
checking paired inputs and checkpoint hashes. Experimental fourth columns are
explicitly labeled; saved D-Flow/P-Flow controls are not presented as fresh reruns.

## Cost and limitations

At rank \(r\), literal transport requires a directional trajectory and forward
and reverse passes involving second derivatives on \(r\) columns. Pure secant
uses \(2N\) local batched JVPs and \(2N\) local batched VJPs; corrected endpoint
iteration uses \(N\) of each. Both use two thin QRs costing \(O(dr^2)\).
All variants retain \(O(Nd+dr)\) state/factor
storage plus model-dependent local derivative workspace; local tapes are
recomputed rather than retained across the full flow. GPU graph storage is extra.
These counts do not guarantee a speedup over the existing specialized DLR solver.

Neither approximate gradient automatically decreases the true objective:
descent requires positive alignment with \(J^\top g_x\), after any optimizer
transformation. Low rank can omit essential directions, literal transport can
accumulate error, and all variants can preserve excessive leading-direction
weighting. Armijo acceptance verifies objective improvement, not image accuracy.
