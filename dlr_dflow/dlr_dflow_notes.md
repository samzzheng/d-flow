# Dynamic Low-Rank D-Flow Notes

These notes combine D-Flow source-point optimization with dynamic low-rank approximation (DLRA) for the flow sensitivity

$$
J_t(z) := \frac{\partial x_t(z)}{\partial z}.
$$

The goal is to use the observed effective low-rank structure of $J_t$ without computing a full SVD of $J_t$ at every inverse-problem iteration.

## 1. Baseline D-Flow formulation

Let a pretrained generative flow be given by the ODE

$$
\frac{d x_t}{d t} = v_\theta(t,x_t), \qquad x_0 = z, \qquad t \in [0,1],
$$

where $z$ is the source/noise variable and $x_1(z)$ is the generated sample.

For an inverse problem, suppose the observation satisfies

$$
y \approx \mathcal{A}(x_\star) + \eta,
$$

where $\mathcal{A}$ is the forward or corruption operator. D-Flow solves a source-point optimization problem

$$
\min_z \; \Phi(z)
 := \ell(\mathcal{A}(x_1(z)),y) + \lambda R_z(z) + \mu R_x(x_1(z)).
$$

For the usual squared data misfit,

$$
\ell(\mathcal{A}(x),y)
= \frac12 \|\mathcal{A}(x)-y\|_{\Gamma^{-1}}^2,
$$

the terminal gradient is

$$
g_x
:= \nabla_x \left[\ell(\mathcal{A}(x),y)+\mu R_x(x)\right]_{x=x_1}
= D\mathcal{A}(x_1)^* \Gamma^{-1}(\mathcal{A}(x_1)-y)
  + \mu \nabla R_x(x_1).
$$

The exact D-Flow source gradient is

$$
\nabla_z \Phi(z)
= J_1(z)^T g_x + \lambda \nabla R_z(z).
$$

The computational bottleneck is therefore the action of $J_1(z)^T$, or equivalently differentiating through the ODE solver. If one explicitly forms or repeatedly compresses $J_t$, the cost is prohibitive for high-dimensional inverse problems.

## 2. Sensitivity equation

Differentiate the flow ODE with respect to $z$. The sensitivity matrix $J_t$ solves

$$
\dot J_t = A_t J_t,
\qquad
J_0 = I,
\qquad
A_t := \partial_x v_\theta(t,x_t).
$$

Here $A_t$ is the Jacobian of the velocity field along the current trajectory $x_t$.

In practice, $A_t$ should not be formed. We only need products

$$
A_t Q
\quad\text{and}\quad
A_t^T Q
$$

with skinny matrices $Q \in \mathbb{R}^{d \times r}$. These are computed with JVPs and VJPs through the neural velocity model.

## 3. Why naive rank-$r$ compression is not enough

The exact flow map of a smooth ODE is locally invertible under standard uniqueness assumptions, so $J_t$ is generally full rank. Also,

$$
J_0 = I.
$$

Therefore a direct approximation

$$
J_t \approx U_t S_t V_t^T,
\qquad
U_t,V_t \in \mathbb{R}^{d \times r},
\qquad
S_t \in \mathbb{R}^{r \times r},
$$

with $r \ll d$ cannot approximate $J_0$ well in Frobenius norm.

The useful assumption is not literal rank deficiency. The useful assumption is effective low-rank action: the important part of $J_t$, $J_t^T g_x$, or $J_tJ_t^T g_x$ is concentrated in a low-dimensional set of directions. This is consistent with the D-Flow intuition that differentiating through the flow projects the terminal loss gradient onto dominant data-manifold directions.

This suggests a low-rank-plus-baseline model rather than a pure low-rank model.

## 4. Low-rank residual model

Use

$$
J_t \approx B_t + Y_t,
\qquad
Y_t = U_t S_t V_t^T,
\qquad
\operatorname{rank}(Y_t)=r.
$$

The simplest baseline is

$$
B_t = \alpha_t I.
$$

Usually start with

$$
\alpha_0=1,
\qquad
Y_0 = 0.
$$

The algebraically simplest version sets $\alpha_t \equiv 1$, so

$$
J_t \approx I + U_t S_t V_t^T.
$$

For the trained circle flow used in the notebook, the identity baseline gives a
very poor sensitivity action. The implemented default therefore lets $\alpha_t$
track average log-volume change:

$$
\frac{d}{dt}\log \alpha_t
\approx \frac{1}{d}\operatorname{tr}(A_t),
$$

with $\operatorname{tr}(A_t)$ estimated by Hutchinson probes. This scalar is a
volume-based baseline, not the Frobenius-optimal scalar approximation to $J_t$,
and it does not imply that every singular direction contracts.

Substitute $J_t \approx B_t+Y_t$ into the sensitivity equation:

$$
\dot Y_t
\approx
F(t,Y_t)
:= A_t(B_t+Y_t)-\dot B_t.
$$

For $B_t=I$,

$$
F(t,Y_t)=A_t(I+Y_t).
$$

The dynamic low-rank idea is to evolve $Y_t$ on the rank-$r$ manifold by projecting the right-hand side onto the tangent space:

$$
\dot Y_t = P_{T_{Y_t}\mathcal{M}_r} F(t,Y_t).
$$

## 5. Tangent-space projection

For

$$
Y = U S V^T,
\qquad
U^TU=I,
\qquad
V^TV=I,
$$

the orthogonal tangent projection of a matrix $Z$ onto the rank-$r$ manifold at $Y$ is

$$
P_Y(Z)
= ZVV^T - UU^T ZVV^T + UU^T Z.
$$

This formula is the central DLRA replacement for repeated SVD truncation. Instead of taking a full matrix step and truncating by SVD, we directly evolve inside the low-rank manifold.

## 6. Projector-splitting update for the residual

The Lubich-Oseledets projector-splitting integrator evolves the factors $U,S,V$ by three smaller substeps. Let

$$
Y_0 = U_0 S_0 V_0^T
$$

be the current low-rank residual at flow time $t_0$, and let $t_1=t_0+h$.

Define

$$
F(t,Y)=A_t(B_t+Y)-\dot B_t.
$$

### K-step

Freeze $V=V_0$, set

$$
K(t_0)=U_0S_0,
$$

and integrate

$$
\dot K(t)=F(t,K(t)V_0^T)V_0.
$$

For $B_t=\alpha_tI$, this becomes

$$
\dot K(t)
=\alpha_tA_tV_0+A_tK(t)-\dot\alpha_tV_0.
$$

Then QR factorize

$$
K(t_1)=U_1\widehat S_1.
$$

### S-step

Freeze $U=U_1$ and $V=V_0$, set

$$
S(t_0)=\widehat S_1,
$$

and integrate

$$
\dot S(t)
= -U_1^T F(t,U_1S(t)V_0^T)V_0.
$$

For $B_t=\alpha_tI$,

$$
\dot S(t)
=-\alpha_tU_1^TA_tV_0-U_1^TA_tU_1S(t)
  +\dot\alpha_tU_1^TV_0.
$$

Set

$$
\widetilde S_0=S(t_1).
$$

### L-step

Freeze $U=U_1$, set

$$
L(t_0)=V_0\widetilde S_0^T,
$$

and integrate

$$
\dot L(t)=F(t,U_1L(t)^T)^T U_1.
$$

For $B_t=\alpha_tI$,

$$
\dot L(t)
=\alpha_tA_t^TU_1+L(t)U_1^TA_t^TU_1
  -\dot\alpha_tU_1.
$$

Then QR factorize

$$
L(t_1)=V_1S_1^T.
$$

The updated residual is

$$
Y_1=U_1S_1V_1^T.
$$

Only skinny QR factorizations of $d\times r$ matrices and small $r\times r$ algebra are required.

## 7. DLR-D-Flow algorithm

At inverse-problem iteration $k$, assume we have source point $z_k$.

1. Integrate the flow state

   $$
   \dot x_t = v_\theta(t,x_t),
   \qquad
   x_0=z_k.
   $$

2. Along the same flow-time grid, evolve the low-rank residual factors

   $$
   Y_t = U_tS_tV_t^T
   $$

   using the projector-splitting steps above.

3. At $t=1$, compute

   $$
   g_x =
   D\mathcal{A}(x_1)^*\Gamma^{-1}(\mathcal{A}(x_1)-y)
   + \mu\nabla R_x(x_1).
   $$

4. Approximate the source gradient by

   $$
   \widehat{\nabla_z\Phi}(z_k)
   =
   B_1^Tg_x + V_1S_1^TU_1^Tg_x
   + \lambda\nabla R_z(z_k).
   $$

   For $B_1=\alpha_1I$,

   $$
   \widehat{\nabla_z\Phi}(z_k)
   =
   \alpha_1g_x + V_1S_1^TU_1^Tg_x
   + \lambda\nabla R_z(z_k).
   $$

5. Update the source variable, for example

   $$
   z_{k+1}=z_k-\eta_k\widehat{\nabla_z\Phi}(z_k),
   $$

   or use LBFGS with this approximate gradient and a line search.

6. Generate the new sample by integrating $x_t(z_{k+1})$.

This gives a D-Flow-like inverse solver, but replaces full Jacobian/SVD operations by dynamic low-rank factor evolution.

## 8. Pseudocode

```text
Input:
  observation y
  forward operator A
  pretrained velocity v_theta(t, x)
  initial source z0
  rank r
  flow time grid 0=t0<...<tN=1
  inverse optimizer steps K

for k = 0,...,K-1:
    x = z_k
    initialize residual factors U,S,V for Y_0
        first inverse iteration:
            use a random orthonormal basis
        later inverse iterations:
            seed the t=0 basis with the previous approximate source gradient
        always reset S to epsilon I
    set B_t = alpha_t I and alpha_0 = 1

    for n = 0,...,N-1:
        define products with A_t = partial_x v_theta(t, x_t):
            A_t Q   by JVP
            A_t^T Q by VJP
        estimate tau_t = tr(A_t) with Hutchinson probes
        set alpha_dot_t = alpha_t tau_t / d

        K-step:
            integrate Kdot = alpha_t A_t V + A_t K - alpha_dot_t V
            QR: K = U_new S_hat

        S-step:
            integrate Sdot = - alpha_t U_new^T A_t V
                             - U_new^T A_t U_new S
                             + alpha_dot_t U_new^T V

        L-step:
            integrate Ldot = alpha_t A_t^T U_new
                             + L U_new^T A_t^T U_new
                             - alpha_dot_t U_new
            QR: L = V_new S_new^T

        U,S,V = U_new,S_new,V_new
        advance x_t and alpha_t to t_{n+1}

    g_x = A'(x_1)^* Gamma^{-1}(A(x_1)-y) + mu grad R_x(x_1)
    g_z_hat = alpha_1 g_x + V S^T U^T g_x + lambda grad R_z(z_k)
    z_{k+1} = optimizer_update(z_k, g_z_hat)

return z_K, x_1(z_K)
```

## 9. How to initialize $Y_0$

Because $Y_0=0$ for the residual model $J_0=I+Y_0$, the rank-$r$ factorization is initially singular. There are three practical choices.

### Option A: gradient-seeded basis

Use the previous iteration's approximate source gradient, which already lives in
the source tangent space, plus random probes:

$$
q_1
=\frac{\widehat{\nabla_z\Phi}(z_{k-1})}
       {\|\widehat{\nabla_z\Phi}(z_{k-1})\|},
\qquad
V_0 = \operatorname{orth}([q_1,\dots,q_r]),
\qquad
U_0=V_0,
\qquad
S_0=\varepsilon I_r.
$$

The first inverse iteration has no previous source gradient and uses a random
basis. This warm start is cheap and targets the directions the optimizer
actually uses. Directly inserting the current endpoint gradient $g_x$ into both
factor spaces is not equivalent to this source-space warm start.

### Option B: short randomized warm start

For a very small first flow step $h$,

$$
Y_h \approx h A_0.
$$

Use randomized range finding with JVP/VJP products to produce low-rank factors of $A_0$ or of $A_0$ applied to a small probe matrix. This uses no full SVD of a $d\times d$ matrix.

### Option C: carry subspaces across inverse iterations

If $z_{k+1}$ is close to $z_k$, reuse the previous $U,V$ subspaces as a warm start, but reset the core to

$$
S_0=\varepsilon I_r
$$

or another tiny core. Do not place the previous final-time residual directly at $t=0$, because the residual initial condition remains $Y_0=0$. This subspace reuse is often the most efficient choice inside a line-search or LBFGS solver.

In all three cases, $S_0$ can be tiny. With $S_0=\varepsilon I$, the implemented
initial condition is

$$
\widehat J_0=\alpha_0I+\varepsilon U_0V_0^T,
$$

so it approximates rather than exactly equals $J_0=I$. Setting $\varepsilon$
small controls this initialization error. The scalar baseline carries the
full-rank part, and the low-rank residual learns important deformation
directions.

## 10. Relationship to adjoint gradients

For one scalar objective and one source-gradient evaluation, the classical adjoint sensitivity equation

$$
\dot a_t = -A_t^Ta_t,
\qquad
a_1=g_x,
$$

already gives the exact gradient

$$
\nabla_z\Phi(z)=a_0+\lambda\nabla R_z(z)
$$

without forming $J_t$. Therefore DLR-D-Flow should not be advertised as automatically cheaper than one exact adjoint gradient when $r>1$.

Its computational role is more specific:

1. It replaces algorithms that form $J_t$ and perform large SVD truncations.
2. It gives a reusable low-rank sensitivity operator, so several terminal gradients or line-search trial gradients can be applied cheaply once the factors are available.
3. It gives a structured projected update that deliberately keeps only dominant flow-sensitivity directions.

So the fair comparison is against full-Jacobian or repeated-SVD D-Flow variants, not against the best possible single-vector adjoint implementation.

## 11. Coupled update when $z$ moves

The previous sections describe how to evolve $U_t,S_t,V_t$ along the generative flow time $t$, for a fixed source point $z$. But in the inverse problem, $z$ itself changes. Therefore the low-rank factors must also move with the inverse-problem gradient flow.

Introduce a second time variable $\tau$ for the inverse solver:

$$
z=z(\tau),
\qquad
\frac{d z}{d\tau}
=-\widehat{\nabla_z\Phi}(z).
$$

For the identity-plus-low-rank model,

$$
\widehat{\nabla_z\Phi}(z)
=g_x+V_1S_1^TU_1^Tg_x+\lambda\nabla R_z(z).
$$

Now define the flow trajectory depending on both variables:

$$
x_t(\tau)=x_t(z(\tau)).
$$

Let

$$
p_t:=\partial_\tau x_t.
$$

Then $p_t$ is the perturbation of the generated flow trajectory induced by the inverse-problem update of $z$. It satisfies

$$
\partial_t p_t=A_t p_t,
\qquad
p_0=\dot z.
$$

Equivalently,

$$
p_t=J_t\dot z.
$$

The velocity Jacobian is

$$
A_t(\tau)=\partial_x v_\theta(t,x_t(\tau)).
$$

Therefore its $\tau$-derivative is

$$
\partial_\tau A_t
=D_xA_t[p_t]
=D_{xx}^2v_\theta(t,x_t)[p_t,\cdot].
$$

Write

$$
H_t[p_t]
:=D_{xx}^2v_\theta(t,x_t)[p_t,\cdot].
$$

This is a linear operator: for any skinny matrix $Q$,

$$
H_t[p_t]Q
=
D_x\left(A_tQ\right)[p_t].
$$

In automatic differentiation terms, this is a JVP through the map $x\mapsto A_tQ$, in direction $p_t$.

Now differentiate the sensitivity equation

$$
\partial_tJ_t=A_tJ_t
$$

with respect to $\tau$. Define

$$
W_t:=\partial_\tau J_t.
$$

Then

$$
\partial_t W_t
=A_tW_t+H_t[p_t]J_t,
\qquad
W_0=0.
$$

This is the missing equation. It says that when the inverse problem moves $z$, the sensitivity matrix changes through a second-variation forcing term

$$
H_t[p_t]J_t.
$$

For

$$
J_t\approx I+Y_t,
\qquad
Y_t=U_tS_tV_t^T,
$$

we approximate

$$
\partial_\tau Y_t
\approx
P_{T_{Y_t}\mathcal M_r}W_t.
$$

Using the standard gauge conditions

$$
U_t^T\partial_\tau U_t=0,
\qquad
V_t^T\partial_\tau V_t=0,
$$

the continuous factor equations are

$$
\partial_\tau U_t
=
(I-U_tU_t^T)W_tV_tS_t^{-1},
$$

$$
\partial_\tau S_t
=
U_t^TW_tV_t,
$$

$$
\partial_\tau V_t
=
(I-V_tV_t^T)W_t^TU_tS_t^{-T}.
$$

These equations explicitly depend on the inverse-problem gradient flow through

$$
\dot z=-\widehat{\nabla_z\Phi}(z),
\qquad
p_0=\dot z,
\qquad
\partial_t p_t=A_tp_t,
$$

and through the Hessian-vector forcing $H_t[p_t]J_t$.

Because the formulas contain $S_t^{-1}$, a robust implementation should use a projector-splitting update in the $\tau$-direction rather than these equations directly when $S_t$ is small.

For a small inverse step $\Delta\tau$, with $W_t$ evaluated at the current $z$, the explicit projector-splitting version is:

1. K-step:

   $$
   K^+=U_tS_t+\Delta\tau\, W_tV_t,
   \qquad
   K^+=U_t^+\widehat S_t.
   $$

2. S-step:

   $$
   \widetilde S_t
   =
   \widehat S_t
   -\Delta\tau\, (U_t^+)^TW_tV_t.
   $$

3. L-step:

   $$
   L^+
   =
   V_t\widetilde S_t^T+\Delta\tau\, W_t^TU_t^+,
   \qquad
   L^+=V_t^+(S_t^+)^T.
   $$

Then

$$
Y_t^+=U_t^+S_t^+(V_t^+)^T.
$$

At $t=0$, one must still enforce

$$
J_0=I,
\qquad
Y_0=0,
\qquad
W_0=0.
$$

So the previous final-time residual should not be copied directly to $t=0$. The low-rank factors may be reused as subspace guesses, but the residual initial condition remains zero.

The coupled DLR-D-Flow outer update is therefore:

1. Compute $g_x$ at $x_1$.
2. Compute the approximate source gradient

   $$
   \widehat{\nabla_z\Phi}
   =
   g_x+V_1S_1^TU_1^Tg_x+\lambda\nabla R_z.
   $$

3. Set

   $$
   \dot z=-\widehat{\nabla_z\Phi}.
   $$

4. Propagate the trajectory perturbation

   $$
   \partial_t p_t=A_tp_t,
   \qquad
   p_0=\dot z.
   $$

5. Propagate the sensitivity perturbation

   $$
   \partial_t W_t=A_tW_t+H_t[p_t](I+U_tS_tV_t^T),
   \qquad
   W_0=0.
   $$

6. Update $U_t,S_t,V_t$ by projecting $W_t$ onto the rank-$r$ tangent space, preferably with the $\tau$-projector-splitting step above.

7. Update

   $$
   z^+=z+\Delta\tau\,\dot z.
   $$

This is the mathematically coupled version. It is not implemented in
dlr_dflow_demo.ipynb. Its extra cost is the second-variation action
$H_t[p_t]Q$, so the practical question is whether this is cheaper than simply
re-evolving the dynamic low-rank sensitivity factors from $t=0$ after each new
$z$. The notebook uses that reinitialize-and-re-evolve strategy: it resets the
core near zero, seeds the basis with the previous approximate source gradient,
and integrates the state and factors together along the new trajectory.

## 12. Computational cost

Let $d$ be the dimension of $x$ and $z$, and let $r\ll d$.

Full sensitivity storage:

$$
O(d^2).
$$

Low-rank residual storage:

$$
O(dr+r^2).
$$

Full Jacobian propagation requires applying $A_t$ to $d$ directions per time step. DLR-D-Flow applies $A_t$ and $A_t^T$ only to $O(r)$ directions per time step.

The dominant costs are:

$$
O(r) \text{ JVP/VJP calls per flow step}
+ O(dr^2) \text{ skinny QR cost}
+ O(r^3) \text{ small core algebra}.
$$

There is no SVD of a $d\times d$ matrix at every inverse iteration. If rank adaptation is used, only small SVDs of $r\times r$ or $2r\times 2r$ core matrices are needed.

## 13. Rank adaptation

A fixed rank $r$ may be too small early in optimization and too large later. Useful rank indicators include:

$$
\|(I-P_{Y_t})F(t,Y_t)\|_F,
$$

estimated by random probes, and the mismatch between the predicted source-gradient action and a few exact VJP checks.

A simple adaptive rule:

1. Start with a small rank, for example $r=8$ or $16$.
2. Every few inverse iterations, sample probes $w_j$ and compare

   $$
   J_1^T w_j
   \quad\text{from an exact adjoint/VJP check}
   $$

   against

   $$
   B_1^Tw_j + V_1S_1^TU_1^Tw_j.
   $$

3. If the relative error is above tolerance, enrich $U,V$ with the missed directions.
4. Compress only the small augmented core, not the full matrix.

## 14. What this method does and does not guarantee

This method removes the full-SVD bottleneck. It does not make the exact flow Jacobian low rank.

The approximation is useful when the inverse-problem update only needs the dominant sensitivity directions. If the forward operator $\mathcal{A}$ needs information in directions discarded by the rank-$r$ model, the update can become biased.

The identity or scalar baseline is important. Without it, the approximation conflicts with $J_0=I$, and the method may incorrectly suppress source directions that are still relevant.

The low-rank factors should be treated as an approximate gradient mechanism or a structured preconditioner. A line search or objective decrease check should remain in the inverse solver.

## 15. Recommended first experiment

Start with the scalar-baseline-plus-low-rank residual model used by the
implementation:

$$
J_t \approx \alpha_tI + U_tS_tV_t^T,
\qquad
\frac{d}{dt}\log\alpha_t
=\frac{1}{d}\operatorname{tr}(A_t).
$$

Use a fixed small rank, for example $r=8,16,32$, and compare four quantities:

1. Data misfit decrease per optimizer step.
2. Runtime and memory against the original D-Flow backpropagation-through-ODE implementation.
3. Agreement of the approximate gradient with exact adjoint gradients on a small subset of iterations.
4. Final reconstruction quality.

The most important diagnostic is not the Frobenius error of $J_t$. The important diagnostic is the action error on relevant terminal gradients:

$$
\frac{
\|J_1^Tg_x - (\alpha_1g_x+V_1S_1^TU_1^Tg_x)\|
}{
\|J_1^Tg_x\|
}.
$$

If this error is small while runtime decreases, the dynamic low-rank idea is helping the D-Flow inverse solver in the way we actually need.

## 16. Refresh cost findings (2026-09-02, circles-64, CPU)

Measured on the pretrained circles-64 model (15 Euler steps, 8 CPU threads),
comparing ways to produce the terminal factors against autograd through the
same discrete flow:

| refresh | time | grad rel. err | cosine |
|---|---|---|---|
| per-step evolution, r=8, 8 probes, 4 enrichment | 34.0 s | 0.055 | 0.9988 |
| per-step evolution, r=4, 4 probes, 2 enrichment | 20.4 s | 0.076 | 0.9978 |
| direct terminal sketch, r=8 (+6 oversampling)   | 16.7 s | 0.005 | 1.0000 |
| direct terminal sketch, r=4 (+4 oversampling)   | 10.3 s | 0.009 | 1.0000 |

Because J_1 is numerically rank 2 here, the direct randomized sketch of J_1
(`enriched.direct_terminal_rank_factors`) dominates the projector-splitting
evolution on both cost and accuracy.  The evolution remains the right tool
only when the sketch's transpose pass (one VJP through the whole flow,
activation memory ~ sketch-width x one backward) does not fit in memory:
the evolution differentiates one frozen step at a time in O(1) memory.

Cost anatomy of the evolution refresh: 60 Jacobian-product columns per flow
step (~2.8 s/step, 42 s over 15 steps), against ~45 column-equivalents for
one exact gradient.  Per column, torch.func forward-mode costs 3.9x a batched
forward element (theory ~2x): convolutions execute 3x (not 2x), and the
GroupNorm JVP decomposition is pathological at ~33x its forward cost (~27%
of JVP time; a custom JVP rule for GroupNorm is worth ~1.3x on everything
forward-mode).  JVP calls also thread-scale worse than forwards (1.4x vs 2.0x
going from 8 to 32 threads).

## 17. References

- Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, Yaron Lipman, "D-Flow: Differentiating through Flows for Controlled Generation", ICML 2024. https://proceedings.mlr.press/v235/ben-hamu24a.html
- Othmar Koch and Christian Lubich, "Dynamical Low-Rank Approximation", SIAM Journal on Matrix Analysis and Applications, 2007. https://doi.org/10.1137/050639703
- Christian Lubich and Ivan Oseledets, "A projector-splitting integrator for dynamical low-rank approximation", BIT Numerical Mathematics, 2014. https://arxiv.org/abs/1301.1058
- Gianluca Ceruti and Christian Lubich, "An unconventional robust integrator for dynamical low-rank approximation", BIT Numerical Mathematics, 2022. https://link.springer.com/article/10.1007/s10543-021-00873-0
