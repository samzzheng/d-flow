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
 := \ell(\mathcal{A}(x_1(z)),y)
$$

For the usual squared data misfit,

$$
\ell(\mathcal{A}(x),y)
= \frac12 \|\mathcal{A}(x)-y\|_{\Gamma^{-1}}^2,
$$

the terminal gradient is

$$
g_x
:= \nabla_x \left[\ell(\mathcal{A}(x),y)\right]_{x=x_1}
= D\mathcal{A}(x_1)^* \Gamma^{-1}(\mathcal{A}(x_1)-y)
$$

The exact D-Flow source gradient is

$$
\nabla_z \Phi(z)
= J_1(z)^\top g_x
$$

The computational bottleneck is therefore the action of $J_1(z)^\top$, or equivalently differentiating through the ODE solver. If one explicitly forms or repeatedly compresses $J_t$, the cost is prohibitive for high-dimensional inverse problems.

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
A_t^\top Q
$$

with skinny matrices $Q \in \mathbb{R}^{d \times r}$. These are computed with JVPs and VJPs through the neural velocity model.

## 3. Why naive rank-$r$ compression is not enough

The exact flow map of a smooth ODE is locally invertible under standard uniqueness assumptions, so $J_t$ is generally full rank. Also,

$$
J_0 = I.
$$

Therefore a direct approximation

$$
J_t \approx U_t S_t V_t^\top,
\qquad
U_t,V_t \in \mathbb{R}^{d \times r},
\qquad
S_t \in \mathbb{R}^{r \times r},
$$

with $r \ll d$ cannot approximate $J_0$ well in Frobenius norm.

The useful assumption is not literal rank deficiency. The useful assumption is effective low-rank action: the important part of $J_t$, $J_t^\top g_x$, or $J_tJ_t^\top g_x$ is concentrated in a low-dimensional set of directions. This is consistent with the D-Flow intuition that differentiating through the flow projects the terminal loss gradient onto dominant data-manifold directions.

This suggests a low-rank-plus-baseline model rather than a pure low-rank model.

## 4. Low-rank residual model

Use

$$
J_t \approx \alpha_t I + Y_t,
\qquad
Y_t = U_t S_t V_t^\top,
\qquad
\operatorname{rank}(Y_t)=r,
$$

where $\alpha_t$ is a scalar baseline and $Y_t$ is a rank-$r$ residual. Usually start with

$$
\alpha_0=1,
\qquad
Y_0 = 0.
$$

The algebraically simplest version sets $\alpha_t \equiv 1$, so

$$
J_t \approx I + U_t S_t V_t^\top.
$$

The construction in Sections 6-8 uses the analytical baseline

$$
\alpha_t=1-t,\qquad \dot\alpha_t=-1,\qquad \alpha_1=0.
$$

This baseline starts from the exact identity sensitivity and vanishes at the terminal time.

Substitute $J_t \approx \alpha_tI+Y_t$ into the sensitivity equation $\dot J_t = A_tJ_t$. Since $\frac{d}{dt}(\alpha_tI)=\dot\alpha_tI$,

$$
\dot Y_t
\approx
F(t,Y_t)
:= A_t(\alpha_tI+Y_t)-\dot\alpha_tI
= \alpha_tA_t + A_tY_t - \dot\alpha_tI.
$$

For $\alpha_t\equiv 1$,

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
Y = U S V^\top,\qquad
U^\top U=I,
\qquad
V^\top V=I,
$$

the orthogonal tangent projection of a matrix $Z$ onto the rank-$r$ manifold at $Y$ is

$$
P_Y(Z)
= ZVV^\top - UU^\top ZVV^\top + UU^\top Z.
$$

This formula is the central DLRA replacement for repeated SVD truncation. Instead of taking a full matrix step and truncating by SVD, we directly evolve inside the low-rank manifold.

## 6. Projector-splitting update for the residual

Use $\alpha_t=1-t$, so

$$
F(t,Y)=(1-t)A_t+A_tY+I,
\qquad A_t=\partial_xv_\theta(t,x_t).
$$

### Initialization

At $t=0$, $x_0=z$, $J_0=I$, and $Y_0=0$. Since $F(0,0)=A_0+I$, the first Euler interval $[0,h]$ gives $Y_h^{\mathrm{Euler}}=h(A_0+I)$. To find its initial subspaces, draw $\Omega\in\mathbb R^{d\times r}$ with independent entries $\pm1$, each with probability $1/2$. Apply the residual and its transpose through JVPs and VJPs, then compute two reduced QR factorizations and one $r\times r$ SVD:

$$
\begin{aligned}
h(A_0\Omega+\Omega)&=Q_1R_1,\\
h(A_0^\top Q_1+Q_1)&=Q_2R_2,\\
R_2&=U_{\mathrm{SVD}}\Sigma_{\mathrm{SVD}}V_{\mathrm{SVD}}^\top.
\end{aligned}
$$

The first QR selects the sampled column subspace; the transpose product recovers its coefficients, and the second QR makes the SVD small. Transposing back gives the factors below, whose product is the projection of the Euler target onto that subspace. Thus initialization preserves $Y_0=0$ and produces factors at $t=h$ without forming a dense Jacobian. Advance the image to $x_h=z+h v_\theta(0,z)$.

$$
U_h=Q_1V_{\mathrm{SVD}},\qquad
S_h=\Sigma_{\mathrm{SVD}},\qquad
V_h=Q_2U_{\mathrm{SVD}},
$$

$$
U_hS_hV_h^\top
=Q_1Q_1^\top Y_h^{\mathrm{Euler}}
\approx Y_h^{\mathrm{Euler}}.
$$

### KSL steps

The Lubich-Oseledets projector-splitting integrator evolves $U,S,V$ by three substeps over the same factor interval $[t_0,t_1]$. Here $0$ and $1$ label this interval's endpoints. Let

$$
Y_0 = U_0 S_0 V_0^\top
$$

be the current low-rank residual at flow time $t_0$, and let $t_1=t_0+h$. Each subproblem below runs over $[t_0,t_1]$, with the preceding stage's output as its initial value at $t_0$. Together the three stages advance one interval of length $h$. The implementation uses one Euler step per subproblem.

Define

$$
F(t,Y)=A_t(\alpha_tI+Y)-\dot\alpha_tI.
$$

#### K-step

Freeze $V=V_0$, set

$$
K(t_0)=U_0S_0,
$$

and integrate

$$
\dot K(t)=F(t,K(t)V_0^\top)V_0,
\qquad t\in[t_0,t_1].
$$

Inserting $F$ and using $V_0^\top V_0=I$, this becomes

$$
\dot K(t)
=A_t\big(\alpha_tV_0+K(t)V_0^\top V_0\big)-\dot\alpha_tV_0
=\alpha_tA_tV_0+A_tK(t)-\dot\alpha_tV_0.
$$

Then QR factorize

$$
K(t_1)=Q_KR_K=U_1\widehat S_1,
\qquad
U_1=Q_K,\quad \widehat S_1=R_K.
$$

#### S-step

Freeze $U=U_1$ and $V=V_0$, set

$$
S(t_0)=\widehat S_1,
$$

and integrate

$$
\dot S(t)
= -U_1^\top F(t,U_1S(t)V_0^\top)V_0,
\qquad t\in[t_0,t_1].
$$

Inserting $F$ and using $V_0^\top V_0=I$,

$$
\dot S(t)
=-U_1^\top\Big[A_t\big(\alpha_tV_0+U_1S(t)\big)-\dot\alpha_tV_0\Big]
=-\alpha_tU_1^\top A_tV_0-U_1^\top A_tU_1S(t)
  +\dot\alpha_tU_1^\top V_0.
$$

Set

$$
\widetilde S_0=S(t_1).
$$

#### L-step

Freeze $U=U_1$, set

$$
L(t_0)=V_0\widetilde S_0^\top,
$$

and integrate

$$
\dot L(t)=F(t,U_1L(t)^\top)^\top U_1,
\qquad t\in[t_0,t_1].
$$

Inserting $F$ and using $U_1^\top U_1=I$,

$$
\dot L(t)
=\Big[A_t\big(\alpha_tI+U_1L(t)^\top\big)-\dot\alpha_tI\Big]^\top U_1
=\alpha_tA_t^\top U_1+L(t)U_1^\top A_t^\top U_1
  -\dot\alpha_tU_1.
$$

Then QR factorize

$$
L(t_1)=Q_LR_L=V_1S_1^\top,
\qquad
V_1=Q_L,\quad S_1=R_L^\top.
$$

The updated residual is

$$
Y_1=U_1S_1V_1^\top.
$$

Only skinny QR factorizations of $d\times r$ matrices and small $r\times r$ algebra are required.

The factor and image grids can be chosen separately, provided $A_t=\partial_xv_\theta(t,x_t)$ is evaluated along a consistent image trajectory. The current default and Section 8 use the same grid for both. With $N$ intervals, one gradient evaluation uses one initialization interval and $N-1$ KSL cycles; applying $VS^\top U^\top g_x$ then requires no further evolution.

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
   Y_t = U_tS_tV_t^\top
   $$

   using the projector-splitting steps above.

3. At $t=1$, compute

   $$
   g_x =
   D\mathcal{A}(x_1)^*\Gamma^{-1}(\mathcal{A}(x_1)-y)
   $$

4. Approximate the source gradient by

   $$
   \widehat{\nabla_z\Phi}(z_k)
   =
   (\alpha_1I + U_1S_1V_1^\top)^\top g_x
   =
   \alpha_1g_x + V_1S_1^\top U_1^\top g_x
   $$

5. Update the source variable, for example

   $$
   z_{k+1}=z_k-\eta_k\widehat{\nabla_z\Phi}(z_k),
   $$

   or use LBFGS with this approximate gradient and a line search.

6. Generate the new sample by integrating $x_t(z_{k+1})$.

This gives a D-Flow-like inverse solver, but replaces full Jacobian/SVD operations by dynamic low-rank factor evolution.

## 8. Pseudocode

Use $\alpha_t=1-t$, $r$ initialization directions, and one Euler step per split subproblem. Each source-gradient evaluation starts with the initialization in Section 6.

```text
Input:
  observation y
  forward operator mathcal_A and measurement weighting Gamma
  pretrained velocity v_theta(t, x)
  initial source z0
  rank r < d
  N flow steps, h = 1/N, t_n = n h
  inverse optimizer steps K

for k = 0,...,K-1:
    x = z_k
    alpha = 1                           # J_0 = I, Y_0 = 0

    initialize over [0,h]:
        define A_0 Q and A_0^T Q at (0,x) using JVPs and VJPs
        draw Omega of shape (d,r), independent entries +1 or -1
        Q_1, R_1 = QR(h (A_0 Omega + Omega))
        Q_2, R_2 = QR(h (A_0^T Q_1 + Q_1))
        U_SVD, Sigma_SVD, V_SVD^T = SVD(R_2)
        U = Q_1 V_SVD
        S = Sigma_SVD
        V = Q_2 U_SVD
        x = x + h v_theta(0,x)
        alpha = 1-h                     # factors now represent time h

    for n = 1,...,N-1:
        t = t_n
        alpha = 1-t
        define A_t Q and A_t^T Q at (t,x) using JVPs and VJPs
        # All three stages cover [t_n,t_{n+1}] with these same products.

        K-step:
            K_start = U S
            K_end = K_start + h (alpha A_t V + A_t K_start + V)
            Q_K, R_K = QR(K_end)
            U_new = Q_K
            S_hat = R_K

        S-step:
            S_tilde = S_hat - h (alpha U_new^T A_t V
                                + U_new^T A_t U_new S_hat
                                + U_new^T V)

        L-step:
            L_start = V S_tilde^T
            L_end = L_start + h (alpha A_t^T U_new
                                + L_start U_new^T A_t^T U_new
                                + U_new)
            Q_L, R_L = QR(L_end)
            V_new = Q_L
            S_new = R_L^T

        U,S,V = U_new,S_new,V_new
        x = x + h v_theta(t,x)
        alpha = 1-t_{n+1}

    alpha = 0                           # exact terminal baseline
    g_x = D mathcal_A(x)^* Gamma^{-1}(mathcal_A(x)-y)
    g_z_hat = V S^T U^T g_x
    z_{k+1} = optimizer_update(z_k, g_z_hat)

integrate the image flow from x_0=z_K to obtain x_1(z_K)
return z_K, x_1(z_K)
```

## 9. How to initialize $Y_0$

Because $Y_0=0$ for the residual model $J_0=I+Y_0$, the rank-$r$ factorization is initially singular. There are three practical choices. Sections 6 and 8 use Option B; Options A and C below describe alternative initializations with a small nonzero core.

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
Y_h \approx h(A_0+I)
$$

for $\alpha_t=1-t$. Use the two QR factorizations and small SVD in Section 6 to produce factors of this first Euler target through JVP/VJP products. This preserves $Y_0=0$ and initializes the factors at time $h$, without inserting an $\varepsilon I$ core. The alternative formula $Y_h\approx hA_0$ applies when $\alpha_t\equiv1$.

### Option C: carry subspaces across inverse iterations

If $z_{k+1}$ is close to $z_k$, reuse the previous $U,V$ subspaces as a warm start, but reset the core to

$$
S_0=\varepsilon I_r
$$

or another tiny core. Do not place the previous final-time residual directly at $t=0$, because the residual initial condition remains $Y_0=0$. This subspace reuse is often the most efficient choice inside a line-search or LBFGS solver.

In Options A and C, $S_0$ can be tiny. With $S_0=\varepsilon I$, the perturbed
initial condition is

$$
\widehat J_0=\alpha_0I+\varepsilon U_0V_0^\top,
$$

so it approximates rather than exactly equals $J_0=I$. Setting $\varepsilon$
small controls this initialization error. The scalar baseline carries the
full-rank part, and the low-rank residual learns important deformation
directions.

## 10. Relationship to adjoint gradients

For one scalar objective and one source-gradient evaluation, the classical adjoint sensitivity equation

$$
\dot a_t = -A_t^\top A_t,
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
=g_x+V_1S_1^\top U_1^\top g_x+\lambda\nabla R_z(z).
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
Y_t=U_tS_tV_t^\top,
$$

we approximate

$$
\partial_\tau Y_t
\approx
P_{T_{Y_t}\mathcal M_r}W_t.
$$

Using the standard gauge conditions

$$
U_t^\top\partial_\tau U_t=0,
\qquad
V_t^\top\partial_\tau V_t=0,
$$

the continuous factor equations are

$$
\partial_\tau U_t
=
(I-U_tU_t^\top)W_tV_tS_t^{-1},
$$

$$
\partial_\tau S_t
=
U_t^\top W_tV_t,
$$

$$
\partial_\tau V_t
=
(I-V_tV_t^\top)W_t^\top U_tS_t^{-T}.
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
   -\Delta\tau\, (U_t^+)^\top W_tV_t.
   $$

3. L-step:

   $$
   L^+
   =
   V_t\widetilde S_t^\top+\Delta\tau\, W_t^\top U_t^+,
   \qquad
   L^+=V_t^+(S_t^+)^\top.
   $$

Then

$$
Y_t^+=U_t^+S_t^+(V_t^+)^\top.
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
   g_x+V_1S_1^\top U_1^\top g_x+\lambda\nabla R_z.
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
   \partial_t W_t=A_tW_t+H_t[p_t](I+U_tS_tV_t^\top),
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

Full Jacobian propagation requires applying $A_t$ to $d$ directions per time step. DLR-D-Flow applies $A_t$ and $A_t^\top$ only to $O(r)$ directions per time step.

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
   J_1^\top w_j
   \quad\text{from an exact adjoint/VJP check}
   $$

   against

   $$
   \alpha_1w_j + V_1S_1^\top U_1^\top W_j.
   $$

3. If the relative error is above tolerance, enrich $U,V$ with the missed directions.
4. Compress only the small augmented core, not the full matrix.

## 14. What this method does and does not guarantee

This method removes the full-SVD bottleneck. It does not make the exact flow Jacobian low rank.

The approximation is useful when the inverse-problem update only needs the dominant sensitivity directions. If the forward operator $\mathcal{A}$ needs information in directions discarded by the rank-$r$ model, the update can become biased.

The identity or scalar baseline is important. Without it, the approximation conflicts with $J_0=I$, and the method may incorrectly suppress source directions that are still relevant.

The low-rank factors should be treated as an approximate gradient mechanism or a structured preconditioner. A line search or objective decrease check should remain in the inverse solver.

## 15. Recommended first experiment

Start with the analytical-baseline-plus-low-rank residual model:

$$
J_t \approx \alpha_tI + U_tS_tV_t^\top,
\qquad
\alpha_t=1-t.
$$

Use a fixed small rank, for example $r=8,16,32$, and compare four quantities:

1. Data misfit decrease per optimizer step.
2. Runtime and memory against the original D-Flow backpropagation-through-ODE implementation.
3. Agreement of the approximate gradient with exact adjoint gradients on a small subset of iterations.
4. Final reconstruction quality.

The most important diagnostic is not the Frobenius error of $J_t$. The important diagnostic is the action error on relevant terminal gradients:

$$
\frac{
\|J_1^\top g_x - (\alpha_1g_x+V_1S_1^\top U_1^\top g_x)\|
}{
\|J_1^\top g_x\|
}.
$$

If this error is small while runtime decreases, the dynamic low-rank idea is helping the D-Flow inverse solver in the way we actually need.

## 16. Refresh cost findings (2026-09-02, circles-64, CPU)

Historical measurements on the pretrained circles-64 model (15 Euler steps, 8 CPU threads), comparing the implementations at that date against autograd through the same discrete flow:

| refresh | time | grad rel. err | cosine |
|---|---|---|---|
| legacy per-step evolution, r=8, 4 enrichment | 34.0 s | 0.055 | 0.9988 |
| legacy per-step evolution, r=4, 2 enrichment | 20.4 s | 0.076 | 0.9978 |
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
