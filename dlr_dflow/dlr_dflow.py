"""DLR-D-Flow: D-Flow source-point optimization with a dynamic low-rank
approximation of the flow sensitivity (see dlr_dflow_notes.md).

The pretrained flow model defines an ODE  x' = v_theta(t, x),  x_0 = z,  and the
generated sample is x_1(z).  For an inverse problem  y ~ A(x_1(z)),  D-Flow
minimizes  Phi(z) = 0.5||A(x_1(z)) - y||^2 + ...  whose exact source gradient is

        grad_z Phi = J_1^T g_x                         (+ lambda grad R_z)
        g_x        = DA(x_1)^* (A(x_1) - y)            (+ mu grad R_x)

with the flow sensitivity  J_t = d x_t / d z  solving the matrix ODE

        J_t' = A_t J_t,   J_0 = I,   A_t = d_x v_theta(t, x_t).

Instead of forming J_t or back-propagating through the ODE, we track a
scalar-baseline-plus-low-rank residual model

        J_t ~ alpha_t I + Y_t,   Y_t = U_t S_t V_t^T,   rank(Y_t) = r,

evolved along the flow by the Lubich-Oseledets projector-splitting integrator
(notes Sec. 6).  Only products A_t Q and A_t^T Q with skinny
matrices are needed; these are JVPs / VJPs through v_theta.  The approximate
source gradient is then

        grad_z Phi ~ alpha_1 g_x + V_1 S_1^T U_1^T g_x (+ lambda grad R_z).

This module provides the sensitivity evolution, the approximate gradient, an
exact-gradient reference (autograd through the same discrete flow) for the
action-error diagnostic of notes Sec. 15, and a small inverse-problem loop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Tuple

import torch

from models.unet.default import Unet

ROOT = Path(__file__).resolve().parents[1]
Factors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (U, S, V), U,V: (d,r), S: (r,r)


# ---------------------------------------------------------------------------
# Model + forward operator
# ---------------------------------------------------------------------------
def load_flow_model(resolution: int = 64, n_circles: int = 1, epoch: int = 50,
                    device: str = "cpu") -> Tuple[torch.nn.Module, int]:
    """Load the pretrained flow-matching UNet (smallest 64x64 n1 model by default)."""
    run = ROOT / f"outputs/fm_unet/circles-{resolution}/n{n_circles}"
    arch = json.loads((run / "config.json").read_text())["arch"]
    model = Unet(ch=arch["ch"], ch_mul=arch["ch_mul"], att_channels=arch["att_channels"],
                 groups=arch["groups"], dropout=0.0).to(device)
    state = torch.load(run / f"checkpoints/epoch_{epoch:04d}.pt",
                       map_location=device, weights_only=True)
    state = {k.removeprefix("module._orig_mod.").removeprefix("module."): v
             for k, v in state.items() if k != "n_averaged"}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, resolution


def make_blur(sigma: float, device: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Separable Gaussian blur forward operator A = H (matches the benchmark)."""
    ks = int(6 * sigma + 1)
    ks = ks if ks % 2 == 1 else ks + 1
    half = ks // 2
    coords = torch.arange(ks, dtype=torch.float32, device=device) - half
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    cx = torch.nn.Conv2d(1, 1, (1, ks), padding=(0, half), bias=False).to(device)
    cy = torch.nn.Conv2d(1, 1, (ks, 1), padding=(half, 0), bias=False).to(device)
    with torch.no_grad():
        cx.weight.copy_(g.view(1, 1, 1, ks))
        cy.weight.copy_(g.view(1, 1, ks, 1))
    for p in (*cx.parameters(), *cy.parameters()):
        p.requires_grad_(False)
    return lambda x: cy(cx(x))


def velocity(model: torch.nn.Module, t: float, X: torch.Tensor) -> torch.Tensor:
    """v_theta(t, X) for a batch X of shape (B,1,H,W) at a single scalar time t."""
    tvec = torch.full((X.shape[0],), float(t), device=X.device, dtype=X.dtype)
    return model(X, tvec)


def euler_flow(model: torch.nn.Module, z: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Generate x_1 from source z by fixed-step Euler (same scheme as the benchmark)."""
    dt = 1.0 / n_steps
    x = z
    for i in range(n_steps):
        x = x + dt * velocity(model, i * dt, x)
    return x


# ---------------------------------------------------------------------------
# Jacobian-of-velocity products  A_t Q = d_x v_theta(t, x) Q  (and transpose)
# ---------------------------------------------------------------------------
# The tangent columns ride a vmap lane while the primal x is captured
# unbatched, so the primal forward (and the frozen-weight tangent terms) are
# computed once per call instead of once per column; only the genuinely
# per-column tangent arithmetic is batched.  (Previously each column carried
# its own copy of the primal through a batched jvp/vjp.)
# Forward-mode products through the hand-written tangent pass of the UNet
# (tangent_unet.unet_jvp) instead of torch.func.jvp: same numbers (fp32 rel
# 2e-7), ~1.4 forward-equivalents per column instead of ~2.5 and 5x fewer
# kernels; flag kept for A/B measurement.
HAND_JVP = True
# Same for reverse mode (tangent_unet.unet_vjp): exact (fp32 rel 3e-7) but only
# faster in the launch-bound regime (1.5x at 64px, 0.85x at 128, 0.75x at 250)
# because autograd's fused backward kernels already beat eager adjoints there.
# Off by default so one code path serves every resolution; opt-in for <=96.
HAND_VJP = False


def make_A_products(model: torch.nn.Module, t: float, x_img: torch.Tensor):
    H, W = x_img.shape[-2:]
    d = H * W
    x1 = x_img.reshape(1, 1, H, W)
    tvec = torch.full((1,), float(t), device=x_img.device, dtype=x_img.dtype)

    def f(X):
        return model(X, tvec)

    def _lanes(Q: torch.Tensor) -> torch.Tensor:
        return Q.T.reshape(-1, 1, 1, H, W)

    def A(Q: torch.Tensor) -> torch.Tensor:            # A_t Q,  (d,r) -> (d,r)
        if HAND_JVP:
            from .tangent_unet import unet_jvp
            primal, out = unet_jvp(model, t, x1, Q.T.reshape(-1, 1, H, W))
            A.last_primal = primal.detach()
            return out.reshape(Q.shape[1], d).T
        # out_dims=(None, 0): the primal v(t, x) does not depend on the tangent
        # lane, so it comes back unbatched; keep it so the flow-state advance
        # can reuse it instead of running a fourth forward pass per step.
        primal, out = torch.func.vmap(
            lambda tangent: torch.func.jvp(f, (x1,), (tangent,)),
            out_dims=(None, 0))(_lanes(Q))
        A.last_primal = primal.detach()
        return out.reshape(Q.shape[1], d).T

    def AT(Q: torch.Tensor) -> torch.Tensor:           # A_t^T Q, (d,r) -> (d,r)
        if HAND_VJP:
            from .tangent_unet import unet_vjp
            _, out = unet_vjp(model, t, x1, Q.T.reshape(-1, 1, H, W))
            return out.reshape(Q.shape[1], d).T
        _, vjp_fn = torch.func.vjp(f, x1)
        out = torch.func.vmap(lambda cot: vjp_fn(cot)[0])(_lanes(Q))
        return out.reshape(Q.shape[1], d).T

    return A, AT


# ---------------------------------------------------------------------------
# Projector-splitting step for the residual Y = U S V^T with a scalar baseline
# B_t = alpha_t I  (notes Sec. 4 & 6).  F(t,Y) = A_t(alpha I + Y) - alpha_dot I,
# integrated by the K/S/L substeps with A_t, alpha_t frozen over the flow step.
# Setting alpha=1, alpha_dot=0 recovers the plain identity-baseline model.
# ---------------------------------------------------------------------------
def projector_split_step(A, AT, U, S, V, h: float, alpha: float, alpha_dot: float) -> Factors:
    r = V.shape[1]
    # K-step: Kdot = alpha A V0 + A K - alpha_dot V0,  K(t0)=U0 S0 -> QR  K=U1 Shat
    AV0_AK = A(torch.cat([V, U @ S], dim=1))           # one JVP: A V0 and A K
    AV0, AK = AV0_AK[:, :r], AV0_AK[:, r:]
    K1 = U @ S + h * (alpha * AV0 + AK - alpha_dot * V)
    U1, Shat = torch.linalg.qr(K1)

    # S-step: Sdot = -alpha U1^T A V0 - (U1^T A U1) S + alpha_dot U1^T V0
    AU1 = A(U1)
    M = U1.T @ AU1                                      # U1^T A U1
    Stil = Shat + h * (-alpha * (U1.T @ AV0) - M @ Shat + alpha_dot * (U1.T @ V))

    # L-step: Ldot = alpha A^T U1 + L (U1^T A^T U1) - alpha_dot U1,  L(t0)=V0 Stil^T
    ATU1 = AT(U1)
    L0 = V @ Stil.T
    L1 = L0 + h * (alpha * ATU1 + L0 @ M.T - alpha_dot * U1)   # U1^T A^T U1 = M^T
    V1, R = torch.linalg.qr(L1)
    return U1, R.T, V1


def init_residual_factors(d: int, r: int, device: str, dtype, eps: float = 1e-4,
                          seed: int = 0, seed_dir: torch.Tensor = None,
                          seed_dirs: torch.Tensor = None) -> Factors:
    """Y_0 ~ 0 seeded on the rank-r manifold (notes Sec. 9, Option A): U0=V0 an
    orthonormal basis, S0 = eps I.  If seed_dir (d,1) is given it becomes the
    first basis column, so the residual can immediately act on that direction."""
    g = torch.Generator(device=device).manual_seed(seed)
    cols = torch.randn(d, r, generator=g, device=device, dtype=dtype)
    if seed_dirs is not None:
        count = min(r, seed_dirs.shape[1])
        cols[:, :count] = seed_dirs.reshape(d, -1)[:, :count]
    elif seed_dir is not None:
        cols[:, :1] = seed_dir.reshape(d, 1)
    U, _ = torch.linalg.qr(cols)
    return U, eps * torch.eye(r, device=device, dtype=dtype), U.clone()


def _hutchinson_trace(A, P: torch.Tensor) -> float:
    """Hutchinson estimate of tr(A_t) = E[p^T A p] using probe columns P (d, k)."""
    return (P * A(P)).sum().item() / P.shape[1]


def evolve_sensitivity(model: torch.nn.Module, z: torch.Tensor, n_steps: int,
                       rank: int, eps: float = 1e-4, seed: int = 0,
                       seed_dir: torch.Tensor = None, scalar_baseline: bool = True,
                       alpha_eps: float = 0.05, baseline: str | None = None,
                       trace_probes: int = 8,
                       seed_dirs: torch.Tensor = None,
                       trace_probe_type: str = "gaussian",
                       fixed_trace_probes: bool = False,
                       return_alpha_path: bool = False):
    """Integrate the flow x_t and the low-rank residual factors together.

    Returns x_1, the final factors (U, S, V), and the terminal baseline alpha_1,
    so that J_1 ~ alpha_1 I + U S V^T. If return_alpha_path is true, also
    returns the N+1 baseline values from t=0 through t=1.

    Baseline choices:
      exponential: alpha(t) = 1 - exp(-(1-t)/alpha_eps), forcing alpha(1)=0;
      hutchinson_trace: d(log alpha)/dt ~= tr(A_t)/d using Gaussian probes;
      identity: alpha(t) = 1.

    baseline=None preserves the legacy scalar_baseline flag: true selects the
    exponential schedule and false selects the identity baseline.
    """
    H, W = z.shape[-2:]
    d = H * W
    dt = 1.0 / n_steps
    x = z
    U, S, V = init_residual_factors(
        d, rank, z.device, z.dtype, eps, seed, seed_dir, seed_dirs)
    if baseline is None:
        baseline = "exponential" if scalar_baseline else "identity"
    if baseline not in {"exponential", "hutchinson_trace", "identity"}:
        raise ValueError(f"unknown scalar baseline: {baseline}")
    if trace_probes < 1:
        raise ValueError("trace_probes must be positive")
    if trace_probe_type not in {"gaussian", "rademacher"}:
        raise ValueError(f"unknown trace probe type: {trace_probe_type}")
    log_alpha = 0.0
    probe_generator = torch.Generator(device=z.device).manual_seed(seed + 1)

    def draw_trace_probes():
        if trace_probe_type == "rademacher":
            return (2 * torch.randint(
                0, 2, (d, trace_probes), generator=probe_generator,
                device=z.device, dtype=torch.int64) - 1).to(dtype=z.dtype)
        return torch.randn(
            d, trace_probes, generator=probe_generator,
            device=z.device, dtype=z.dtype)

    fixed_probes = (draw_trace_probes()
                    if baseline == "hutchinson_trace" and fixed_trace_probes
                    else None)
    alpha_path = [1.0]
    for i in range(n_steps):
        t = i * dt
        A, AT = make_A_products(model, t, x)
        if baseline == "exponential":
            e = math.exp(-(1.0 - t) / alpha_eps)
            alpha = 1.0 - e
            alpha_dot = -e / alpha_eps
            next_t = (i + 1) * dt
            next_alpha = 1.0 - math.exp(-(1.0 - next_t) / alpha_eps)
        elif baseline == "hutchinson_trace":
            probes = fixed_probes if fixed_probes is not None else draw_trace_probes()
            trace_estimate = _hutchinson_trace(A, probes)
            alpha = math.exp(log_alpha)
            mean_log_rate = trace_estimate / d
            alpha_dot = alpha * mean_log_rate
            log_alpha += dt * mean_log_rate
            next_alpha = math.exp(log_alpha)
        else:
            alpha, alpha_dot = 1.0, 0.0
            next_alpha = 1.0
        U, S, V = projector_split_step(A, AT, U, S, V, dt, alpha, alpha_dot)
        x = x + dt * velocity(model, t, x)             # advance flow after freezing A_t
        alpha_path.append(next_alpha)
    alpha1 = alpha_path[-1]
    result = (x, (U, S, V), alpha1)
    return (*result, alpha_path) if return_alpha_path else result


# ---------------------------------------------------------------------------
# Source gradients
# ---------------------------------------------------------------------------
def terminal_gradient(x1: torch.Tensor, A_op: Callable, y: torch.Tensor) -> torch.Tensor:
    """g_x = DA(x1)^* (A(x1) - y) for the squared data misfit (Gamma = I)."""
    x = x1.detach().requires_grad_(True)
    cost = 0.5 * (A_op(x) - y).pow(2).sum()
    (g,) = torch.autograd.grad(cost, x)
    return g


def approx_source_gradient(factors: Factors, g_x: torch.Tensor, alpha1: float = 1.0
                           ) -> torch.Tensor:
    """DLR gradient  alpha_1 g_x + V S^T U^T g_x  = (alpha_1 I + U S V^T)^T g_x."""
    U, S, V = factors
    gx = g_x.reshape(-1, 1)
    gz = alpha1 * gx + V @ (S.T @ (U.T @ gx))
    return gz.reshape(g_x.shape)


def exact_source_gradient(model: torch.nn.Module, z: torch.Tensor, n_steps: int,
                          g_x: torch.Tensor) -> torch.Tensor:
    """Exact  J_1^T g_x  by autograd through the identical discrete Euler flow."""
    zl = z.detach().requires_grad_(True)
    x1 = euler_flow(model, zl, n_steps)
    (gz,) = torch.autograd.grad((x1 * g_x.detach()).sum(), zl)
    return gz


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def selftest_jacobian_products(model: torch.nn.Module, device: str) -> None:
    """Check the batched JVP/VJP helpers against autograd on a single column,
    and that <A q, w> = <q, A^T w>."""
    H = W = 64
    x = torch.randn(1, 1, H, W, device=device)
    A, AT = make_A_products(model, 0.37, x)
    q = torch.randn(H * W, 1, device=device)
    w = torch.randn(H * W, 1, device=device)

    # autograd JVP on a single column
    def f(xi):
        return velocity(model, 0.37, xi.reshape(1, 1, H, W)).reshape(-1)
    _, Aq_ref = torch.autograd.functional.jvp(f, x.reshape(-1), q.reshape(-1))
    rel = (A(q).reshape(-1) - Aq_ref).norm() / Aq_ref.norm()
    inner = (A(q) * w).sum().item()
    inner_T = (q * AT(w)).sum().item()
    print(f"[selftest] JVP vs autograd rel.err = {rel.item():.2e}")
    print(f"[selftest] <Aq,w> = {inner:.6f}  vs  <q,A^T w> = {inner_T:.6f}  "
          f"(rel {abs(inner - inner_T) / (abs(inner) + 1e-12):.2e})")


def action_error(model, z, n_steps, ranks, A_op, y, seed=0, scalar_baseline=True,
                 seed_grad=True, method="enriched"):
    """Relative action error ||J_1^T g_x - (alpha_1 g_x + V S^T U^T g_x)|| /
    ||J_1^T g_x|| over a list of ranks (notes Sec. 15 diagnostic).

    Returns the identity-only baseline, the scalar-baseline-only error, and per-rank
    (rank, rel.err, cosine) rows.
    """
    x1 = euler_flow(model, z, n_steps)
    g_x = terminal_gradient(x1, A_op, y)
    gz_exact = exact_source_gradient(model, z, n_steps, g_x)
    denom = gz_exact.norm().item()
    seed_dir = None
    if seed_grad:
        seed_dir = g_x.reshape(-1, 1) / (g_x.norm() + 1e-12)
    rows = []
    alpha1 = 1.0
    for r in ranks:
        if method == "legacy":
            # Legacy path: scalar_baseline=True selects the hand-tuned
            # exponential schedule, which forces alpha_1 = 0 exactly and is the
            # worst of the three baselines.  Kept only for regression checks.
            _, factors, alpha1 = evolve_sensitivity(
                model, z, n_steps, rank=r, seed=seed, seed_dir=seed_dir,
                scalar_baseline=scalar_baseline)
        else:
            _, factors, alpha1 = sensitivity_factors(
                model, z, n_steps, rank=r, method=method, seed=seed)
        gz_approx = approx_source_gradient(factors, g_x, alpha1)
        rel = (gz_exact - gz_approx).norm().item() / denom
        cos = torch.nn.functional.cosine_similarity(
            gz_exact.reshape(1, -1), gz_approx.reshape(1, -1)).item()
        rows.append((r, rel, cos))
    err_identity = (gz_exact - g_x).norm().item() / denom          # alpha=1, r=0
    err_scalar = (gz_exact - alpha1 * g_x).norm().item() / denom   # scalar baseline, r=0
    return err_identity, err_scalar, alpha1, rows, denom


# ---------------------------------------------------------------------------
# Preferred entry points (see notes Sec. 4 / 9 revision).
#
# Measured on circles-64 (d=4096, N=15): the terminal sensitivity J_1 has
# numerical rank 2 (singular values 56.1, 20.5, 0.163, ...), so a rank-4
# model is already adequate and the additive scalar baseline is only needed
# for small t, where J_t is still close to I.  The eps-seeded projector
# splitting in evolve_sensitivity cannot find the dominant subspace -- with
# S_0 = eps I the tangent projection is degenerate and the basis never
# rotates -- so the enriched evolution (deferred sketch initialization plus
# per-step basis enrichment) is the default here.
# ---------------------------------------------------------------------------
def sensitivity_factors(model: torch.nn.Module, z: torch.Tensor, n_steps: int,
                        rank: int = 4, method: str = "enriched",
                        zero_terminal_baseline: bool = True, **kwargs):
    """Return x_1, factors (U,S,V) and alpha_1 with J_1 ~ alpha_1 I + U S V^T."""
    if method == "enriched":
        from .enriched import evolve_trace_centered_enriched
        return evolve_trace_centered_enriched(
            model, z, n_steps, rank=rank,
            zero_terminal_baseline=zero_terminal_baseline, **kwargs)
    if method == "legacy":
        return evolve_sensitivity(model, z, n_steps, rank=rank, **kwargs)
    if method == "direct":
        from .enriched import direct_terminal_rank_factors
        factors = direct_terminal_rank_factors(
            model, z, n_steps, rank=rank, alpha=0.0,
            oversampling=kwargs.get("oversampling", 6),
            seed=kwargs.get("seed", 0))
        return euler_flow(model, z, n_steps), factors, 0.0
    raise ValueError(f"unknown sensitivity method: {method}")


def solve_inverse_lbfgs(model: torch.nn.Module, z0: torch.Tensor, A_op: Callable,
                        y: torch.Tensor, n_steps: int = 15, rank: int = 4,
                        outer_steps: int = 40, lr: float = 1.0,
                        history_size: int = 100, refresh_every: int = 5,
                        method: str = "enriched", x_true: torch.Tensor = None,
                        verbose: bool = False, target_phi: float = None,
                        sketch_steps: int = None, warm_start: bool = False,
                        time_limit: float = None, safeguard: bool = True,
                        blowup_factor: float = 4.0, min_lr: float = 1e-3,
                        reset_history_on_refresh: bool = False,
                        reg_lam: float = 0.0, shell_lam: float = 0.0,
                        shell_project: bool = False, lam_kurt: float = 0.0,
                        lam_tv: float = 0.0, init_mode: str = "random",
                        **kwargs):
    """D-Flow source optimization with the low-rank gradient driving LBFGS.

    Mirrors the repository's fixed-step LBFGS setup (max_iter=1, no line
    search, persistent history): each optimizer call costs one forward flow
    for the objective plus the frozen-factor action for the gradient; the
    factors are refreshed every refresh_every outer iterations.  No autograd
    passes through the flow anywhere.

    safeguard guards against the failure mode of driving LBFGS with an
    approximate gradient: with no line search, a single bad step has nothing
    to reject it and the iterate can run away (observed blowing up to NaN).
    When the objective goes non-finite or grows past blowup_factor times the
    best value seen, the source is rolled back to the best iterate, the LBFGS
    history is discarded (its curvature pairs are poisoned by the bad step),
    the step size is halved and the factors are refreshed."""
    import time as _time
    z = z0.detach().clone().requires_grad_(True)
    current_lr = lr
    optimizer = torch.optim.LBFGS(
        [z], lr=current_lr, max_iter=1, history_size=history_size,
        tolerance_grad=1e-7, tolerance_change=1e-9)
    factors = alpha1 = None
    history = []
    best_phi, best_z = float("inf"), z.detach().clone()
    since_best = 0                       # iterations since the last new best
    loop_start = _time.perf_counter()
    for k in range(1, outer_steps + 1):
        if time_limit is not None and _time.perf_counter() - loop_start > time_limit:
            break
        outer_t0 = _time.perf_counter()
        current_reg = reg_lam(k) if callable(reg_lam) else reg_lam
        refreshed = factors is None or (k - 1) % refresh_every == 0
        if refreshed:
            if method == "direct":
                # Skip sensitivity_factors' x_1 flow solve; the closure
                # recomputes x_1 anyway and the factors don't need it.
                from .enriched import direct_terminal_rank_factors
                previous_basis = (factors[2] if (warm_start and factors is not None)
                                  else None)
                factors = direct_terminal_rank_factors(
                    model, z.detach(), n_steps, rank=rank, alpha=0.0,
                    oversampling=kwargs.get("oversampling", 6),
                    seed=kwargs.get("seed", 0) + k,
                    sketch_steps=sketch_steps, warm_start=previous_basis)
                alpha1 = 0.0
            else:
                if method == "enriched" and kwargs.pop("use_cuda_graphs", False):
                    # capture once; replay across every refresh of this solve
                    from .graphed_products import GraphedAProducts
                    kwargs["products"] = GraphedAProducts(model, z.detach())
                if method == "enriched" and init_mode != "random":
                    # Notes Sec. 9: option A seeds with the previous source
                    # gradient, option C carries the previous V subspace.
                    if init_mode == "grad" and z.grad is not None:
                        kwargs["seed_basis"] = z.grad.detach().reshape(-1, 1)
                    elif init_mode == "carry" and factors is not None:
                        kwargs["seed_basis"] = factors[2].detach()
                _, factors, alpha1 = sensitivity_factors(
                    model, z.detach(), n_steps, rank=rank, method=method,
                    **kwargs)
            if reset_history_on_refresh and k > 1:
                # A refresh replaces one gradient approximation with another,
                # so curvature pairs (s, y) straddling the change compare
                # gradients of two different functions.  LBFGS treats them as
                # one, and the poisoned inverse-Hessian model produces bad
                # directions.  Discard the memory whenever the factors change.
                optimizer = torch.optim.LBFGS(
                    [z], lr=current_lr, max_iter=1, history_size=history_size,
                    tolerance_grad=1e-7, tolerance_change=1e-9)

        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                x1 = euler_flow(model, z, n_steps)
            g_x = terminal_gradient(x1, A_op, y)
            if lam_tv > 0.0:
                # anisotropic total variation on the generated image; its
                # gradient rides through the same J_1^T action as the misfit
                xl = x1.detach().requires_grad_(True)
                tv = ((xl[..., 1:, :] - xl[..., :-1, :]).abs().sum()
                      + (xl[..., :, 1:] - xl[..., :, :-1]).abs().sum())
                (g_tv,) = torch.autograd.grad(lam_tv * tv, xl)
                g_x = g_x + g_tv
            grad = approx_source_gradient(factors, g_x, alpha1).reshape_as(z)
            if lam_kurt > 0.0:
                # penalize heavy-tailed source coordinates: (kurtosis - 3)^2
                zl = z.detach().requires_grad_(True)
                zc = (zl - zl.mean()) / zl.std()
                excess = (zc ** 4).mean() - 3.0
                (g_k,) = torch.autograd.grad(lam_kurt * excess ** 2, zl)
                grad = grad + g_k
            if shell_lam > 0.0:
                # Penalize distance from the Gaussian SHELL ||z|| = sqrt(d),
                # not from the origin: in high dimension the mode is atypical,
                # and the measured pathology is shell departure, not norm size.
                zn = z.detach().norm()
                grad = grad + shell_lam * (1.0 - math.sqrt(z.numel()) / zn) * z.detach()
            if current_reg > 0.0:
                # Gaussian-prior term 0.5*reg_lam*||z||^2 (the standard D-Flow
                # source regularizer): keeps z near the typical set where the
                # low-rank premise on J_1 holds.  phi below stays the pure
                # measurement misfit so stopping/reporting are unchanged.
                grad = grad + current_reg * z.detach()
            z.grad = grad
            with torch.no_grad():
                return 0.5 * (A_op(x1) - y).pow(2).sum()

        z_eval = z.detach().clone()          # phi below is evaluated at this z
        phi = float(optimizer.step(closure))
        if shell_project:
            with torch.no_grad():
                z.mul_(math.sqrt(z.numel()) / z.norm())
        rolled_back = False
        if phi < best_phi:
            best_phi, best_z = phi, z_eval
            since_best = 0
        else:
            since_best += 1
        if phi >= best_phi and safeguard and (not math.isfinite(phi)
                            or phi > blowup_factor * best_phi):
            with torch.no_grad():
                z.copy_(best_z)
            current_lr = max(min_lr, current_lr * 0.5)
            optimizer = torch.optim.LBFGS(
                [z], lr=current_lr, max_iter=1, history_size=history_size,
                tolerance_grad=1e-7, tolerance_change=1e-9)
            factors = None                   # stale factors: force a refresh
            phi = best_phi
            rolled_back = True
        # adaptive fidelity: a coarse factor grid (factor_stride > 1) that
        # rolls back or stagnates has hit a landscape needing full per-step
        # factors (e.g. the 128px plateau) - drop to stride 1 and continue
        if (kwargs.get("factor_stride", 1) > 1
                and (rolled_back or since_best >= 8)):
            kwargs["factor_stride"] = 1
            factors = None
            since_best = 0
            if verbose:
                print(f"  step {k}: factor_stride -> 1 (fidelity fallback)")
        row = {"step": k, "phi": phi, "refreshed": refreshed,
               "rolled_back": rolled_back, "lr": current_lr,
               "time_s": _time.perf_counter() - outer_t0}
        if x_true is not None:
            with torch.no_grad():
                xk = euler_flow(model, z, n_steps)
                row["image_mse"] = (xk.clamp(0, 1) - x_true).pow(2).mean().item()
        history.append(row)
        if target_phi is not None and phi <= target_phi:
            break
        if verbose and (k % 5 == 0 or k == 1):
            print(f"  step {k:3d}: phi={phi:.4e}")
    # Return the iterate whose phi was actually recorded (the closure evaluates
    # before the update), not the post-update z the loop ends on.
    return best_z, history


def solve_inverse_lm(model: torch.nn.Module, z0: torch.Tensor, A_op: Callable,
                     y: torch.Tensor, n_steps: int = 100, rank: int = 4,
                     outer_steps: int = 60, refresh_every: int = 2,
                     oversampling: int = 4, sketch_steps: int = None,
                     lam0: float = 10.0, lam_up: float = 4.0,
                     lam_down: float = 0.5, max_rejects: int = 3,
                     target_phi: float = None, time_limit: float = None,
                     seed: int = 0, verbose: bool = False):
    """Sketched Levenberg-Marquardt D-Flow solver.

    Uses the terminal factors J_1 ~ U S V^T as an explicit Gauss-Newton model
    instead of feeding their action to a quasi-Newton method: for a step
    z <- z + V w the linearized measurement residual is r + A(U S) w, so the
    damped normal equations are rank x rank.  There is no optimizer history,
    so factor refreshes poison nothing (the LBFGS failure mode here), and a
    rejected trial raises the damping instead of running away (the
    no-line-search failure mode)."""
    import time as _time
    from .enriched import direct_terminal_rank_factors
    z = z0.detach().clone()
    H, W = z.shape[-2:]

    def phi_of(zz):
        with torch.no_grad():
            return 0.5 * (A_op(euler_flow(model, zz, n_steps)) - y).pow(2).sum().item()

    phi = phi_of(z)
    lam = lam0
    history = []
    factors = None
    accepted_since_refresh = rejects_since_refresh = sketches = 0
    t_start = _time.perf_counter()
    for k in range(1, outer_steps + 1):
        if time_limit is not None and _time.perf_counter() - t_start > time_limit:
            break
        t0 = _time.perf_counter()
        refreshed = False
        if (factors is None or accepted_since_refresh >= refresh_every
                or rejects_since_refresh >= max_rejects):
            factors = direct_terminal_rank_factors(
                model, z, n_steps, rank=rank, alpha=0.0,
                oversampling=oversampling, seed=seed + k,
                sketch_steps=sketch_steps)
            sketches += 1
            accepted_since_refresh = rejects_since_refresh = 0
            refreshed = True
        U, S, V = factors
        with torch.no_grad():
            x1 = euler_flow(model, z, n_steps)
            residual = (A_op(x1) - y).reshape(-1)
            US = U @ S
            B = A_op(US.T.reshape(-1, 1, H, W)).reshape(US.shape[1], -1).T
            BtB = B.T @ B
            rhs = -(B.T @ residual)
        eye = torch.eye(BtB.shape[0], device=z.device, dtype=z.dtype)
        accepted = False
        for _ in range(max_rejects):
            w = torch.linalg.solve(BtB + lam * eye, rhs)
            z_try = z + (V @ w).reshape_as(z)
            phi_try = phi_of(z_try)
            if phi_try < phi:
                z, phi = z_try, phi_try
                lam = max(lam * lam_down, 1e-8)
                accepted = True
                accepted_since_refresh += 1
                break
            lam *= lam_up
        if not accepted:
            rejects_since_refresh += 1
        history.append({"step": k, "phi": phi, "refreshed": refreshed,
                        "accepted": accepted, "lam": lam, "sketches": sketches,
                        "time_s": _time.perf_counter() - t0})
        if target_phi is not None and phi <= target_phi:
            break
        if verbose and (k % 5 == 0 or k == 1):
            print(f"  step {k:3d}: phi={phi:.4e} lam={lam:.2e}")
    return z, history


def objective(model: torch.nn.Module, z: torch.Tensor, n_steps: int,
              A_op: Callable, y: torch.Tensor) -> float:
    """Phi(z) = 0.5 ||A(x_1(z)) - y||^2, one forward flow solve."""
    with torch.no_grad():
        return 0.5 * (A_op(euler_flow(model, z, n_steps)) - y).pow(2).sum().item()


def solve_inverse(model: torch.nn.Module, z0: torch.Tensor, A_op: Callable,
                  y: torch.Tensor, n_steps: int = 15, rank: int = 4,
                  outer_steps: int = 40, step0: float = 1.0,
                  method: str = "enriched", line_search: bool = True,
                  armijo_c: float = 1e-4, max_backtracks: int = 12,
                  x_true: torch.Tensor = None, verbose: bool = False,
                  refresh_every: int = 1, target_phi: float = None,
                  **kwargs):
    """D-Flow source optimization with the low-rank gradient and a backtracking
    Armijo line search.  The factors are reused across all trial steps, which is
    the reuse argument of notes Sec. 10 -- each trial costs one forward flow.

    refresh_every=k additionally freezes the factors (U, S, V, alpha_1) for k
    outer iterations at a time: between refreshes only x_1 and the terminal
    gradient g_x are recomputed (one forward flow each), amortizing the
    expensive sensitivity evolution across k source updates."""
    import time as _time
    z = z0.detach().clone()
    phi = objective(model, z, n_steps, A_op, y)
    step = step0
    history = []
    factors = alpha1 = None
    for k in range(1, outer_steps + 1):
        outer_t0 = _time.perf_counter()
        refreshed = factors is None or (k - 1) % refresh_every == 0
        if refreshed:
            x1, factors, alpha1 = sensitivity_factors(
                model, z, n_steps, rank=rank, method=method, **kwargs)
        else:
            with torch.no_grad():
                x1 = euler_flow(model, z, n_steps)
        g_x = terminal_gradient(x1, A_op, y)
        g = approx_source_gradient(factors, g_x, alpha1).reshape_as(z)
        gnorm2 = g.pow(2).sum().item()

        if line_search:
            while True:
                trial = step * 2.0                   # allow the step to grow back
                accepted = False
                for _ in range(max_backtracks):
                    z_try = z - trial * g
                    phi_try = objective(model, z_try, n_steps, A_op, y)
                    if phi_try <= phi - armijo_c * trial * gnorm2:
                        z, phi, step, accepted = z_try, phi_try, trial, True
                        break
                    trial *= 0.5
                if accepted or refreshed:
                    break
                # A stale frozen gradient failed the search: refresh the
                # factors at the current z and retry the search once.
                x1, factors, alpha1 = sensitivity_factors(
                    model, z, n_steps, rank=rank, method=method, **kwargs)
                refreshed = True
                g_x = terminal_gradient(x1, A_op, y)
                g = approx_source_gradient(factors, g_x, alpha1).reshape_as(z)
                gnorm2 = g.pow(2).sum().item()
            if not accepted:                          # fresh gradient also failed
                step *= 0.5
        else:
            z = z - step * g
            phi = objective(model, z, n_steps, A_op, y)

        row = {"step": k, "phi": phi, "step_size": step, "grad_norm": gnorm2 ** 0.5,
               "refreshed": refreshed, "time_s": _time.perf_counter() - outer_t0}
        if x_true is not None:
            with torch.no_grad():
                xk = euler_flow(model, z, n_steps)
                row["image_mse"] = (xk.clamp(0, 1) - x_true).pow(2).mean().item()
        history.append(row)
        if target_phi is not None and phi <= target_phi:
            break
        if verbose and (k % 5 == 0 or k == 1):
            extra = f" image_mse={row['image_mse']:.3e}" if x_true is not None else ""
            print(f"  step {k:3d}: phi={phi:.4e} eta={step:.3e}{extra}")
    return z, history
