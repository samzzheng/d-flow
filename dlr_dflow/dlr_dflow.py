"""DLR D-Flow sensitivity and source-gradient computation.

Approximate the flow sensitivity as J_t = alpha_t I + U_t S_t V_t.T.
The trace-centered integrator evolves the factors using velocity Jacobian
products, then applies their transpose to a caller's terminal loss gradient.
Source optimization lives in optimization.py; backends and utilities in implementation/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .implementation.types import EvolutionResult, Factors, TensorOperator

if TYPE_CHECKING:
    from .implementation.tangent_unet import UnetLinearization

# Backend switches for velocity Jacobian products.
HAND_JVP = True
HAND_VJP = True
SHARE_LINEARIZATION = True


def velocity(model: torch.nn.Module, t: float, X: torch.Tensor) -> torch.Tensor:
    """v_theta(t, X) for a batch X of shape (B,1,H,W) at a single scalar time t."""
    tvec = torch.full((X.shape[0],), float(t), device=X.device, dtype=X.dtype)
    return model(X, tvec)


def make_A_products(model: torch.nn.Module, t: float, x_img: torch.Tensor,
                    *, checkpoint_blocks: bool = False) -> tuple[TensorOperator, TensorOperator]:
    """Build exact local Jacobian actions, optionally recomputing block tapes."""
    if checkpoint_blocks and not (SHARE_LINEARIZATION and HAND_JVP and HAND_VJP):
        raise ValueError("checkpoint_blocks requires shared hand-written JVP/VJP products")
    H, W = x_img.shape[-2:]
    d = H * W
    x1 = x_img.reshape(1, 1, H, W)
    tvec = torch.full((1,), float(t), device=x_img.device, dtype=x_img.dtype)

    def f(X: torch.Tensor) -> torch.Tensor:
        return model(X, tvec)

    def _lanes(Q: torch.Tensor) -> torch.Tensor:
        return Q.T.reshape(-1, 1, 1, H, W)

    linearization = None

    def shared_tape() -> UnetLinearization:
        nonlocal linearization
        if linearization is None:
            from .implementation.tangent_unet import unet_linearize
            linearization = unet_linearize(
                model, t, x1, checkpoint_blocks=checkpoint_blocks)
        return linearization

    def prepare_linearization() -> None:
        # Concurrent callers can build the shared primal on their producer
        # stream before recording the dependency for derivative streams.
        if SHARE_LINEARIZATION and HAND_JVP and HAND_VJP:
            shared_tape()

    def A(Q: torch.Tensor) -> torch.Tensor:            # A_t Q,  (d,r) -> (d,r)
        if SHARE_LINEARIZATION and HAND_JVP and HAND_VJP:
            tape = shared_tape()
            A.last_primal = tape.primal.detach()
            return tape.jvp(Q.T.reshape(-1, 1, H, W)).reshape(Q.shape[1], d).T
        if HAND_JVP:
            from .implementation.tangent_unet import unet_jvp
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
        if SHARE_LINEARIZATION and HAND_JVP and HAND_VJP:
            return shared_tape().vjp(Q.T.reshape(-1, 1, H, W)).reshape(Q.shape[1], d).T
        if HAND_VJP:
            from .implementation.tangent_unet import unet_vjp
            _, out = unet_vjp(model, t, x1, Q.T.reshape(-1, 1, H, W))
            return out.reshape(Q.shape[1], d).T
        _, vjp_fn = torch.func.vjp(f, x1)
        out = torch.func.vmap(lambda cot: vjp_fn(cot)[0])(_lanes(Q))
        return out.reshape(Q.shape[1], d).T

    A.prepare_linearization = prepare_linearization
    return A, AT


def terminal_gradient(x1: torch.Tensor, A_op: TensorOperator, y: torch.Tensor) -> torch.Tensor:
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


def sensitivity_factors(
    model: torch.nn.Module,
    z: torch.Tensor,
    n_steps: int,
    rank: int = 4,
    method: str = "enriched",
    zero_terminal_baseline: bool = True,
    speed_profile: str | None = None,
    checkpoint_blocks: bool = False,
    **kwargs: object,
) -> EvolutionResult:
    """Return the terminal image, factors (U, S, V), and scalar baseline.

    Alpha follows the trace estimate in the current integrator. Setting
    trace_probes=0 selects its analytic alpha_t=1-t option. The terminal
    baseline can be set to zero independently via zero_terminal_baseline.

    The fast profile uses four trace probes and no enrichment; reference
    uses eight probes and four enrichment directions. Explicit counts take
    precedence. Optional alpha paths and diagnostics are passed through.
    """
    from .implementation.enriched import evolve_trace_centered_enriched

    if method != "enriched":
        raise ValueError("DLR D-Flow supports only method='enriched'")
    if speed_profile is not None:
        if speed_profile not in {"reference", "fast"}:
            raise ValueError("speed_profile must be 'reference' or 'fast'")
        probes, enrichment = (4, 0) if speed_profile == "fast" else (8, 4)
        kwargs.setdefault("trace_probes", probes)
        kwargs.setdefault("enrichment_probes", enrichment)
    return evolve_trace_centered_enriched(
        model, z, n_steps, rank=rank,
        zero_terminal_baseline=zero_terminal_baseline,
        checkpoint_blocks=checkpoint_blocks, **kwargs)
