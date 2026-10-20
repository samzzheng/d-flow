"""Optimizer-independent source optimization with DLR D-Flow gradients.

Pass an optimizer factory to optimize; optimizer selection, learning rates,
and line-search rules belong to the caller or the convenience adapters.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from sampler.ode_samplers import euler_flow
from sampler.optimizer_state import reset_optimizer_history

from . import dlr_dflow as core
from .implementation.objectives import squared_data_misfit
from .implementation.types import OptimizationResult, TensorOperator

OptimizerFactory = Callable[[list[torch.Tensor]], torch.optim.Optimizer]


def objective(model: torch.nn.Module, z: torch.Tensor, n_steps: int,
              A_op: TensorOperator, y: torch.Tensor) -> float:
    """Phi(z) = 0.5 ||A(x_1(z)) - y||^2, one forward flow solve."""
    with torch.no_grad():
        return squared_data_misfit(A_op(euler_flow(model, z, n_steps)) - y).item()


def optimize(model: torch.nn.Module, z0: torch.Tensor, A_op: TensorOperator,
             y: torch.Tensor, *, optimizer_factory: OptimizerFactory,
             n_steps: int = 15, rank: int = 4,
             outer_steps: int = 40, refresh_every: int = 5,
             method: str = "enriched", x_true: torch.Tensor | None = None,
             verbose: bool = False, target_phi: float | None = None,
             time_limit: float | None = None, safeguard: bool = True,
             blowup_factor: float = 4.0, min_lr: float = 1e-3,
             reset_history_on_refresh: bool = False,
             reg_lam: float | Callable[[int], float] = 0.0, shell_lam: float = 0.0,
             shell_project: bool = False, lam_kurt: float = 0.0,
             lam_tv: float = 0.0, init_mode: str = "random",
             dlr_activation_checkpointing: bool = False,
             reset_optimizer: Callable[[torch.optim.Optimizer], None] = reset_optimizer_history,
             **kwargs: Any) -> OptimizationResult:
    """Optimize a source using DLR gradients and a caller-supplied optimizer.

    The factory receives [z] and returns a closure-compatible optimizer:
    step(closure) must evaluate the closure before its first source update.
    Its return value is ignored. Repeated closure calls evaluate the current
    source, while factors remain frozen until the next scheduled refresh.
    Choose max_iter=1 when each internal update needs a sensitivity refresh.

    History and stopping use the initial measurement misfit of each outer
    step; image metrics use that same source, or the best source on rollback.
    Return the best of those evaluated sources, rather than the final
    unevaluated update. Regularizers contribute to both closure loss and
    gradient, while measurement reporting remains separate.

    Refreshes and rollbacks use reset_optimizer; supply a custom reset hook
    if your optimizer keeps history outside its standard state dictionary.
    """
    import time as _time
    if method != "enriched":
        raise ValueError("DLR D-Flow supports only method='enriched'")
    # Keep the lower-level keyword usable by existing callers of this wrapper.
    checkpoint_blocks = kwargs.pop("checkpoint_blocks", dlr_activation_checkpointing)
    if dlr_activation_checkpointing and not checkpoint_blocks:
        raise ValueError("conflicting DLR checkpointing settings")
    z = z0.detach().clone().requires_grad_(True)
    if outer_steps < 1 or refresh_every < 1:
        raise ValueError("outer_steps and refresh_every must be positive")
    optimizer = optimizer_factory([z])
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
        refreshed_x1 = None
        if refreshed:
            if kwargs.pop("use_cuda_graphs", False):
                # capture once; replay across every refresh of this solve
                from .implementation.graphed_products import GraphedAProducts
                kwargs["products"] = GraphedAProducts(
                    model, z.detach(), checkpoint_blocks=checkpoint_blocks)
            if init_mode != "random":
                # Notes Sec. 9: option A seeds with the previous source
                # gradient, option C carries the previous V subspace.
                if init_mode == "grad" and z.grad is not None:
                    kwargs["seed_basis"] = z.grad.detach().reshape(-1, 1)
                elif init_mode == "carry" and factors is not None:
                    kwargs["seed_basis"] = factors[2].detach()
            refreshed_x1, factors, alpha1 = core.sensitivity_factors(
                model, z.detach(), n_steps, rank=rank, method=method,
                checkpoint_blocks=checkpoint_blocks, **kwargs)
            if reset_history_on_refresh and k > 1:
                # Avoid carrying optimizer history across a changed
                # gradient approximation when requested by the caller.
                reset_optimizer(optimizer)

        initial_phi = None

        @torch.enable_grad()
        def closure() -> torch.Tensor:
            nonlocal refreshed_x1, initial_phi
            optimizer.zero_grad()
            # A sensitivity refresh already advanced this same source through
            # the flow. Consume that terminal state once; any later closure
            # evaluation must evaluate its own (possibly updated) source.
            x1, refreshed_x1 = refreshed_x1, None
            if x1 is None:
                with torch.no_grad():
                    x1 = euler_flow(model, z, n_steps)
            g_x = core.terminal_gradient(x1, A_op, y)
            with torch.no_grad():
                loss = squared_data_misfit(A_op(x1) - y)
            if initial_phi is None:
                initial_phi = float(loss)
            penalty = loss.new_zeros(())
            if lam_tv > 0.0:
                # anisotropic total variation on the generated image; its
                # gradient rides through the same J_1^T action as the misfit
                xl = x1.detach().requires_grad_(True)
                tv = ((xl[..., 1:, :] - xl[..., :-1, :]).abs().sum()
                      + (xl[..., :, 1:] - xl[..., :, :-1]).abs().sum())
                (g_tv,) = torch.autograd.grad(lam_tv * tv, xl)
                g_x = g_x + g_tv
                penalty = penalty + lam_tv * tv.detach()
            grad = core.approx_source_gradient(factors, g_x, alpha1).reshape_as(z)
            if lam_kurt > 0.0:
                # penalize heavy-tailed source coordinates: (kurtosis - 3)^2
                zl = z.detach().requires_grad_(True)
                zc = (zl - zl.mean()) / zl.std()
                excess = (zc ** 4).mean() - 3.0
                (g_k,) = torch.autograd.grad(lam_kurt * excess ** 2, zl)
                grad = grad + g_k
                penalty = penalty + lam_kurt * excess.detach().square()
            if shell_lam > 0.0:
                # Penalize distance from the Gaussian SHELL ||z|| = sqrt(d),
                # not from the origin: in high dimension the mode is atypical,
                # and the measured pathology is shell departure, not norm size.
                zn = z.detach().norm()
                grad = grad + shell_lam * (1.0 - math.sqrt(z.numel()) / zn) * z.detach()
                penalty = penalty + 0.5 * shell_lam * (zn - math.sqrt(z.numel())).square()
            if current_reg > 0.0:
                # Gaussian-prior term 0.5*reg_lam*||z||^2 (the standard D-Flow
                # source regularizer): keeps z near the typical set where the
                # low-rank premise on J_1 holds.  phi below stays the pure
                # measurement misfit so stopping/reporting are unchanged.
                grad = grad + current_reg * z.detach()
                penalty = penalty + 0.5 * current_reg * z.detach().square().sum()
            z.grad = grad
            return loss + penalty

        z_eval = z.detach().clone()          # phi below is evaluated at this z
        optimizer.step(closure)
        if initial_phi is None:
            raise RuntimeError("optimizer.step(closure) must evaluate the closure")
        phi = initial_phi
        if shell_project:
            with torch.no_grad():
                z.mul_(math.sqrt(z.numel()) / z.norm())
        rolled_back = False
        if math.isfinite(phi) and phi < best_phi:
            best_phi, best_z = phi, z_eval
            since_best = 0
        else:
            since_best += 1
        if safeguard and (not math.isfinite(phi)
                          or phi > blowup_factor * best_phi):
            with torch.no_grad():
                z.copy_(best_z)
            for group in optimizer.param_groups:
                group["lr"] = max(min_lr, group["lr"] * 0.5)
            reset_optimizer(optimizer)
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
               "rolled_back": rolled_back, "lr": optimizer.param_groups[0]["lr"],
               "dlr_activation_checkpointing": bool(checkpoint_blocks),
               "time_s": _time.perf_counter() - outer_t0}
        if x_true is not None:
            with torch.no_grad():
                xk = euler_flow(model, best_z if rolled_back else z_eval, n_steps)
                row["image_mse"] = (xk.clamp(0, 1) - x_true).pow(2).mean().item()
        history.append(row)
        if target_phi is not None and phi <= target_phi:
            break
        if verbose and (k % 5 == 0 or k == 1):
            print(f"  step {k:3d}: phi={phi:.4e}")
    # Return the iterate whose phi was actually recorded (the closure evaluates
    # before the update), not the post-update z the loop ends on.
    return best_z, history
