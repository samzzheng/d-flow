"""Convenience adapters for fixed-step LBFGS and Armijo gradient descent.

The optimizer-independent driver is dlr_dflow.optimization.optimize.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from sampler.ode_samplers import euler_flow
from sampler.optimizer_state import LBFGS_HISTORY_RESET_POLICY, LBFGS_INITIAL_STEP_POLICY

from .. import dlr_dflow as core
from ..optimization import objective, optimize
from .types import OptimizationResult, TensorOperator


def solve_inverse_lbfgs(model: torch.nn.Module, z0: torch.Tensor, A_op: TensorOperator,
                        y: torch.Tensor, n_steps: int = 15, rank: int = 4,
                        outer_steps: int = 40, lr: float = 1.0,
                        history_size: int = 100, refresh_every: int = 5,
                        method: str = "enriched", x_true: torch.Tensor | None = None,
                        verbose: bool = False, target_phi: float | None = None,
                        time_limit: float | None = None, safeguard: bool = True,
                        blowup_factor: float = 4.0, min_lr: float = 1e-3,
                        reset_history_on_refresh: bool = False,
                        reg_lam: float | Callable[[int], float] = 0.0, shell_lam: float = 0.0,
                        shell_project: bool = False, lam_kurt: float = 0.0,
                        lam_tv: float = 0.0, init_mode: str = "random",
                        dlr_activation_checkpointing: bool = False,
                        **kwargs: Any) -> OptimizationResult:
    """Fixed-step LBFGS convenience adapter for the shared DLR driver."""
    result, history = optimize(
        model, z0, A_op, y,
        optimizer_factory=lambda parameters: torch.optim.LBFGS(
            parameters, lr=lr, max_iter=1, history_size=history_size,
            tolerance_grad=1e-7, tolerance_change=1e-9),
        n_steps=n_steps, rank=rank, outer_steps=outer_steps,
        refresh_every=refresh_every, method=method, x_true=x_true,
        verbose=verbose, target_phi=target_phi, time_limit=time_limit,
        safeguard=safeguard, blowup_factor=blowup_factor, min_lr=min_lr,
        reset_history_on_refresh=reset_history_on_refresh,
        reg_lam=reg_lam, shell_lam=shell_lam, shell_project=shell_project,
        lam_kurt=lam_kurt, lam_tv=lam_tv, init_mode=init_mode,
        dlr_activation_checkpointing=dlr_activation_checkpointing, **kwargs)
    for row in history:
        row.update(lbfgs_initial_step_policy=LBFGS_INITIAL_STEP_POLICY,
                   lbfgs_history_reset_policy=LBFGS_HISTORY_RESET_POLICY)
    return result, history


def solve_inverse(model: torch.nn.Module, z0: torch.Tensor, A_op: TensorOperator,
                  y: torch.Tensor, n_steps: int = 15, rank: int = 4,
                  outer_steps: int = 40, step0: float = 1.0,
                  method: str = "enriched", line_search: bool = True,
                  armijo_c: float = 1e-4, max_backtracks: int = 12,
                  x_true: torch.Tensor | None = None, verbose: bool = False,
                  refresh_every: int = 1, target_phi: float | None = None,
                  **kwargs: Any) -> OptimizationResult:
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
            x1, factors, alpha1 = core.sensitivity_factors(
                model, z, n_steps, rank=rank, method=method, **kwargs)
        else:
            with torch.no_grad():
                x1 = euler_flow(model, z, n_steps)
        g_x = core.terminal_gradient(x1, A_op, y)
        g = core.approx_source_gradient(factors, g_x, alpha1).reshape_as(z)
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
                x1, factors, alpha1 = core.sensitivity_factors(
                    model, z, n_steps, rank=rank, method=method, **kwargs)
                refreshed = True
                g_x = core.terminal_gradient(x1, A_op, y)
                g = core.approx_source_gradient(factors, g_x, alpha1).reshape_as(z)
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
