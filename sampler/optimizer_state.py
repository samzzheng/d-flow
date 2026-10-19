"""History reset that preserves native L-BFGS's lifetime first-step schedule."""
from __future__ import annotations

import torch


LBFGS_INITIAL_STEP_POLICY = "first_iteration_only"
LBFGS_HISTORY_RESET_POLICY = "preserve_iteration_schedule"


def reset_optimizer_history(optimizer) -> None:
    """Discard curvature/moments while preserving the native L-BFGS schedule.

    Torch's plain state.clear() also clears n_iter and repeats its initial
    1/||g||_1 step. A rollback or factor refresh is within the same solve, so
    retain n_iter and cumulative function evaluations. Seed a zero previous
    step, empty curvature lists and identity inverse-Hessian scaling: the next
    step cannot form a secant pair across the reset and starts at the usual lr.

    Other optimizers retain their original state.clear() behavior. Custom
    Armijo L-BFGS keeps its lifetime counter outside state and handles cache
    invalidation at its next step. Construct a new optimizer for a new solve.
    """
    if not isinstance(optimizer, torch.optim.LBFGS):
        optimizer.state.clear()
        return
    parameters = optimizer.param_groups[0]["params"]
    previous = optimizer.state.get(parameters[0], {})
    iterations = previous.get("n_iter", 0)
    evaluations = previous.get("func_evals", 0)
    optimizer.state.clear()
    state = optimizer.state[parameters[0]]
    state.update(n_iter=iterations, func_evals=evaluations)
    if iterations == 0:
        return
    # Native LBFGS flattens complex gradients into real and imaginary entries.
    zeros = torch.cat([
        parameter.real.new_zeros(2 * parameter.numel()) if parameter.is_complex()
        else parameter.new_zeros(parameter.numel())
        for parameter in parameters
    ])
    state.update(old_dirs=[], old_stps=[], ro=[], H_diag=1.0,
                 prev_flat_grad=zeros.clone(), d=zeros, t=0.0)
