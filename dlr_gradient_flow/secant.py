"""Endpoint KSL with an integrated secant and correction of the old factors.

For old factors Y=USV.T, freeze the additive increment J(new_source)-Y
through one KSL step. The algebra reduces to K=J(new_source)V followed by
L=J(new_source).T U_new. This is a warm-started endpoint subspace iteration,
not the literal Hessian-based sensitivity transport from Section 11.

The update corrects old factor error as well as source movement. It retains
only the flow states, recomputes local linearizations on the reverse pass,
and never forms the full Jacobian. Periodic fresh initialization can recover
important directions absent from the old right basis.
"""

from __future__ import annotations

import torch

from dlr_dflow.implementation.graphed_products import GraphedAProducts
from dlr_dflow.implementation.types import Factors


@torch.no_grad()
def secant_factors(
    model: torch.nn.Module,
    source: torch.Tensor,
    factors: Factors,
    n_steps: int = 100,
    *,
    products: GraphedAProducts | None = None,
) -> tuple[torch.Tensor, Factors]:
    """Return the Euler endpoint and one corrected rank-preserving KSL update.

    The terminal baseline is zero. Only the old V is needed because old U,S
    cancel algebraically in this corrected update. With ``products=None``,
    torch.func differentiates the native model. A supplied graph provider
    uses its shared primal; callers must validate that backend against the
    canonical Euler objective and construct it for this model and shape.
    """
    x = source.detach().clone()
    shape = x.shape
    dimension, rank = source.numel(), factors[2].shape[1]
    step = 1.0 / n_steps
    columns = factors[2].detach().clone()
    states: list[torch.Tensor] = []

    for index in range(n_steps):
        states.append(x)
        time = index * step
        if products is None:
            time_tensor = x.new_full((shape[0],), time)
            primal, actions = torch.func.vmap(
                lambda tangent: torch.func.jvp(
                    lambda value: model(value, time_tensor), (x,), (tangent,)),
                out_dims=(None, 0),
            )(columns.T.reshape(rank, *shape))
            derivative = actions.reshape(rank, dimension).T
        else:
            products.set_state(time, x)
            derivative = products.A(columns)
            primal = products.A.last_primal
        columns = columns + step * derivative
        x = x + step * primal

    u_new, _ = torch.linalg.qr(columns, mode="reduced")
    columns = u_new
    for index in reversed(range(n_steps)):
        state = states[index]
        time = index * step
        if products is None:
            time_tensor = state.new_full((shape[0],), time)
            _, pullback = torch.func.vjp(
                lambda value: model(value, time_tensor), state)
            actions = torch.func.vmap(lambda cotangent: pullback(cotangent)[0])(
                columns.T.reshape(rank, *shape))
            derivative = actions.reshape(rank, dimension).T
        else:
            products.set_state(time, state)
            derivative = products.AT(columns)
        columns = columns + step * derivative

    v_new, r_l = torch.linalg.qr(columns, mode="reduced")
    return x.detach(), (u_new.detach(), r_l.T.detach(), v_new.detach())
