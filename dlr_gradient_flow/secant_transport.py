"""Pure Jacobian-secant transport with a frozen-increment K/S/L step.

The increment is Delta=J(new_source)-J(old_source), the path integral of
D_z J(z)[new_source-old_source]. This avoids second derivatives but retains
the old factor error. Unlike corrected endpoint iteration in secant.py, it
does not replace that error by the new Jacobian. Freezing Delta in a K/S/L
step still introduces low-rank projection/discretization error.
"""

from __future__ import annotations

import torch

from dlr_dflow.implementation.graphed_products import GraphedAProducts
from dlr_dflow.implementation.types import Factors
from sampler.ode_samplers import euler_flow


@torch.no_grad()
def secant_transport_factors(
    model: torch.nn.Module,
    old_source: torch.Tensor,
    new_source: torch.Tensor,
    factors: Factors,
    n_steps: int = 100,
    *,
    products: GraphedAProducts | None = None,
) -> tuple[torch.Tensor, Factors]:
    """Return the new Euler endpoint and transported terminal factors.

    Four matrix-free derivative passes compute Delta V and Delta.T U_new.
    Native torch.func products differentiate the actual Euler map; an
    optional graph provider must agree with that map. Only old/new primal
    states and rank-r columns are retained, not full-map derivative tapes.
    Alpha_1=0, so the factors represent the full terminal sensitivity.
    """
    if torch.equal(old_source, new_source):
        return euler_flow(model, new_source, n_steps).detach(), tuple(
            value.detach().clone() for value in factors)
    U, S, V = factors
    dimension, rank = V.shape
    shape, step = new_source.shape, 1.0 / n_steps
    trajectories: list[list[torch.Tensor]] = []
    forward_results: list[torch.Tensor] = []
    for source in (old_source, new_source):
        x, columns = source.detach().clone(), V.detach().clone()
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
                derivative, primal = products.A(columns), products.A.last_primal
            columns, x = columns + step * derivative, x + step * primal
        trajectories.append(states)
        forward_results.append(columns)
    endpoint = x.detach()
    delta_v = forward_results[1] - forward_results[0]
    u_new, s_hat = torch.linalg.qr(U @ S + delta_v, mode="reduced")
    s_tilde = s_hat - u_new.T @ delta_v
    reverse_results: list[torch.Tensor] = []
    for states in trajectories:
        columns = u_new
        for index in reversed(range(n_steps)):
            state, time = states[index], index * step
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
        reverse_results.append(columns)
    delta_transpose_u = reverse_results[1] - reverse_results[0]
    v_new, r_l = torch.linalg.qr(
        V @ s_tilde.T + delta_transpose_u, mode="reduced")
    return endpoint, (u_new.detach(), r_l.T.detach(), v_new.detach())
