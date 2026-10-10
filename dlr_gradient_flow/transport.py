"""Matrix-free sensitivity transport for the discrete Euler flow.

For a source displacement ``delta``, ``W = D_z J(z)[delta]`` is the
first-order change in the terminal Jacobian.  This module computes its
actions and applies a K/S/L projector-splitting update in optimization time.
The analytical baseline is zero at terminal flow time, so the transported
factors represent ``J`` itself; no identity term is added.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from dlr_dflow.implementation.types import Factors, TensorOperator

TransportBackend = Literal["exact", "finite_difference"]
AugmentedState = tuple[torch.Tensor, torch.Tensor]


def variation_products(
    model: torch.nn.Module,
    source: torch.Tensor,
    displacement: torch.Tensor,
    n_steps: int = 100,
    *,
    backend: TransportBackend = "exact",
    difference_step: float = 1e-3,
    lane_chunk_size: int = 4,
) -> tuple[TensorOperator, TensorOperator]:
    """Return actions of ``W`` and ``W.T`` on matrices with columns as lanes.

    ``exact`` differentiates the actual Euler recurrence, including its
    second derivatives.  Only the primal and directional state at each
    time step are stored; local derivative tapes are recomputed and freed.
    ``finite_difference`` uses the explicitly approximate centered quotient
    ``(J(z+epsilon*delta)-J(z-epsilon*delta))/(2*epsilon)``.  It is mainly a
    second-derivative-free reference; float32 cancellation limits epsilon.

    The model must be deterministic (evaluation mode) and use differentiable
    native operations.  Detached hand-written first derivatives must not be
    substituted into the exact backend.
    """
    step = 1.0 / n_steps
    shape = source.shape
    dimension = source.numel()
    source = source.detach()
    displacement = displacement.detach()

    def velocity(index: int) -> Callable[[torch.Tensor], torch.Tensor]:
        time = source.new_full((source.shape[0],), index * step)

        def evaluate(x: torch.Tensor) -> torch.Tensor:
            return model(x, time)

        return evaluate

    if backend == "exact":
        states: list[AugmentedState] = []
        x, p = source, displacement
        with torch.no_grad():
            for index in range(n_steps):
                states.append((x, p))
                value, tangent = torch.func.jvp(velocity(index), (x,), (p,))
                x, p = (x + step * value).detach(), (p + step * tangent).detach()

        def augmented_velocity(index: int) -> Callable[..., AugmentedState]:
            evaluate = velocity(index)

            def evaluate_pair(x: torch.Tensor, p: torch.Tensor) -> AugmentedState:
                return torch.func.jvp(evaluate, (x,), (p,))

            return evaluate_pair

        def action(columns: torch.Tensor) -> torch.Tensor:
            q = columns.T.reshape((-1,) + shape).detach()
            w = torch.zeros_like(q)
            with torch.no_grad():
                for index, state in enumerate(states):
                    _, pullback = torch.func.vjp(augmented_velocity(index), *state)
                    # Transposing this linear pullback gives the augmented
                    # JVP without double-forward AD (unsupported by native
                    # SiLU).  This still differentiates the native model.
                    _, pushforward = torch.func.vjp(
                        pullback, tuple(torch.zeros_like(value) for value in state),
                    )

                    def tangent(q_lane: torch.Tensor, w_lane: torch.Tensor) -> AugmentedState:
                        return pushforward((q_lane, w_lane))[0]

                    dq, dw = torch.func.vmap(tangent, chunk_size=lane_chunk_size)(q, w)
                    q, w = (q + step * dq).detach(), (w + step * dw).detach()
                    del pushforward, pullback
            return w.reshape(-1, dimension).T.contiguous()

        def transpose_action(columns: torch.Tensor) -> torch.Tensor:
            p_adjoint = columns.T.reshape((-1,) + shape).detach()
            x_adjoint = torch.zeros_like(p_adjoint)
            for index in reversed(range(n_steps)):
                _, pullback = torch.func.vjp(augmented_velocity(index), *states[index])

                def transpose(x_lane: torch.Tensor, p_lane: torch.Tensor) -> AugmentedState:
                    return pullback((x_lane, p_lane))

                dx, dp = torch.func.vmap(transpose, chunk_size=lane_chunk_size)(x_adjoint, p_adjoint)
                x_adjoint = (x_adjoint + step * dx).detach()
                p_adjoint = (p_adjoint + step * dp).detach()
                del pullback
            return x_adjoint.reshape(-1, dimension).T.contiguous()

        return action, transpose_action

    if backend != "finite_difference":
        raise ValueError(f"Unknown sensitivity-transport backend: {backend}")
    if difference_step <= 0:
        raise ValueError("difference_step must be positive")
    trajectories: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for sign in (1, -1):
            x = source + sign * difference_step * displacement
            trajectory = []
            for index in range(n_steps):
                trajectory.append(x.detach())
                x = x + step * velocity(index)(x)
            trajectories.append(trajectory)

    def action(columns: torch.Tensor) -> torch.Tensor:
        results = []
        with torch.no_grad():
            for trajectory in trajectories:
                q = columns.T.reshape((-1,) + shape).detach()
                for index, x in enumerate(trajectory):
                    evaluate = velocity(index)

                    def tangent(lane: torch.Tensor) -> torch.Tensor:
                        return torch.func.jvp(evaluate, (x,), (lane,))[1]

                    dq = torch.func.vmap(tangent, chunk_size=lane_chunk_size)(q)
                    q = (q + step * dq).detach()
                results.append(q)
        return ((results[0] - results[1]) / (2 * difference_step)).reshape(-1, dimension).T.contiguous()

    def transpose_action(columns: torch.Tensor) -> torch.Tensor:
        results = []
        for trajectory in trajectories:
            q = columns.T.reshape((-1,) + shape).detach()
            for index in reversed(range(n_steps)):
                _, pullback = torch.func.vjp(velocity(index), trajectory[index])

                def transpose(lane: torch.Tensor) -> torch.Tensor:
                    return pullback(lane)[0]

                dq = torch.func.vmap(transpose, chunk_size=lane_chunk_size)(q)
                q = (q + step * dq).detach()
                del pullback
            results.append(q)
        return ((results[0] - results[1]) / (2 * difference_step)).reshape(-1, dimension).T.contiguous()

    return action, transpose_action


def transport_factors(
    model: torch.nn.Module,
    source: torch.Tensor,
    displacement: torch.Tensor,
    factors: Factors,
    n_steps: int = 100,
    *,
    backend: TransportBackend = "exact",
    difference_step: float = 1e-3,
    lane_chunk_size: int = 4,
) -> Factors:
    """Transport terminal factors through one frozen-variation K/S/L step.

    The optimization-time step is already contained in ``displacement``.
    This approximates the Jacobian at ``source + displacement`` to first
    order, in addition to existing low-rank and tangent-projection error.
    It does not differentiate or rerun the original DLR factor solver.
    """
    action, transpose_action = variation_products(
        model, source, displacement, n_steps,
        backend=backend, difference_step=difference_step,
        lane_chunk_size=lane_chunk_size,
    )
    U, S, V = factors
    WV = action(V)
    U_next, S_hat = torch.linalg.qr(U @ S + WV, mode="reduced")
    S_tilde = S_hat - U_next.T @ WV
    V_next, S_transpose = torch.linalg.qr(
        V @ S_tilde.T + transpose_action(U_next), mode="reduced",
    )
    return U_next.detach(), S_transpose.T.detach(), V_next.detach()
