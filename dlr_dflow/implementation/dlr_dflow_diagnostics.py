"""Diagnostics for DLR-D-Flow Jacobian products and gradient actions.

These routines intentionally live outside the solver module. They compare the
low-rank action with autograd or print numerical checks, but are not needed by
the optimization path.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from sampler.ode_samplers import euler_flow

from .. import dlr_dflow as core


ActionError = tuple[float, float, float, list[tuple[int, float, float]], float]


def exact_source_gradient(
    model: torch.nn.Module,
    z: torch.Tensor,
    n_steps: int,
    g_x: torch.Tensor,
) -> torch.Tensor:
    """Return ``J_1.T @ g_x`` through the discrete Euler flow."""
    source = z.detach().requires_grad_(True)
    terminal = euler_flow(model, source, n_steps)
    (gradient,) = torch.autograd.grad((terminal * g_x.detach()).sum(), source)
    return gradient


def selftest_jacobian_products(model: torch.nn.Module, device: str) -> None:
    """Compare product helpers with autograd and check the transpose identity."""
    height = width = 64
    x = torch.randn(1, 1, height, width, device=device)
    products, transpose_products = core.make_A_products(model, 0.37, x)
    q = torch.randn(height * width, 1, device=device)
    w = torch.randn(height * width, 1, device=device)

    _, reference = torch.autograd.functional.jvp(
        lambda value: core.velocity(
            model, 0.37, value.reshape(1, 1, height, width)).reshape(-1), x.reshape(-1), q.reshape(-1))
    relative_error = (products(q).reshape(-1) - reference).norm() / reference.norm()
    inner = (products(q) * w).sum().item()
    transpose_inner = (q * transpose_products(w)).sum().item()
    print(f"[selftest] JVP vs autograd rel.err = {relative_error.item():.2e}")
    print(
        f"[selftest] <Aq,w> = {inner:.6f}  vs  <q,A^T w> = {transpose_inner:.6f} "
        f"(rel {abs(inner - transpose_inner) / (abs(inner) + 1e-12):.2e})")


def action_error(
    model: torch.nn.Module,
    z: torch.Tensor,
    n_steps: int,
    ranks: Sequence[int],
    forward_operator: Callable[[torch.Tensor], torch.Tensor],
    observation: torch.Tensor,
    seed: int = 0,
) -> ActionError:
    """Measure low-rank source-gradient action error over several ranks."""
    terminal = euler_flow(model, z, n_steps)
    terminal_gradient = core.terminal_gradient(terminal, forward_operator, observation)
    exact = exact_source_gradient(model, z, n_steps, terminal_gradient)
    denominator = exact.norm().item()
    rows: list[tuple[int, float, float]] = []
    alpha = 1.0
    for rank in ranks:
        _, factors, alpha = core.sensitivity_factors(
            model, z, n_steps, rank=rank, seed=seed)
        estimate = core.approx_source_gradient(factors, terminal_gradient, alpha)
        relative_error = (exact - estimate).norm().item() / denominator
        cosine = torch.nn.functional.cosine_similarity(
            exact.reshape(1, -1), estimate.reshape(1, -1)).item()
        rows.append((rank, relative_error, cosine))

    identity_error = (exact - terminal_gradient).norm().item() / denominator
    scalar_error = (exact - alpha * terminal_gradient).norm().item() / denominator
    return identity_error, scalar_error, alpha, rows, denominator
