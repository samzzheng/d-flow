"""Experimental Gaussian residual sketches of an exact Euler prefix."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .. import dlr_dflow as core
from .enriched import _thin_svd
from .types import Factors


@torch.no_grad()
def residual_sketch_startup(
    model: torch.nn.Module, z: torch.Tensor, n_steps: int, rank: int,
    oversampling: int, minimum_steps: int, generator: torch.Generator,
    *, checkpoint_blocks: bool = False, products: Any = None,
    seed_basis: torch.Tensor | None = None,
    factor_callback: Callable[[int, float, Factors], None] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, Factors, int]:
    """Sketch J_m - (1-mh)I without subtracting identity-sized actions.

    QR plus a small SVD resolves the sketch range. The roundoff cutoff is a
    heuristic based on dtype, matrix size and accumulated product magnitudes,
    not a certified error bound for a learned-model JVP. A zero sketch extends
    the same prefix and probes, up to the original terminal grid point.
    """
    h, dimension = 1.0 / n_steps, z.numel()
    width = min(dimension, rank + oversampling)
    omega = torch.randn(dimension, width, generator=generator, device=z.device, dtype=z.dtype)
    if seed_basis is not None and seed_basis.numel():
        count = min(seed_basis.shape[1], width)
        seeds = seed_basis[:, :count].detach().to(z)
        omega[:, :count] = seeds / seeds.norm(dim=0).clamp_min(1e-12) * dimension ** .5
    residual = torch.zeros_like(omega)
    omega_norm = omega.norm()
    roundoff_scale = z.new_zeros(())
    eps = torch.finfo(z.dtype).eps
    points = []
    x = z.detach()
    empty = z.new_empty(dimension, 0)
    factors = empty, z.new_empty(0, 0), empty
    for index in range(n_steps):
        points.append(x)
        if products is None:
            a, _ = core.make_A_products(model, index * h, x, checkpoint_blocks=checkpoint_blocks)
        else:
            products.set_state(index * h, x)
            a = products.A
        action = a(residual + (1.0 - index * h) * omega)
        roundoff_scale = roundoff_scale + residual.norm() + h * (action.norm() + omega_norm)
        residual = residual + h * (action + omega)
        primal = getattr(a, "last_primal", None)
        if primal is None:
            primal = core.velocity(model, index * h, x)
        x = x + h * primal.reshape_as(x)
        completed = index + 1
        if completed < minimum_steps and factor_callback is None and diagnostics is None:
            continue
        if not torch.isfinite(residual).all():
            raise FloatingPointError("Nonfinite Euler-prefix residual sketch")
        left, values, _ = _thin_svd(residual)
        cutoff = max(float(32 * eps * roundoff_scale),
                     float(values[0]) * max(residual.shape) * eps)
        resolved = int((values > cutoff).sum())
        factors = empty, z.new_empty(0, 0), empty
        if resolved:
            left = left[:, :resolved]
            transpose_residual = torch.zeros_like(left)
            transpose_scale = z.new_zeros(())
            # Reverse the Jacobian order: M_0^T ... M_{m-1}^T Q.
            for reverse_index, point_index in enumerate(range(completed - 1, -1, -1)):
                if products is None:
                    _, at = core.make_A_products(
                        model, point_index * h, points[point_index],
                        checkpoint_blocks=checkpoint_blocks)
                else:
                    products.set_state(point_index * h, points[point_index])
                    at = products.AT
                action = at(transpose_residual + (1.0 - reverse_index * h) * left)
                transpose_scale = transpose_scale + transpose_residual.norm() + h * (action.norm() + left.norm())
                transpose_residual = transpose_residual + h * (action + left)
            right, values, left_coordinates_t = _thin_svd(transpose_residual)
            cutoff = max(float(32 * eps * transpose_scale),
                         float(values[0]) * max(transpose_residual.shape) * eps)
            retained = min(rank, int((values > cutoff).sum()))
            factors = (left @ left_coordinates_t.T[:, :retained],
                       torch.diag(values[:retained]), right[:, :retained])
        if factor_callback is not None:
            factor_callback(completed, 1.0 - completed * h, factors)
        if diagnostics is not None:
            diagnostics.append(dict(t=completed * h, alpha=1.0 - completed * h,
                                    residual_singular_values=factors[1].diag().cpu().tolist(),
                                    exact_prefix_startup=True))
        if completed >= minimum_steps and factors[1].numel():
            break
    return x, factors, completed
