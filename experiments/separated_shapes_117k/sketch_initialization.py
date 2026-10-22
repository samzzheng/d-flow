"""Experimental first-residual initialization with an explicitly retained seed."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow.implementation.enriched import _canonicalize_factors, _rademacher, _thin_svd
from dlr_dflow.implementation.types import Factors


@torch.no_grad()
def retained_seed_startup(
    model: torch.nn.Module,
    z: torch.Tensor,
    n_steps: int,
    rank: int,
    oversampling: int,
    minimum_steps: int,
    generator: torch.Generator,
    *,
    checkpoint_blocks: bool = False,
    products: Any = None,
    seed_basis: torch.Tensor | None = None,
    factor_callback: Callable[[int, float, Factors], None] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    retain_seed: bool = True,
) -> tuple[torch.Tensor, Factors, int]:
    """Use the ordinary Rademacher startup, optionally retaining its first seed.

    Let B=h(A0+I). The control returns the existing sketch factors. In the
    retained-seed variant, w is the supplied unit seed and r-1 complementary
    directions approximate the old factorization on w-perp. Recompute B V
    and QR it, giving Y=B V V.T with w in span(V). This changes initialization
    only and needs r additional JVP columns. Later KSL steps do not constrain V.
    """
    if minimum_steps != 1:
        raise ValueError("The retained-seed control only replaces the first residual step")
    h, dimension = 1.0/n_steps, z.numel()
    alpha = 1.0-h
    if products is None:
        A, AT = core.make_A_products(model, 0., z, checkpoint_blocks=checkpoint_blocks)
    else:
        products.set_state(0., z)
        A, AT = products.A, products.AT
    width = min(dimension, rank+oversampling)
    sketch = _rademacher(dimension, width, generator, z.device, z.dtype)
    if seed_basis is not None and seed_basis.numel():
        seeds = seed_basis.detach().reshape(dimension, -1).to(z)
        seeds = seeds/seeds.norm(dim=0, keepdim=True).clamp_min(1e-12)*dimension**.5
        count = min(seeds.shape[1], width)
        sketch = torch.cat([seeds[:, :count], sketch[:, count:]], dim=1)
    image = h*A(sketch)+(1.0-alpha)*sketch
    range_basis, _ = torch.linalg.qr(image, mode="reduced")
    transpose_image = h*AT(range_basis)+(1.0-alpha)*range_basis
    right, values, coordinates_t = _thin_svd(transpose_image)
    factors = (range_basis @ coordinates_t.T[:, :rank], torch.diag(values[:rank]), right[:, :rank])
    if retain_seed:
        if seed_basis is None or not seed_basis.numel():
            raise ValueError("Retaining a direction requires a supplied sketch seed")
        anchor = seed_basis[:, :1].detach().to(z)
        norm = anchor.norm()
        if not torch.isfinite(norm) or norm == 0:
            raise FloatingPointError("The retained sketch seed must be finite and nonzero")
        anchor = anchor/norm
        _, core_matrix, right = factors
        complement = (right-anchor @ (anchor.T @ right)) @ core_matrix.T
        complement_basis, _, _ = _thin_svd(complement)
        right, _ = torch.linalg.qr(torch.cat([anchor, complement_basis[:, :rank-1]], dim=1))
        left, core_matrix = torch.linalg.qr(h*A(right)+(1.0-alpha)*right)
        factors = _canonicalize_factors((left, core_matrix, right))
    primal = getattr(A, "last_primal", None)
    if primal is None:
        primal = core.velocity(model, 0., z)
    endpoint = z.detach()+h*primal.reshape_as(z)
    if factor_callback is not None:
        factor_callback(1, alpha, factors)
    if diagnostics is not None:
        diagnostics.append(dict(t=h, alpha=alpha,
            residual_singular_values=factors[1].diag().cpu().tolist(), retained_sketch_seed=retain_seed))
    return endpoint, factors, 1
