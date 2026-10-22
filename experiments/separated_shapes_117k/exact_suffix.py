"""Experimental DLR-prefix/exact-adjoint-suffix gradient, not terminal rank-r DLR."""
from __future__ import annotations

from typing import Any

import torch

from dlr_dflow.api import sensitivity_factors
from dlr_dflow.implementation.types import Factors
from experiments.ct_mri_sweep import run


class _PrefixReady(Exception):
    """End the observer-driven experiment at the requested prefix grid point."""


def suffix_gradient(
    model: torch.nn.Module, source: torch.Tensor, terminal: torch.Tensor,
    steps: int, suffix_steps: int, options: dict[str, Any], seed: int,
    products: Any = None, seed_basis: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return J_prefix_hat.T times the exact remaining Euler-map pullback.

    The callback exits the existing factor evolution without changing its time
    grid or public API. Earlier observer canonicalizations do not alter its
    internal factors. The native prefix is recomputed to build only the short
    suffix graph. This prototype prioritizes verification over avoiding these
    additional prefix forward evaluations.
    """
    prefix_steps = steps-suffix_steps
    h = 1.0/steps
    if not 0 < prefix_steps < steps:
        raise ValueError("The hybrid requires both a nonempty DLR prefix and exact suffix")
    captured: Factors | None = None
    baseline = 1.-prefix_steps*h

    def capture(step: int, alpha: float, factors: Factors) -> None:
        nonlocal captured
        if step == prefix_steps:
            if abs(alpha-baseline) > 1e-6:
                raise AssertionError("Prefix alpha differs from 1-t")
            captured = tuple(value.detach().clone() for value in factors)
            raise _PrefixReady

    try:
        sensitivity_factors(model, source.detach(), steps, **{**options, "factor_callback": capture},
                            seed=seed, products=products, seed_basis=seed_basis)
    except _PrefixReady:
        pass
    if captured is None:
        raise RuntimeError("Factor evolution did not reach the requested prefix")
    with torch.no_grad():
        image = source.detach()
        for index in range(prefix_steps):
            image = image + h*model(image, image.new_full((image.shape[0],), index*h))
    with torch.enable_grad():
        variable = image.detach().requires_grad_(True)
        image = variable
        for index in range(prefix_steps, steps):
            image = image + h*model(image, image.new_full((image.shape[0],), index*h))
        pullback, = torch.autograd.grad(image, variable, terminal.detach())
    U, S, V = captured
    vector = pullback.detach().reshape(-1, 1)
    gradient = baseline*vector + V @ (S.T @ (U.T @ vector))
    return gradient.reshape_as(source), V


class ExactSuffixEvaluator(run.Evaluator):
    """Use ordinary objective/optimizer settings; explicitly label this hybrid."""

    suffix_steps: int = 1

    def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
        if method != "DLR D-Flow":
            return super().gradient(source, method)
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        _, terminal = run.endpoint_gradient(self.image, self.operator, self.observed)
        gradient, basis = suffix_gradient(self.model, source, terminal, self.settings["flow_steps"],
            self.suffix_steps, self.settings["dlr"], self.seed, self.products, self.previous_v)
        if self.settings.get("dlr_reuse_previous_v", False):
            self.previous_v = basis.detach().clone()
        self.counts["gradients"] += 1
        self.counts["native_suffix_vjps"] = self.counts.get("native_suffix_vjps", 0) + self.suffix_steps
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite hybrid source gradient")
        return gradient.detach()
