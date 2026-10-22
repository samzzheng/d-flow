"""Experimental exact-suffix and goal-oriented terminal projection controls."""
from __future__ import annotations

from typing import Any

import torch

from dlr_dflow.api import sensitivity_factors
from dlr_dflow.implementation.types import Factors
from experiments.ct_mri_sweep import run


class _PrefixReady(Exception):
    """End the observer-driven experiment at the requested prefix grid point."""


def prefix_factors(
    model: torch.nn.Module, source: torch.Tensor, steps: int, prefix_steps: int,
    options: dict[str, Any], seed: int,
    products: Any = None, seed_basis: torch.Tensor | None = None,
) -> Factors:
    """Stop existing factor evolution on its original grid through an observer."""
    if not 0 < prefix_steps < steps:
        raise ValueError("The control requires a nonempty, nonterminal DLR prefix")
    captured: Factors | None = None
    baseline = 1.-prefix_steps/steps

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
    return captured


def prefix_image(model: torch.nn.Module, source: torch.Tensor, steps: int,
                 prefix_steps: int) -> torch.Tensor:
    """Recompute only native primal prefix states; preserve the exact Euler grid."""
    h = 1.0/steps
    with torch.no_grad():
        image = source.detach()
        for index in range(prefix_steps):
            image = image + h*model(image, image.new_full((image.shape[0],), index*h))
    return image


def suffix_gradient(
    model: torch.nn.Module, source: torch.Tensor, terminal: torch.Tensor,
    steps: int, suffix_steps: int, options: dict[str, Any], seed: int,
    products: Any = None, seed_basis: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the shifted-low-rank prefix Jacobian to the exact suffix pullback."""
    prefix_steps, h = steps-suffix_steps, 1.0/steps
    U, S, V = prefix_factors(model, source, steps, prefix_steps, options, seed, products, seed_basis)
    image = prefix_image(model, source, steps, prefix_steps)
    with torch.enable_grad():
        variable = image.detach().requires_grad_(True)
        image = variable
        for index in range(prefix_steps, steps):
            image = image + h*model(image, image.new_full((image.shape[0],), index*h))
        pullback, = torch.autograd.grad(image, variable, terminal.detach())
    vector = pullback.detach().reshape(-1, 1)
    gradient = (1.-prefix_steps*h)*vector + V @ (S.T @ (U.T @ vector))
    return gradient.reshape_as(source), V


@torch.no_grad()
def task_terminal_factors(
    factors: Factors, terminal: torch.Tensor, h: float, products: Any,
    alpha: float | None = None,
) -> Factors:
    """Rank-preserving Euler residual projection exact on the supplied covector.

    Let T=(I+h A)Y+h(alpha A+I). The default alpha=h is the terminal step.
    Choose an orthonormal Q containing gx, then QR(T.T Q)=V R. The matrix
    Q R.T V.T has rank at most r and its transpose action on gx equals
    T.T gx. It changes the final projection criterion, not the prefix KSL.
    """
    alpha = h if alpha is None else alpha
    U, S, V = factors
    US = U @ S
    left, _ = torch.linalg.qr(US + h*(products.A(alpha*V + US)+V))
    direction = terminal.reshape(-1, 1)
    direction = direction/direction.norm().clamp_min(1e-30)
    left, _ = torch.linalg.qr(torch.cat((direction, left[:, :-1]), dim=1))
    transpose = products.AT(left)
    image = V @ (S.T @ (U.T @ (left+h*transpose))) + h*(alpha*transpose+left)
    right, triangular = torch.linalg.qr(image)
    return left, triangular.T, right


def task_gradient(
    model: torch.nn.Module, source: torch.Tensor, terminal: torch.Tensor,
    steps: int, goal_steps: int, options: dict[str, Any], seed: int,
    products: Any, seed_basis: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve the transported loss covector in the final goal_steps projections.

    Transport needs goal_steps-1 extra native single-vector adjoints. Every
    Euler factor update keeps rank r and alpha=1-t; the native flow is unchanged.
    In real arithmetic the resulting gradient equals the corresponding exact-
    suffix hybrid, while the returned right basis is the terminal task basis.
    """
    cut, h = steps-goal_steps, 1./steps
    factors = prefix_factors(model, source, steps, cut, options, seed, products, seed_basis)
    points = []
    with torch.no_grad():
        point = prefix_image(model, source, steps, cut)
        for index in range(cut, steps):
            points.append(point)
            if index+1 < steps:
                point = point + h*model(point, point.new_full((point.shape[0],), index*h))
    goals = {steps: terminal.detach()}
    for index in range(steps-1, cut, -1):
        with torch.enable_grad():
            variable = points[index-cut].detach().requires_grad_(True)
            image = variable + h*model(variable, variable.new_full((variable.shape[0],), index*h))
            pullback, = torch.autograd.grad(image, variable, goals[index+1])
        goals[index] = pullback.detach()
    for index, point in enumerate(points, start=cut):
        products.set_state(index*h, point)
        factors = task_terminal_factors(factors, goals[index+1], h, products, alpha=1.-index*h)
    U, S, V = factors
    gradient = (V @ (S.T @ (U.T @ terminal.reshape(-1, 1)))).reshape_as(source)
    return gradient.detach(), V


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


class TaskTerminalEvaluator(run.Evaluator):
    """Keep terminal rank r while making its projection specific to this loss."""

    goal_steps: int = 1

    def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
        if method != "DLR D-Flow":
            return super().gradient(source, method)
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        _, terminal = run.endpoint_gradient(self.image, self.operator, self.observed)
        gradient, V = task_gradient(self.model, source, terminal, self.settings["flow_steps"],
            self.goal_steps, self.settings["dlr"], self.seed, self.products, self.previous_v)
        if self.settings.get("dlr_reuse_previous_v", False):
            self.previous_v = V.detach().clone()
        self.counts["gradients"] += 1
        self.counts["native_goal_vjps"] = self.counts.get("native_goal_vjps", 0) + self.goal_steps-1
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite task-projected source gradient")
        return gradient.detach()
