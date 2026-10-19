"""Isolated projected LBFGS adapter with the project's Armijo policies.

Based on ArmijoLBFGS in scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py.
Importing that script currently requires the removed parallel_bug module, so
this experiment keeps a standalone adapter and leaves production scripts alone.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch

from .p_flow import project_source

Closure = Callable[[], torch.Tensor]
Objective = Callable[[torch.Tensor], float]


def lbfgs_direction(gradient: torch.Tensor, displacements: list[torch.Tensor],
                    changes: list[torch.Tensor]) -> torch.Tensor:
    """Literal two-loop recursion shared by eager and captured execution."""
    direction = -gradient.clone()
    coefficients = []
    for s, y in reversed(list(zip(displacements, changes))):
        rho = 1. / y.dot(s)
        coefficient = rho * s.dot(direction)
        direction -= coefficient * y
        coefficients.append((rho, coefficient))
    if displacements:
        direction *= displacements[-1].dot(changes[-1]) / changes[-1].dot(changes[-1])
    for (s, y), (rho, coefficient) in zip(zip(displacements, changes), reversed(coefficients)):
        direction += s * (coefficient - rho * y.dot(direction))
    return direction


class ArmijoLBFGS:
    """Ambient LBFGS directions with optional sphere projection and line search.

    With projection enabled, project directions onto the current sphere's
    tangent plane, evaluate retracted trials, and store actual displacements
    in curvature pairs. This is a projected LBFGS variant, not a claim of a
    full Riemannian BFGS implementation. Approximate gradients determine the
    predicted slope; every accepted step must lower the true objective.
    """

    initial_step_policy = "first_iteration_only"
    progress_policy = "strict_decrease_nonzero_step"

    def __init__(self, source: torch.Tensor, objective: Objective, lr: float = 1.,
                 history_size: int = 100, max_ls: int = 30, c1: float = 1e-4,
                 step_cap: float = 0., project: bool = True,
                 backtracking: Literal["geometric", "interpolation"] = "geometric") -> None:
        if backtracking not in {"geometric", "interpolation"}:
            raise ValueError(f"Unknown Armijo backtracking policy: {backtracking}")
        self.source, self.objective = source, objective
        self.lr, self.history_size, self.max_ls, self.c1 = lr, history_size, max_ls, c1
        self.step_cap, self.project = step_cap, project
        self.backtracking = backtracking
        self.S: list[torch.Tensor] = []
        self.Y: list[torch.Tensor] = []
        self.state: dict[str, bool] = {}
        self.f: float | None = None
        self.g: torch.Tensor | None = None
        self.step_attempts = 0
        self.consecutive_failures = 0
        self.failed_searches = 0
        self.last_function_evaluations = 0
        self.capped_steps = 0

    def _direction(self) -> torch.Tensor:
        return lbfgs_direction(self.g, self.S, self.Y)

    def step(self, closure: Closure) -> float:
        if not self.state:
            self.S, self.Y, self.f, self.g = [], [], None, None
            self.consecutive_failures = 0
        if self.f is None:
            self.f = float(closure())
            self.g = self.source.grad.detach().flatten().clone()
            self.state["initialized"] = True
        f, g = self.f, self.g
        direction = self._direction()
        z = self.source.detach().flatten().clone()
        if self.project:
            direction -= z * z.dot(direction) / z.dot(z)
        slope = float(g.dot(direction))
        if not math.isfinite(slope) or slope >= 0:
            self.S, self.Y = [], []
            direction = -g.clone()
            if self.project:
                direction -= z * z.dot(direction) / z.dot(z)
            slope = float(g.dot(direction))
        valid = math.isfinite(f) and math.isfinite(slope) and slope < 0
        rate = 0.
        if valid:
            rate = self.lr * (min(1., 1. / float(g.abs().sum())) if self.step_attempts == 0 else 1.)
            if self.step_cap > 0 and rate * float(direction.norm()) > self.step_cap:
                rate = self.step_cap / float(direction.norm())
                self.capped_steps += 1
        self.step_attempts += 1
        self.last_function_evaluations = 0
        accepted = False
        for _ in range(self.max_ls):
            if not valid or not math.isfinite(rate) or rate <= 0:
                break
            candidate = (z + rate * direction).reshape_as(self.source)
            if self.project:
                candidate = project_source(candidate)
            if torch.equal(candidate.flatten(), z):
                break
            value = self.objective(candidate)
            self.last_function_evaluations += 1
            if math.isfinite(value) and value < f and value - f <= self.c1 * rate * slope:
                accepted = True
                break
            if self.backtracking == "geometric":
                rate *= .5
            else:
                denominator = 2 * (value - f - slope * rate)
                interpolated = -slope * rate * rate / denominator if math.isfinite(value) and denominator > 0 else .1 * rate
                rate = min(max(interpolated, .1 * rate), .5 * rate)
        if not accepted:
            self.S, self.Y = [], []
            self.failed_searches += 1
            self.consecutive_failures += 1
            return f
        with torch.no_grad():
            self.source.copy_(candidate)
        new_value = float(closure())
        new_gradient = self.source.grad.detach().flatten().clone()
        s, y = candidate.flatten() - z, new_gradient - g
        if float(y.dot(s)) > 1e-10:
            self.S.append(s)
            self.Y.append(y)
            if len(self.S) > self.history_size:
                self.S.pop(0)
                self.Y.pop(0)
        self.f, self.g = new_value, new_gradient
        self.consecutive_failures = 0
        return new_value
