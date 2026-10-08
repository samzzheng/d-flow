"""Experimental curvature filtering and geometric Armijo; production stays unchanged."""
from __future__ import annotations

import math
from typing import Any

import torch

from p_flow.armijo import ArmijoLBFGS, Closure, Objective
from p_flow.p_flow import project_source

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {"history_size": 100, "curvature_cosine": 0., "geometric": False},
    "short": {"history_size": 10, "curvature_cosine": 0., "geometric": False},
    "filtered": {"history_size": 10, "curvature_cosine": .01, "geometric": False},
    "geometric": {"history_size": 100, "curvature_cosine": 0., "geometric": True},
    "combined": {"history_size": 10, "curvature_cosine": .01, "geometric": True},
}


class ExperimentalArmijo(ArmijoLBFGS):
    """Mirror the existing recurrence so ablations change only their named policy.

    The relative filter requires s.T@y > .01*||s||*||y|| in addition to the
    original absolute threshold. It does not certify an accurate Hessian pair.
    Geometric backtracking halves the rate, retaining the same 30-trial budget.
    """

    def __init__(self, source: torch.Tensor, objective: Objective, variant: str,
                 lr: float = 1., max_ls: int = 30, c1: float = 1e-4,
                 step_cap: float = .64, project: bool = False) -> None:
        settings = VARIANTS[variant]
        super().__init__(source, objective, lr, settings["history_size"], max_ls, c1, step_cap, project)
        self.geometric = settings["geometric"]
        self.curvature_cosine = settings["curvature_cosine"]
        self.last_diagnostics: dict[str, Any] = {}

    def step(self, closure: Closure) -> float:
        if not self.state:
            self.S, self.Y, self.f, self.g = [], [], None, None
            self.consecutive_failures = 0
        if self.f is None:
            self.f = float(closure())
            self.g = self.source.grad.detach().flatten().clone()
            self.state["initialized"] = True
        f, g = self.f, self.g
        self.last_diagnostics = {"history_before": len(self.S), "trials": [], "stop_reason": "trial_budget"}
        direction = -g.clone()
        coefficients = []
        for s, y in reversed(list(zip(self.S, self.Y))):
            rho = 1. / y.dot(s)
            coefficient = rho * s.dot(direction)
            direction -= coefficient * y
            coefficients.append((rho, coefficient))
        if self.S:
            direction *= self.S[-1].dot(self.Y[-1]) / self.Y[-1].dot(self.Y[-1])
        for (s, y), (rho, coefficient) in zip(zip(self.S, self.Y), reversed(coefficients)):
            direction += s * (coefficient - rho * y.dot(direction))
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
        self.last_diagnostics.update(initial_rate=rate, slope=slope, gradient_norm=float(g.norm()),
                                     direction_norm=float(direction.norm()), first_attempt=self.step_attempts == 0)
        self.step_attempts += 1
        self.last_function_evaluations = 0
        accepted = False
        for _ in range(self.max_ls):
            if not valid or not math.isfinite(rate) or rate <= 0:
                self.last_diagnostics["stop_reason"] = "invalid_direction_or_rate"
                break
            candidate = (z + rate * direction).reshape_as(self.source)
            if self.project:
                candidate = project_source(candidate)
            if torch.equal(candidate.flatten(), z):
                self.last_diagnostics["stop_reason"] = "unchanged_float32_source"
                break
            value = self.objective(candidate)
            self.last_diagnostics["trials"].append({"rate": rate, "phi": value if math.isfinite(value) else None,
                "displacement": float((candidate.flatten() - z).norm())})
            self.last_function_evaluations += 1
            if math.isfinite(value) and value < f and value - f <= self.c1 * rate * slope:
                accepted = True
                self.last_diagnostics["stop_reason"] = "accepted"
                break
            if self.geometric:
                rate *= .5
                continue
            denominator = 2 * (value - f - slope * rate)
            interpolated = -slope * rate * rate / denominator if math.isfinite(value) and denominator > 0 else .1 * rate
            rate = min(max(interpolated, .1 * rate), .5 * rate)
        self.last_diagnostics.update(accepted=accepted, accepted_rate=rate if accepted else None)
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
        curvature = float(y.dot(s))
        norm_product = float(y.norm()) * float(s.norm())
        keep = curvature > max(1e-10, self.curvature_cosine * norm_product)
        self.last_diagnostics.update(curvature=curvature,
            curvature_cosine=curvature / norm_product if norm_product else None, pair_kept=keep,
            filter_rejected=curvature > 1e-10 and not keep)
        if keep:
            self.S.append(s)
            self.Y.append(y)
            if len(self.S) > self.history_size:
                self.S.pop(0)
                self.Y.pop(0)
        self.f, self.g = new_value, new_gradient
        self.consecutive_failures = 0
        return new_value
