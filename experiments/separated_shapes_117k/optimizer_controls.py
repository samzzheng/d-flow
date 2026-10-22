"""Isolated recovery control for approximate-gradient LBFGS histories."""
from typing import Any

import torch

from experiments.ct_mri_sweep.optimizer import SweepLBFGS
from p_flow.armijo import Closure, Objective


class HistoryFallbackLBFGS(SweepLBFGS):
    """On rejection, discard the newer half before abandoning all curvature."""

    def __init__(self, source: torch.Tensor, objective: Objective, **kwargs: Any) -> None:
        super().__init__(source, objective, **kwargs)
        self.history_backtracks = 0

    def step(self, closure: Closure) -> float:
        old_s, old_y = self.S.copy(), self.Y.copy()
        value = super().step(closure)
        if self.consecutive_failures and len(old_s) > 1:
            retained = len(old_s) // 2
            self.S, self.Y = old_s[:retained], old_y[:retained]
            self.history_backtracks += 1
            # This prepares a different direction for the next normal attempt.
            # Total failed searches and step attempts still count every retry.
            self.consecutive_failures = 0
        return value


class ScaleRecoveryLBFGS(SweepLBFGS):
    """After a rejected search, test the cached fallback gradient at the cap.

    Ordinary attempts keep the established first-attempt scaling. Recovery
    changes only the starting length of the existing history-free direction;
    geometric Armijo, strict decrease, the displacement cap and failure/attempt
    accounting remain in the parent optimizer. No gradient refresh is forced.
    """

    def _direction(self) -> torch.Tensor:
        direction = super()._direction()
        if self.consecutive_failures and not self.S and self.step_cap > 0 and self.lr > 0:
            norm = float(direction.norm())
            if 0 < norm < float("inf"):
                direction = direction * (self.step_cap / (self.lr * norm))
                self.scaled_recoveries = getattr(self, "scaled_recoveries", 0) + 1
        return direction


def safeguard_direction(direction: torch.Tensor, gradient: torch.Tensor,
                        minimum_cosine: float) -> torch.Tensor:
    """Rotate toward negative gradient only when needed, retaining step length."""
    length, gradient_norm = direction.norm(), gradient.norm()
    if not torch.isfinite(length+gradient_norm) or length == 0 or gradient_norm == 0:
        return direction
    downhill = -gradient/gradient_norm
    parallel = direction.dot(downhill)
    if parallel >= minimum_cosine*length:
        return direction
    perpendicular = direction-parallel*downhill
    perpendicular_norm = perpendicular.norm()
    if perpendicular_norm == 0:
        return length*downhill
    return length*(minimum_cosine*downhill +
                   (1-minimum_cosine**2)**.5*perpendicular/perpendicular_norm)


class AngleSafeguardedLBFGS(SweepLBFGS):
    """Preserve curvature history but bound the direction's descent angle."""

    def __init__(self, source: torch.Tensor, objective: Objective,
                 minimum_cosine: float = .1, **kwargs: Any) -> None:
        if not 0 < minimum_cosine < 1:
            raise ValueError("minimum_cosine must lie strictly between zero and one")
        if kwargs.get("project", True):
            raise ValueError("Angle safeguarding currently supports unconstrained sources only")
        super().__init__(source, objective, **kwargs)
        self.minimum_cosine = minimum_cosine

    def _direction(self) -> torch.Tensor:
        return safeguard_direction(super()._direction(), self.g, self.minimum_cosine)


class CapStartedLBFGS(SweepLBFGS):
    """Start later searches at the existing cap, preserving the LBFGS ray.

    The first attempt retains the established initial scaling. Later attempts
    may require more objective trials; geometric Armijo and its budget remain
    unchanged, and no additional gradient evaluation is introduced.
    """

    def _direction(self) -> torch.Tensor:
        direction = super()._direction()
        if self.step_attempts > 0 and self.step_cap > 0 and self.lr > 0:
            norm = float(direction.norm())
            if 0 < norm < float("inf"):
                direction = direction*(self.step_cap/(self.lr*norm))
        return direction
