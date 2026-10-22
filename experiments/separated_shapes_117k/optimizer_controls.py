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
