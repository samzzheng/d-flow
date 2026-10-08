"""Policy invariants for the isolated optimizer experiment (CPU only)."""
from __future__ import annotations

from copy import deepcopy
import unittest

import torch

from p_flow.armijo import ArmijoLBFGS
from .optimizer_ablation import ExperimentalArmijo, VARIANTS


class OptimizerChecks(unittest.TestCase):
    def test_baseline_matches_production_and_resume(self) -> None:
        for projected in (False, True):
            initial = torch.linspace(-1, 1, 12).reshape(1, -1)
            trajectories = []
            for experimental in (False, True):
                source = initial.clone().requires_grad_(True)
                diagonal = torch.logspace(-1, 2, source.numel())
                candidates = []

                def objective(value: torch.Tensor) -> float:
                    candidates.append(value.detach().clone())
                    return float((.5 * diagonal * value.square()).sum().detach())

                def closure() -> torch.Tensor:
                    source.grad = diagonal * source.detach()
                    return source.new_tensor(objective(source.detach()))

                optimizer = (ExperimentalArmijo(source, objective, "baseline", project=projected, step_cap=.2)
                             if experimental else ArmijoLBFGS(source, objective, project=projected, step_cap=.2))
                trajectory = []
                for step in range(40):
                    if experimental and step == 20:
                        state = deepcopy({key: value for key, value in vars(optimizer).items()
                                          if key not in {"source", "objective"}})
                        optimizer = ExperimentalArmijo(source, objective, "baseline", project=projected, step_cap=.2)
                        for key, value in state.items():
                            setattr(optimizer, key, value)
                    optimizer.step(closure)
                    trajectory.append(source.detach().clone())
                trajectories.append((trajectory, candidates, optimizer))
            for left, right in zip(trajectories[0][:2], trajectories[1][:2]):
                self.assertEqual(len(left), len(right))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))
            self.assertEqual(trajectories[0][2].failed_searches, trajectories[1][2].failed_searches)

    def test_filter_rejects_nearly_orthogonal_positive_pair(self) -> None:
        for variant in ("short", "filtered"):
            source = torch.tensor([1., 0.], requires_grad=True)

            def objective(value: torch.Tensor) -> float:
                return float(.5 * value.detach().square().sum())

            def closure() -> torch.Tensor:
                displacement = source.detach()[0] - 1
                source.grad = torch.stack((1 + 1e-4 * displacement, -displacement))
                return source.new_tensor(objective(source))

            optimizer = ExperimentalArmijo(source, objective, variant, step_cap=.2)
            optimizer.step(closure)
            self.assertGreater(optimizer.last_diagnostics["curvature"], 1e-10)
            self.assertLess(optimizer.last_diagnostics["curvature_cosine"], .01)
            self.assertEqual(len(optimizer.S), int(variant == "short"))

    def test_caps_history_halving_and_lifetime_scaling(self) -> None:
        for variant in VARIANTS:
            source = torch.linspace(.5, 1.5, 16, requires_grad=True)
            diagonal = torch.logspace(0, 3, source.numel())

            def objective(value: torch.Tensor) -> float:
                return float((.5 * diagonal * value.detach().square()).sum())

            def closure() -> torch.Tensor:
                source.grad = diagonal * source.detach()
                return source.new_tensor(objective(source))

            optimizer = ExperimentalArmijo(source, objective, variant, step_cap=.2)
            for step in range(40):
                before = source.detach().clone()
                old_value = objective(source)
                optimizer.step(closure)
                self.assertLessEqual(float((source.detach() - before).norm()), .20001)
                self.assertLessEqual(objective(source), old_value)
                self.assertLessEqual(len(optimizer.S), optimizer.history_size)
                self.assertEqual(optimizer.last_diagnostics["first_attempt"], step == 0)
                trials = optimizer.last_diagnostics["trials"]
                if optimizer.geometric:
                    for left, right in zip(trials, trials[1:]):
                        self.assertEqual(right["rate"], .5 * left["rate"])
                if step == 10:
                    optimizer.S, optimizer.Y = [], []
            self.assertTrue(any(row["displacement"] > 0 for row in trials))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
