"""Check projected trial acceptance, step caps, and lifetime scaling."""

from __future__ import annotations

import unittest

import torch

from .armijo import ArmijoLBFGS
from .p_flow import project_source


class ArmijoTests(unittest.TestCase):
    def test_projected_trials_are_the_accepted_points_and_respect_cap(self) -> None:
        z = project_source(torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)).requires_grad_(True)
        target = project_source(torch.tensor([[4., 3., 2., 1.]], dtype=torch.float64))
        trials = []

        def objective(candidate: torch.Tensor) -> float:
            trials.append(candidate.detach().clone())
            self.assertAlmostEqual(float(candidate.norm()), 2., places=12)
            return float(.5 * (candidate - target).square().sum())

        def closure() -> torch.Tensor:
            value = .5 * (z - target).square().sum()
            z.grad, = torch.autograd.grad(value, z)
            return value.detach()

        optimizer = ArmijoLBFGS(z, objective, step_cap=.02)
        previous = float(closure())
        for _ in range(4):
            before = z.detach().clone()
            value = optimizer.step(closure)
            self.assertLess(value, previous)
            self.assertLessEqual(float((z.detach() - before).norm()), .02 + 1e-12)
            torch.testing.assert_close(z.detach(), trials[-1], atol=0, rtol=0)
            previous = value

    def test_reset_does_not_repeat_initial_scaling(self) -> None:
        z = torch.tensor([[3., 4.]], dtype=torch.float64, requires_grad=True)
        optimizer = ArmijoLBFGS(z, lambda x: float(.5 * x.square().sum()), lr=.1, project=False)

        def closure() -> torch.Tensor:
            z.grad = z.detach().clone()
            return .5 * z.detach().square().sum()

        before = z.detach().clone()
        optimizer.step(closure)
        torch.testing.assert_close(z.detach(), before - .1 * before / before.abs().sum())
        optimizer.state.clear()
        before = z.detach().clone()
        optimizer.step(closure)
        torch.testing.assert_close(z.detach(), .9 * before)
        self.assertEqual(optimizer.step_attempts, 2)

    def test_bad_proxy_direction_cannot_accept_increased_objective(self) -> None:
        z = torch.tensor([[3., 4.]], dtype=torch.float64, requires_grad=True)
        before = z.detach().clone()
        optimizer = ArmijoLBFGS(z, lambda x: float(.5 * x.square().sum()), project=False, max_ls=8)

        def closure() -> torch.Tensor:
            z.grad = -z.detach().clone()
            return .5 * z.detach().square().sum()

        optimizer.step(closure)
        torch.testing.assert_close(z.detach(), before, atol=0, rtol=0)
        self.assertEqual(optimizer.failed_searches, 1)


if __name__ == "__main__":
    unittest.main()
