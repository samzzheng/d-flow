"""Optimizer independence and closure correctness for the DLR driver."""

from __future__ import annotations

import unittest
from typing import Callable
from unittest.mock import patch

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow import optimization
from dlr_dflow.implementation.optimizer_adapters import solve_inverse_lbfgs
from dlr_dflow.implementation.types import SensitivityState
from sampler.ode_samplers import euler_flow


class LinearVelocity(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x


class ProbeOptimizer(torch.optim.Optimizer):
    """Evaluate at two sources under no_grad and deliberately return None."""

    def __init__(self, parameters: list[torch.Tensor], lr: float = 0.02) -> None:
        super().__init__(parameters, dict(lr=lr))
        self.evaluations: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> None:
        z = self.param_groups[0]["params"][0]
        for _ in range(2):
            loss = closure()
            self.evaluations.append((z.detach().clone(), loss.clone(), z.grad.clone()))
            z.add_(z.grad, alpha=-self.param_groups[0]["lr"])


class DLROptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LinearVelocity()
        self.z = torch.tensor([0.8, -0.3, 0.5, 1.2], dtype=torch.float64).reshape(1, 1, 2, 2)
        self.y = torch.tensor([0.2, -0.1, 0.3, 0.4], dtype=torch.float64).reshape_as(self.z)
        self.weights = torch.tensor([1., 2., .7, 1.3], dtype=torch.float64).reshape_as(self.z)
        identity = torch.eye(4, dtype=self.z.dtype)
        self.factors = identity, 2 * identity, identity

    def operator(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights

    def sensitivity(self, model: torch.nn.Module, z: torch.Tensor,
                    n_steps: int, **kwargs: object) -> SensitivityState:
        return 2 * z.detach(), self.factors, 0.

    def test_standard_optimizers_match_exact_linear_problem(self) -> None:
        factories = (
            lambda p: torch.optim.SGD(p, lr=.02, momentum=.8),
            lambda p: torch.optim.Adam(p, lr=.02),
            lambda p: torch.optim.LBFGS(p, lr=.1, max_iter=1),
        )
        for factory in factories:
            with self.subTest(optimizer=factory([self.z]).__class__.__name__):
                reference = self.z.clone().requires_grad_(True)
                optimizer = factory([reference])
                values, sources = [], []

                def closure() -> torch.Tensor:
                    optimizer.zero_grad()
                    value = .5 * (self.operator(2 * reference) - self.y).square().sum()
                    value.backward()
                    return value

                for _ in range(4):
                    sources.append(reference.detach().clone())
                    values.append(float(optimizer.step(closure).detach()))
                with patch.object(core, "sensitivity_factors", side_effect=self.sensitivity) as refresh, \
                        patch.object(optimization, "euler_flow", wraps=euler_flow) as flow:
                    actual, history = optimization.optimize(
                        self.model, self.z, self.operator, self.y, optimizer_factory=factory,
                        n_steps=1, outer_steps=4, refresh_every=2, safeguard=False)
                torch.testing.assert_close(actual, sources[min(range(4), key=values.__getitem__)])
                torch.testing.assert_close(torch.tensor([row["phi"] for row in history]), torch.tensor(values))
                self.assertEqual(refresh.call_count, 2)
                self.assertEqual(flow.call_count, 2)
                self.assertTrue(all("lbfgs_initial_step_policy" not in row for row in history))

    def test_repeated_closure_and_none_return_include_regularization(self) -> None:
        optimizers = []

        def factory(parameters: list[torch.Tensor]) -> ProbeOptimizer:
            optimizer = ProbeOptimizer(parameters)
            optimizers.append(optimizer)
            return optimizer

        with patch.object(core, "sensitivity_factors", side_effect=self.sensitivity), \
                patch.object(optimization, "euler_flow", wraps=euler_flow) as flow:
            _, history = optimization.optimize(
                self.model, self.z, self.operator, self.y, optimizer_factory=factory,
                n_steps=1, outer_steps=2, refresh_every=1, safeguard=False,
                reg_lam=.03, shell_lam=.02, lam_tv=.01, lam_kurt=.001)
        self.assertEqual(flow.call_count, 2)
        evaluations = optimizers[0].evaluations
        for z, value, gradient in evaluations:
            z = z.requires_grad_(True)
            x = 2 * z
            misfit = .5 * (self.operator(x) - self.y).square().sum()
            tv = (x[..., 1:, :] - x[..., :-1, :]).abs().sum() + (x[..., :, 1:] - x[..., :, :-1]).abs().sum()
            excess = (((z - z.mean()) / z.std()) ** 4).mean() - 3
            loss = misfit + .015 * z.square().sum() + .01 * (z.norm() - 2).square() + .01 * tv + .001 * excess.square()
            expected_gradient, = torch.autograd.grad(loss, z)
            torch.testing.assert_close(value, loss.detach())
            torch.testing.assert_close(gradient, expected_gradient)
        for index, row in enumerate(history):
            z = evaluations[2 * index][0].detach()
            self.assertAlmostEqual(row["phi"], float(.5 * (self.operator(2 * z) - self.y).square().sum()))

    def test_custom_reset_hook_is_used_on_factor_refresh(self) -> None:
        resets = []
        with patch.object(core, "sensitivity_factors", side_effect=self.sensitivity):
            optimization.optimize(
                self.model, self.z, self.operator, self.y,
                optimizer_factory=lambda p: torch.optim.Adam(p, lr=.01),
                n_steps=1, outer_steps=3, refresh_every=1, safeguard=False,
                reset_history_on_refresh=True, reset_optimizer=resets.append)
        self.assertEqual(len(resets), 2)
        self.assertIs(resets[0], resets[1])

    def test_lbfgs_adapter_preserves_lifetime_initial_step_scaling(self) -> None:
        original_step = torch.optim.LBFGS.step
        sources, counters = [], []

        def step(optimizer: torch.optim.LBFGS, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
            value = original_step(optimizer, closure)
            z = optimizer.param_groups[0]["params"][0]
            sources.append(z.detach().clone())
            counters.append(optimizer.state[z]["n_iter"])
            return value

        with patch.object(core, "sensitivity_factors", side_effect=self.sensitivity), \
                patch.object(torch.optim.LBFGS, "step", step):
            _, history = solve_inverse_lbfgs(
                self.model, self.z, self.operator, self.y, n_steps=1,
                outer_steps=3, refresh_every=1, lr=.1,
                reset_history_on_refresh=True, safeguard=False)
        expected = self.z.clone()
        for index, actual in enumerate(sources):
            gradient = 2 * self.weights * (self.operator(2 * expected) - self.y)
            rate = .1 * min(1., 1. / float(gradient.abs().sum())) if index == 0 else .1
            expected -= rate * gradient
            torch.testing.assert_close(actual, expected)
        self.assertEqual(counters, [1, 2, 3])
        self.assertTrue(all(row["lbfgs_initial_step_policy"] == "first_iteration_only" for row in history))


if __name__ == "__main__":
    unittest.main()
