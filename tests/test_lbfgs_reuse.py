"""Check that reusing a refreshed terminal state preserves LBFGS behavior."""

import unittest
from unittest.mock import patch

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow import enriched


class DenseVelocity(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, x, t):
        return (x.flatten(1) @ self.matrix.T).reshape_as(x)


class LBFGSTerminalReuseTests(unittest.TestCase):
    def setUp(self):
        dtype = torch.float64
        matrix = torch.tensor([
            [-.2, .3, 0., .1], [.1, -.4, .2, 0.],
            [0., .2, -.1, .1], [.3, 0., .1, -.3]], dtype=dtype)
        self.model = DenseVelocity(matrix)
        self.z = torch.tensor([.8, -.3, .5, 1.2], dtype=dtype).reshape(1, 1, 2, 2)
        self.y = torch.tensor([.2, -.1, .3, .4], dtype=dtype).reshape_as(self.z)
        self.weights = torch.tensor([1., 2., .7, 1.3], dtype=dtype).reshape_as(self.z)
        self.steps = 3
        self.jacobian = torch.linalg.matrix_power(torch.eye(4, dtype=dtype) + matrix / self.steps,
                                                 self.steps)
        u, s, vt = torch.linalg.svd(self.jacobian)
        self.factors = u, torch.diag(s), vt.T

    def run_solver(self, cache_enabled, refresh_every, method="enriched", repeated_closure=False):
        original_step = torch.optim.LBFGS.step
        original_flow = core.euler_flow
        trajectory = []
        closures = []

        def sensitivity(model, z, n_steps, **kwargs):
            with torch.no_grad():
                x1 = core.euler_flow(model, z, n_steps)
            return x1 if cache_enabled else None, self.factors, 0.

        def recording_step(optimizer, closure):
            z = optimizer.param_groups[0]["params"][0]

            def record_closure():
                source = z.detach().clone()
                result = closure()
                closures.append((source, z.grad.detach().clone(), result.detach().clone()))
                return result

            if repeated_closure:
                record_closure()
                with torch.no_grad():
                    z.add_(.01)
                result = record_closure()
            else:
                result = original_step(optimizer, record_closure)
            trajectory.append(z.detach().clone())
            return result

        with patch.object(core, "euler_flow", wraps=original_flow) as flow, \
                patch.object(core, "sensitivity_factors", side_effect=sensitivity) as refresh, \
                patch.object(enriched, "direct_terminal_rank_factors", return_value=self.factors) as direct, \
                patch.object(torch.optim.LBFGS, "step", recording_step):
            result, history = core.solve_inverse_lbfgs(
                self.model, self.z, lambda x: x * self.weights, self.y,
                n_steps=self.steps, rank=4, outer_steps=1 if repeated_closure else 4,
                lr=.2, refresh_every=refresh_every, method=method,
                safeguard=False, reg_lam=.03)
        return result, history, trajectory, closures, flow.call_count, refresh.call_count, direct.call_count

    def assert_equivalent(self, reference, actual):
        torch.testing.assert_close(actual[0], reference[0], atol=1e-14, rtol=1e-14)
        self.assertEqual(len(actual[1]), len(reference[1]))
        for actual_row, reference_row in zip(actual[1], reference[1]):
            self.assertEqual({k: v for k, v in actual_row.items() if k != "time_s"},
                             {k: v for k, v in reference_row.items() if k != "time_s"})
        for actual_source, reference_source in zip(actual[2], reference[2]):
            torch.testing.assert_close(actual_source, reference_source, atol=1e-14, rtol=1e-14)
        self.assertEqual(len(actual[3]), len(reference[3]))
        for actual_closure, reference_closure in zip(actual[3], reference[3]):
            for actual_value, reference_value in zip(actual_closure, reference_closure):
                torch.testing.assert_close(actual_value, reference_value, atol=1e-14, rtol=1e-14)
            source, gradient, objective = actual_closure
            x1 = (self.jacobian @ source.flatten()).reshape_as(source)
            residual = self.weights * x1 - self.y
            expected_gradient = (self.jacobian.T @ (self.weights * residual).flatten()).reshape_as(source)
            expected_gradient += .03 * source
            torch.testing.assert_close(gradient, expected_gradient, atol=1e-14, rtol=1e-14)
            torch.testing.assert_close(objective, .5 * residual.square().sum(), atol=1e-14, rtol=1e-14)

    def test_refresh_reuse_preserves_trajectory_and_saves_one_flow_per_refresh(self):
        for refresh_every in (1, 2):
            with self.subTest(refresh_every=refresh_every):
                reference = self.run_solver(False, refresh_every)
                actual = self.run_solver(True, refresh_every)
                self.assert_equivalent(reference, actual)
                refreshes = 4 // refresh_every
                self.assertEqual(actual[5], refreshes)
                self.assertEqual(actual[4], 4)
                self.assertEqual(reference[4], 4 + refreshes)

    def test_direct_method_keeps_its_existing_flow_count_and_trajectory(self):
        reference = self.run_solver(False, 2, method="direct")
        actual = self.run_solver(True, 2, method="direct")
        self.assert_equivalent(reference, actual)
        self.assertEqual(actual[4], 4)
        self.assertEqual(actual[5], 0)
        self.assertEqual(actual[6], 2)

    def test_cache_is_consumed_before_a_second_closure_at_an_updated_source(self):
        reference = self.run_solver(False, 1, repeated_closure=True)
        actual = self.run_solver(True, 1, repeated_closure=True)
        self.assert_equivalent(reference, actual)
        self.assertEqual(len(actual[3]), 2)
        self.assertEqual(actual[4], 2)
        self.assertEqual(reference[4], 3)


if __name__ == "__main__":
    unittest.main()
