"""Numerical invariants for the isolated study (no pretrained model required)."""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow.api import sensitivity_factors, make_terminal_flow_products
from dlr_dflow.implementation import enriched
from sampler.ode_samplers import euler_flow
from .diagnostics import comparison, reference_svd
from .study import Problem, Radon, Run, Tensor, candidate, project, tangent


class LinearFlow(torch.nn.Module):
    def __init__(self, matrix: Tensor) -> None:
        super().__init__()
        self.matrix = matrix

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return (x.flatten(1) @ self.matrix.T).reshape_as(x)


class Quadratic:
    def __init__(self, source: Tensor, reject: bool = False) -> None:
        self.target = torch.arange(source.numel(), dtype=source.dtype).reshape_as(source) / 10
        self.counts = {"exact_gradients": 0, "dlr_gradients": 0, "trial_flows": 0}
        self.trials: list[Tensor] = []
        self.reject = reject

    def evaluate(self, z: Tensor, estimator: str, rank: int = 8) -> tuple[Tensor, float, Tensor]:
        self.counts["exact_gradients"] += 1
        return z, float(.5 * (z - self.target).square().sum()), z - self.target

    def trial(self, z: Tensor) -> tuple[Tensor, float]:
        self.trials.append(z.clone())
        self.counts["trial_flows"] += 1
        return z, float("inf") if self.reject else float(.5 * (z - self.target).square().sum())

    def metrics(self, x: Tensor) -> dict[str, float]:
        return {"relative_image_error": float((x - self.target).norm())}


class StudyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(71)

    def test_analytic_alpha_and_no_trace_columns(self) -> None:
        matrix = torch.randn(16, 16, dtype=torch.float64) / 10
        model = LinearFlow(matrix)
        source = torch.randn(1, 1, 4, 4, dtype=torch.float64)
        widths = []

        def products(model: LinearFlow, t: float, x: Tensor, **kwargs: object) -> tuple:
            def forward(q: Tensor) -> Tensor:
                widths.append(q.shape[1])
                return model.matrix @ q
            return forward, lambda q: model.matrix.T @ q

        with patch.object(core, "make_A_products", products), patch.object(enriched, "_trace_probes", wraps=enriched._trace_probes) as probes:
            endpoint, _, alpha, path = sensitivity_factors(model, source, 5, rank=2, speed_profile="fast",
                trace_probes=0, enrichment_probes=0, initialization_oversampling=8,
                zero_terminal_baseline=True, return_alpha_path=True)
        self.assertEqual(probes.call_args.args[1], 0)
        self.assertEqual(widths, [10, 2, 2, 2, 2])
        self.assertEqual(alpha, 0.)
        torch.testing.assert_close(torch.tensor(path), torch.linspace(1, 0, 6), atol=1e-7, rtol=0)
        torch.testing.assert_close(endpoint, euler_flow(model, source, 5))

    def test_discrete_flow_products_and_spectral_reference(self) -> None:
        matrix = torch.diag(torch.linspace(-.8, 1, 16, dtype=torch.float64))
        model = LinearFlow(matrix)
        source = torch.randn(1, 1, 4, 4, dtype=torch.float64)
        J, JT = make_terminal_flow_products(model, source, 7)
        dense = torch.linalg.matrix_power(torch.eye(16, dtype=torch.float64) + matrix / 7, 7)
        q = torch.randn(16, 3, dtype=torch.float64)
        torch.testing.assert_close(J(q), dense @ q)
        torch.testing.assert_close(JT(q), dense.T @ q)
        gx = torch.randn(16, 1, dtype=torch.float64)
        exact = dense.T @ gx
        output = {}
        factors = reference_svd(J, JT, gx, exact, 8, 4, time.monotonic() + 30, output)
        u, s, vh = torch.linalg.svd(dense)
        expected = vh[:8].T @ (s[:8, None] * (u[:, :8].T @ gx))
        left, center, right = factors
        torch.testing.assert_close(right @ center.T @ left.T @ gx, expected)
        self.assertEqual(output["status"], "certified_numerical")

    def test_projected_curve_derivative_and_actual_trials(self) -> None:
        z = project(torch.randn(1, 1, 4, 4, dtype=torch.float64))
        direction = tangent(torch.randn_like(z), z, True)
        rate = .17
        raw = z + rate * direction
        derivative = 4 / raw.norm() * tangent(direction, raw, True)
        numeric = (candidate(z, direction, rate + 1e-6, True) - candidate(z, direction, rate - 1e-6, True)) / 2e-6
        torch.testing.assert_close(derivative, numeric)
        for optimizer in ("gd", "adam", "lbfgs"):
            problem = Quadratic(z)
            run = Run(optimizer, "exact", optimizer, True, 1., z.clone())
            for _ in range(3):
                run.advance(problem)
            self.assertTrue(all(abs(float(trial.norm()) - 4) < 1e-10 for trial in problem.trials))
            self.assertTrue(all(h["source_step_norm"] <= .04000001 for h in run.history[1:]))
            self.assertLess(run.history[1]["initial_scale"], 1)
            self.assertEqual(run.history[2]["initial_scale"], 1)
            if optimizer != "adam":
                self.assertTrue(all(b["phi"] < a["phi"] for a, b in zip(run.history, run.history[1:])))

    def test_failed_attempt_does_not_repeat_scaling(self) -> None:
        z = torch.full((1, 1, 4, 4), 10., dtype=torch.float64)
        problem = Quadratic(z, reject=True)
        run = Run("failure", "exact", "gd", False, 1., z.clone())
        run.advance(problem)
        torch.testing.assert_close(run.z, z)
        problem.reject = False
        run.advance(problem)
        self.assertEqual(run.history[-1]["initial_scale"], 1.)
        self.assertTrue(run.history[-1]["accepted"])

    def test_exact_gradient_and_radon_finite_difference(self) -> None:
        matrix = torch.randn(16, 16, dtype=torch.float64) / 20
        z = torch.randn(1, 1, 4, 4, dtype=torch.float64)
        operator = Radon(4, 3, torch.float64)
        truth = torch.rand_like(z)
        clean = operator(truth)
        problem = Problem(LinearFlow(matrix), truth, z, operator, clean, clean + .01, 1, steps=5)
        _, _, gradient = problem.evaluate(z, "exact")
        direction = torch.randn_like(z)
        _, plus = problem.trial(z + 1e-5 * direction)
        _, minus = problem.trial(z - 1e-5 * direction)
        self.assertAlmostEqual((plus - minus) / 2e-5, float((gradient * direction).sum()), places=8)

    def test_subspace_decomposition_and_undefined_metrics(self) -> None:
        v, _ = torch.linalg.qr(torch.randn(16, 3, dtype=torch.float64))
        exact = torch.randn(16, 1, dtype=torch.float64)
        approximate = v @ torch.randn(3, 1, dtype=torch.float64)
        subspace = v @ v.T @ exact
        torch.testing.assert_close((approximate - exact).square().sum(),
                                  (approximate - subspace).square().sum() + (subspace - exact).square().sum())
        self.assertIsNone(comparison(exact, torch.zeros_like(exact))["cosine"])
        self.assertIsNone(comparison(exact, torch.zeros_like(exact))["relative_error"])


if __name__ == "__main__":
    unittest.main()
