"""Regression tests for the lifetime first-step policy of custom Armijo L-BFGS."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
from benchmark_dlr_speedup import load_points
from optimizer_settings_audit import AuditedArmijo


class ArmijoInitialScalingTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.full((64,), 10.0, dtype=torch.float64, requires_grad=True)
        self.trials = []
        self.reject = False
        self.gradient_calls = 0

        def objective(candidate):
            self.trials.append(candidate.detach().clone())
            return float(self.loss(self.source.detach()) + 1) if self.reject else float(self.loss(candidate))

        self.optimizer = bench.ArmijoLBFGS(self.source, objective, step_cap=2.5, max_ls=1)

    @staticmethod
    def loss(source):
        # Negative curvature makes all new secant pairs fail the positive test.
        return -.5 * source.square().sum()

    def closure(self):
        self.optimizer.zero_grad()
        value = self.loss(self.source)
        value.backward()
        self.gradient_calls += 1
        return value.detach()

    def test_negative_curvature_does_not_repeat_l1_scaling(self):
        norms = []
        for _ in range(3):
            previous = self.source.detach().clone()
            self.optimizer.step(self.closure)
            norms.append(float((self.source.detach() - previous).norm()))
            self.assertEqual(self.optimizer.S, [])
            self.assertEqual(self.optimizer.Y, [])
        self.assertAlmostEqual(norms[0], .125)
        self.assertAlmostEqual(norms[1], 2.5)
        self.assertAlmostEqual(norms[2], 2.5)
        self.assertEqual(self.optimizer.step_attempts, 3)
        self.assertEqual(self.optimizer.total_function_evaluations, 3)

    def test_failed_first_search_consumes_initial_scaling(self):
        previous = self.source.detach().clone()
        self.reject = True
        self.optimizer.step(self.closure)
        torch.testing.assert_close(self.source.detach(), previous, rtol=0, atol=0)
        self.assertEqual(self.optimizer.step_attempts, 1)
        self.assertEqual(self.optimizer.consecutive_failures, 1)
        self.assertAlmostEqual(float((self.trials[-1] - previous).norm()), .125)
        self.reject = False
        self.optimizer.step(self.closure)
        self.assertAlmostEqual(float((self.source.detach() - previous).norm()), 2.5)
        self.assertEqual(self.optimizer.step_attempts, 2)
        self.assertEqual(self.optimizer.consecutive_failures, 0)

    def test_state_clear_refreshes_values_without_restarting_scaling(self):
        self.optimizer.step(self.closure)
        with torch.no_grad():
            self.source.fill_(9.)
        self.optimizer.S = [torch.ones_like(self.source)]
        self.optimizer.Y = [torch.ones_like(self.source)]
        self.optimizer.consecutive_failures = 2
        self.optimizer.state.clear()
        previous = self.source.detach().clone()
        old_gradient_calls = self.gradient_calls
        self.optimizer.step(self.closure)
        self.assertEqual(self.gradient_calls, old_gradient_calls + 2)
        self.assertAlmostEqual(float((self.source.detach() - previous).norm()), 2.5)
        self.assertEqual(self.optimizer.step_attempts, 2)
        self.assertEqual(self.optimizer.S, [])
        self.assertEqual(self.optimizer.consecutive_failures, 0)
        self.assertEqual(self.optimizer.f, float(self.loss(self.source.detach())))

    def test_non_descent_reset_keeps_lifetime_step_schedule(self):
        self.optimizer.step(self.closure)
        previous = self.source.detach().clone()
        self.optimizer.S = [torch.ones_like(self.source)]
        self.optimizer.Y = [torch.ones_like(self.source)]
        with patch.object(self.optimizer, "_direction", return_value=self.optimizer.g.clone()):
            self.optimizer.step(self.closure)
        self.assertAlmostEqual(float((self.source.detach() - previous).norm()), 2.5)
        self.assertEqual(self.optimizer.step_attempts, 2)
        self.assertEqual(self.optimizer.S, [])

    def test_cap_and_true_armijo_backtracking_are_preserved(self):
        source = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
        trials = []

        def objective(candidate):
            trials.append(candidate.detach().clone())
            return float(50 * candidate.square().sum())

        optimizer = bench.ArmijoLBFGS(source, objective, step_cap=2.5, max_ls=5)

        def closure():
            optimizer.zero_grad()
            value = 50 * source.square().sum()
            value.backward()
            return value.detach()

        optimizer.step(closure)
        self.assertAlmostEqual(float(source.detach()), 1.)
        optimizer.S.clear()
        optimizer.Y.clear()
        previous = source.detach().clone()
        trials.clear()
        optimizer.step(closure)
        self.assertAlmostEqual(float((trials[0] - previous).norm()), 2.5)
        self.assertGreater(objective(trials[0]), objective(previous))
        self.assertEqual(optimizer.last_function_evaluations, 2)
        self.assertAlmostEqual(float(source.detach()), 0.)

    def test_positive_curvature_matches_explicit_historical_policy(self):
        outcomes = []
        for factory in (bench.ArmijoLBFGS, lambda *args, **kwargs: AuditedArmijo(*args, scaling="legacy", **kwargs)):
            source = torch.linspace(-1., 2., 10, dtype=torch.float64).requires_grad_(True)
            weights = torch.linspace(1., 7., 10, dtype=torch.float64)

            def value(candidate):
                return .5 * (weights * candidate.square()).sum()

            optimizer = factory(source, lambda candidate: float(value(candidate)), step_cap=2.5)

            def closure():
                optimizer.zero_grad()
                result = value(source)
                result.backward()
                return result.detach()

            rows = []
            for _ in range(5):
                optimizer.step(closure)
                self.assertGreater(len(optimizer.S), 0)
                rows.append((source.detach().clone(), optimizer.f, len(optimizer.S), optimizer.total_function_evaluations))
            outcomes.append(rows)
        for actual, historical in zip(*outcomes):
            torch.testing.assert_close(actual[0], historical[0], rtol=0, atol=0)
            self.assertEqual(actual[1:], historical[1:])

    def test_diagnostic_default_matches_production_with_empty_history(self):
        outcomes = []
        for factory in (bench.ArmijoLBFGS, AuditedArmijo):
            source = torch.full((64,), 10., dtype=torch.float64, requires_grad=True)
            optimizer = factory(source, lambda candidate: float(self.loss(candidate)), step_cap=2.5)

            def closure():
                optimizer.zero_grad()
                value = self.loss(source)
                value.backward()
                return value.detach()

            for _ in range(3):
                optimizer.step(closure)
            self.assertEqual(optimizer.initial_step_policy, "first_iteration_only")
            self.assertEqual(optimizer.step_attempts, 3)
            outcomes.append(source.detach().clone())
        torch.testing.assert_close(*outcomes, rtol=0, atol=0)


class ArmijoResumePolicyTests(unittest.TestCase):
    def test_legacy_armijo_output_cannot_mix_with_new_policy(self):
        old = {"lbfgs_line_search": "armijo"}
        new = {**old, "armijo_initial_step_policy": bench.ArmijoLBFGS.initial_step_policy}
        with self.assertRaisesRegex(RuntimeError, "armijo_initial_step_policy"):
            bench.validate_resume_settings(old, new)
        bench.validate_resume_settings(new, new)

    def test_new_armijo_metadata_does_not_invalidate_other_optimizers(self):
        old = {"optimizer": "adam", "lbfgs_line_search": "none"}
        bench.validate_resume_settings(old, {**old, "armijo_initial_step_policy": None})

    def test_legacy_point_cache_is_not_reused_for_new_optimizer_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = torch.zeros(1, 1, 2, 2)
            metadata = {"point_optimizer": "exact Armijo L-BFGS"}
            torch.save({"metadata": metadata, "points": {0: source}, "references": {}},
                       path / "res250_sample0.pt")
            args = SimpleNamespace(points_cache=path, iterates=[0], generate_missing_points=False)
            current = {**metadata, "armijo_initial_step_policy": bench.ArmijoLBFGS.initial_step_policy}
            with self.assertRaisesRegex(ValueError, "Point-cache protocol mismatch"):
                load_points(args, None, None, 250, 0, source, None, None, current)


if __name__ == "__main__":
    unittest.main()
