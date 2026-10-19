"""Numerical Armijo acceptance must represent an actual decreasing step."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
from optimizer_settings_audit import AuditedArmijo
from benchmark_dlr_speedup import load_points


class ArmijoNumericalProgressTests(unittest.TestCase):
    def run_constant_objective(self, factory=bench.ArmijoLBFGS):
        source = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
        calls = {"gradient": 0, "objective": 0}

        def objective(candidate):
            calls["objective"] += 1
            return 1e20

        optimizer = factory(source, objective, max_ls=3)

        def closure():
            calls["gradient"] += 1
            source.grad = torch.ones_like(source)
            return 1e20

        optimizer.step(closure)
        return source, optimizer, calls

    def test_equal_loss_is_not_accepted_when_armijo_rhs_rounds_to_old_loss(self):
        self.assertEqual(1e20 - 1e-4, 1e20)
        source, optimizer, calls = self.run_constant_objective()
        self.assertEqual(float(source.detach()), 1.)
        self.assertEqual(optimizer.failed_searches, 1)
        self.assertEqual(optimizer.consecutive_failures, 1)
        self.assertEqual(calls, {"gradient": 1, "objective": 3})
        self.assertEqual(optimizer.total_function_evaluations, 3)
        self.assertEqual(optimizer.S, [])

    def test_zero_displacement_stops_before_evaluating_objective(self):
        source = torch.tensor([1e8], dtype=torch.float32, requires_grad=True)
        calls = []
        optimizer = bench.ArmijoLBFGS(source, lambda candidate: calls.append(candidate) or 1e8)

        def closure():
            source.grad = torch.ones_like(source)
            return 1e8

        optimizer.step(closure)
        self.assertEqual(calls, [])
        self.assertEqual(optimizer.last_function_evaluations, 0)
        self.assertEqual(optimizer.failed_searches, 1)
        self.assertEqual(float(source.detach()), 1e8)

    def test_zero_gradient_does_not_divide_by_zero_or_evaluate_trials(self):
        source = torch.tensor([1.], requires_grad=True)
        calls = []
        optimizer = bench.ArmijoLBFGS(source, lambda candidate: calls.append(candidate) or 0.)

        def closure():
            source.grad = torch.zeros_like(source)
            return 0.

        optimizer.step(closure)
        optimizer.step(closure)
        self.assertEqual(calls, [])
        self.assertEqual(optimizer.step_attempts, 2)
        self.assertEqual(optimizer.failed_searches, 2)
        self.assertTrue(optimizer.stalled)
        self.assertEqual(float(source.detach()), 1.)

    def test_nonfinite_slope_after_fallback_never_evaluates_a_trial(self):
        for gradient in (float("nan"), float("inf")):
            with self.subTest(gradient=gradient):
                source = torch.tensor([1.], requires_grad=True)
                calls = []
                optimizer = bench.ArmijoLBFGS(source, lambda candidate: calls.append(candidate) or 0.)

                def closure():
                    source.grad = torch.full_like(source, gradient)
                    return 1.

                optimizer.step(closure)
                self.assertEqual(calls, [])
                self.assertEqual(optimizer.failed_searches, 1)
                self.assertEqual(float(source.detach()), 1.)

    def test_diagnostic_default_matches_guard_and_legacy_requires_explicit_choice(self):
        for factory in (bench.ArmijoLBFGS, AuditedArmijo):
            source, optimizer, calls = self.run_constant_objective(factory)
            self.assertEqual(float(source.detach()), 1.)
            self.assertEqual(optimizer.failed_searches, 1)
            self.assertEqual(optimizer.progress_policy, "strict_decrease_nonzero_step")
        source, optimizer, _ = self.run_constant_objective(
            lambda *args, **kwargs: AuditedArmijo(*args, scaling="once", progress="legacy", **kwargs))
        self.assertEqual(float(source.detach()), 0.)
        self.assertEqual(optimizer.failed_searches, 0)
        self.assertEqual(optimizer.progress_policy, "legacy_rounded_rhs")
        self.assertEqual(optimizer.initial_step_policy, "first_iteration_only")

    def test_failed_search_clears_bad_memory_then_recovers_without_l1_restart(self):
        source = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
        objective = lambda candidate: float(.5 * candidate.square().sum())
        optimizer = bench.ArmijoLBFGS(source, objective, lr=.1, max_ls=1)
        optimizer.state = {"initialised": True}
        optimizer.f, optimizer.g = 2., torch.tensor([2.], dtype=source.dtype)
        optimizer.S = [torch.ones(1, dtype=source.dtype)]
        optimizer.Y = [.01 * torch.ones(1, dtype=source.dtype)]
        optimizer.step_attempts = 1
        gradient_calls = []

        def closure():
            gradient_calls.append(True)
            source.grad = source.detach().clone()
            return objective(source.detach())

        optimizer.step(closure)
        self.assertEqual(float(source.detach()), 2.)
        self.assertEqual(optimizer.S, [])
        self.assertEqual(gradient_calls, [])
        self.assertEqual(optimizer.consecutive_failures, 1)
        optimizer.step(closure)
        self.assertAlmostEqual(float(source.detach()), 1.8)
        self.assertEqual(len(gradient_calls), 1)
        self.assertEqual(optimizer.step_attempts, 3)
        self.assertEqual(optimizer.consecutive_failures, 0)
        self.assertEqual(optimizer.failed_searches, 1)

    def test_progress_policy_is_part_of_production_resume_identity(self):
        old = {"optimizer": "lbfgs", "lbfgs_line_search": "armijo",
               "armijo_initial_step_policy": "first_iteration_only"}
        current = {**old, "armijo_progress_policy": "strict_decrease_nonzero_step"}
        with self.assertRaisesRegex(RuntimeError, "armijo_progress_policy"):
            bench.validate_resume_settings(old, current)
        bench.validate_resume_settings(current, current)
        adam = {"optimizer": "adam", "lbfgs_line_search": "none"}
        bench.validate_resume_settings(adam, {**adam, "armijo_progress_policy": None})

    def test_point_cache_cannot_mix_old_progress_with_current_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = torch.zeros(1, 1, 2, 2)
            metadata = {"point_optimizer": "exact Armijo L-BFGS",
                        "armijo_initial_step_policy": bench.ArmijoLBFGS.initial_step_policy}
            torch.save({"metadata": metadata, "points": {0: source}, "references": {}},
                       path / "res250_sample0.pt")
            args = SimpleNamespace(points_cache=path, iterates=[0], generate_missing_points=False)
            current = {**metadata, "armijo_progress_policy": bench.ArmijoLBFGS.progress_policy}
            with self.assertRaisesRegex(ValueError, "Point-cache protocol mismatch"):
                load_points(args, None, None, 250, 0, source, None, None, current)


if __name__ == "__main__":
    unittest.main()
