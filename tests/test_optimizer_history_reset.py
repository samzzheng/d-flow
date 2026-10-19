"""Native L-BFGS history resets must not repeat first-iteration scaling."""
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import torch

from dlr_dflow import dlr_dflow as core
from sampler.optimizer_state import reset_optimizer_history

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as gpu_bench
import benchmark_dflow_sparse_ct_circles_cpu as cpu_bench


class NativeHistoryResetTests(unittest.TestCase):
    def make_problem(self, line_search=None):
        source = torch.tensor([10., 7.], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([2., 3.], dtype=torch.float64)
        optimizer = torch.optim.LBFGS([source], lr=.1, max_iter=1, max_eval=20,
                                      line_search_fn=line_search,
                                      tolerance_grad=0., tolerance_change=0.)
        evaluations = []

        def closure():
            optimizer.zero_grad()
            evaluations.append(source.detach().clone())
            value = .5 * (weights * source.square()).sum()
            value.backward()
            return value

        return source, weights, optimizer, evaluations, closure

    def test_reset_discards_curvature_without_rescaling_next_gradient_step(self):
        source, weights, optimizer, _, closure = self.make_problem()
        optimizer.step(closure)
        optimizer.step(closure)
        state = optimizer.state[source]
        self.assertGreater(len(state["old_dirs"]), 0)
        old_iterations, old_evaluations = state["n_iter"], state["func_evals"]
        with torch.no_grad():
            source.copy_(torch.tensor([3., 4.]))
        before = source.detach().clone()
        reset_optimizer_history(optimizer)
        state = optimizer.state[source]
        self.assertEqual(state["n_iter"], old_iterations)
        self.assertEqual(state["func_evals"], old_evaluations)
        self.assertEqual(state["old_dirs"], [])
        optimizer.step(closure)
        torch.testing.assert_close(source.detach(), before - .1 * weights * before, rtol=0, atol=1e-15)
        self.assertEqual(optimizer.state[source]["n_iter"], old_iterations + 1)
        # The seeded zero previous step cannot create a cross-reset secant pair.
        self.assertEqual(optimizer.state[source]["old_dirs"], [])

    def test_strong_wolfe_uses_normal_lr_after_reset_and_still_checks_objective(self):
        source, weights, optimizer, evaluations, closure = self.make_problem("strong_wolfe")
        optimizer.step(closure)
        optimizer.step(closure)
        with torch.no_grad():
            source.copy_(torch.tensor([3., 4.]))
        before = source.detach().clone()
        reset_optimizer_history(optimizer)
        evaluations.clear()
        optimizer.step(closure)
        self.assertGreaterEqual(len(evaluations), 2)
        torch.testing.assert_close(evaluations[0], before, rtol=0, atol=0)
        torch.testing.assert_close(evaluations[1], before - .1 * weights * before, rtol=0, atol=1e-15)
        self.assertLess(float((weights * source.detach().square()).sum()),
                        float((weights * before.square()).sum()))

    def test_new_optimizer_keeps_original_first_step_and_uninterrupted_trajectory(self):
        for line_search in (None, "strong_wolfe"):
            outcomes = []
            for reset_before_start in (False, True):
                source, _, optimizer, evaluations, closure = self.make_problem(line_search)
                if reset_before_start:
                    reset_optimizer_history(optimizer)
                for _ in range(3):
                    optimizer.step(closure)
                outcomes.append((source.detach().clone(), evaluations))
            torch.testing.assert_close(outcomes[0][0], outcomes[1][0], rtol=0, atol=0)
            self.assertEqual(len(outcomes[0][1]), len(outcomes[1][1]))
            for actual, expected in zip(outcomes[0][1], outcomes[1][1]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_adam_reset_keeps_adams_existing_update_rule(self):
        source = torch.tensor([3., 4.], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([source], lr=.05)

        def step(opt, parameter):
            opt.zero_grad()
            parameter.square().sum().backward()
            opt.step()

        step(optimizer, source)
        reset_optimizer_history(optimizer)
        self.assertEqual(dict(optimizer.state), {})
        reference = source.detach().clone().requires_grad_(True)
        reference_optimizer = torch.optim.Adam([reference], lr=.05)
        step(optimizer, source)
        step(reference_optimizer, reference)
        torch.testing.assert_close(source.detach(), reference.detach(), rtol=0, atol=0)

    def test_native_resume_metadata_distinguishes_reset_schedules(self):
        old = {"lbfgs_line_search": "strong_wolfe"}
        new = {**old, "lbfgs_initial_step_policy": "first_iteration_only",
               "lbfgs_history_reset_policy": "preserve_iteration_schedule"}
        for benchmark in (cpu_bench, gpu_bench):
            with self.assertRaisesRegex(RuntimeError, "lbfgs_history_reset_policy"):
                benchmark.validate_resume_settings(old, new)
            benchmark.validate_resume_settings(new, new)


class IdentityVelocity(torch.nn.Module):
    def forward(self, x, time):
        return x


class InverseSolverResetTests(unittest.TestCase):
    def run_solver(self, factor_sign=1., safeguard=False, nan_at_refresh=None):
        source = torch.tensor([2., -1., 3., .5], dtype=torch.float64).reshape(1, 1, 2, 2)
        identity = torch.eye(4, dtype=source.dtype)
        factors = identity, factor_sign * 2 * identity, identity

        refresh_count = 0

        def sensitivity(model, z, *args, **kwargs):
            nonlocal refresh_count
            refresh_count += 1
            terminal = 2 * z.detach()
            if refresh_count == nan_at_refresh:
                terminal = torch.full_like(terminal, float("nan"))
            return terminal, factors, 0.

        original_step = torch.optim.LBFGS.step
        counters, sources, optimizer_ids = [], [], []

        def recording_step(optimizer, closure):
            value = original_step(optimizer, closure)
            parameter = optimizer.param_groups[0]["params"][0]
            counters.append(optimizer.state[parameter]["n_iter"])
            sources.append(parameter.detach().clone())
            optimizer_ids.append(id(optimizer))
            return value

        with patch.object(core, "sensitivity_factors", side_effect=sensitivity), \
                patch.object(torch.optim.LBFGS, "step", recording_step):
            result, history = core.solve_inverse_lbfgs(
                IdentityVelocity(), source, lambda x: x, torch.zeros_like(source),
                n_steps=1, rank=4, outer_steps=3, lr=.1, refresh_every=1,
                reset_history_on_refresh=True, safeguard=safeguard, blowup_factor=1.001)
        return source, result, history, counters, sources, optimizer_ids

    def test_factor_refresh_preserves_first_only_scaling_within_solve(self):
        source, _, history, counters, sources, optimizer_ids = self.run_solver()
        expected = source.clone()
        for index, actual in enumerate(sources):
            gradient = 4 * expected
            rate = .1 * min(1., 1. / float(gradient.abs().sum())) if index == 0 else .1
            expected -= rate * gradient
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
        self.assertEqual(counters, [1, 2, 3])
        self.assertEqual(len(set(optimizer_ids)), 1)
        self.assertTrue(all(row["lbfgs_initial_step_policy"] == "first_iteration_only" for row in history))

    def test_safeguard_rollback_preserves_schedule_while_halving_lr(self):
        _, _, history, counters, _, optimizer_ids = self.run_solver(factor_sign=-1., safeguard=True)
        self.assertTrue(any(row["rolled_back"] for row in history))
        self.assertLess(history[-1]["lr"], .1)
        self.assertEqual(counters, [1, 2, 3])
        self.assertEqual(len(set(optimizer_ids)), 1)

    def test_nan_objective_rolls_back_and_next_step_recovers_with_preserved_schedule(self):
        source, result, history, counters, sources, optimizer_ids = self.run_solver(
            safeguard=True, nan_at_refresh=2)
        self.assertFalse(torch.isfinite(sources[1]).all())
        self.assertEqual([row["rolled_back"] for row in history], [False, True, False])
        self.assertEqual(history[1]["phi"], history[0]["phi"])
        self.assertEqual([row["lr"] for row in history], [.1, .05, .05])
        self.assertTrue(all(torch.isfinite(torch.tensor(row["phi"])) for row in history))
        torch.testing.assert_close(result, source, rtol=0, atol=0)
        # The rollback restores the best source and discards the NaN gradient;
        # its next update uses the halved rate without repeating L1 scaling.
        torch.testing.assert_close(sources[2], source - .05 * 4 * source, rtol=0, atol=1e-15)
        self.assertEqual(counters, [1, 2, 3])
        self.assertEqual(len(set(optimizer_ids)), 1)


if __name__ == "__main__":
    unittest.main()
