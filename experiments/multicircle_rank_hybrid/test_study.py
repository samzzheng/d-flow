"""Check independent spectra and estimator switching, not reconstruction quality."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from p_flow.armijo import ArmijoLBFGS
from .hybrid import AlternatingLBFGS, SCHEDULES, solve
from .ranks import spectrum


class StudyTests(unittest.TestCase):
    def test_resume_preserves_schedule_basis_and_optimizer(self) -> None:
        class QuadraticEvaluator:
            def __init__(self, interrupted: list[bool], pause_at: int = 0) -> None:
                self.settings = {"optimizer": {"lr": .5, "max_ls": 10}, "stopping": {"max_updates": 8}}
                self.observed = torch.ones(1, 1, 2, 2, dtype=torch.float64)
                self.source = self.image = None
                self.previous_v = None
                self.value = 0.
                self.counts = {"gradients": 0, "forward_objectives": 0}
                self.interrupted, self.pause_at = interrupted, pause_at

            def operator(self, image: torch.Tensor) -> torch.Tensor:
                return image

            def objective(self, source: torch.Tensor) -> float:
                self.source = source.detach().clone()
                self.image = self.source.clone()
                self.value = float(.5 * (source - self.observed).square().sum())
                self.counts["forward_objectives"] += 1
                return self.value

            def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
                self.objective(source)
                self.counts["gradients"] += 1
                if self.pause_at == self.counts["gradients"]:
                    self.interrupted[0] = True
                if method == "DLR D-Flow":
                    self.previous_v = source.detach().reshape(-1, 1).clone()
                return (source.detach() - self.observed) * (1. if method == "P-Flow" else 2.)

        inputs = {"initial": torch.full((1, 1, 2, 2), 3., dtype=torch.float64),
                  **{key: torch.ones(1, 1, 2, 2, dtype=torch.float64) for key in ("truth", "clean", "observed")}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            interrupted = [False]
            uninterrupted = solve(QuadraticEvaluator(interrupted), "p3_dlr1", inputs,
                                  path / "full.pt", {}, interrupted)
            paused = solve(QuadraticEvaluator(interrupted, pause_at=4), "p3_dlr1", inputs,
                           path / "resumed.pt", {}, interrupted)
            self.assertEqual(paused["status"], "paused")
            interrupted[0] = False
            resumed = solve(QuadraticEvaluator(interrupted), "p3_dlr1", inputs,
                            path / "resumed.pt", {}, interrupted)
        torch.testing.assert_close(resumed["source"], uninterrupted["source"], rtol=0, atol=0)
        torch.testing.assert_close(resumed["previous_v"], uninterrupted["previous_v"], rtol=0, atol=0)
        self.assertEqual(resumed["history"], uninterrupted["history"])
        self.assertEqual(resumed["optimizer"]["step_attempts"], 8)

    def test_rank_reference_matches_dense_svd(self) -> None:
        torch.manual_seed(4)
        left = torch.linalg.qr(torch.randn(24, 24, dtype=torch.float64))[0]
        right = torch.linalg.qr(torch.randn(24, 24, dtype=torch.float64))[0]
        singular = torch.tensor([20., 10., 4., 1., .1] + [.00001] * 19, dtype=torch.float64)
        matrix = (left * singular) @ right.T
        with tempfile.TemporaryDirectory() as folder:
            result = spectrum(lambda v: matrix @ v, lambda v: matrix.T @ v,
                              torch.zeros(24, dtype=torch.float64), Path(folder) / "rank.json",
                              [12], 3, 64, 123, {})
        exact = torch.linalg.svdvals(matrix)
        for target in (.99, .999, .9999):
            rank = int(torch.where(exact.square().cumsum(0) >= target * exact.square().sum())[0][0]) + 1
            self.assertEqual(result["latest"]["effective_ranks"][str(target)], rank)
        self.assertEqual(result["status"], "resolved_estimate")
        self.assertLess(result["adjoint_scaled_error"], 1e-12)

    def test_uncaptured_flat_spectrum_is_not_reported_low_rank(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            result = spectrum(lambda v: v, lambda v: v, torch.zeros(64, dtype=torch.float64),
                              Path(folder) / "rank.json", [8], 2, 128, 7, {})
        self.assertIsNone(result["latest"]["effective_ranks"]["0.99"])
        self.assertEqual(result["status"], "unresolved_at_budget")

    def test_switches_recompute_and_never_reuse_cross_estimator_pairs(self) -> None:
        for variant in ("p1_dlr1", "p3_dlr1"):
            source = torch.tensor([4., 3.], dtype=torch.float64, requires_grad=True)
            optimizer = ArmijoLBFGS(source, lambda z: float(.5 * z.square().sum()),
                                    lr=.01, project=False)
            alternating = AlternatingLBFGS(optimizer, SCHEDULES[variant])
            calls = []

            def closure(method: str) -> torch.Tensor:
                calls.append(method)
                source.grad = source.detach().clone() * (1. if method == "P-Flow" else 2.)
                return .5 * source.detach().square().sum()

            for attempt in range(8):
                start = len(calls)
                before = source.detach().clone()
                method, switched, pairs = alternating.step(closure)
                self.assertTrue(all(value == method for value in calls[start:]))
                self.assertEqual(optimizer.step_attempts, attempt + 1)
                if switched:
                    self.assertEqual(pairs, 0)
                    scale = 1. if method == "P-Flow" else 2.
                    # Global first-attempt L1 scaling must not repeat on a switch.
                    torch.testing.assert_close(source.detach(), before * (1 - .01 * scale))
                if variant == "p1_dlr1":
                    self.assertEqual(pairs, 0)

    def test_failure_does_not_prevent_other_estimator_turn(self) -> None:
        source = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
        optimizer = ArmijoLBFGS(source, lambda z: float(.5 * z.square().sum()), project=False, max_ls=3)
        alternating = AlternatingLBFGS(optimizer, SCHEDULES["p1_dlr1"])

        def closure(method: str) -> torch.Tensor:
            source.grad = source.detach().clone() * (-1. if method == "P-Flow" else 1.)
            return .5 * source.detach().square().sum()

        before = source.detach().clone()
        self.assertEqual(alternating.step(closure)[0], "P-Flow")
        torch.testing.assert_close(source.detach(), before)
        self.assertEqual(alternating.step(closure)[0], "DLR D-Flow")
        self.assertLess(float(source.detach().norm()), float(before.norm()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
