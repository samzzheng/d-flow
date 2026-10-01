"""Convergence labels and exact continuation of an interrupted GPU schedule."""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .gpu_study import schedule, stopping
from .study import Run, Tensor
from .test_study import Quadratic


class ConvergenceTests(unittest.TestCase):
    def test_plateau_is_not_small_iteration_count_or_image_error(self) -> None:
        decreasing = [{"phi": 1 / (i + 1), "relative_image_error": 1.} for i in range(601)]
        self.assertEqual(stopping(decreasing, 500, 2000, 100, .001)[0], "running")
        constant = [{"phi": 1.} for _ in range(501)]
        self.assertEqual(stopping(constant[:101], 500, 2000, 100, .001)[0], "running")
        self.assertEqual(stopping(constant, 500, 2000, 100, .001)[0], "objective_plateau")
        oscillatory = [{"phi": 1 + .1 * (i % 2)} for i in range(501)]
        self.assertEqual(stopping(oscillatory, 500, 2000, 100, .001)[0], "oscillatory_plateau")
        self.assertEqual(stopping(decreasing, 500, 600, 100, .001)[0], "step_limit")

    def test_resume_preserves_adam_and_lifetime_scaling(self) -> None:
        torch.set_num_threads(2)
        source = torch.full((1, 1, 4, 4), 3., dtype=torch.float64)
        expected = Run("resume", "exact", "adam", False, .1, source.clone())
        expected_problem = Quadratic(source)
        for _ in range(20):
            expected.advance(expected_problem)
        interrupted = [False]

        class Interruptible(Quadratic):
            def evaluate(self, z: Tensor, estimator: str, rank: int = 8) -> tuple[Tensor, float, Tensor]:
                result = super().evaluate(z, estimator, rank)
                if self.counts["exact_gradients"] == 4:
                    interrupted[0] = True
                return result

        with tempfile.TemporaryDirectory() as directory, patch("dlr_dflow.experiments.first_order_study.gpu_report.render"):
            out = Path(directory)
            (out / "states").mkdir()
            args = argparse.Namespace(out=out, device="cpu", min_updates=20, max_updates=20,
                                      window=5, plateau_relative=.001)
            run = Run("resume", "exact", "adam", False, .1, source.clone())
            with self.assertRaises(KeyboardInterrupt):
                schedule([run], [Interruptible(source)], "comparisons", {}, args, interrupted)
            self.assertEqual(run.attempts, 4)
            restored = Run("resume", "exact", "adam", False, .1, source.clone())
            schedule([restored], [Quadratic(source)], "comparisons", {}, args, [False])
            torch.testing.assert_close(restored.z, expected.z, rtol=0, atol=0)
            for key in ("m", "v"):
                torch.testing.assert_close(restored.state[key], expected.state[key], rtol=0, atol=0)
            self.assertEqual(restored.attempts, 20)
            self.assertTrue(all(row["initial_scale"] == 1 for row in restored.history[2:]))
            self.assertEqual(restored.status, "step_limit")


if __name__ == "__main__":
    unittest.main()
