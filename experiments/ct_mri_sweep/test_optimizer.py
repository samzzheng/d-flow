"""Dense inverse-BFGS reference and optional CUDA replay/resume checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from p_flow.armijo import ArmijoLBFGS, lbfgs_direction
from . import run
from .optimizer import SweepLBFGS
from .test_sweep import LinearVelocity


class DirectionTests(unittest.TestCase):
    def test_two_loop_matches_dense_inverse_bfgs(self) -> None:
        generator = torch.Generator().manual_seed(41)
        s = list(torch.randn(5, 8, generator=generator, dtype=torch.float64).unbind())
        curvature = torch.diag(torch.linspace(1, 4, 8, dtype=torch.float64))
        y = [curvature @ v for v in s]
        identity = torch.eye(8, dtype=torch.float64)
        inverse = identity * s[-1].dot(y[-1]) / y[-1].dot(y[-1])
        for displacement, change in zip(s, y):
            rho = 1 / change.dot(displacement)
            left = identity - rho * displacement[:, None] @ change[None, :]
            inverse = left @ inverse @ left.T + rho * displacement[:, None] @ displacement[None, :]
        g = torch.randn(8, generator=generator, dtype=torch.float64)
        torch.testing.assert_close(lbfgs_direction(g, s, y), -inverse @ g, atol=1e-13, rtol=1e-13)

    @unittest.skipUnless(os.getenv("SWEEP_TEST_DEVICE"), "Set SWEEP_TEST_DEVICE to an explicitly reserved CUDA device")
    def test_cuda_replay_refreshes_history_and_survives_reset(self) -> None:
        device = torch.device(os.environ["SWEEP_TEST_DEVICE"])
        source = torch.ones(4096, device=device)
        optimizer = SweepLBFGS(source, lambda x: 0., history_size=8, project=False)
        generator = torch.Generator(device=device).manual_seed(41)
        for length in (0, 3, 8, 8, 0, 8, 8):
            optimizer.g = torch.randn(4096, device=device, generator=generator)
            optimizer.S = list(torch.randn(length, 4096, device=device, generator=generator).unbind())
            optimizer.Y = [v * torch.linspace(1, 2, 4096, device=device) for v in optimizer.S]
            actual = optimizer._direction()
            torch.testing.assert_close(actual, lbfgs_direction(optimizer.g, optimizer.S, optimizer.Y), rtol=0, atol=0)
            actual.neg_()  # The caller may project/change the output; replay must replace it.

    @unittest.skipUnless(os.getenv("SWEEP_TEST_DEVICE"), "Set SWEEP_TEST_DEVICE to an explicitly reserved CUDA device")
    def test_cuda_solver_matches_eager_and_resumes_without_serializing_graph(self) -> None:
        device = torch.device(os.environ["SWEEP_TEST_DEVICE"])
        torch.cuda.set_device(device)
        settings = json.loads(Path(__file__).with_name("config_fp32.json").read_text())
        settings["flow_steps"] = 5
        settings["optimizer"]["history_size"] = 8
        settings["stopping"].update(min_updates=20, max_updates=20)
        settings["execution"]["checkpoint_every"] = 2
        initial = torch.linspace(1, 4, 64, device=device).reshape(1, 1, 8, 8)
        truth = torch.ones_like(initial)
        operator = run.MRI(8, 1, 1, device)
        clean = operator(truth)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(run, "SweepLBFGS", ArmijoLBFGS):
                eager = run.solve(run.Evaluator(LinearVelocity(), operator, clean, settings, 0),
                    "D-Flow", initial, truth, clean, root / "eager.pt", {}, [False])
            interrupted = [False]
            evaluator = run.Evaluator(LinearVelocity(), operator, clean, settings, 0)
            gradient = evaluator.gradient

            def interrupt_after_capture(source: torch.Tensor, method: str) -> torch.Tensor:
                value = gradient(source, method)
                interrupted[0] = evaluator.counts["gradients"] == 13
                return value

            with patch.object(evaluator, "gradient", interrupt_after_capture):
                paused = run.solve(evaluator, "D-Flow", initial, truth, clean, root / "resumed.pt", {}, interrupted)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(len(paused["optimizer"]["S"]), 8)
            saved = torch.load(root / "resumed.pt", weights_only=True)
            self.assertNotIn("_direction_graph", saved["optimizer"])
            resumed = run.solve(run.Evaluator(LinearVelocity(), operator, clean, settings, 0),
                "D-Flow", initial, truth, clean, root / "resumed.pt", {}, [False])
            torch.testing.assert_close(eager["source"], resumed["source"], rtol=0, atol=0)
            self.assertEqual([row["phi"] for row in eager["history"]], [row["phi"] for row in resumed["history"]])


if __name__ == "__main__":
    unittest.main()
