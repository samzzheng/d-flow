"""Nontrivial invariants for rank refinement and optimizer transitions."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from .adaptive import select_rank, solve


class AdaptiveTests(unittest.TestCase):
    def test_pairwise_agreement_is_not_an_exact_gradient_certificate(self) -> None:
        full = torch.zeros(64, dtype=torch.float64)
        full[:3] = 1
        full[20] = 10

        def evaluate(rank: int) -> torch.Tensor:
            value = full.clone()
            value[rank:] = 0
            return value

        # No rank-comparison heuristic can detect a direction missed by BOTH
        # members of its first pair. This fixture starts at16 so32 exposes it.
        rank, record = select_rank(16, .05, evaluate)
        self.assertEqual(rank, 8)  # explicitly demonstrate the shared-bias limitation
        rank, record = select_rank(32, .05, evaluate)
        self.assertEqual(rank, 32)
        self.assertFalse(record["cap_unvalidated"])

    def test_growth_and_unresolved_cap(self) -> None:
        rank, record = select_rank(8, .01, lambda r: torch.arange(1, 65, dtype=torch.float64).clamp(max=r))
        self.assertEqual(rank, 64)
        self.assertTrue(record["cap_unvalidated"])
        self.assertEqual([(r["small"], r["large"]) for r in record["comparisons"]],
                         [(4, 8), (8, 16), (16, 32), (32, 64)])

    def test_shrinking_uses_tighter_threshold(self) -> None:
        def evaluate(rank: int) -> torch.Tensor:
            return torch.tensor([.97 if rank == 4 else 1.], dtype=torch.float64)

        rank, record = select_rank(8, .05, evaluate)
        self.assertEqual(rank, 8)
        self.assertAlmostEqual(record["comparisons"][0]["threshold"], .025)

    def test_rank_change_and_resume_preserve_optimizer_invariants(self) -> None:
        class QuadraticEvaluator:
            def __init__(self, interrupted: list[bool], pause_at: int = 0) -> None:
                self.settings = {"optimizer": {"lr": .5, "max_ls": 10}, "stopping": {"max_updates": 12}}
                self.observed = torch.ones(1, 1, 2, 2, dtype=torch.float64)
                self.source = self.image = None
                self.previous_v = None
                self.value = 0.
                self.counts = {"gradients": 0, "factor_evaluations": 0, "forward_objectives": 0}
                self.interrupted, self.pause_at = interrupted, pause_at

            def operator(self, image: torch.Tensor) -> torch.Tensor:
                return image

            def objective(self, source: torch.Tensor) -> float:
                self.source = source.detach().clone()
                self.image = self.source.clone()
                self.value = float(.5 * (source.detach() - self.observed).square().sum())
                self.counts["forward_objectives"] += 1
                return self.value

            def rank_gradient(self, source: torch.Tensor, rank: int, commit_basis: bool) -> torch.Tensor:
                self.objective(source)
                self.counts["factor_evaluations"] += 1
                if commit_basis:
                    self.counts["gradients"] += 1
                    self.previous_v = source.detach().reshape(-1, 1).clone()
                    if self.pause_at == self.counts["gradients"]:
                        self.interrupted[0] = True
                # Initially all ranks agree; later rank4 misses a component.
                scale = .5 if rank == 4 and float(source.detach().mean()) < 2.97 else 1.
                return (source.detach() - self.observed) * scale

        inputs = {"initial": torch.full((1, 1, 2, 2), 3., dtype=torch.float64),
                  **{key: torch.ones(1, 1, 2, 2, dtype=torch.float64) for key in ("truth", "clean", "observed")}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            interrupted = [False]
            full = solve(QuadraticEvaluator(interrupted), inputs, "adaptive05", path / "full.pt", {}, interrupted)
            paused = solve(QuadraticEvaluator(interrupted, pause_at=7), inputs, "adaptive05", path / "resume.pt", {}, interrupted)
            self.assertEqual(paused["status"], "paused")
            interrupted[0] = False
            resumed = solve(QuadraticEvaluator(interrupted), inputs, "adaptive05", path / "resume.pt", {}, interrupted)
        torch.testing.assert_close(full["source"], resumed["source"], rtol=0, atol=0)
        self.assertEqual(full["history"], resumed["history"])
        self.assertEqual(full["rank_checks"], resumed["rank_checks"])
        self.assertEqual({row["rank"] for row in full["history"][1:]}, {4, 8})
        for row in full["history"][1:]:
            if row["rank_changed"]:
                self.assertEqual(row["curvature_pairs_used"], 0)
        self.assertEqual(full["optimizer"]["step_attempts"], 12)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
