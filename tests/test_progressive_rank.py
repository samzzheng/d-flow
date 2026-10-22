"""Dense linear-flow checks for the isolated progressive rank policy."""
import unittest

import torch

from dlr_dflow.api import sensitivity_factors
from dlr_dflow.implementation import enriched
from experiments.separated_shapes_117k.progressive_rank import progressive_rank
from experiments.ct_mri_sweep.validation import ChunkedAutodiffProducts


class LinearFlow(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        diagonal = torch.linspace(-.95, 1., x.numel(), dtype=x.dtype).reshape_as(x)
        return diagonal * x


class ProgressiveRankTests(unittest.TestCase):
    def test_rank_bounds_and_unchanged_primal(self) -> None:
        torch.set_num_threads(2)
        source = torch.ones(1, 1, 4, 4, dtype=torch.float64)
        options = dict(rank=8, adaptive_rank=True, rank_min=1, rank_max=8,
                       rank_tol=.999, trace_probes=0, enrichment_probes=0,
                       initialization_oversampling=8, seed=12,
                       products=ChunkedAutodiffProducts(LinearFlow().eval(), columns=4))
        original = enriched.evolve_trace_centered_enriched
        original_step = enriched._fused_enriched_step
        baseline = sensitivity_factors(LinearFlow(), source, 5, **options)
        for fraction in (.5, 1.):
            ranks = []

            def record(step: int, alpha: float, factors: tuple[torch.Tensor, ...]) -> None:
                ranks.append(factors[1].shape[0])
                self.assertAlmostEqual(alpha, 1-step/5)

            with progressive_rank(1, fraction):
                image, factors, alpha = sensitivity_factors(
                    LinearFlow(), source, 5, **options, factor_callback=record)
            self.assertIs(enriched.evolve_trace_centered_enriched, original)
            self.assertIs(enriched._fused_enriched_step, original_step)
            self.assertEqual(ranks, [0, 8, 4, 2, 1, 1] if fraction == .5 else [0, 8, 8, 8, 8, 8])
            self.assertEqual(alpha, 0.)
            torch.testing.assert_close(image, baseline[0], rtol=0, atol=0)
            for basis in (factors[0], factors[2]):
                torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1], dtype=basis.dtype))
            if fraction == 1.:
                control = sensitivity_factors(LinearFlow(), source, 5, **{**options, "rank_min": 8})
                for actual, expected in zip(factors, control[1]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_restores_implementation_on_error(self) -> None:
        original = enriched.evolve_trace_centered_enriched
        with self.assertRaises(ValueError), progressive_rank(2):
            sensitivity_factors(LinearFlow(), torch.ones(1, 1, 4, 4), 5, adaptive_rank=False)
        self.assertIs(enriched.evolve_trace_centered_enriched, original)

    def test_full_rank_reproduces_dense_euler_jacobian(self) -> None:
        source = torch.ones(1, 1, 4, 4, dtype=torch.float64)
        model = LinearFlow().eval()
        with progressive_rank(16, .5):
            _, (u, s, v), alpha = sensitivity_factors(model, source, 5,
                rank=16, adaptive_rank=True, rank_min=16, rank_max=16,
                trace_probes=0, enrichment_probes=0, initialization_oversampling=0,
                products=ChunkedAutodiffProducts(model, columns=4))
        expected = torch.diag((1+.2*torch.linspace(-.95, 1., 16, dtype=source.dtype))**5)
        torch.testing.assert_close(u@s@v.T+alpha*torch.eye(16), expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
