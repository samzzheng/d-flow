"""Acceptance invariants for experimental proxy recovery (CPU only)."""
import unittest

import torch

from experiments.separated_shapes_117k.proxy_fallback import proxy_step


class ProxyFallbackTests(unittest.TestCase):
    def test_accepts_actual_decrease_despite_overlarge_proxy_slope(self) -> None:
        source = torch.tensor([[1.]], dtype=torch.float64)
        objective = lambda value: float(value.square().sum())
        result = proxy_step(source, torch.full_like(source, 1e8), objective, .5, 10, False)
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.value, .25)
        self.assertEqual(float(source), 1.)
        self.assertLessEqual(float((result.source-source).norm()), .5)

    def test_rejects_uphill_and_unreliable_decreases(self) -> None:
        source = torch.tensor([[1.]], dtype=torch.float32)
        uphill = proxy_step(source, -source, lambda value: float(value.square().sum()), .5, 8, False)
        self.assertFalse(uphill.accepted)
        torch.testing.assert_close(uphill.source, source, rtol=0, atol=0)
        tiny = proxy_step(source, source, lambda value: 1.+1e-9*float(value.sum()), .5, 8, False)
        self.assertFalse(tiny.accepted)

    def test_backtracks_and_preserves_existing_projection(self) -> None:
        scalar = torch.tensor([[1.]], dtype=torch.float64)
        backtracked = proxy_step(scalar, scalar, lambda value: float(value.square().sum()), 4., 10, False)
        self.assertTrue(backtracked.accepted)
        self.assertEqual(len(backtracked.trials), 3)
        source = torch.tensor([[1., 1.]], dtype=torch.float64)
        target = torch.tensor([[2.**.5, 0.]], dtype=torch.float64)
        result = proxy_step(source, source-target,
            lambda value: float((value-target).square().sum()), .3, 10, True)
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(float(result.source.norm()), 2.**.5)
        self.assertLessEqual(float((result.source-source).norm()), .3*(1+1e-6))


if __name__ == "__main__":
    unittest.main()
