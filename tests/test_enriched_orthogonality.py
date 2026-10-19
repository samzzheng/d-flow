"""Enrichment must remain orthogonal when its forcing has deficient rank."""

import unittest
from unittest.mock import patch

import torch

from dlr_dflow import enriched


def reconstruct(factors):
    u, s, v = factors
    return u @ s @ v.T


class EnrichedOrthogonalityTests(unittest.TestCase):
    def make_problem(self, dtype):
        generator = torch.Generator().manual_seed(12)
        dimension, rank = 64, 8
        u = torch.linalg.qr(torch.randn(
            dimension, rank, generator=generator, dtype=dtype))[0]
        v = torch.linalg.qr(torch.randn(
            dimension, rank, generator=generator, dtype=dtype))[0]
        s = torch.diag(torch.linspace(2, .2, rank, dtype=dtype))
        probes = torch.randn(dimension, 4, generator=generator, dtype=dtype)
        return (u, s, v), probes, generator

    def evolve_step(self, matrix, factors, probes, via_transpose):
        with patch.object(enriched, "RECOMPRESS_FROM_TRANSPOSE", via_transpose):
            return enriched._fused_enriched_step(
                lambda q: matrix @ q, lambda q: matrix.T @ q,
                factors, probes, probes, .03, 0., 4,
                torch.Generator().manual_seed(3),
                alpha_rate=lambda derivative, u: 0.,
            )[0]

    def assert_orthogonal(self, factors, tolerance):
        u, _, v = factors
        identity = torch.eye(u.shape[1], dtype=u.dtype)
        torch.testing.assert_close(u.T @ u, identity, atol=tolerance, rtol=0)
        torch.testing.assert_close(v.T @ v, identity, atol=tolerance, rtol=0)

    def test_zero_velocity_preserves_residual(self):
        """With alpha=alpha_dot=A=0 the residual ODE is exactly stationary."""
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                factors, probes, _ = self.make_problem(dtype)
                matrix = torch.zeros(64, 64, dtype=dtype)
                actual = self.evolve_step(matrix, factors, probes, True)
                reference = self.evolve_step(matrix, factors, probes, False)
                tolerance = 3e-6 if dtype == torch.float32 else 1e-12
                self.assert_orthogonal(actual, tolerance)
                torch.testing.assert_close(reconstruct(actual), reconstruct(factors),
                                           atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(reconstruct(actual), reconstruct(reference),
                                           atol=tolerance, rtol=tolerance)

    def test_deficient_forcing_matches_explicit_recompression(self):
        """Exercise zero and rank-one enrichment outside the K-step basis."""
        for dtype in (torch.float32, torch.float64):
            factors, probes, generator = self.make_problem(dtype)
            u = factors[0]
            inner = torch.randn(8, 8, generator=generator, dtype=dtype)
            left = torch.randn(64, 1, generator=generator, dtype=dtype)
            right = torch.randn(64, 1, generator=generator, dtype=dtype)
            for case, matrix in (("range_in_u", u @ inner @ u.T),
                                 ("rank_one", left @ right.T)):
                with self.subTest(dtype=dtype, case=case):
                    actual = self.evolve_step(matrix, factors, probes, True)
                    reference = self.evolve_step(matrix, factors, probes, False)
                    tolerance = 5e-6 if dtype == torch.float32 else 2e-12
                    self.assert_orthogonal(actual, tolerance)
                    torch.testing.assert_close(reconstruct(actual), reconstruct(reference),
                                               atol=tolerance, rtol=tolerance)


if __name__ == "__main__":
    unittest.main()
