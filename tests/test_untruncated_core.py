"""A general square core must preserve the untruncated DLRA evolution."""

import unittest
from unittest.mock import patch

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow import enriched


class DenseVelocity(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, x, t):
        # State dependence and a changing velocity Jacobian exercise more
        # than repeated applications of a constant normal matrix.
        linear = x.flatten(1) @ self.matrix.T
        scale = 1 + .2 * t.reshape(-1, 1)
        return (scale * linear + .03 * x.flatten(1).square()).reshape_as(x)


def reconstruct(factors):
    u, s, v = factors
    return u @ s @ v.T


class UntruncatedCoreTests(unittest.TestCase):
    def problem(self, dimension, dtype):
        generator = torch.Generator().manual_seed(217)
        matrix = torch.randn(dimension, dimension, generator=generator, dtype=dtype)
        matrix = matrix / dimension ** .5 - .6 * torch.eye(dimension, dtype=dtype)
        z = torch.randn(1, 1, 4, dimension // 4, generator=generator, dtype=dtype)
        direction = torch.randn(z.shape, generator=generator, dtype=dtype)
        return DenseVelocity(matrix), z, direction

    def run_evolution(self, model, z, skip, **kwargs):
        with patch.multiple(core, HAND_JVP=False, HAND_VJP=False,
                            SHARE_LINEARIZATION=False), \
                patch.object(enriched, "SKIP_UNTRUNCATED_SVD", skip):
            return enriched.evolve_trace_centered_enriched(
                model, z, n_steps=20, rank=4, trace_probes=4,
                initialization_oversampling=3, enrichment_probes=0,
                seed=73, **kwargs)

    def assert_canonical(self, factors, tolerance):
        u, s, v = factors
        values = torch.diag(s)
        torch.testing.assert_close(s, torch.diag(values), atol=0, rtol=0)
        self.assertTrue(bool((values >= 0).all()))
        self.assertTrue(bool((values[:-1] >= values[1:]).all()))
        identity = torch.eye(s.shape[0], dtype=s.dtype)
        torch.testing.assert_close(u.T @ u, identity, atol=tolerance, rtol=0)
        torch.testing.assert_close(v.T @ v, identity, atol=tolerance, rtol=0)

    def test_multistep_actions_and_public_factor_contract(self):
        for dimension in (32, 64):
            for dtype in (torch.float64, torch.float32):
                for transpose_core in (False, True):
                    with self.subTest(dimension=dimension, dtype=dtype,
                                      transpose_core=transpose_core):
                        model, z, terminal_grad = self.problem(dimension, dtype)
                        with patch.object(enriched, "RECOMPRESS_FROM_TRANSPOSE", transpose_core):
                            reference = self.run_evolution(model, z, False, return_alpha_path=True)
                            actual = self.run_evolution(model, z, True, return_alpha_path=True)
                        tolerance = 2e-10 if dtype == torch.float64 else 8e-5
                        torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
                        torch.testing.assert_close(torch.tensor(actual[3]), torch.tensor(reference[3]),
                                                   atol=0, rtol=0)
                        torch.testing.assert_close(reconstruct(actual[1]), reconstruct(reference[1]),
                                                   atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(
                            core.approx_source_gradient(actual[1], terminal_grad, actual[2]),
                            core.approx_source_gradient(reference[1], terminal_grad, reference[2]),
                            atol=tolerance, rtol=tolerance)
                        self.assert_canonical(actual[1], tolerance)

    def test_observers_preserve_evolution_and_singular_values(self):
        model, z, _ = self.problem(32, torch.float64)
        snapshots = []

        def callback(step, alpha, factors):
            self.assert_canonical(factors, 2e-11)
            snapshots.append((step, alpha, reconstruct(factors), torch.diag(factors[1])))

        plain = self.run_evolution(model, z, True)
        observed = self.run_evolution(model, z, True, factor_callback=callback,
                                      return_diagnostics=True)
        self.assertEqual(len(snapshots), 21)
        for actual, expected in zip(observed[1], plain[1]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for step, (_, _, residual, values) in enumerate(snapshots[1:], start=1):
            expected = torch.linalg.svdvals(residual)[:values.numel()]
            torch.testing.assert_close(values, expected, atol=2e-11, rtol=2e-11)
            torch.testing.assert_close(
                torch.tensor(observed[3][step]["residual_singular_values"], dtype=z.dtype),
                values, atol=0, rtol=0)
        diagnostics_only = self.run_evolution(model, z, True, return_diagnostics=True)
        for actual, expected in zip(diagnostics_only[3], observed[3]):
            torch.testing.assert_close(
                torch.tensor(actual["residual_singular_values"]),
                torch.tensor(expected["residual_singular_values"]))

    def test_skips_intermediate_svd_and_redundant_qr(self):
        model, z, _ = self.problem(32, torch.float64)
        counts = []
        original_svd, original_qr = torch.linalg.svd, torch.linalg.qr
        for skip in (False, True):
            with patch.object(torch.linalg, "svd", wraps=original_svd) as svd, \
                    patch.object(torch.linalg, "qr", wraps=original_qr) as qr:
                self.run_evolution(model, z, skip)
            counts.append((svd.call_count, qr.call_count))
        self.assertEqual(counts[0][0], 20)
        self.assertEqual(counts[1][0], 2)  # initialization and public output
        self.assertEqual(counts[0][1] - counts[1][1], 19)

    def test_custom_alpha_rate_keeps_original_basis(self):
        model, z, _ = self.problem(32, torch.float64)

        def rate(step, derivative, u):
            # Intentionally basis dependent: sign/gauge changes must not
            # alter a user hook merely because the fast path is enabled.
            return -.1 + (.03 * u[0, 0] if u.shape[1] else 0.)

        reference = self.run_evolution(model, z, False, alpha_rate=rate)
        actual = self.run_evolution(model, z, True, alpha_rate=rate)
        for value, expected in zip(actual[1], reference[1]):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        self.assertEqual(actual[2], reference[2])

    def test_adaptive_rank_still_canonicalizes_each_step(self):
        model, z, _ = self.problem(32, torch.float64)
        kwargs = dict(adaptive_rank=True, rank_min=2, rank_max=5, rank_tol=.2)
        reference = self.run_evolution(model, z, False, **kwargs)
        actual = self.run_evolution(model, z, True, **kwargs)
        for value, expected in zip(actual[1], reference[1]):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        self.assert_canonical(actual[1], 2e-11)

    def test_custom_rate_multirate_remainder_returns_diagonal_core(self):
        model, z, _ = self.problem(32, torch.float64)
        kwargs = dict(alpha_rate=lambda step, derivative, u: -.1, factor_stride=3)
        reference = self.run_evolution(model, z, False, **kwargs)
        actual = self.run_evolution(model, z, True, **kwargs)
        self.assert_canonical(actual[1], 2e-11)
        torch.testing.assert_close(reconstruct(actual[1]), reconstruct(reference[1]),
                                   atol=2e-11, rtol=2e-11)


if __name__ == "__main__":
    unittest.main()
