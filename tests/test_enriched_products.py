"""Regression checks for algebraic reductions in the enriched sensitivity step."""

import unittest
from unittest.mock import patch

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow import enriched


def reconstruct(factors):
    u, s, v = factors
    return u @ s @ v.T


class DenseVelocity(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return (x.flatten(1) @ self.matrix.T).reshape_as(x)


class EnrichedProductTests(unittest.TestCase):
    def test_zero_enrichment_reuses_transpose_for_recompression(self):
        """Check the multi-step action against the explicit-JVP core."""
        dimension, steps, rank, probes = 16, 12, 4, 4
        for dtype in (torch.float64, torch.float32):
            generator = torch.Generator().manual_seed(71)
            matrix = torch.randn(dimension, dimension, generator=generator, dtype=dtype) / 5
            matrix -= .8 * torch.eye(dimension, dtype=dtype)
            z = torch.randn(1, 1, 4, 4, generator=generator, dtype=dtype)
            terminal_grad = torch.randn(z.shape, generator=generator, dtype=dtype)
            results, counts = [], []
            for transpose_recompression in (False, True):
                model = DenseVelocity(matrix)
                with patch.multiple(core, HAND_JVP=False, HAND_VJP=False), \
                        patch.object(enriched, "RECOMPRESS_FROM_TRANSPOSE", transpose_recompression):
                    results.append(enriched.evolve_trace_centered_enriched(
                        model, z, n_steps=steps, rank=rank, trace_probes=probes,
                        initialization_oversampling=3, enrichment_probes=0,
                        seed=37, return_alpha_path=True))
                counts.append(model.calls)
            reference, actual = results
            tolerance = 5e-11 if dtype == torch.float64 else 1e-4
            torch.testing.assert_close(actual[0], reference[0], atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(reconstruct(actual[1]), reconstruct(reference[1]),
                                       atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(
                core.approx_source_gradient(actual[1], terminal_grad, actual[2]),
                core.approx_source_gradient(reference[1], terminal_grad, reference[2]),
                atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(torch.tensor(actual[3]), torch.tensor(reference[3]))
            self.assertEqual(counts, [3 * steps - 1, 2 * steps])

    def test_fused_step_matches_unfused_action_and_saves_rank_columns(self):
        """Nonsymmetric products exercise both transpose identities."""
        dimension, rank, probes = 32, 4, 5
        for dtype in (torch.float64, torch.float32):
            generator = torch.Generator().manual_seed(17)
            matrix = torch.randn(dimension, dimension, generator=generator, dtype=dtype) / 4
            u = torch.linalg.qr(torch.randn(dimension, rank, generator=generator, dtype=dtype))[0]
            v = torch.linalg.qr(torch.randn(dimension, rank, generator=generator, dtype=dtype))[0]
            s = torch.diag(torch.tensor([3., 1., .4, .1], dtype=dtype))
            p = torch.randn(dimension, probes, generator=generator, dtype=dtype)
            sensitivities = p + .1 * (matrix @ p)
            terminal_grad = torch.randn(dimension, 1, generator=generator, dtype=dtype)
            for enrichment in (0, 3):
                for alpha in (0., .6):
                    for transpose_core in (False, True):
                        for transpose_recompression in (False, True):
                            with self.subTest(dtype=dtype, enrichment=enrichment, alpha=alpha,
                                              transpose_core=transpose_core,
                                              transpose_recompression=transpose_recompression):
                                def run(fused):
                                    widths = []

                                    def a(q):
                                        widths.append(q.shape[1])
                                        return matrix @ q

                                    with patch.multiple(enriched, FUSE_K_IMAGE=fused,
                                                        M_FROM_TRANSPOSE=transpose_core,
                                                        RECOMPRESS_FROM_TRANSPOSE=transpose_recompression):
                                        result = enriched._fused_enriched_step(
                                            a, lambda q: matrix.T @ q, (u, s, v),
                                            sensitivities, p, .03, alpha, enrichment,
                                            torch.Generator().manual_seed(29))
                                    return result, widths

                                (reference, ref_derivative, ref_alpha_dot), ref_widths = run(False)
                                (actual, derivative, alpha_dot), widths = run(True)
                                self.assertEqual(widths[0], probes + rank + enrichment)
                                self.assertEqual(sum(ref_widths) - sum(widths), rank)
                                tolerance = 2e-11 if dtype == torch.float64 else 3e-5
                                torch.testing.assert_close(reconstruct(actual), reconstruct(reference),
                                                           atol=tolerance, rtol=tolerance)
                                torch.testing.assert_close(
                                    core.approx_source_gradient(actual, terminal_grad, alpha),
                                    core.approx_source_gradient(reference, terminal_grad, alpha),
                                    atol=tolerance, rtol=tolerance)
                                torch.testing.assert_close(derivative, ref_derivative)
                                torch.testing.assert_close(alpha_dot, ref_alpha_dot)

    def test_initialization_matches_direct_svd_and_reuses_primal(self):
        dimension, rank, probes, oversampling, seed = 16, 3, 4, 3, 9
        generator = torch.Generator().manual_seed(5)
        matrix = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64) / 10
        z = torch.randn(1, 1, 4, 4, generator=generator, dtype=torch.float64)
        model = DenseVelocity(matrix)
        trace_vectors, sketch_generator = enriched._trace_probes(
            dimension, probes, seed + 1, z.device, z.dtype)
        sketch = enriched._rademacher(
            dimension, rank + oversampling, sketch_generator, z.device, z.dtype)
        exact_jacobian = torch.eye(dimension, dtype=z.dtype) + matrix
        alpha = (trace_vectors * (exact_jacobian @ trace_vectors)).sum() / (probes * dimension)
        residual = exact_jacobian - alpha * torch.eye(dimension, dtype=z.dtype)
        basis = torch.linalg.qr(residual @ sketch, mode="reduced")[0]
        right, values, left_t = torch.linalg.svd(residual.T @ basis, full_matrices=False)
        reference = (basis @ left_t.T[:, :rank]) @ torch.diag(values[:rank]) @ right[:, :rank].T

        for share_primal in (False, True):
            with self.subTest(share_primal=share_primal):
                model.calls = 0
                with patch.multiple(core, HAND_JVP=False, HAND_VJP=False), \
                        patch.object(enriched, "SHARE_PRIMAL", share_primal):
                    x, factors, actual_alpha, diagnostics = enriched.evolve_trace_centered_enriched(
                        model, z, n_steps=1, rank=rank, trace_probes=probes,
                        initialization_oversampling=oversampling, enrichment_probes=2,
                        seed=seed, return_diagnostics=True)
                torch.testing.assert_close(reconstruct(factors), reference, atol=2e-12, rtol=2e-12)
                torch.testing.assert_close(x.flatten(), exact_jacobian @ z.flatten())
                self.assertAlmostEqual(actual_alpha, alpha.item(), places=12)
                self.assertEqual(model.calls, 2 if share_primal else 3)
                self.assertEqual(len(diagnostics), 2)
                self.assertIsInstance(diagnostics[1]["alpha"], float)

    def test_multistep_dense_flow_preserves_gradient_and_terminal_state(self):
        """Check accumulated float32 error, including a nonnormal velocity."""
        dimension, steps = 16, 12
        for dtype in (torch.float64, torch.float32):
            generator = torch.Generator().manual_seed(53)
            matrix = torch.randn(dimension, dimension, generator=generator, dtype=dtype) / 5
            matrix -= .8 * torch.eye(dimension, dtype=dtype)
            model = DenseVelocity(matrix)
            z = torch.randn(1, 1, 4, 4, generator=generator, dtype=dtype)
            terminal_grad = torch.randn(z.shape, generator=generator, dtype=dtype)
            results = []
            for fused in (False, True):
                with patch.multiple(core, HAND_JVP=False, HAND_VJP=False), \
                        patch.object(enriched, "FUSE_K_IMAGE", fused):
                    results.append(enriched.evolve_trace_centered_enriched(
                        model, z, n_steps=steps, rank=4, trace_probes=4,
                        initialization_oversampling=3, enrichment_probes=2,
                        seed=37, return_alpha_path=True))
            reference, actual = results
            tolerance = 5e-11 if dtype == torch.float64 else 1e-4
            torch.testing.assert_close(actual[0], reference[0], atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(
                core.approx_source_gradient(actual[1], terminal_grad, actual[2]),
                core.approx_source_gradient(reference[1], terminal_grad, reference[2]),
                atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(torch.tensor(actual[3]), torch.tensor(reference[3]))
            exact = torch.linalg.matrix_power(torch.eye(dimension, dtype=dtype) + matrix / steps, steps)
            torch.testing.assert_close(actual[0].flatten(), exact @ z.flatten(),
                                       atol=tolerance, rtol=tolerance)


if __name__ == "__main__":
    unittest.main()
