"""Equivalent p=0 algebra retains QR stability and the projected Euler step."""

import unittest
from unittest.mock import patch

import torch

from dlr_dflow import enriched


def residual(factors):
    u, s, v = factors
    return u @ s @ v.T


class ZeroEnrichmentSpecializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.old_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def problem(self, dtype, deficient, alpha):
        generator = torch.Generator().manual_seed(319)
        randn = lambda *shape: torch.randn(*shape, dtype=dtype, generator=generator)
        dimension, rank = 64, 8
        u = torch.linalg.qr(randn(dimension, rank))[0]
        v = torch.linalg.qr(randn(dimension, rank))[0]
        values = (torch.tensor([1., 1e-3, 1e-7, 1e-12, 0., 0., 0., 0.], dtype=dtype)
                  if deficient else torch.linspace(1., .2, rank, dtype=dtype))
        s = torch.diag(values)
        a = .2 * randn(dimension, dimension) / dimension ** .5 - .3 * torch.eye(dimension, dtype=dtype)
        probes = (2 * torch.randint(0, 2, (dimension, 4), generator=generator) - 1).to(dtype)
        sensitivities = alpha * probes + u @ (s @ (v.T @ probes))
        return a, (u, s, v), probes, sensitivities, randn(dimension, 3)

    @staticmethod
    def step(a, factors, sensitivities, probes, alpha, specialized):
        with patch.object(enriched, "SPECIALIZE_ZERO_ENRICHMENT", specialized):
            return enriched._fused_enriched_step(
                lambda q: a @ q, lambda q: a.T @ q,
                factors, sensitivities, probes, .01, alpha, 0, None)

    def test_one_hundred_steps_preserve_represented_matrix_and_gradient_action(self):
        for dtype in (torch.float32, torch.float64):
            for deficient in (False, True):
                for alpha0 in (0., .7):
                    with self.subTest(dtype=dtype, deficient=deficient, alpha=alpha0):
                        a, initial, probes, q0, cotangents = self.problem(dtype, deficient, alpha0)
                        results = []
                        for specialized in (False, True):
                            factors, sensitivities, alpha = initial, q0, alpha0
                            for _ in range(100):
                                factors, derivative, _ = self.step(
                                    a, factors, sensitivities, probes, alpha, specialized)
                                sensitivities = sensitivities + .01 * derivative
                                alpha = (probes * sensitivities).sum() / probes.numel()
                            results.append((factors, sensitivities, alpha))
                        expected, actual = results
                        tolerance = 3e-5 if dtype == torch.float32 else 3e-12
                        expected_matrix, actual_matrix = residual(expected[0]), residual(actual[0])
                        self.assertLess(float((actual_matrix - expected_matrix).norm() / expected_matrix.norm()),
                                        tolerance)
                        expected_action, actual_action = expected_matrix.T @ cotangents, actual_matrix.T @ cotangents
                        self.assertLess(float((actual_action - expected_action).norm() / expected_action.norm()),
                                        tolerance)
                        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
                        for basis in (actual[0][0], actual[0][2]):
                            self.assertLess(float((basis.T @ basis - torch.eye(8, dtype=dtype)).norm()),
                                            2e-6 if dtype == torch.float32 else 3e-14)

    def test_core_matches_explicit_projected_euler_target(self):
        for dtype in (torch.float32, torch.float64):
            for alpha in (0., .7):
                with self.subTest(dtype=dtype, alpha=alpha):
                    a, initial, probes, sensitivities, _ = self.problem(dtype, True, alpha)
                    factors, derivative, alpha_dot = self.step(a, initial, sensitivities, probes, alpha, True)
                    u, s, v = factors
                    identity = torch.eye(a.shape[0], dtype=dtype)
                    y = residual(initial)
                    euler_target = y + .01 * (a @ (alpha * identity + y) - alpha_dot * identity)
                    expected_core = u.T @ euler_target @ v
                    torch.testing.assert_close(s, expected_core,
                                               atol=3e-7 if dtype == torch.float32 else 2e-15,
                                               rtol=2e-5 if dtype == torch.float32 else 2e-12)

    def test_zero_residual_and_zero_velocity_remain_zero(self):
        a, initial, probes, sensitivities, _ = self.problem(torch.float64, True, .7)
        initial = (initial[0], torch.zeros_like(initial[1]), initial[2])
        for alpha in (0., .7):
            factors, derivative, alpha_dot = self.step(
                torch.zeros_like(a), initial, sensitivities, probes, alpha, True)
            self.assertEqual(float(residual(factors).norm()), 0.)
            self.assertEqual(float(derivative.norm()), 0.)
            self.assertEqual(float(alpha_dot), 0.)

    def test_specialization_retains_product_widths_and_exactly_two_qr_calls(self):
        a, factors, probes, sensitivities, _ = self.problem(torch.float64, True, .7)
        original_qr = torch.linalg.qr
        with patch.object(enriched, "SPECIALIZE_ZERO_ENRICHMENT", True), \
                patch.object(torch.linalg, "qr", wraps=original_qr) as qr:
            widths = []
            def action(kind, q):
                widths.append((kind, q.shape[1]))
                return (a if kind == "A" else a.T) @ q
            enriched._fused_enriched_step(
                lambda q: action("A", q), lambda q: action("AT", q),
                factors, sensitivities, probes, .01, .7, 0, None)
        self.assertEqual(widths, [("A", 12), ("AT", 8)])
        self.assertEqual(qr.call_count, 2)

    def test_unsupported_algebra_options_keep_the_reference_path(self):
        a, factors, probes, sensitivities, _ = self.problem(torch.float64, False, .7)
        for disabled_flag in ("FUSE_K_IMAGE", "M_FROM_TRANSPOSE", "RECOMPRESS_FROM_TRANSPOSE",
                              "SKIP_UNTRUNCATED_SVD"):
            with self.subTest(flag=disabled_flag), patch.object(enriched, "SPECIALIZE_ZERO_ENRICHMENT", True), \
                    patch.object(enriched, disabled_flag, False), \
                    patch.object(enriched, "_zero_enrichment_step", side_effect=AssertionError("unexpected specialization")):
                enriched._fused_enriched_step(lambda q: a @ q, lambda q: a.T @ q,
                                              factors, sensitivities, probes, .01, .7, 0, None)
        for options in ({"alpha_rate": lambda derivative, basis: .05},
                        {"adaptive_rank": True, "rank_min": 4, "rank_max": 8}):
            with self.subTest(options=tuple(options)), patch.object(enriched, "SPECIALIZE_ZERO_ENRICHMENT", True), \
                    patch.object(enriched, "_zero_enrichment_step", side_effect=AssertionError("unexpected specialization")):
                enriched._fused_enriched_step(lambda q: a @ q, lambda q: a.T @ q,
                                              factors, sensitivities, probes, .01, .7, 0, None, **options)
        with patch.object(enriched, "SPECIALIZE_ZERO_ENRICHMENT", True), \
                patch.object(enriched, "_zero_enrichment_step", side_effect=AssertionError("unexpected specialization")):
            enriched._fused_enriched_step(
                lambda q: a @ q, lambda q: a.T @ q, factors, sensitivities,
                probes, .01, .7, 2, torch.Generator().manual_seed(72))


if __name__ == "__main__":
    unittest.main()
