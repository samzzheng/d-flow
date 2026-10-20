"""Exact discrete-prefix products, degeneracy and retained-rank safeguards."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow.implementation.enriched import evolve_trace_centered_enriched
from dlr_dflow.implementation.residual_startup import residual_sketch_startup
from sampler.ode_samplers import euler_flow


class TimeVaryingVelocity(torch.nn.Module):
    def __init__(self, matrices: torch.Tensor) -> None:
        super().__init__()
        self.matrices = matrices

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        matrix = self.matrices[0] + t[0] * self.matrices[1]
        return (x.flatten(1) @ matrix.T + .03 * x.flatten(1).square()).reshape_as(x)


class CovarianceVelocity(torch.nn.Module):
    def __init__(self, covariance: torch.Tensor) -> None:
        super().__init__()
        self.covariance = covariance

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        return (-flat / (1 - t[:, None]) +
                t[:, None] / (1 - t[:, None]) ** 3 * (flat @ self.covariance.T)).reshape_as(x)


class ResidualStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(92)
        self.z = torch.randn(1, 1, 2, 4, dtype=torch.float64)
        self.flags = patch.multiple(core, HAND_JVP=False, HAND_VJP=False, SHARE_LINEARIZATION=False)
        self.flags.start()
        self.addCleanup(self.flags.stop)

    def test_exact_prefix_order_and_original_grid(self) -> None:
        model = TimeVaryingVelocity(.2 * torch.randn(2, 8, 8, dtype=torch.float64))
        for steps in (1, 2, 3):
            x, (u, s, v), completed = residual_sketch_startup(
                model, self.z, 5, 8, 0, steps, torch.Generator().manual_seed(91))

            def prefix(source: torch.Tensor) -> torch.Tensor:
                for index in range(steps):
                    source = source + .2 * model(source, source.new_tensor([.2 * index]))
                return source

            jacobian = torch.func.jacrev(prefix)(self.z).reshape(8, 8)
            residual = jacobian - (1 - .2 * steps) * torch.eye(8, dtype=self.z.dtype)
            torch.testing.assert_close(u @ s @ v.T, residual, atol=3e-14, rtol=3e-13)
            torch.testing.assert_close(x, prefix(self.z), atol=0, rtol=0)
            self.assertEqual(completed, steps)
            self.assertGreater(float((residual - residual.T).norm()), .01)

    def test_ideal_first_step_zero_extends_and_retains_only_resolved_rank(self) -> None:
        q = torch.linalg.qr(torch.randn(8, 2, dtype=self.z.dtype))[0]
        covariance = q @ torch.diag(self.z.new_tensor([.4, .2])) @ q.T
        model = CovarianceVelocity(covariance)
        results = []
        for minimum in (1, 2):
            result = residual_sketch_startup(
                model, self.z, 5, 6, 2, minimum, torch.Generator().manual_seed(91))
            x, (u, s, v), completed = result
            self.assertEqual(completed, 2)
            self.assertEqual(s.shape, (2, 2))
            torch.testing.assert_close(u @ s @ v.T, .2**2 / .8**2 * covariance,
                                       atol=2e-15, rtol=2e-13)
            results.append(result)
        for left, right in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_zero_residual_reaches_terminal_without_invented_basis(self) -> None:
        model = CovarianceVelocity(torch.zeros(8, 8, dtype=self.z.dtype))
        x, factors, alpha, path = evolve_trace_centered_enriched(
            model, self.z, 5, rank=4, trace_probes=0, enrichment_probes=0,
            residual_sketch_steps=2, zero_terminal_baseline=True, return_alpha_path=True)
        self.assertEqual(factors[1].shape, (0, 0))
        self.assertEqual(alpha, 0)
        self.assertEqual(len(path), 6)
        torch.testing.assert_close(x, euler_flow(model, self.z, 5), atol=0, rtol=0)

    def test_callback_and_diagnostics_do_not_change_evolution(self) -> None:
        model = TimeVaryingVelocity(.2 * torch.randn(2, 8, 8, dtype=self.z.dtype))
        callbacks = []
        options = dict(rank=3, trace_probes=0, enrichment_probes=0,
                       residual_sketch_steps=2, seed=3, zero_terminal_baseline=True)
        plain = evolve_trace_centered_enriched(model, self.z, 5, **options)
        observed = evolve_trace_centered_enriched(
            model, self.z, 5, **options, return_diagnostics=True,
            factor_callback=lambda step, alpha, factors: callbacks.append((step, factors)))
        torch.testing.assert_close(plain[0], euler_flow(model, self.z, 5), atol=0, rtol=0)
        self.assertEqual([step for step, _ in callbacks], list(range(6)))
        for left, right in zip(plain[1], observed[1]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        for _, (_, s, _) in callbacks:
            torch.testing.assert_close(s, torch.diag(s.diag()), atol=0, rtol=0)

    def test_unsupported_protocols_fail_explicitly(self) -> None:
        model = CovarianceVelocity(torch.eye(8, dtype=self.z.dtype))
        defaults = dict(trace_probes=0, enrichment_probes=0, rank=2, residual_sketch_steps=2)
        for override in (dict(trace_probes=1), dict(adaptive_rank=True), dict(factor_stride=2),
                         dict(residual_sketch_steps=6), dict(alpha_rate=lambda *args: -1)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                evolve_trace_centered_enriched(model, self.z, 5, **{**defaults, **override})


if __name__ == '__main__':
    unittest.main()
