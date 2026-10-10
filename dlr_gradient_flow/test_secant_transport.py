"""Dense-reference invariants for pure Jacobian-secant transport."""

from __future__ import annotations

import unittest

import torch

from sampler.ode_samplers import euler_flow

from .secant_transport import secant_transport_factors
from .test_transport import SmoothFlow
from .transport import transport_factors


class SecantTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(1063)
        self.model = SmoothFlow(torch.randn(6, 6, dtype=torch.float64) / 5)
        self.source = torch.randn(1, 1, 2, 3, dtype=torch.float64) / 3
        self.displacement = torch.randn_like(self.source) / 4
        self.steps = 7

    def test_full_rank_adds_actual_jacobian_difference(self) -> None:
        source_new = self.source + self.displacement
        factors = tuple(torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))[0]
                        for _ in range(3))
        jacobians = [torch.func.jacrev(
            lambda value: euler_flow(self.model, value, self.steps).flatten()
        )(source).reshape(6, 6) for source in (self.source, source_new)]
        endpoint, (u, s, v) = secant_transport_factors(
            self.model, self.source, source_new, factors, self.steps)
        expected = factors[0] @ factors[1] @ factors[2].T + jacobians[1] - jacobians[0]
        torch.testing.assert_close(u @ s @ v.T, expected, atol=2e-12, rtol=2e-11)
        torch.testing.assert_close(endpoint, euler_flow(self.model, source_new, self.steps))

    def test_zero_displacement_preserves_factors_exactly(self) -> None:
        factors = (torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64))[0],
                   torch.randn(3, 3, dtype=torch.float64),
                   torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64))[0])
        endpoint, updated = secant_transport_factors(
            self.model, self.source, self.source.clone(), factors, self.steps)
        for old, new in zip(factors, updated):
            self.assertTrue(torch.equal(old, new))
        torch.testing.assert_close(endpoint, euler_flow(self.model, self.source, self.steps))

    def test_secant_and_literal_transport_agree_to_first_order(self) -> None:
        factors = (torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64))[0],
                   torch.diag(torch.linspace(.8, 1.7, 3, dtype=torch.float64)),
                   torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64))[0])
        errors = []
        for step in (.08, .04, .02):
            displacement = step * self.displacement
            u_exact, s_exact, v_exact = transport_factors(
                self.model, self.source, displacement, factors, self.steps)
            _, (u, s, v) = secant_transport_factors(
                self.model, self.source, self.source + displacement, factors, self.steps)
            errors.append(float((u @ s @ v.T - u_exact @ s_exact @ v_exact.T).norm()))
            torch.testing.assert_close(u.T @ u, torch.eye(3, dtype=u.dtype), atol=2e-12, rtol=0)
            torch.testing.assert_close(v.T @ v, torch.eye(3, dtype=v.dtype), atol=2e-12, rtol=0)
        for coarser, finer in zip(errors, errors[1:]):
            self.assertGreater(coarser / finer, 3.5)
            self.assertLess(coarser / finer, 4.5)


if __name__ == "__main__":
    unittest.main()
