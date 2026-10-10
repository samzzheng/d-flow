"""CPU checks of optimization-time sensitivity transport for the Euler map.

Dense references are deliberately restricted to a six-dimensional nonlinear
model. No pretrained checkpoints, GPU jobs, or reconstruction runs are needed.
"""

from __future__ import annotations

import unittest

import torch

from dlr_dflow.dlr_dflow import approx_source_gradient
from dlr_dflow.implementation.types import Factors
from sampler.ode_samplers import euler_flow

from .transport import transport_factors, variation_products


Tensor = torch.Tensor


class SmoothFlow(torch.nn.Module):
    """A smooth, time-dependent velocity with noncommuting dense derivatives."""

    def __init__(self, matrix: Tensor) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        flat = x.flatten(1)
        velocity = (1 + 0.2 * t[:, None]) * (flat @ self.matrix.T).sin()
        return (velocity + 0.1 * flat.square()).reshape_as(x)


class TransportTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(1729)
        self.dimension = 6
        self.steps = 7
        self.model = SmoothFlow(torch.randn(6, 6, dtype=torch.float64) / 5)
        self.source = torch.randn(1, 1, 2, 3, dtype=torch.float64) / 3
        self.displacement = torch.randn_like(self.source) / 4

    def jacobian(self, source: Tensor) -> Tensor:
        return torch.func.jacrev(
            lambda z: euler_flow(self.model, z, self.steps).flatten()
        )(source).reshape(self.dimension, self.dimension)

    def factors(self, rank: int) -> Factors:
        left, _ = torch.linalg.qr(torch.randn(6, rank, dtype=torch.float64))
        right, _ = torch.linalg.qr(torch.randn(6, rank, dtype=torch.float64))
        core = torch.diag(torch.linspace(0.8, 1.7, rank, dtype=torch.float64))
        return left, core, right

    def test_exact_second_variation_actions_and_adjoint(self) -> None:
        _, dense = torch.func.jvp(
            self.jacobian, (self.source,), (self.displacement,)
        )
        columns = torch.randn(6, 5, dtype=torch.float64)
        cotangents = torch.randn(6, 5, dtype=torch.float64)
        for chunk in (1, 3):
            with self.subTest(lane_chunk_size=chunk):
                action, transpose = variation_products(
                    self.model, self.source, self.displacement,
                    n_steps=self.steps, backend="exact", lane_chunk_size=chunk,
                )
                forward = action(columns)
                backward = transpose(cotangents)
                torch.testing.assert_close(forward, dense @ columns, atol=2e-12, rtol=2e-11)
                torch.testing.assert_close(backward, dense.T @ cotangents, atol=2e-12, rtol=2e-11)
                torch.testing.assert_close((forward * cotangents).sum(),
                                           (columns * backward).sum(), atol=2e-12, rtol=2e-11)

    def test_finite_difference_products_converge_to_second_variation(self) -> None:
        _, dense = torch.func.jvp(
            self.jacobian, (self.source,), (self.displacement,)
        )
        columns = torch.randn(6, 3, dtype=torch.float64)
        errors: list[float] = []
        for step in (0.08, 0.04, 0.02):
            action, transpose = variation_products(
                self.model, self.source, self.displacement, n_steps=self.steps,
                backend="finite_difference", difference_step=step, lane_chunk_size=2,
            )
            forward_error = (action(columns) - dense @ columns).norm()
            backward_error = (transpose(columns) - dense.T @ columns).norm()
            errors.append(float(forward_error + backward_error))
        for coarser, finer in zip(errors, errors[1:]):
            self.assertGreater(coarser / finer, 3.5)
            self.assertLess(coarser / finer, 4.5)

    def test_transport_linearization_has_quadratic_displacement_remainder(self) -> None:
        jacobian = self.jacobian(self.source)
        action, _ = variation_products(
            self.model, self.source, self.displacement, n_steps=self.steps,
        )
        variation = action(torch.eye(6, dtype=torch.float64))
        errors = [float((self.jacobian(self.source + step * self.displacement)
                         - jacobian - step * variation).norm())
                  for step in (0.08, 0.04, 0.02)]
        for coarser, finer in zip(errors, errors[1:]):
            self.assertGreater(coarser / finer, 3.5)
            self.assertLess(coarser / finer, 4.5)

    def test_full_rank_ksl_is_exact_additive_update(self) -> None:
        factors = self.factors(6)
        left, core, right = factors
        _, dense = torch.func.jvp(
            self.jacobian, (self.source,), (self.displacement,)
        )
        updated_left, updated_core, updated_right = transport_factors(
            self.model, self.source, self.displacement, factors, n_steps=self.steps,
        )
        expected = left @ core @ right.T + dense
        torch.testing.assert_close(updated_left @ updated_core @ updated_right.T,
                                   expected, atol=2e-12, rtol=2e-11)
        terminal_gradient = torch.randn_like(self.source)
        transported_gradient = approx_source_gradient(
            (updated_left, updated_core, updated_right), terminal_gradient, alpha1=0.0,
        )
        torch.testing.assert_close(transported_gradient.flatten(),
                                   expected.T @ terminal_gradient.flatten(),
                                   atol=2e-12, rtol=2e-11)

    def test_low_rank_ksl_matches_tangent_projection_to_first_order(self) -> None:
        factors = self.factors(3)
        left, core, right = factors
        matrix = left @ core @ right.T
        _, dense = torch.func.jvp(
            self.jacobian, (self.source,), (self.displacement,)
        )
        tangent = (dense @ right @ right.T
                   - left @ left.T @ dense @ right @ right.T
                   + left @ left.T @ dense)
        errors: list[float] = []
        for step in (0.04, 0.02, 0.01):
            updated_left, updated_core, updated_right = transport_factors(
                self.model, self.source, step * self.displacement, factors,
                n_steps=self.steps,
            )
            updated = updated_left @ updated_core @ updated_right.T
            errors.append(float((updated - matrix - step * tangent).norm()))
            identity = torch.eye(3, dtype=torch.float64)
            torch.testing.assert_close(updated_left.T @ updated_left, identity)
            torch.testing.assert_close(updated_right.T @ updated_right, identity)
            self.assertEqual(updated_core.shape, (3, 3))
        for coarser, finer in zip(errors, errors[1:]):
            self.assertGreater(coarser / finer, 3.5)
            self.assertLess(coarser / finer, 4.5)

    def test_zero_displacement_preserves_the_represented_matrix(self) -> None:
        factors = self.factors(3)
        left, core, right = factors
        updated_left, updated_core, updated_right = transport_factors(
            self.model, self.source, torch.zeros_like(self.source), factors,
            n_steps=self.steps,
        )
        torch.testing.assert_close(updated_left @ updated_core @ updated_right.T,
                                   left @ core @ right.T, atol=2e-12, rtol=2e-11)

    def test_nearly_singular_core_needs_no_inverse(self) -> None:
        left, core, right = self.factors(3)
        core[-1, -1] = 1e-14
        updated_left, updated_core, updated_right = transport_factors(
            self.model, self.source, self.displacement, (left, core, right),
            n_steps=self.steps,
        )
        self.assertTrue(bool(torch.isfinite(updated_core).all()))
        identity = torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(updated_left.T @ updated_left, identity)
        torch.testing.assert_close(updated_right.T @ updated_right, identity)


if __name__ == "__main__":
    unittest.main()
