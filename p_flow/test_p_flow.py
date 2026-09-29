"""Check the identity proxy and spherical update independently of learned models."""

from __future__ import annotations

import unittest

import torch

from .p_flow import project_source, proxy_gradient, step


class LinearFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
        self.grad_modes: list[bool] = []

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        self.grad_modes.append(torch.is_grad_enabled())
        return self.weight * x


class PFlowTests(unittest.TestCase):
    def test_one_step_is_endpoint_gradient_not_source_gradient(self) -> None:
        model = LinearFlow()
        z = torch.tensor([1., -2., 3., .5], dtype=torch.float64).reshape(1, 1, 2, 2)
        matrix = torch.tensor([[1., 2., 0., 0.], [0., 0., 3., -1.]], dtype=z.dtype)
        y = torch.tensor([[.5, -.2]], dtype=z.dtype)
        operator = lambda x: x.flatten(1) @ matrix.T
        actual, (endpoint, loss, gradient) = step(model, z, operator, y, .02, n_steps=1)
        residual = operator(2 * z) - y
        expected_gradient = (residual @ matrix).reshape_as(z)
        torch.testing.assert_close(endpoint, 2 * z)
        torch.testing.assert_close(gradient, expected_gradient)
        torch.testing.assert_close(loss, .5 * residual.square().sum())
        torch.testing.assert_close(actual, project_source(z - .02 * expected_gradient))
        self.assertFalse(torch.allclose(gradient, 2 * expected_gradient))
        self.assertIsNone(model.weight.grad)
        self.assertFalse(endpoint.requires_grad)
        self.assertEqual(model.grad_modes, [False])

    def test_shell_projection_is_per_sample_and_idempotent(self) -> None:
        z = torch.randn(3, 2, 4, 5, dtype=torch.float64)
        projected = project_source(z)
        torch.testing.assert_close(projected.flatten(1).norm(dim=1), torch.full((3,), 40 ** .5, dtype=z.dtype))
        torch.testing.assert_close(project_source(projected), projected)
        with self.assertRaises(FloatingPointError):
            project_source(torch.zeros_like(z))

    def test_proxy_still_works_in_no_grad_context(self) -> None:
        model = LinearFlow()
        z = torch.ones(1, 1, 2, 2, dtype=torch.float64)
        with torch.no_grad():
            endpoint, _, gradient = proxy_gradient(model, z, lambda x: x, torch.zeros_like(z), 2)
        torch.testing.assert_close(endpoint, 2.25 * z)
        torch.testing.assert_close(gradient, endpoint)
        self.assertIsNone(model.weight.grad)


if __name__ == "__main__":
    unittest.main()
