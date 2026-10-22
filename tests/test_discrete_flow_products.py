"""The audit differentiates the complete discrete map in the correct order."""
import unittest

import torch

from experiments.separated_shapes_117k.gradient_audit import DiscreteProducts
from experiments.separated_shapes_117k.ksl_stability import stable_sl_step
from experiments.separated_shapes_117k.optimizer_controls import HistoryFallbackLBFGS
from sampler.ode_samplers import euler_flow


class NonlinearVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("matrix", torch.randn(9, 9, dtype=torch.float64) / 3)

    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        values = image.flatten(1)
        return (torch.tanh(values @ self.matrix.T) + time[:, None] * values.square()).reshape_as(image)


class DiscreteProductTests(unittest.TestCase):
    def test_history_recovery_preserves_armijo_and_changes_rejected_direction(self) -> None:
        source = torch.tensor([1., 0.], dtype=torch.float64, requires_grad=True)
        objective = lambda value: float(.5*value[0].square() + .1*value[1].square())
        optimizer = HistoryFallbackLBFGS(source, objective, project=False, max_ls=12)
        optimizer.state["initialized"] = True
        optimizer.f = objective(source.detach())
        optimizer.g = torch.tensor([1., -2.], dtype=source.dtype)
        optimizer.S = [torch.tensor([1., 0.], dtype=source.dtype), torch.tensor([0., 1.], dtype=source.dtype)]
        optimizer.Y = [torch.tensor([1., 0.], dtype=source.dtype), torch.tensor([-.9, 1.], dtype=source.dtype)]

        def closure() -> torch.Tensor:
            source.grad = torch.tensor([1., -2.], dtype=source.dtype)
            return source.new_tensor(objective(source.detach()))

        before = source.detach().clone()
        optimizer.step(closure)
        torch.testing.assert_close(source, before)
        self.assertEqual(optimizer.history_backtracks, 1)
        self.assertEqual(optimizer.failed_searches, 1)
        optimizer.step(closure)
        self.assertLess(objective(source.detach()), objective(before))
        self.assertEqual(optimizer.step_attempts, 2)

    def test_stable_sl_matches_resolved_substeps_and_euler_recompression(self) -> None:
        torch.manual_seed(19)
        dimension, rank, h, alpha = 12, 4, .2, .6
        matrix = torch.randn(dimension, dimension, dtype=torch.float64) - 5 * torch.eye(dimension)
        U, _ = torch.linalg.qr(torch.randn(dimension, rank, dtype=torch.float64))
        V, _ = torch.linalg.qr(torch.randn(dimension, rank, dtype=torch.float64))
        S = torch.randn(rank, rank, dtype=torch.float64)
        empty = U[:, :0]
        (U1, actual_core, V1), _, _ = stable_sl_step(
            lambda value: matrix @ value, lambda value: matrix.T @ value,
            (U, S, V), empty, empty, h, alpha)
        K1 = U @ S + h * (matrix @ (alpha * V + U @ S) + V)
        M, W = U1.T @ matrix @ U1, alpha * matrix.T @ U1 + U1
        core = U1.T @ K1
        C = W.T @ V
        # Independently integrate S and then L with fine RK4 substeps.
        dt = h / 400
        for _ in range(400):
            k1 = -M @ core - C
            k2 = -M @ (core + dt * k1 / 2) - C
            k3 = -M @ (core + dt * k2 / 2) - C
            k4 = -M @ (core + dt * k3) - C
            core = core + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        L = V @ core.T
        for _ in range(400):
            k1 = L @ M.T + W
            k2 = (L + dt * k1 / 2) @ M.T + W
            k3 = (L + dt * k2 / 2) @ M.T + W
            k4 = (L + dt * k3) @ M.T + W
            L = L + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        reference_V, _ = torch.linalg.qr(L)
        torch.testing.assert_close(V1 @ V1.T, reference_V @ reference_V.T, atol=1e-10, rtol=1e-10)
        target = U @ S @ V.T
        identity = torch.eye(dimension, dtype=target.dtype)
        target = target + h * (matrix @ (alpha * identity + target) + identity)
        torch.testing.assert_close(actual_core, U1.T @ target @ V1, atol=1e-10, rtol=1e-10)

    def test_native_products_match_dense_jacobian_and_finite_difference(self) -> None:
        torch.manual_seed(207)
        model = NonlinearVelocity()
        source = .1 * torch.randn(1, 1, 3, 3, dtype=torch.float64)
        products = DiscreteProducts(model, source, steps=5)
        dense = torch.func.jacrev(lambda value: euler_flow(model, value, 5))(source).reshape(9, 9)
        directions = torch.randn(9, 7, dtype=torch.float64)  # Exercises two chunks.
        torch.testing.assert_close(products.apply(directions), dense @ directions)
        torch.testing.assert_close(products.apply(directions, transpose=True), dense.T @ directions)
        torch.testing.assert_close(products.image, euler_flow(model, source, 5), rtol=0, atol=0)
        direction = directions[:, :1].reshape_as(source)
        finite_difference = (euler_flow(model, source + 1e-6 * direction, 5)
                             - euler_flow(model, source - 1e-6 * direction, 5)) / 2e-6
        torch.testing.assert_close(products.apply(directions[:, :1]).reshape_as(source),
                                   finite_difference, rtol=1e-8, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
