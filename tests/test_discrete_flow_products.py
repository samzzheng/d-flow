"""The audit differentiates the complete discrete map in the correct order."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from functools import partial

import torch

from experiments.separated_shapes_117k.gradient_audit import DiscreteProducts
from experiments.separated_shapes_117k.ksl_stability import increment_ksl_step, stable_sl_step, refined_increment_step, pure_ksl_step, subdivided_increment_step
from experiments.separated_shapes_117k.optimizer_controls import HistoryFallbackLBFGS, ScaleRecoveryLBFGS
from experiments.separated_shapes_117k.scale_recovery import FrozenSeedEvaluator
from experiments.ct_mri_sweep import run
from sampler.ode_samplers import euler_flow
from experiments.ct_mri_sweep.validation import ChunkedAutodiffProducts
from experiments.separated_shapes_117k.exact_suffix import suffix_gradient, task_terminal_factors, task_gradient
from dlr_dflow.api import sensitivity_factors
from dlr_dflow.implementation.enriched import _thin_svd
from dlr_dflow.implementation import residual_startup
from experiments.separated_shapes_117k.sketch_initialization import retained_seed_startup


class NonlinearVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("matrix", torch.randn(9, 9, dtype=torch.float64) / 3)

    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        values = image.flatten(1)
        return (torch.tanh(values @ self.matrix.T) + time[:, None] * values.square()).reshape_as(image)


class DiscreteProductTests(unittest.TestCase):
    def test_retained_seed_startup_keeps_direction_and_dense_residual_action(self) -> None:
        torch.manual_seed(92)
        model = NonlinearVelocity()
        source = .1*torch.randn(1, 1, 3, 3, dtype=torch.float64)
        direction = torch.randn(9, 1, dtype=source.dtype)
        direction /= direction.norm()
        products = ChunkedAutodiffProducts(model)
        time = source.new_tensor([0.])
        matrix = torch.func.jacrev(lambda value: model(value, time))(source).reshape(9, 9)
        target = .2*(matrix+torch.eye(9, dtype=source.dtype))
        for rank in (1, 3, 9):
            image, (left, core, right), completed = retained_seed_startup(
                model, source, 5, rank, 2, 1, torch.Generator().manual_seed(7),
                products=products, seed_basis=direction)
            torch.testing.assert_close(right @ (right.T @ direction), direction, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(left @ core @ right.T, target @ right @ right.T, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(image, source+.2*model(source,time), atol=1e-11, rtol=1e-11)
            self.assertEqual(completed, 1)
            self.assertEqual(core.shape, (rank,rank))
            if rank == 9:
                torch.testing.assert_close(left @ core @ right.T, target, atol=1e-11, rtol=1e-11)

    def test_startup_control_reproduces_existing_evolution(self) -> None:
        torch.manual_seed(93)
        model = NonlinearVelocity()
        source = .1*torch.randn(1, 1, 3, 3, dtype=torch.float64)
        seed = torch.randn(9, 2, dtype=source.dtype)
        options = dict(rank=3, trace_probes=0, enrichment_probes=0,
                       initialization_oversampling=2, seed=14, seed_basis=seed)
        ordinary_image, ordinary, ordinary_alpha = sensitivity_factors(
            model, source, 5, products=ChunkedAutodiffProducts(model), **options)
        with patch.object(residual_startup, "residual_sketch_startup", partial(retained_seed_startup, retain_seed=False)):
            image, actual, alpha = sensitivity_factors(model, source, 5,
                products=ChunkedAutodiffProducts(model), residual_sketch_steps=1, **options)
        torch.testing.assert_close(image, ordinary_image, atol=0, rtol=0)
        torch.testing.assert_close(actual[0] @ actual[1] @ actual[2].T,
                                   ordinary[0] @ ordinary[1] @ ordinary[2].T, atol=1e-11, rtol=1e-11)
        self.assertEqual(alpha, ordinary_alpha)

    def test_notes_ksl_matches_three_dense_euler_stages_without_recompression(self) -> None:
        torch.manual_seed(88)
        d, h, alpha = 12, .2, .6
        matrix = torch.randn(d, d, dtype=torch.float64)
        identity = torch.eye(d, dtype=matrix.dtype)
        for rank in (3, d):
            U, _ = torch.linalg.qr(torch.randn(d, rank, dtype=matrix.dtype))
            V, _ = torch.linalg.qr(torch.randn(d, rank, dtype=matrix.dtype))
            S = torch.randn(rank, rank, dtype=matrix.dtype)
            Y = U @ S @ V.T
            K = U @ S + h*(matrix @ (alpha*identity + Y) + identity) @ V
            left, core = torch.linalg.qr(K)
            core = core - h*left.T @ (matrix @ (alpha*identity + left @ core @ V.T) + identity) @ V
            L = V @ core.T
            L = L + h*(matrix @ (alpha*identity + left @ L.T) + identity).T @ left
            right, triangular = torch.linalg.qr(L)
            factors, _, _ = pure_ksl_step(lambda q: matrix @ q, lambda q: matrix.T @ q,
                (U, S, V), U[:, :0], U[:, :0], h, alpha)
            actual = factors[0] @ factors[1] @ factors[2].T
            torch.testing.assert_close(actual, left @ triangular.T @ right.T, atol=1e-11, rtol=1e-10)
            self.assertEqual(factors[1].shape, (rank, rank))
            if rank == d:
                unsplit = Y + h*(matrix @ (alpha*identity+Y) + identity)
                self.assertGreater(float((actual-unsplit).norm()), 1e-3)

    def test_native_compressed_svd_reference_matches_dense_truncated_gradients(self) -> None:
        torch.manual_seed(81)
        model = NonlinearVelocity()
        source = .1*torch.randn(1, 1, 3, 3, dtype=torch.float64)
        products = DiscreteProducts(model, source, steps=5)
        dense = torch.func.jacrev(lambda value: euler_flow(model, value, 5))(source).reshape(9, 9)
        dense_left, dense_values, dense_right_t = torch.linalg.svd(dense)
        basis, _ = torch.linalg.qr(products.apply(torch.randn(9, 9, dtype=source.dtype)))
        right, values, coordinates = _thin_svd(products.apply(basis, transpose=True))
        left = basis @ coordinates.T
        torch.testing.assert_close(values, dense_values, atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(products.apply(right), left*values, atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(products.apply(left, transpose=True), right*values, atol=1e-11, rtol=1e-10)
        terminal = torch.randn(9, 1, dtype=source.dtype)
        for rank in (1, 3, 6):
            actual = right[:, :rank] @ (values[:rank, None]*(left[:, :rank].T @ terminal))
            expected = dense_right_t[:rank].T @ (dense_values[:rank, None]*(dense_left[:, :rank].T @ terminal))
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)

    def test_transported_goal_projections_match_exact_suffix_with_low_rank_factors(self) -> None:
        torch.manual_seed(76)
        model = NonlinearVelocity()
        source = .1*torch.randn(1, 1, 3, 3, dtype=torch.float64)
        terminal = torch.randn_like(source)
        options = dict(rank=3, trace_probes=0, enrichment_probes=0, initialization_oversampling=0)
        for count in (1, 2, 4):
            expected, _ = suffix_gradient(model, source, terminal, 5, count, options, 8,
                                          products=ChunkedAutodiffProducts(model))
            actual, basis = task_gradient(model, source, terminal, 5, count, options, 8,
                                         products=ChunkedAutodiffProducts(model))
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
            self.assertEqual(basis.shape, (9, 3))
            torch.testing.assert_close(basis.T @ basis, torch.eye(3, dtype=source.dtype), atol=1e-11, rtol=1e-11)

    def test_task_terminal_preserves_rank_and_exact_target_gradient_action(self) -> None:
        torch.manual_seed(71)
        d, h = 16, .2
        matrix = torch.randn(d, d, dtype=torch.float64)
        products = SimpleNamespace(A=lambda q: matrix @ q, AT=lambda q: matrix.T @ q)
        for rank in (1, 4, 16):
            U, _ = torch.linalg.qr(torch.randn(d, rank, dtype=matrix.dtype))
            V, _ = torch.linalg.qr(torch.randn(d, rank, dtype=matrix.dtype))
            S = torch.randn(rank, rank, dtype=matrix.dtype)
            terminal = torch.randn(d, 1, dtype=matrix.dtype)
            target = (torch.eye(d, dtype=matrix.dtype)+h*matrix) @ (h*torch.eye(d, dtype=matrix.dtype)+U @ S @ V.T)
            left, core, right = task_terminal_factors((U, S, V), terminal, h, products)
            self.assertEqual(core.shape, (rank, rank))
            torch.testing.assert_close(left.T @ left, torch.eye(rank, dtype=matrix.dtype), atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(right.T @ right, torch.eye(rank, dtype=matrix.dtype), atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(right @ core.T @ left.T @ terminal, target.T @ terminal, atol=1e-10, rtol=1e-10)
            if rank < d:
                self.assertGreater(float((left @ core @ right.T-target).norm()), 1e-3)

    def test_exact_suffix_uses_low_rank_prefix_instead_of_full_gradient(self) -> None:
        torch.manual_seed(61)
        model = NonlinearVelocity()
        source = .1*torch.randn(1, 1, 3, 3, dtype=torch.float64)
        terminal = torch.randn_like(source)
        options = dict(rank=3, trace_probes=0, enrichment_probes=0, initialization_oversampling=0)
        recorded = {}
        sensitivity_factors(model, source, 5, **options, seed=9,
            products=ChunkedAutodiffProducts(model),
            factor_callback=lambda step, alpha, factors: recorded.update({step: (alpha, factors)}))
        alpha, (U, S, V) = recorded[4]
        prefix_matrix = alpha*torch.eye(9, dtype=source.dtype)+U @ S @ V.T
        point = DiscreteProducts(model, source, 5).points[4]
        local = torch.func.jacrev(lambda value: value + .2*model(value, value.new_tensor([.8])))(point).reshape(9, 9)
        expected = prefix_matrix.T @ local.T @ terminal.flatten()
        actual, _ = suffix_gradient(model, source, terminal, 5, 1, options, 9,
                                    products=ChunkedAutodiffProducts(model))
        torch.testing.assert_close(actual.flatten(), expected, rtol=1e-10, atol=1e-11)
        exact = DiscreteProducts(model, source, 5).apply(terminal.reshape(-1, 1), transpose=True)
        self.assertGreater(float((actual.flatten()-exact.flatten()).norm()), 1e-5)

    def test_exact_suffix_full_rank_matches_native_gradient_and_stops_prefix(self) -> None:
        torch.manual_seed(52)
        model = NonlinearVelocity()
        source = .1*torch.randn(1, 1, 3, 3, dtype=torch.float64)
        terminal = torch.randn_like(source)
        variable = source.detach().requires_grad_(True)
        exact, = torch.autograd.grad(euler_flow(model, variable, 5), variable, terminal)
        for suffix_steps in (1, 4):
            products = ChunkedAutodiffProducts(model)
            with patch.object(products, "set_state", wraps=products.set_state) as calls:
                gradient, basis = suffix_gradient(model, source, terminal, 5, suffix_steps,
                    dict(rank=9, trace_probes=0, enrichment_probes=0, initialization_oversampling=0),
                    seed=7, products=products)
            torch.testing.assert_close(gradient, exact, rtol=1e-10, atol=1e-11)
            self.assertEqual(basis.shape, (9, 9))
            self.assertLessEqual(max(call.args[0] for call in calls.call_args_list), (4-suffix_steps)/5+1e-15)

    def test_refined_increment_preserves_target_and_reduces_projection_error(self) -> None:
        torch.manual_seed(49)
        dimension, h, alpha = 16, .2, .6
        matrix = torch.randn(dimension, dimension, dtype=torch.float64)
        for rank in (3, dimension):
            U, _ = torch.linalg.qr(torch.randn(dimension, rank, dtype=matrix.dtype))
            V, _ = torch.linalg.qr(torch.randn(dimension, rank, dtype=matrix.dtype))
            S = torch.randn(rank, rank, dtype=matrix.dtype)
            Y = U @ S @ V.T
            target = Y + h*(matrix @ (alpha*torch.eye(dimension, dtype=matrix.dtype)+Y)
                            + torch.eye(dimension, dtype=matrix.dtype))
            ordinary, _, _ = increment_ksl_step(lambda q: matrix @ q, lambda q: matrix.T @ q,
                (U, S, V), U[:, :0], U[:, :0], h, alpha)
            refined, _, _ = refined_increment_step(lambda q: matrix @ q, lambda q: matrix.T @ q,
                (U, S, V), U[:, :0], U[:, :0], h, alpha)
            left, core, right = refined
            torch.testing.assert_close(core, left.T @ target @ right, atol=1e-11, rtol=1e-11)
            before = (ordinary[0] @ ordinary[1] @ ordinary[2].T-target).norm()
            after = (left @ core @ right.T-target).norm()
            self.assertLessEqual(float(after), float(before)+1e-10)
            if rank == dimension:
                torch.testing.assert_close(left @ core @ right.T, target, atol=1e-11, rtol=1e-11)

    def test_subdivided_increment_converges_to_dense_projected_ode(self) -> None:
        torch.manual_seed(71)
        dimension, h, alpha = 7, .1, .6
        matrix = .2 * torch.randn(dimension, dimension, dtype=torch.float64)
        identity = torch.eye(dimension, dtype=matrix.dtype)
        for rank in (2, dimension):
            U, _ = torch.linalg.qr(torch.randn(dimension, rank, dtype=matrix.dtype))
            V, _ = torch.linalg.qr(torch.randn(dimension, rank, dtype=matrix.dtype))
            S = torch.diag(torch.linspace(1., 2., rank, dtype=matrix.dtype))
            Y = U @ S @ V.T
            forcing = matrix @ (alpha * identity + Y) + identity

            def field(value: torch.Tensor) -> torch.Tensor:
                left, _, right_t = torch.linalg.svd(value, full_matrices=False)
                left, right = left[:, :rank], right_t[:rank].T
                return forcing @ right @ right.T + left @ left.T @ forcing - left @ left.T @ forcing @ right @ right.T

            reference = Y.clone()
            delta = h / 256
            for _ in range(256):
                k1 = field(reference)
                k2 = field(reference + delta*k1/2)
                k3 = field(reference + delta*k2/2)
                k4 = field(reference + delta*k3)
                reference = reference + delta*(k1 + 2*k2 + 2*k3 + k4)/6
            errors = []
            for substeps in (1, 2, 4, 8):
                actual, _, _ = subdivided_increment_step(lambda value: matrix @ value,
                    lambda value: matrix.T @ value, (U, S, V), U[:, :0], U[:, :0], h, alpha,
                    substeps=substeps)
                reconstructed = actual[0] @ actual[1] @ actual[2].T
                errors.append(float((reconstructed-reference).norm()))
                if substeps == 1:
                    ordinary, _, _ = increment_ksl_step(lambda value: matrix @ value,
                        lambda value: matrix.T @ value, (U, S, V), U[:, :0], U[:, :0], h, alpha)
                    torch.testing.assert_close(reconstructed, ordinary[0] @ ordinary[1] @ ordinary[2].T,
                                               atol=1e-11, rtol=1e-11)
                if rank == dimension:
                    torch.testing.assert_close(reconstructed, Y+h*forcing, atol=1e-11, rtol=1e-11)
            if rank < dimension:
                self.assertTrue(all(after < .6*before for before, after in zip(errors, errors[1:])))

    def test_increment_ksl_matches_dense_frozen_field_and_full_rank_euler(self) -> None:
        torch.manual_seed(31)
        dimension, h, alpha = 12, .2, .6
        matrix = torch.randn(dimension, dimension, dtype=torch.float64)
        identity = torch.eye(dimension, dtype=matrix.dtype)
        for rank in (4, dimension):
            U, _ = torch.linalg.qr(torch.randn(dimension, rank, dtype=matrix.dtype))
            V, _ = torch.linalg.qr(torch.randn(dimension, rank, dtype=matrix.dtype))
            S = torch.randn(rank, rank, dtype=matrix.dtype)
            Y = U @ S @ V.T
            forcing = matrix @ (alpha * identity + Y) + identity
            expected_U, expected_S = torch.linalg.qr(U @ S + h * forcing @ V)
            expected_S = expected_S - h * expected_U.T @ forcing @ V
            expected_V, R_L = torch.linalg.qr(V @ expected_S.T + h * forcing.T @ expected_U)
            actual, _, _ = increment_ksl_step(lambda value: matrix @ value,
                lambda value: matrix.T @ value, (U, S, V), U[:, :0], U[:, :0], h, alpha)
            reconstructed = actual[0] @ actual[1] @ actual[2].T
            torch.testing.assert_close(reconstructed, expected_U @ R_L.T @ expected_V.T,
                                       atol=1e-11, rtol=1e-11)
            if rank == dimension:
                torch.testing.assert_close(reconstructed, Y + h * forcing, atol=1e-11, rtol=1e-11)

    def test_frozen_seed_does_not_accumulate_or_override_basis_reuse(self) -> None:
        seed = torch.tensor([[1.], [0.], [0.]])
        next_basis = torch.eye(3)[:, 1:]
        for reuse in (False, True):
            evaluator = object.__new__(FrozenSeedEvaluator)
            evaluator.settings = {"dlr_reuse_previous_v": reuse}
            evaluator.previous_v = next_basis.clone() if reuse else None
            evaluator.seed_direction = seed
            seen = []

            def base_gradient(current: run.Evaluator, source: torch.Tensor, method: str) -> torch.Tensor:
                seen.append(current.previous_v.clone())
                if current.settings["dlr_reuse_previous_v"]:
                    current.previous_v = next_basis.clone()
                return torch.zeros_like(source)

            with patch.object(run.Evaluator, "gradient", base_gradient):
                for _ in range(3):
                    evaluator.gradient(torch.ones(3), "DLR D-Flow")
            expected = torch.cat([seed, next_basis], dim=1) if reuse else seed
            for actual in seen:
                torch.testing.assert_close(actual, expected)
            if not reuse:
                self.assertIsNone(evaluator.previous_v)
            previous = evaluator.previous_v
            with patch.object(run.Evaluator, "gradient", side_effect=FloatingPointError("test")):
                with self.assertRaises(FloatingPointError):
                    evaluator.gradient(torch.ones(3), "DLR D-Flow")
            self.assertIs(evaluator.previous_v, previous)

    def test_scaled_recovery_escapes_an_unrepresentable_fallback_step(self) -> None:
        from experiments.ct_mri_sweep.optimizer import SweepLBFGS
        for kind in (SweepLBFGS, ScaleRecoveryLBFGS):
            source = torch.tensor([1.], requires_grad=True)
            objective = lambda value: float(1e-8 * value.square().sum())
            optimizer = kind(source, objective, lr=1., max_ls=30, step_cap=.1, project=False)

            def closure() -> torch.Tensor:
                loss = 1e-8 * source.square().sum()
                source.grad, = torch.autograd.grad(loss, source)
                return loss

            before = objective(source.detach())
            optimizer.step(closure)
            self.assertEqual(float(source), 1.)
            self.assertEqual(optimizer.consecutive_failures, 1)
            optimizer.step(closure)
            self.assertEqual(optimizer.step_attempts, 2)
            if kind is ScaleRecoveryLBFGS:
                self.assertLess(objective(source.detach()), before)
                self.assertAlmostEqual(float(source), .9, places=6)
                self.assertLessEqual(abs(float(source)-1.), .100001)
                self.assertEqual(optimizer.scaled_recoveries, 1)
                self.assertEqual(optimizer.consecutive_failures, 0)
            else:
                self.assertEqual(float(source), 1.)
                self.assertEqual(optimizer.consecutive_failures, 2)

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
        (stable_U, stable_S, stable_V), _, _ = stable_sl_step(
            lambda value: matrix @ value, lambda value: matrix.T @ value,
            (U, S, V), empty, empty, h, alpha, recompress=False)
        torch.testing.assert_close(stable_U @ stable_S @ stable_V.T, U1 @ L.T,
                                   atol=1e-10, rtol=1e-10)
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
