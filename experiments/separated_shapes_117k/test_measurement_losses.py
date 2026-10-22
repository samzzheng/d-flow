"""Complex gradients, filtering bounds and evaluator consistency on CPU."""
from copy import deepcopy
import unittest

import torch

from experiments.ct_mri_sweep.operators import MRI
from experiments.ct_mri_sweep.run import Evaluator
from experiments.separated_shapes_117k.measurement_losses import LossEvaluator, measurement_loss


class ToyFlow(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return .1 * x.sin() + .03 * t[:, None, None, None] * x


class LossTests(unittest.TestCase):
    def test_loss_gradients_and_filter_bounds(self) -> None:
        torch.manual_seed(7)
        for problem, dtype in (("ct", torch.float64), ("mri", torch.complex128)):
            residual = torch.randn(1, 1, 8, 8, dtype=dtype, requires_grad=True)
            ordinary = measurement_loss(residual, problem, "l2")
            for kind, sigma in (("coarse", 0.), ("coarse", 1.1), ("intensity", 0.)):
                with self.subTest(problem=problem, kind=kind, sigma=sigma):
                    def loss(x: torch.Tensor) -> torch.Tensor:
                        return measurement_loss(x, problem, kind, sigma)
                    self.assertTrue(torch.autograd.gradcheck(loss, (residual,), fast_mode=True))
                    value = loss(residual)
                    if kind == "coarse":
                        self.assertGreaterEqual(float(value), .1 * float(ordinary))
                        self.assertLessEqual(float(value), float(ordinary) * (1 + 1e-14))
                        if sigma == 0:
                            torch.testing.assert_close(value, ordinary, rtol=0, atol=0)
                    else:
                        self.assertGreaterEqual(float(value), float(ordinary))
            if problem == "mri":
                for kind in ("l2", "coarse", "intensity"):
                    torch.testing.assert_close(measurement_loss(residual, problem, kind, 1.),
                                               measurement_loss(1j * residual, problem, kind, 1.))

    def test_exact_and_terminal_gradients_use_same_loss(self) -> None:
        torch.manual_seed(10)
        model = ToyFlow()
        operator = MRI(8, 4, 12, torch.device("cpu"))
        source = torch.randn(1, 1, 8, 8, dtype=torch.float64, requires_grad=True)
        observed = operator(torch.randn_like(source))
        direction = torch.randn_like(source)
        direction /= direction.norm()
        for kind, sigma in (("l2", 0.), ("coarse", 1.), ("intensity", 0.)):
            settings = {"flow_steps": 5, "measurement_loss": {"kind": kind, "sigma": sigma}}
            evaluator = LossEvaluator(model, operator, observed, settings, 1)
            evaluator.method = "D-Flow"
            exact = evaluator.gradient(source, "D-Flow")
            repeated = evaluator.gradient(source, "D-Flow")
            torch.testing.assert_close(exact, repeated)
            finite = (evaluator.objective(source + 1e-5 * direction)
                      - evaluator.objective(source - 1e-5 * direction)) / 2e-5
            self.assertAlmostEqual(finite, float((exact * direction).sum()), places=7)
            evaluator.method = "P-Flow"
            proxy = evaluator.gradient(source, "P-Flow")
            endpoint = evaluator.image.detach().requires_grad_(True)
            reference, = torch.autograd.grad(evaluator.loss(operator(endpoint)), endpoint)
            torch.testing.assert_close(proxy, reference)
            if kind == "l2":
                original = Evaluator(model, operator, observed, deepcopy(settings), 1)
                original.method = "D-Flow"
                torch.testing.assert_close(original.gradient(source, "D-Flow"), exact, rtol=0, atol=0)
                original.method = "P-Flow"
                torch.testing.assert_close(original.gradient(source, "P-Flow"), proxy, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
