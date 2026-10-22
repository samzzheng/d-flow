"""Geometric and inexact-gradient guarantees of the experimental direction cap."""
import unittest

import torch

from experiments.ct_mri_sweep.optimizer import SweepLBFGS
from experiments.separated_shapes_117k.optimizer_controls import CapStartedLBFGS, safeguard_direction


class DirectionSafeguardTests(unittest.TestCase):
    def test_cap_started_preserves_ray_and_first_attempt(self) -> None:
        source = torch.ones(2, dtype=torch.float64, requires_grad=True)
        standard = SweepLBFGS(source, lambda z: float(.5*z.square().sum()), project=False, lr=.7, step_cap=.5)
        changed = CapStartedLBFGS(source, standard.objective, project=False, lr=.7, step_cap=.5)
        for optimizer in (standard, changed):
            optimizer.g = source.detach().clone()
            optimizer.S = [torch.tensor([1e-4, 0.], dtype=torch.float64)]
            optimizer.Y = [torch.tensor([1., 0.], dtype=torch.float64)]
        self.assertTrue(torch.equal(changed._direction(), standard._direction()))
        changed.step_attempts = 10
        proposal, reference = changed._direction(), standard._direction()
        torch.testing.assert_close(proposal*changed.lr, .5*reference/reference.norm())

    def test_cap_started_recovers_from_tiny_history_scale(self) -> None:
        values = []
        for kind in (SweepLBFGS, CapStartedLBFGS):
            source = torch.ones(2, dtype=torch.float64, requires_grad=True)
            optimizer = kind(source, lambda z: float(.5*z.square().sum()), project=False, step_cap=.5)
            optimizer.g = source.detach().clone()
            optimizer.f = 1.
            optimizer.state = {"initialized": True}
            optimizer.step_attempts = 10
            optimizer.S = [torch.tensor([1e-4, 0.], dtype=torch.float64)]
            optimizer.Y = [torch.tensor([1., 0.], dtype=torch.float64)]

            def closure() -> torch.Tensor:
                source.grad = source.detach().clone()
                return .5*source.detach().square().sum()

            optimizer.step(closure)
            self.assertEqual(optimizer.consecutive_failures, 0)
            self.assertLessEqual(float((source.detach()-1).norm()), .5+1e-12)
            values.append(float(.5*source.detach().square().sum()))
        self.assertLess(values[1], values[0]-.1)

    def test_length_angle_and_bounded_gradient_error(self) -> None:
        generator = torch.Generator().manual_seed(121)
        for threshold in (.05, .1, .25):
            for _ in range(100):
                gradient = torch.randn(50, generator=generator, dtype=torch.float64)
                direction = torch.randn(50, generator=generator, dtype=torch.float64)
                result = safeguard_direction(direction, gradient, threshold)
                torch.testing.assert_close(result.norm(), direction.norm(), atol=1e-12, rtol=1e-12)
                cosine = -result.dot(gradient)/(result.norm()*gradient.norm())
                self.assertGreaterEqual(float(cosine), threshold-1e-12)
                if -direction.dot(gradient) >= threshold*direction.norm()*gradient.norm():
                    self.assertTrue(torch.equal(result, direction))
                error = torch.randn(50, generator=generator, dtype=torch.float64)
                error *= .99*threshold*gradient.norm()/error.norm()
                self.assertLess(float((gradient+error).dot(result)), 0.)

    def test_degenerate_and_antiparallel_directions(self) -> None:
        gradient = torch.tensor([2., 0.], dtype=torch.float64)
        torch.testing.assert_close(safeguard_direction(gradient, gradient, .1), -gradient)
        for direction, g in ((torch.zeros_like(gradient), gradient),
                             (gradient, torch.zeros_like(gradient))):
            self.assertTrue(torch.equal(safeguard_direction(direction, g, .1), direction))


if __name__ == "__main__":
    unittest.main()
