"""Geometric and inexact-gradient guarantees of the experimental direction cap."""
import unittest

import torch

from experiments.separated_shapes_117k.optimizer_controls import safeguard_direction


class DirectionSafeguardTests(unittest.TestCase):
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
