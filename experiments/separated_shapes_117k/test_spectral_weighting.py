"""Algebraic invariants for terminal weighting, without GPU/model execution."""
import unittest

import torch

from experiments.separated_shapes_117k.spectral_weighting import median_capped_core


class SpectralWeightingTests(unittest.TestCase):
    def test_cap_and_coordinate_invariance(self) -> None:
        generator = torch.Generator().manual_seed(12)
        core = torch.randn(6, 6, generator=generator, dtype=torch.float64)
        original = core.clone()
        capped, singular = median_capped_core(core)
        torch.testing.assert_close(core, original, rtol=0, atol=0)
        torch.testing.assert_close(torch.linalg.svdvals(capped), singular.clamp(max=singular.median()))
        q_left = torch.linalg.qr(torch.randn(6, 6, generator=generator, dtype=torch.float64)).Q
        q_right = torch.linalg.qr(torch.randn(6, 6, generator=generator, dtype=torch.float64)).Q
        changed, _ = median_capped_core(q_left.T@core@q_right)
        torch.testing.assert_close(q_left@changed@q_right.T, capped, rtol=1e-12, atol=1e-12)

    def test_gradient_subspace_alignment_and_nonexpansion(self) -> None:
        generator = torch.Generator().manual_seed(17)
        left = torch.linalg.qr(torch.randn(23, 6, generator=generator, dtype=torch.float64)).Q
        right = torch.linalg.qr(torch.randn(23, 6, generator=generator, dtype=torch.float64)).Q
        core = torch.randn(6, 6, generator=generator, dtype=torch.float64)
        capped, _ = median_capped_core(core)
        terminal = torch.randn(23, generator=generator, dtype=torch.float64)
        original = right@core.T@left.T@terminal
        gradient = right@capped.T@left.T@terminal
        torch.testing.assert_close(right@right.T@gradient, gradient)
        self.assertLessEqual(float(gradient.norm()), float(original.norm())+1e-12)
        self.assertGreater(float(original.dot(gradient)), 0)
        matched = gradient*(original.norm()/gradient.norm())
        torch.testing.assert_close(matched.norm(), original.norm())


if __name__ == "__main__":
    unittest.main()
