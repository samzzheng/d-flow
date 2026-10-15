"""Geometry and rasterization checks for the separated-shapes PDF recipe."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from dataset.separated_shapes import Shape, _rasterize_binary, rasterize, sample_geometry


class SeparatedShapesTests(unittest.TestCase):
    def test_continuous_constraints_and_triangle_construction(self) -> None:
        counts: set[int] = set()
        kinds: set[str] = set()
        for seed in range(1000):
            shapes = sample_geometry(seed)
            counts.add(len(shapes))
            for i, shape in enumerate(shapes):
                kinds.add(shape.kind)
                self.assertTrue(np.all(np.abs(shape.center) + shape.enclosing_radius <= 1))
                for other in shapes[:i]:
                    distance = np.linalg.norm(np.subtract(shape.center, other.center))
                    self.assertGreaterEqual(distance, shape.enclosing_radius + other.enclosing_radius + 0.05)
                if shape.kind == "disk":
                    self.assertTrue(0.10 <= shape.radius <= 0.20)
                else:
                    vertices = np.array(shape.vertices)
                    np.testing.assert_allclose(vertices.mean(axis=0), shape.center, atol=1e-15)
                    lengths = np.linalg.norm(vertices[[1, 0, 0]] - vertices[[2, 2, 1]], axis=1)
                    np.testing.assert_allclose(lengths, shape.side_lengths, atol=1e-15)
                    self.assertTrue(np.all((lengths >= 0.20) & (lengths <= 0.30)))
                    self.assertTrue(0 <= shape.orientation < 2 * np.pi)
                    a, b, c = lengths
                    cosines = np.array([(b*b+c*c-a*a)/(2*b*c),
                                        (a*a+c*c-b*b)/(2*a*c),
                                        (a*a+b*b-c*c)/(2*a*b)])
                    self.assertTrue(np.all((cosines >= -1/8 - 1e-14) & (cosines <= 7/9 + 1e-14)))
                    self.assertAlmostEqual(shape.enclosing_radius,
                                           np.linalg.norm(vertices - shape.center, axis=1).max())
        self.assertEqual(counts, {1, 2, 3, 4})
        self.assertEqual(kinds, {"disk", "triangle"})

    def test_rejection_retries_only_centers_and_cap_raises(self) -> None:
        rng = Mock()
        rng.integers.side_effect = [1, 0]
        rng.uniform.side_effect = [0.15, np.array([[0.99, 0.99]]), np.array([[0.0, 0.0]])]
        with patch("dataset.separated_shapes.np.random.default_rng", return_value=rng):
            shapes = sample_geometry(42, max_attempts=2)
        self.assertEqual(shapes, [Shape("disk", (0.0, 0.0), 0.15, radius=0.15)])
        self.assertEqual(rng.integers.call_count, 2)
        self.assertEqual(rng.uniform.call_count, 3)
        rng.reset_mock(side_effect=True)
        rng.integers.side_effect = [1, 0]
        rng.uniform.side_effect = [0.15, np.array([[0.99, 0.99]])]
        with patch("dataset.separated_shapes.np.random.default_rng", return_value=rng):
            with self.assertRaisesRegex(RuntimeError, "Shapes were not redrawn"):
                sample_geometry(42, max_attempts=1)

    def test_pixel_centers_and_filled_triangle_not_enclosing_disk(self) -> None:
        disk = Shape("disk", (0.25, -0.25), 0.26, radius=0.26)
        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[1, 2] = 1
        np.testing.assert_array_equal(_rasterize_binary([disk], 4), expected)
        triangle = Shape("triangle", (0.0, 0.0), np.sqrt(2),
                         vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)))
        # Interior is x + y <= 0, including its diagonal boundary.
        expected = (np.indices((4, 4)).sum(axis=0) <= 3).astype(np.uint8)
        np.testing.assert_array_equal(_rasterize_binary([triangle], 4), expected)

    def test_seed_reproducibility_and_raster_area(self) -> None:
        for seed in range(20):
            shapes = sample_geometry(seed)
            self.assertEqual(shapes, sample_geometry(seed))
            image = rasterize(shapes)
            self.assertEqual(image.dtype, np.float32)
            self.assertEqual((image.min(), image.max()), (0, 1))
            self.assertTrue(np.any((image > 0) & (image < 1)))
            area, perimeter = 0.0, 0.0
            for shape in shapes:
                if shape.kind == "disk":
                    area += np.pi * shape.radius ** 2
                    perimeter += 2 * np.pi * shape.radius
                else:
                    a, b, c = shape.side_lengths
                    s = (a + b + c) / 2
                    area += np.sqrt(s * (s-a) * (s-b) * (s-c))
                    perimeter += 2 * s
            # Pixel-center area error is confined to a boundary strip.
            pixel_width = 2 / image.shape[0]
            self.assertLess(abs(image.sum() * pixel_width**2 - area), perimeter * pixel_width)

    def test_antialiasing_preserves_interiors_and_estimates_pixel_coverage(self) -> None:
        # The line x+y=0 cuts each diagonal pixel exactly in half.
        triangle = Shape("triangle", (0.0, 0.0), np.sqrt(2),
                         vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)))
        indices = np.indices((4, 4)).sum(axis=0)
        expected = np.where(indices < 3, 1.0, np.where(indices == 3, 0.5, 0.0))
        coarse = rasterize([triangle], 4)
        refined = rasterize([triangle], 16).reshape(4, 4, 4, 4).mean(axis=(1, 3))
        self.assertEqual(coarse.dtype, np.float32)
        # Subpixel centers on the edge are included, with error at most 1/(2s).
        np.testing.assert_allclose(coarse, expected, atol=1 / 16)
        np.testing.assert_allclose(refined, expected, atol=1 / 64)
        self.assertLess(np.linalg.norm(refined - expected), np.linalg.norm(coarse - expected))
        for seed in range(10):
            shapes = sample_geometry(seed)
            coverage = rasterize(shapes, 64)
            self.assertTrue(np.all((coverage >= 0) & (coverage <= 1)))
            self.assertTrue(np.any((coverage > 0) & (coverage < 1)))
            self.assertEqual(shapes, sample_geometry(seed))


if __name__ == "__main__":
    unittest.main()
