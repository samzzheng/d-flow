"""Geometric invariants for the offline template-similarity diagnostic."""
import unittest

import numpy as np
import torch

from scripts.audit_shape_templates import component_templates, nearest_iou


class ShapeTemplateTests(unittest.TestCase):
    def test_integer_translation_preserves_centered_components(self) -> None:
        for height in (4, 5, 6, 7):
            images = torch.zeros(9, 1, 64, 64)
            for index, (dy, dx) in enumerate((dy, dx) for dy in range(3) for dx in range(3)):
                images[index, 0, 20 + dy:20 + dy + height, 20 + dx:26 + dx] = 1
            components = component_templates(images)
            np.testing.assert_array_equal(components[0], np.repeat(components[0][:1], 9, axis=0))
            scores, _ = nearest_iou(components, components)
            np.testing.assert_array_equal(scores, np.ones(9))

    def test_size_and_cropped_area_remain_in_iou(self) -> None:
        small = torch.zeros(1, 1, 64, 64)
        small[0, 0, 20:26, 20:26] = 1
        templates = component_templates(small)
        oversized = component_templates(torch.ones(1, 1, 64, 64))
        self.assertEqual(oversized[1][0], 4096)
        self.assertEqual(oversized[0].sum(), 1024)
        scores, _ = nearest_iou(oversized, templates)
        self.assertAlmostEqual(float(scores[0]), 36 / 4096)


if __name__ == '__main__':
    unittest.main()
