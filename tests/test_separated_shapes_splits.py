"""Ensure raster collisions cannot cross the final dataset split boundary."""

from collections import Counter
import unittest

import numpy as np

from dataset.generate_separated_shapes import generate_batch, separate_duplicate_images
from dataset.separated_shapes import rasterize, sample_geometry


class SeparatedShapesSplitTests(unittest.TestCase):
    def test_grayscale_coverage_survives_storage_and_duplicate_detection(self) -> None:
        images, _ = generate_batch((42, 2, 64, 8))
        expected = rasterize(sample_geometry(42), 64, supersample=8)
        np.testing.assert_allclose(images[0, 0].astype(float) / 255, expected, atol=0.5 / 255)
        self.assertTrue(np.any((images > 0) & (images < 255)))
        # Equal support with distinct coverage must not count as a duplicate.
        splits = {"train": np.array([[[[128]]]], dtype=np.uint8),
                  "test": np.array([[[[255]]]], dtype=np.uint8)}
        order, swaps = separate_duplicate_images(splits)
        self.assertEqual(swaps, 0)
        np.testing.assert_array_equal(order, [0, 1])

    def test_duplicate_groups_preserve_population_sizes_and_metadata_order(self) -> None:
        # Repeated masks within and across all three splits, plus unique donors.
        labels = {"train": [1, 1, 2, 3, 4, 5], "val": [1, 6, 7, 8], "test": [1, 6, 9]}
        splits = {name: np.array([[(value >> bit) & 1 for bit in range(4)] for value in values],
                                dtype=np.uint8).reshape(-1, 1, 2, 2) * 255
                  for name, values in labels.items()}
        original = np.concatenate(list(splits.values())).copy()
        order, swaps = separate_duplicate_images(splits)
        self.assertEqual(swaps, 3)
        np.testing.assert_array_equal(np.concatenate(list(splits.values())), original[order])
        self.assertEqual(Counter(order.tolist()), Counter(range(len(original))))
        self.assertEqual({name: len(images) for name, images in splits.items()},
                         {name: len(values) for name, values in labels.items()})
        masks = [set(image.tobytes() for image in images) for images in splits.values()]
        for i, group in enumerate(masks):
            for other in masks[:i]:
                self.assertTrue(group.isdisjoint(other))
        # Once isolated, a second pass leaves both ordering and arrays alone.
        order, swaps = separate_duplicate_images(splits)
        self.assertEqual(swaps, 0)
        np.testing.assert_array_equal(order, np.arange(len(original)))


if __name__ == "__main__":
    unittest.main()
