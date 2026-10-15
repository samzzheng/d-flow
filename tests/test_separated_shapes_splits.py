"""Ensure raster collisions cannot cross the final dataset split boundary."""

from collections import Counter
import unittest

import numpy as np

from dataset.generate_separated_shapes import separate_duplicate_images


class SeparatedShapesSplitTests(unittest.TestCase):
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
