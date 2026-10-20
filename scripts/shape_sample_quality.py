"""Descriptive morphology checks; these do not certify valid disks/triangles."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
import torch


def shape_statistics(samples: torch.Tensor) -> dict:
    """Compare raw generated images with identically processed real images."""
    images = samples.detach().cpu().float().numpy()[:, 0]
    if not np.isfinite(images).all():
        raise ValueError('Nonfinite generated images')
    outside = np.maximum(-images, 0) + np.maximum(images - 1, 0)
    result: dict = {'samples': len(images), 'outside_unit_interval_rms':
                    np.sqrt(np.mean(outside**2, axis=(1, 2))).tolist(), 'thresholds': {}}
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for threshold in (0.25, 0.5, 0.75):
        rows = []
        for image in images:
            labels, count = ndimage.label(image > threshold, structure=np.ones((3, 3)))
            sizes = np.bincount(labels.ravel())[1:]
            # Four pixels removes specks while retaining the dataset's small triangles.
            selected = np.flatnonzero(sizes >= 4) + 1
            solidities = []
            distances = []
            for position, label in enumerate(selected):
                points = np.argwhere(labels == label)
                hull = ConvexHull((points[:, None, :] + corners).reshape(-1, 2))
                solidities.append(len(points) / hull.volume)
                if position + 1 < len(selected):
                    distance = ndimage.distance_transform_edt(labels != label)
                    distances.append(float(distance[np.isin(labels, selected[position + 1:])].min()))
            rows.append(dict(components=len(selected), specks=int(count - len(selected)),
                             foreground_fraction=float(np.mean(image > threshold)),
                             minimum_solidity=min(solidities) if solidities else None,
                             minimum_component_distance_pixels=min(distances) if distances else None))
        result['thresholds'][str(threshold)] = {
            'component_count_histogram': {str(count): sum(row['components'] == count for row in rows)
                                          for count in range(5)},
            'more_than_four_components': sum(row['components'] > 4 for row in rows),
            'samples': rows,
        }
    return result


def compare_shape_statistics(reference: torch.Tensor, generated: dict[str, torch.Tensor]) -> dict:
    """Keep reference and generated morphology together, without a synthetic pass score."""
    return {
        'protocol': '8-connected thresholded components; retain area >=4 pixels; '
                    'solidity is pixel area divided by the convex hull of pixel corners; '
                    'distance is nearest foreground pixel-center distance between components.',
        'limitation': 'Unconditional samples have no target object count. Matching count, '
                      'spacing, or convexity does not certify circles/triangles, correct sizes, '
                      'or separation of touching objects. Inspect the paired sample panels too.',
        'reference': shape_statistics(reference),
        'generated': {name: shape_statistics(samples) for name, samples in generated.items()},
    }
