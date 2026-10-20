"""CPU morphology audit against held-out-calibrated real 64x64 shape templates.

Run with OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4.
This measures template similarity, not certified geometric validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
import torch

Components = tuple[NDArray[np.float32], NDArray[np.float32], list[dict[str, int]]]


def component_templates(samples: torch.Tensor) -> Components:
    """Center binary components without resizing, retaining their physical size."""
    patches, areas, records = [], [], []
    for sample, image in enumerate(samples[:, 0].numpy()):
        labels, count = ndimage.label(image > 0.5, structure=np.ones((3, 3)))
        for component in range(1, count + 1):
            points = np.argwhere(labels == component)
            if len(points) < 4:
                continue
            centered = points - np.floor(points.mean(axis=0) + 0.5).astype(int) + 16
            inside = np.all((centered >= 0) & (centered < 32), axis=1)
            patch = np.zeros((32, 32), dtype=np.float32)
            patch[tuple(centered[inside].T)] = 1
            patches.append(patch.ravel())
            # Keep the original area in the union even if the crop truncates a blob.
            areas.append(len(points))
            records.append(dict(sample=sample, component=component, area=len(points)))
    return (np.asarray(patches, dtype=np.float32).reshape(-1, 1024),
            np.asarray(areas, dtype=np.float32), records)


def nearest_iou(components: Components, templates: Components) -> tuple[NDArray, NDArray]:
    """Exact binary intersection-over-union with every centered training template."""
    patches, areas, _ = components
    scores = np.empty(len(patches), dtype=np.float32)
    matches = np.empty(len(patches), dtype=np.int64)
    for start in range(0, len(patches), 128):
        intersection = patches[start:start + 128] @ templates[0].T
        iou = intersection / (areas[start:start + 128, None] + templates[1] - intersection)
        indices = np.argmax(iou, axis=1)
        scores[start:start + 128] = iou[np.arange(len(indices)), indices]
        matches[start:start + 128] = indices
    return scores, matches


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparisons', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/separated-shapes-64.pt'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = torch.load(args.data, weights_only=True, mmap=True, map_location='cpu')
    if data['metadata']['resolution'] != 64:
        raise ValueError('This template audit is calibrated for resolution 64 only')
    templates = component_templates(data['train'][:4096].float() / 255)
    reference = component_templates(data['val'][:1024].float() / 255)
    reference_scores, _ = nearest_iou(reference, templates)
    cutoff = float(np.quantile(reference_scores, 0.01))
    reference_bad = {row['sample'] for row, score in zip(reference[2], reference_scores) if score < cutoff}
    for comparison in args.comparisons:
        provenance = json.loads((comparison / 'baseline_comparison.json').read_text())
        report: dict = dict(
            protocol='Threshold0.5, 8-connected components of area>=4 pixels; center by rounded '
                     'centroid without scaling/rotation. Best binary IoU against components from '
                     'first4096 training images. Lower1% cutoff calibrated on first1024 validation images.',
            limitation='A finite-template similarity outlier is not a certified invalid shape. '
                       'Count, relative spacing, sub-threshold fragments, intensity, and boundary '
                       'placement require separate checks. No reconstruction-accuracy claim.',
            dataset=str(args.data), training_templates=len(templates[0]),
            reference_components=len(reference[0]), cutoff=cutoff,
            reference_iou_quantiles=np.quantile(reference_scores, [0.01, 0.1, 0.5]).tolist(),
            reference_images_with_outlier=len(reference_bad), reference_images=1024,
            source_sha256=provenance['source_sha256'], models=provenance['models'], generated={})
        generated = torch.load(comparison / 'eval/capacity_comparison_samples.pt', weights_only=True)
        for name, images in generated.items():
            components = component_templates(images)
            scores, matches = nearest_iou(components, templates)
            outliers = sorted({row['sample'] for row, score in zip(components[2], scores) if score < cutoff})
            report['generated'][name] = dict(
                samples=len(images), components=len(scores), outlier_images=outliers,
                iou_quantiles=np.quantile(scores, [0.01, 0.1, 0.5]).tolist() if len(scores) else None,
                component_scores=[dict(**row, iou=float(score),
                                       nearest_training_component=templates[2][match])
                                  for row, score, match in zip(components[2], scores, matches)])
            if name.endswith('_100') and len(scores):
                fig, axes = plt.subplots(2, 8, figsize=(14, 4))
                for column, index in enumerate(np.argsort(scores)[:8]):
                    axes[0, column].imshow(components[0][index].reshape(32, 32), cmap='gray', vmin=0, vmax=1)
                    axes[1, column].imshow(templates[0][matches[index]].reshape(32, 32), cmap='gray', vmin=0, vmax=1)
                    axes[0, column].set_title(f'Sample {components[2][index]["sample"]}\nIoU {scores[index]:.3f}')
                for ax in axes.ravel():
                    ax.set_xticks([])
                    ax.set_yticks([])
                axes[0, 0].set_ylabel('Generated\ncomponent')
                axes[1, 0].set_ylabel('Nearest real\ntemplate')
                fig.suptitle(f'{name}: lowest template similarity; thresholded masks, unchanged sizes')
                fig.tight_layout()
                fig.savefig(comparison / 'eval' / f'template_outliers_{name}.png', dpi=150)
                plt.close(fig)
        (comparison / 'template_quality.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(comparison, 'cutoff', cutoff,
              {name: len(result['outlier_images']) for name, result in report['generated'].items()}, flush=True)


if __name__ == '__main__':
    main()
