#!/usr/bin/env python3
"""Preview or regenerate the scale-relative one-circle datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cmocean
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.circles_64 import (
    BORDER_GAP_FRACTION,
    RADIUS_MAX_FRACTION,
    RADIUS_MIN_FRACTION,
    generate_all_resolutions,
    generate_fixed_n,
    geometry_for_resolution,
)


DEFAULT_RESOLUTIONS = (128, 160, 64, 96)
DEFAULT_SPLITS = {"train": 48_000, "val": 1_000, "test": 1_000}


def save_preview(
    resolutions: list[int],
    samples_per_resolution: int,
    seed: int,
    output: Path,
    radius_min_fraction: float,
    radius_max_fraction: float,
    border_gap_fraction: float,
) -> None:
    fig, axes = plt.subplots(
        len(resolutions), samples_per_resolution,
        figsize=(2.7 * samples_per_resolution, 2.45 * len(resolutions)),
        squeeze=False,
    )
    for row, resolution in enumerate(resolutions):
        geometry = geometry_for_resolution(
            resolution,
            radius_min_fraction,
            radius_max_fraction,
            border_gap_fraction,
        )
        images = generate_fixed_n(
            1,
            num_samples=samples_per_resolution,
            im_size=resolution,
            radius_min_fraction=radius_min_fraction,
            radius_max_fraction=radius_max_fraction,
            border_gap_fraction=border_gap_fraction,
            show_progress=False,
            seed=seed + resolution,
            output_dtype=torch.float32,
        )
        for column, (axis, image) in enumerate(zip(axes[row], images), start=1):
            axis.imshow(
                image.numpy(), cmap=cmocean.cm.dense_r,
                vmin=0, vmax=1, interpolation="nearest",
            )
            axis.set_title(f"{resolution}×{resolution} · sample {column}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 1:
                axis.set_ylabel(
                    f"radius {geometry['radius_min']:.1f}–"
                    f"{geometry['radius_max']:.1f} px\n"
                    f"border gap {geometry['border_gap']} px",
                    fontsize=8,
                )
    fig.suptitle(
        "Proposed one-circle datasets: background 0, circle 1, "
        f"radius {radius_min_fraction:.0%}–{radius_max_fraction:.0%} of resolution",
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview: {output}")


def delete_old_circle_datasets(data_dir: Path) -> None:
    targets = sorted(data_dir.glob("circles-*.pt"))
    print("Deleting old circle datasets:")
    for path in targets:
        print(f"  {path}")
    for path in targets:
        path.unlink()


def validate_dataset(path: Path, resolution: int,
                     split_sizes: dict[str, int] = DEFAULT_SPLITS) -> None:
    saved = torch.load(path, weights_only=True, mmap=True)
    expected = {
        split: (count, 1, resolution, resolution)
        for split, count in split_sizes.items()
    }
    for split, shape in expected.items():
        tensor = saved[split]
        if tuple(tensor.shape) != shape:
            raise RuntimeError(f"{path}: {split} has {tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != torch.uint8:
            raise RuntimeError(f"{path}: {split} has dtype {tensor.dtype}, expected uint8")
    metadata = saved.get("metadata", {})
    if metadata.get("background_value") != 0 or metadata.get("circle_value") != 1:
        raise RuntimeError(f"{path}: invalid polarity metadata")
    print(f"Validated: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview-only", action="store_true")
    mode.add_argument("--generate", action="store_true")
    parser.add_argument("--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--samples-per-resolution", type=int, default=2)
    parser.add_argument(
        "--preview-output", type=Path,
        default=Path("outputs/circles_preview_all_resolutions.png"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius-min-fraction", type=float, default=RADIUS_MIN_FRACTION)
    parser.add_argument("--radius-max-fraction", type=float, default=RADIUS_MAX_FRACTION)
    parser.add_argument("--border-gap-fraction", type=float, default=BORDER_GAP_FRACTION)
    parser.add_argument(
        "--delete-old", action="store_true",
        help="Delete every data-dir/circles-*.pt file before full generation.",
    )
    parser.add_argument(
        "--keep-existing", action="store_true",
        help="Add the requested resolutions without touching existing datasets.",
    )
    args = parser.parse_args()

    if args.samples_per_resolution < 1:
        parser.error("--samples-per-resolution must be positive")
    if args.preview_only:
        if args.delete_old:
            parser.error("--delete-old cannot be used with --preview-only")
        save_preview(
            args.resolutions,
            args.samples_per_resolution,
            args.seed,
            args.preview_output,
            args.radius_min_fraction,
            args.radius_max_fraction,
            args.border_gap_fraction,
        )
        return

    if args.delete_old and args.keep_existing:
        parser.error("--delete-old and --keep-existing are mutually exclusive")
    if not (args.delete_old or args.keep_existing):
        parser.error(
            "full generation requires --delete-old (regenerate everything) or "
            "--keep-existing (add resolutions) after preview approval"
        )
    if args.delete_old:
        delete_old_circle_datasets(args.data_dir)
    paths = generate_all_resolutions(
        args.resolutions,
        circle_counts=(1,),
        out_dir=args.data_dir,
        split_sizes=DEFAULT_SPLITS,
        seed=args.seed,
        radius_min_fraction=args.radius_min_fraction,
        radius_max_fraction=args.radius_max_fraction,
        border_gap_fraction=args.border_gap_fraction,
        output_dtype=torch.uint8,
    )
    for (resolution, n_circles), path in paths.items():
        if n_circles != 1:
            raise RuntimeError("one-circle workflow generated an unexpected circle count")
        validate_dataset(path, resolution)


if __name__ == "__main__":
    main()
