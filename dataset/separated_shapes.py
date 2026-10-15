"""Generate preview samples following separated_shapes_dataset.pdf.

Run: .venv/bin/python -m dataset.separated_shapes
Coordinates use [-1, 1]^2; arrays use image[row=y, column=x]. Output is
binary uint8 (0/1), without antialiasing. No dataset splits are created.
Lengths are resolution independent: a domain length l spans l*resolution/2
pixels, so increasing resolution preserves shape proportions. Default: 64x64.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


Point = tuple[float, float]
Vertices = tuple[Point, Point, Point]
BinaryImage = NDArray[np.uint8]


@dataclass(frozen=True)
class Shape:
    kind: str
    center: Point
    enclosing_radius: float
    radius: float | None = None
    vertices: Vertices | None = None
    side_lengths: tuple[float, float, float] | None = None
    orientation: float | None = None


def sample_geometry(
    seed: int, separation: float = 0.05, max_attempts: int = 10_000,
) -> list[Shape]:
    """Sample fixed shapes, then jointly reject centers until they fit.

    A trial-cap failure raises instead of redrawing the shapes and biasing
    their count, types, sizes, or orientations.
    """
    if separation < 0 or not np.isfinite(separation) or max_attempts < 1:
        raise ValueError("Require finite nonnegative separation and positive max_attempts")
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, 5))
    shapes: list[Shape] = []
    for _ in range(count):
        if rng.integers(2) == 0:
            radius = float(rng.uniform(0.10, 0.20))
            shapes.append(Shape("disk", (0.0, 0.0), radius, radius=radius))
        else:
            a, b, c = map(float, rng.uniform(0.20, 0.30, size=3))
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            q = (b * b + c * c - a * a) / (2.0 * c)
            vertices = np.array([[0.0, 0.0], [c, 0.0], [q, np.sqrt(b * b - q * q)]])
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
            vertices = (vertices - vertices.mean(axis=0)) @ rotation.T
            shapes.append(Shape(
                "triangle", (0.0, 0.0), float(np.linalg.norm(vertices, axis=1).max()),
                vertices=tuple(map(tuple, vertices.tolist())),
                side_lengths=(a, b, c), orientation=theta,
            ))

    radii = np.array([shape.enclosing_radius for shape in shapes])
    pairs = np.triu_indices(count, k=1)
    required_distances = (radii[:, None] + radii[None, :] + separation)[pairs]
    for _ in range(max_attempts):
        centers = rng.uniform(-1.0, 1.0, size=(count, 2))
        inside = np.all(np.abs(centers) + radii[:, None] <= 1.0)
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        if inside and np.all(distances[pairs] >= required_distances):
            break
    else:
        raise RuntimeError(
            f"Placement failed for seed={seed} after {max_attempts} trials; "
            "retry the same seed with a larger max_attempts. Shapes were not redrawn."
        )

    return [Shape(
        shape.kind, tuple(center.tolist()), shape.enclosing_radius,
        radius=shape.radius,
        vertices=(None if shape.vertices is None else
                  tuple(map(tuple, (np.array(shape.vertices) + center).tolist()))),
        side_lengths=shape.side_lengths, orientation=shape.orientation,
    ) for shape, center in zip(shapes, centers)]


def rasterize(shapes: list[Shape], resolution: int = 64) -> BinaryImage:
    """Evaluate actual filled shapes at pixel centers, not bounding disks."""
    if resolution < 1:
        raise ValueError("resolution must be positive")
    coordinates = -1.0 + 2.0 * (np.arange(resolution) + 0.5) / resolution
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    image = np.zeros((resolution, resolution), dtype=np.uint8)
    for shape in shapes:
        if shape.kind == "disk":
            mask = (x - shape.center[0]) ** 2 + (y - shape.center[1]) ** 2 <= shape.radius ** 2
        else:
            vertices = np.array(shape.vertices)
            # Construction gives counterclockwise vertices. Interior points lie
            # on the left of every directed edge, including the boundary.
            mask = np.ones_like(image, dtype=bool)
            for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
                edge = end - start
                mask &= edge[0] * (y - start[1]) - edge[1] * (x - start[0]) >= 0.0
        image[mask] = 1
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/separated_shapes_preview"))
    args = parser.parse_args()
    if args.samples < 1 or args.resolution < 1 or args.seed < 0 or args.max_attempts < 1:
        parser.error("samples, resolution, and max-attempts must be positive; seed must be nonnegative")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = list(range(args.seed, args.seed + args.samples))
    geometries = [sample_geometry(seed, max_attempts=args.max_attempts) for seed in seeds]
    images = np.stack([rasterize(shapes, args.resolution) for shapes in geometries])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "samples.npz", images=images[:, None], seeds=seeds)
    metadata = {
        "specification": "separated_shapes_dataset.pdf",
        "resolution": args.resolution, "domain": [[-1, 1], [-1, 1]],
        "separation": 0.05, "max_attempts": args.max_attempts,
        "storage": "uint8 binary 0/1, (sample, channel, row=y, column=x)",
        "pixel_centers": "-1 + 2*(index + 0.5)/resolution; display origin=lower",
        "numpy_version": np.__version__, "rng": "default_rng/PCG64",
        "samples": [{"index": i, "seed": seed, "count": len(shapes),
                     "shapes": list(map(asdict, shapes))}
                    for i, (seed, shapes) in enumerate(zip(seeds, geometries))],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    columns = min(4, args.samples)
    rows = (args.samples + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows),
                             squeeze=False, layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i < args.samples:
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=1, origin="lower",
                      interpolation="nearest", extent=(-1, 1, -1, 1))
            ax.set_title(f"Sample {i} · N = {len(geometries[i])} · seed {seeds[i]}", fontsize=10)
    fig.suptitle(f"Separated circles and triangles · {args.resolution} × {args.resolution}")
    fig.savefig(args.output / "preview.png", dpi=160)
    plt.close(fig)
    print(f"Saved {args.samples} preview samples, geometry, seeds, and preview.png to {args.output}")


if __name__ == "__main__":
    main()
