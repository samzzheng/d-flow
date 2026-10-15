"""Generate training-ready separated shapes with per-sample geometry.

Run: .venv/bin/python -m dataset.generate_separated_shapes
Unlike the preview NPZ, uint8 training tensors encode foreground as 255 so
the existing training loader recovers binary float images in [0, 1].
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from dataset.separated_shapes import rasterize, sample_geometry


BatchJob = tuple[int, int, int]
BatchResult = tuple[NDArray[np.uint8], list[str]]


def generate_batch(job: BatchJob) -> BatchResult:
    """Top-level worker function required by the multiprocessing pool."""
    first_seed, count, resolution = job
    images = np.empty((count, 1, resolution, resolution), dtype=np.uint8)
    records: list[str] = []
    for index, seed in enumerate(range(first_seed, first_seed + count)):
        shapes = sample_geometry(seed)
        images[index, 0] = rasterize(shapes, resolution) * 255
        records.append(json.dumps({"seed": seed, "count": len(shapes),
                                   "shapes": list(map(asdict, shapes))}, separators=(",", ":")))
    return images, records


def separate_duplicate_images(
    splits: dict[str, NDArray[np.uint8]],
) -> tuple[NDArray[np.int64], int]:
    """Group identical rasters by swapping with unique images; never resample.

    Return each final sample's original global index for geometry/seed ordering.
    Duplicates within a split remain, preserving the generated population.
    """
    owners: dict[bytes, tuple[str, int]] = {}
    hashes: dict[str, list[bytes]] = {}
    collisions: list[tuple[tuple[str, int], tuple[str, int]]] = []
    offsets: dict[str, int] = {}
    total = 0
    for name, images in splits.items():
        offsets[name] = total
        total += len(images)
        hashes[name] = []
        for start in range(0, len(images), 1024):
            block = images[start:start + 1024]
            packed = np.packbits(block.reshape(len(block), -1), axis=1)
            for local, image in enumerate(packed):
                digest = hashlib.sha256(image.tobytes()).digest()
                location = (name, start + local)
                owner = owners.setdefault(digest, location)
                hashes[name].append(digest)
                if owner[0] != name:
                    collisions.append((owner, location))
    frequencies = Counter(digest for values in hashes.values() for digest in values)
    donors = {name: iter([i for i, digest in enumerate(values) if frequencies[digest] == 1])
              for name, values in hashes.items()}
    order = np.arange(total, dtype=np.int64)
    for (owner, _), (other, index) in collisions:
        donor = next(donors[owner], None)
        if donor is None:
            raise RuntimeError(f"Insufficient unique images in {owner} to isolate duplicate groups")
        saved = splits[owner][donor].copy()
        splits[owner][donor] = splits[other][index]
        splits[other][index] = saved
        a, b = offsets[owner] + donor, offsets[other] + index
        order[a], order[b] = order[b], order[a]
    return order, len(collisions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=80_000)
    parser.add_argument("--val", type=int, default=10_000)
    parser.add_argument("--test", type=int, default=10_000)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=202610150000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("data/separated-shapes-64.pt"))
    args = parser.parse_args()
    if min(args.train, args.val, args.test, args.resolution, args.workers) < 1 or args.seed < 0:
        parser.error("split sizes, resolution, and workers must be positive; seed must be nonnegative")
    geometry_path = args.output.with_suffix(".geometry.jsonl")
    summary_path = args.output.with_suffix(".json")
    for path in (args.output, geometry_path, summary_path):
        if path.exists():
            parser.error(f"Output already exists: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(1)
    started = perf_counter()
    split_sizes = {"train": args.train, "val": args.val, "test": args.test}
    metadata = {
        "format_version": 1, "resolution": args.resolution, "split_sizes": split_sizes,
        "domain": [[-1, 1], [-1, 1]], "separation": 0.05,
        "count_distribution": "Uniform integers 1..4",
        "type_distribution": "Independent disk/triangle, probability 1/2 each",
        "disk_radius_range": [0.10, 0.20], "triangle_side_range": [0.20, 0.30],
        "triangle_orientation_range": [0.0, 2.0 * np.pi],
        "placement": "Joint uniform center rejection with fixed shape parameters; enclosing disks",
        "rasterization": "Binary pixel centers; row=y, column=x; no antialiasing",
        "storage": "uint8 0/255, NCHW; divide by 255 to obtain binary 0/1",
        "geometry_file": geometry_path.name,
        "geometry_order": "One JSON line per sample, train then val then test, in tensor order",
        "seed": args.seed, "initial_split_seed_ranges": {}, "rng": "default_rng/PCG64",
        "split_policy": "Independent seeds, then swap exact cross-split raster duplicates into one split",
        "seed_lookup": "sample_seeds[split][index] in the .pt file; geometry JSONL follows final tensor order",
        "numpy_version": np.__version__,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path(__file__).with_name("separated_shapes.py"))},
        "created_at": datetime.now().astimezone().isoformat(),
    }
    splits: dict[str, NDArray[np.uint8]] = {}
    first_seed = args.seed
    with geometry_path.open("x") as geometry_file, get_context("spawn").Pool(args.workers) as pool:
        for name, count in split_sizes.items():
            images = np.empty((count, 1, args.resolution, args.resolution), dtype=np.uint8)
            metadata["initial_split_seed_ranges"][name] = [first_seed, first_seed + count - 1]
            jobs = ((first_seed + offset, min(256, count - offset), args.resolution)
                    for offset in range(0, count, 256))
            offset = 0
            for batch, records in pool.imap(generate_batch, jobs):
                images[offset:offset + len(batch)] = batch
                geometry_file.write("\n".join(records) + "\n")
                offset += len(batch)
                if offset % 25_600 == 0 or offset == count:
                    print(f"{name}: {offset:,}/{count:,} ({perf_counter() - started:.1f}s)", flush=True)
            splits[name] = images
            first_seed += count
    order, swaps = separate_duplicate_images(splits)
    metadata["cross_split_duplicate_swaps"] = swaps
    if swaps:
        changed = set(np.flatnonzero(order != np.arange(len(order))).tolist())
        with geometry_path.open() as geometry_file:
            moved_records = {i: line for i, line in enumerate(geometry_file) if i in changed}
        temporary = geometry_path.with_suffix(".tmp")
        with geometry_path.open() as source, temporary.open("x") as destination:
            for i, line in enumerate(source):
                destination.write(moved_records[int(order[i])] if i in changed else line)
        temporary.replace(geometry_path)
    print(f"Isolated {swaps} cross-split duplicate images without resampling", flush=True)
    sample_seeds: dict[str, torch.Tensor] = {}
    offset = 0
    for name, count in split_sizes.items():
        sample_seeds[name] = torch.from_numpy(order[offset:offset + count] + args.seed)
        offset += count
    dataset = {name: torch.from_numpy(images) for name, images in splits.items()}
    torch.save({**dataset, "sample_seeds": sample_seeds, "metadata": metadata}, args.output)
    metadata["generation_seconds"] = perf_counter() - started
    metadata["dataset_bytes"] = args.output.stat().st_size
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output} and geometry in {metadata['generation_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
