"""Generate antialiased white-circle-on-black datasets.

Circles are kept strictly away from the image boundary and from each other by a
constant gap using rejection sampling — any proposed image that cannot satisfy the
constraints is discarded entirely and redrawn, preserving manifold structure.

Antialiasing uses signed distance to blend linearly over a one-pixel band.
Pixels fully inside a circle are 1 and pixels fully outside are 0.
"""
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import tqdm as _tqdm



# ── primitives ────────────────────────────────────────────────────────────────

RADIUS_MIN_FRACTION = 0.10
RADIUS_MAX_FRACTION = 0.25
BORDER_GAP_FRACTION = 4 / 64
ANTIALIAS_WIDTH = 1.0


def _circle_values(im_size: int, cx: float, cy: float, r: float) -> torch.Tensor:
    """Return a float32 antialiased circle with background=0 and fill=1."""
    xs = torch.arange(im_size, dtype=torch.float32).view(-1, 1)
    ys = torch.arange(im_size, dtype=torch.float32).view(1, -1)
    dist = torch.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) - r  # signed: neg inside
    return torch.clamp(0.5 - dist, 0.0, 1.0)


def _overlaps(cx: int, cy: int, r: float,
              placed: List[Tuple[int, int, float]], gap: int) -> bool:
    for cx2, cy2, r2 in placed:
        if ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5 < r + r2 + gap:
            return True
    return False


def _try_place(
    n: int,
    rng: random.Random,
    im_size: int,
    r_min: float,
    r_max: float,
    border_gap: int,
    circle_gap: int,
    attempts_per_circle: int,
) -> Optional[List[Tuple[int, int, float]]]:
    """Try once to place n non-overlapping circles. Returns None on failure."""
    placed: List[Tuple[int, int, float]] = []
    for _ in range(n):
        for _ in range(attempts_per_circle):
            r = rng.uniform(r_min, r_max)
            lo = math.ceil(r + border_gap)
            hi = math.floor((im_size - 1) - r - border_gap)
            if lo > hi:
                return None
            cx = rng.randint(lo, hi)
            cy = rng.randint(lo, hi)
            if not _overlaps(cx, cy, r, placed, gap=circle_gap):
                placed.append((cx, cy, r))
                break
        else:
            return None   # could not place this circle → reject entire image
    return placed


def geometry_for_resolution(
    im_size: int,
    radius_min_fraction: float = RADIUS_MIN_FRACTION,
    radius_max_fraction: float = RADIUS_MAX_FRACTION,
    border_gap_fraction: float = BORDER_GAP_FRACTION,
) -> Dict[str, float | int]:
    """Resolve scale-relative geometry settings for one resolution."""
    if im_size < 1:
        raise ValueError("im_size must be positive")
    if not 0 < radius_min_fraction <= radius_max_fraction < 0.5:
        raise ValueError("radius fractions must satisfy 0 < min <= max < 0.5")
    if border_gap_fraction < 0:
        raise ValueError("border_gap_fraction must be nonnegative")
    border_gap = max(0, round(im_size * border_gap_fraction))
    r_min = im_size * radius_min_fraction
    r_max = im_size * radius_max_fraction
    if 2 * (r_max + border_gap) > im_size - 1:
        raise ValueError(
            "maximum radius plus border gap does not fit within the image"
        )
    return {
        "radius_min": r_min,
        "radius_max": r_max,
        "border_gap": border_gap,
    }


# ── per-count generator ───────────────────────────────────────────────────────

def generate_fixed_n(
    n_circles: int,
    num_samples: int = 500_000,
    im_size: int = 64,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    radius_min_fraction: float = RADIUS_MIN_FRACTION,
    radius_max_fraction: float = RADIUS_MAX_FRACTION,
    border_gap: Optional[int] = None,
    border_gap_fraction: float = BORDER_GAP_FRACTION,
    circle_gap: int = 4,   # pixels between circle edges
    attempts_per_circle: int = 500,
    show_progress: bool = True,
    seed: Optional[int] = None,
    output_dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
    """Return a tensor of fixed-count, antialiased circle images.

    Every image contains exactly n_circles non-overlapping circles.
    Circles are separated from the border and from each other by at least
    border_gap / circle_gap pixels (rejection sampling — failed attempts are
    fully discarded so the output distribution is exact).

    Pixel values: background = 0, circles = 1. Compact uint8 output uses the
    full 0--255 range and is converted back to [0, 1] by the training loader.
    """
    geometry = geometry_for_resolution(
        im_size,
        radius_min_fraction=radius_min_fraction,
        radius_max_fraction=radius_max_fraction,
        border_gap_fraction=border_gap_fraction,
    )
    if (r_min is None) != (r_max is None):
        raise ValueError("r_min and r_max must be provided together")
    resolved_r_min = geometry["radius_min"] if r_min is None else float(r_min)
    resolved_r_max = geometry["radius_max"] if r_max is None else float(r_max)
    resolved_border_gap = geometry["border_gap"] if border_gap is None else border_gap
    if output_dtype not in (torch.uint8, torch.float32):
        raise ValueError("output_dtype must be torch.uint8 or torch.float32")

    rng  = random.Random(seed)
    data = torch.zeros(num_samples, im_size, im_size, dtype=output_dtype)

    it = range(num_samples)
    if show_progress and _tqdm is not None:
        it = _tqdm.tqdm(it, desc=f"Generating n={n_circles} circles")

    for idx in it:
        while True:
            placed = _try_place(n_circles, rng, im_size,
                                resolved_r_min, resolved_r_max,
                                resolved_border_gap, circle_gap,
                                attempts_per_circle)
            if placed is not None:
                break   # valid image found

        img = torch.zeros(im_size, im_size, dtype=torch.float32)
        for cx, cy, r in placed:
            img = torch.maximum(img, _circle_values(im_size, cx, cy, r))
        data[idx] = ((img * 255).round().to(torch.uint8)
                     if output_dtype == torch.uint8 else img)

    return data


# ── split helper ──────────────────────────────────────────────────────────────

def split(
    data: torch.Tensor,
    train: int = 480_000,
    val: int = 10_000,
    test: int = 10_000,
) -> Dict[str, torch.Tensor]:
    assert train + val + test == len(data)
    return {
        "train": data[:train],
        "val":   data[train : train + val],
        "test":  data[train + val :],
    }


def generate_fixed_n_splits(
    n_circles: int,
    split_sizes: Dict[str, int],
    im_size: int = 64,
    seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Generate one tensor per split without first materializing all samples."""
    splits: Dict[str, Any] = {}
    seed_base = 0 if seed is None else seed
    for idx, (name, count) in enumerate(split_sizes.items()):
        splits[name] = generate_fixed_n(
            n_circles,
            num_samples=count,
            im_size=im_size,
            seed=seed_base + idx,
            **kwargs,
        ).unsqueeze(1)
    geometry = geometry_for_resolution(
        im_size,
        radius_min_fraction=kwargs.get("radius_min_fraction", RADIUS_MIN_FRACTION),
        radius_max_fraction=kwargs.get("radius_max_fraction", RADIUS_MAX_FRACTION),
        border_gap_fraction=kwargs.get("border_gap_fraction", BORDER_GAP_FRACTION),
    )
    splits["metadata"] = {
        "format_version": 2,
        "resolution": im_size,
        "n_circles": n_circles,
        "split_sizes": dict(split_sizes),
        "seed": seed_base,
        "background_value": 0,
        "circle_value": 1,
        "storage_dtype": str(splits[next(iter(split_sizes))].dtype),
        "radius_min_fraction": kwargs.get("radius_min_fraction", RADIUS_MIN_FRACTION),
        "radius_max_fraction": kwargs.get("radius_max_fraction", RADIUS_MAX_FRACTION),
        "radius_min_pixels": geometry["radius_min"],
        "radius_max_pixels": geometry["radius_max"],
        "border_gap_pixels": geometry["border_gap"],
        "antialias_width_pixels": ANTIALIAS_WIDTH,
    }
    return splits


# ── generate all four datasets ────────────────────────────────────────────────

def generate_all(
    out_dir: Path = Path("data"),
    num_samples: int = 500_000,
    seed: int = 42,
    **kwargs,
) -> Dict[int, Path]:
    """Generate and save one .pt file per circle count (n=1,2,3,4)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for n in range(1, 5):
        data   = generate_fixed_n(n, num_samples=num_samples,
                                  seed=seed + n, **kwargs)
        splits = split(data.unsqueeze(1))
        path   = out_dir / f"circles-64-n{n}.pt"
        torch.save(splits, path)
        print(f"  Saved n={n} → {path}  "
              f"train {splits['train'].shape}  "
              f"val {splits['val'].shape}  "
              f"test {splits['test'].shape}")
        paths[n] = path
    return paths


def generate_all_resolutions(
    resolutions: Iterable[int],
    circle_counts: Iterable[int] = (1, 2, 3, 4),
    out_dir: Path = Path("data"),
    split_sizes: Optional[Dict[str, int]] = None,
    seed: int = 42,
    **kwargs,
) -> Dict[Tuple[int, int], Path]:
    """Generate and save datasets for each resolution and circle count."""
    if split_sizes is None:
        split_sizes = {"train": 48_000, "val": 1_000, "test": 1_000}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[Tuple[int, int], Path] = {}
    for im_size in resolutions:
        for n in circle_counts:
            splits = generate_fixed_n_splits(
                n,
                split_sizes=split_sizes,
                im_size=im_size,
                seed=seed + im_size * 10 + n * 100,
                **kwargs,
            )
            path = out_dir / f"circles-{im_size}-n{n}.pt"
            torch.save(splits, path)
            print(f"  Saved {path}  "
                  f"train {splits['train'].shape}  "
                  f"val {splits['val'].shape}  "
                  f"test {splits['test'].shape}")
            paths[(im_size, n)] = path
            del splits
    return paths


# ── legacy wrapper (kept for backwards compatibility) ─────────────────────────
    
# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating 4 × 500k circle datasets …")
    generate_all(num_samples=500_000, seed=42)
    print("Done.")
