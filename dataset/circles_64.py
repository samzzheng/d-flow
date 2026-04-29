"""Generate 64×64 grayscale circles datasets (one per fixed circle count 1–4).

Circles are kept strictly away from the image boundary and from each other by a
constant gap using rejection sampling — any proposed image that cannot satisfy the
constraints is discarded entirely and redrawn, preserving manifold structure.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import tqdm as _tqdm



# ── primitives ────────────────────────────────────────────────────────────────

def _circle_mask(im_size: int, cx: int, cy: int, r: int) -> torch.Tensor:
    xs = torch.arange(im_size, dtype=torch.int32).view(-1, 1)
    ys = torch.arange(im_size, dtype=torch.int32).view(1, -1)
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r**2


def _overlaps(cx: int, cy: int, r: int,
              placed: List[Tuple[int, int, int]], gap: int) -> bool:
    for cx2, cy2, r2 in placed:
        if ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5 < r + r2 + gap:
            return True
    return False


def _try_place(
    n: int,
    rng: random.Random,
    im_size: int,
    r_min: int,
    r_max: int,
    border_gap: int,
    circle_gap: int,
    attempts_per_circle: int,
) -> Optional[List[Tuple[int, int, int]]]:
    """Try once to place n non-overlapping circles. Returns None on failure."""
    placed: List[Tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(attempts_per_circle):
            r  = rng.randint(r_min, r_max)
            lo = r + border_gap
            hi = (im_size - 1) - r - border_gap
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


# ── per-count generator ───────────────────────────────────────────────────────

def generate_fixed_n(
    n_circles: int,
    num_samples: int = 20_000,
    im_size: int = 64,
    r_min: int = 4,
    r_max: int = 10,
    border_gap: int = 4,   # pixels between circle edge and image border
    circle_gap: int = 4,   # pixels between circle edges
    attempts_per_circle: int = 500,
    show_progress: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Return (num_samples, im_size, im_size) float32 tensor.

    Every image contains exactly n_circles non-overlapping circles.
    Circles are separated from the border and from each other by at least
    border_gap / circle_gap pixels (rejection sampling — failed attempts are
    fully discarded so the output distribution is exact).

    Pixel values: background = 1, circles = 0.
    """
    rng  = random.Random(seed)
    data = torch.zeros(num_samples, im_size, im_size, dtype=torch.float32)

    it = range(num_samples)
    if show_progress and _tqdm is not None:
        it = _tqdm.tqdm(it, desc=f"Generating n={n_circles} circles")

    for idx in it:
        while True:
            placed = _try_place(n_circles, rng, im_size,
                                r_min, r_max, border_gap, circle_gap,
                                attempts_per_circle)
            if placed is not None:
                break   # valid image found

        img = torch.ones(im_size, im_size, dtype=torch.float32)
        for cx, cy, r in placed:
            img[_circle_mask(im_size, cx, cy, r)] = 0.0
        data[idx] = img

    return data


# ── split helper ──────────────────────────────────────────────────────────────

def split(
    data: torch.Tensor,
    train: int = 18_000,
    val: int = 1_000,
    test: int = 1_000,
) -> Dict[str, torch.Tensor]:
    assert train + val + test == len(data)
    return {
        "train": data[:train],
        "val":   data[train : train + val],
        "test":  data[train + val :],
    }


# ── generate all four datasets ────────────────────────────────────────────────

def generate_all(
    out_dir: Path = Path("data"),
    num_samples: int = 20_000,
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


# ── legacy wrapper (kept for backwards compatibility) ─────────────────────────

def generate(
    num_samples: int = 20_000,
    im_size: int = 64,
    r_min: int = 4,
    r_max: int = 10,
    border_pad: int = 4,
    min_circles: int = 1,
    max_circles: int = 4,
    circle_gap: int = 4,
    max_attempts: int = 500,
    show_progress: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Mixed-count dataset (n ~ Unif{min_circles..max_circles})."""
    rng  = random.Random(seed)
    data = torch.zeros(num_samples, im_size, im_size, dtype=torch.float32)

    it = range(num_samples)
    if show_progress and _tqdm is not None:
        it = _tqdm.tqdm(it, desc="Generating circles 64×64")

    for idx in it:
        n = rng.randint(min_circles, max_circles)
        while True:
            placed = _try_place(n, rng, im_size, r_min, r_max,
                                border_pad, circle_gap, max_attempts)
            if placed is not None:
                break
        img = torch.ones(im_size, im_size, dtype=torch.float32)
        for cx, cy, r in placed:
            img[_circle_mask(im_size, cx, cy, r)] = 0.0
        data[idx] = img

    return data


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating 4 × 20k circle datasets …")
    generate_all(num_samples=20_000, seed=42)
    print("Done.")
