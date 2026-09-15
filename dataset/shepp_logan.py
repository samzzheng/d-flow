"""Generate reproducible transformed Shepp-Logan phantom datasets.

The canonical phantom is imported from scikit-image. Random affine and
intensity transforms provide a distribution of phantoms rather than repeated
copies of one image. Samples are stored as uint8 and converted to [0, 1] by the
trainer, keeping the 256x256 dataset reasonably compact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from tqdm import tqdm


SPLIT_SIZES = {"train": 80_000, "val": 10_000, "test": 10_000}


def canonical_phantom(resolution: int, device: torch.device) -> torch.Tensor:
    phantom = shepp_logan_phantom().astype(np.float32)
    phantom = resize(
        phantom,
        (resolution, resolution),
        order=1,
        mode="constant",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    return torch.from_numpy(phantom).to(device).view(1, 1, resolution, resolution)


def generate_split(
    resolution: int,
    count: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    base = canonical_phantom(resolution, device)
    result = torch.empty((count, 1, resolution, resolution), dtype=torch.uint8)
    generator = torch.Generator(device=device).manual_seed(seed)

    for start in tqdm(range(0, count, batch_size), desc=f"{resolution}px", unit="batch"):
        size = min(batch_size, count - start)
        angle = (torch.rand(size, device=device, generator=generator) * 30.0 - 15.0)
        angle = torch.deg2rad(angle)
        scale_x = 0.90 + 0.20 * torch.rand(size, device=device, generator=generator)
        scale_y = 0.90 + 0.20 * torch.rand(size, device=device, generator=generator)
        translate_x = 0.16 * torch.rand(size, device=device, generator=generator) - 0.08
        translate_y = 0.16 * torch.rand(size, device=device, generator=generator) - 0.08

        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        theta = torch.zeros((size, 2, 3), device=device)
        theta[:, 0, 0] = cosine / scale_x
        theta[:, 0, 1] = -sine / scale_y
        theta[:, 1, 0] = sine / scale_x
        theta[:, 1, 1] = cosine / scale_y
        theta[:, 0, 2] = translate_x
        theta[:, 1, 2] = translate_y

        grid = F.affine_grid(
            theta,
            (size, 1, resolution, resolution),
            align_corners=False,
        )
        images = F.grid_sample(
            base.expand(size, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        gamma = 0.85 + 0.30 * torch.rand(
            (size, 1, 1, 1), device=device, generator=generator
        )
        intensity = 0.90 + 0.10 * torch.rand(
            (size, 1, 1, 1), device=device, generator=generator
        )
        images = images.clamp_(0, 1).pow_(gamma).mul_(intensity).clamp_(0, 1)
        result[start:start + size].copy_((images * 255).round().to(torch.uint8).cpu())

    return result


def generate_dataset(
    resolution: int,
    output: Path,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> None:
    split_tensors = {}
    for split_index, (name, count) in enumerate(SPLIT_SIZES.items()):
        split_tensors[name] = generate_split(
            resolution,
            count,
            seed + split_index,
            batch_size,
            device,
        )

    split_tensors["metadata"] = {
        "name": "transformed Shepp-Logan phantom",
        "source": "skimage.data.shepp_logan_phantom",
        "resolution": resolution,
        "split_sizes": SPLIT_SIZES,
        "seed": seed,
        "storage_dtype": "uint8",
        "value_range": [0.0, 1.0],
        "augmentation": {
            "rotation_degrees": [-15.0, 15.0],
            "axis_scale": [0.90, 1.10],
            "translation_fraction": [-0.08, 0.08],
            "gamma": [0.85, 1.15],
            "intensity": [0.90, 1.00],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(split_tensors, output)
    print(f"Saved {output} ({sum(SPLIT_SIZES.values()):,} samples)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    for resolution in args.resolutions:
        output = args.out_dir / f"shepp-logan-{resolution}.pt"
        if output.exists() and not args.overwrite:
            print(f"Skipping existing {output}")
            continue
        generate_dataset(
            resolution,
            output,
            args.seed,
            args.batch_size,
            device,
        )


if __name__ == "__main__":
    main()
