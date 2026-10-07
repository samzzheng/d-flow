"""Linear CT and single-coil MRI operators with paired measurement noise."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class CT:
    """Uniform angles in [0, pi); the established rotate-and-average Radon map."""

    def __init__(self, resolution: int, angles: int, device: torch.device) -> None:
        self.angles = torch.arange(angles, device=device) * (math.pi / angles)
        theta = torch.zeros(angles, 2, 3, device=device)
        cosine, sine = self.angles.cos(), self.angles.sin()
        theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
        theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
        # Bound the 180-angle operator's temporary memory at large resolutions.
        self.grids = tuple(F.affine_grid(part, (len(part), 1, resolution, resolution),
                                        align_corners=False) for part in theta.split(30))
        self.metadata = {"kind": "ct", "angle_count": angles, "range_degrees": [0, 180],
                         "normalization": "detector line mean", "align_corners": False}

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        projections = [F.grid_sample(image.expand(len(grid), -1, -1, -1), grid,
                                     align_corners=False).mean(dim=-2) for grid in self.grids]
        return torch.cat(projections).transpose(0, 1).unsqueeze(0)


class MRI:
    """Centered unitary Fourier transform with a seeded variable-density mask.

    Cartesian masks retain round(N/R) full phase-encoding lines, including DC.
    Point masks retain round(N*N/R) coefficients. Actual acceleration is saved.
    Real circle images are a zero-phase single-coil phantom; data remain complex.
    """

    def __init__(self, resolution: int, acceleration: int, seed: int,
                 device: torch.device, pattern: str = "cartesian") -> None:
        generator = torch.Generator().manual_seed(seed)
        coordinate = (torch.arange(resolution) - resolution // 2) / (resolution / 2)
        distance = coordinate.abs() if pattern == "cartesian" else (
            coordinate[:, None].square() + coordinate[None, :].square()).sqrt().flatten()
        count = max(1, round(distance.numel() / acceleration))
        dc = resolution // 2 if pattern == "cartesian" else (resolution // 2) * resolution + resolution // 2
        weights = (1 - distance.clamp(max=1)).pow(4) + 0.01
        weights[dc] = 0
        selected = torch.cat((torch.tensor([dc]), torch.multinomial(
            weights, count - 1, replacement=False, generator=generator))) if count > 1 else torch.tensor([dc])
        mask = torch.zeros(distance.numel(), dtype=torch.bool)
        mask[selected] = True
        mask = mask[:, None].expand(resolution, resolution) if pattern == "cartesian" else mask.reshape(resolution, resolution)
        self.mask = mask.to(device).reshape(1, 1, resolution, resolution)
        self.metadata = {"kind": "mri", "pattern": pattern, "nominal_acceleration": acceleration,
                         "actual_acceleration": resolution ** 2 / int(mask.sum()),
                         "acquired_coefficients": int(mask.sum()), "mask_seed": seed,
                         "normalization": "centered orthonormal FFT", "zero_phase_phantom": True}

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.mask * torch.fft.fftshift(torch.fft.fft2(image, norm="ortho"), dim=(-2, -1))


def noisy_measurement(clean: torch.Tensor, relative_noise: float, seed: int,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
    """Fix ||noise||/||clean|| exactly; MRI noise exists only on acquired samples."""
    if relative_noise == 0:
        return clean
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    if clean.is_complex():
        noise = torch.complex(noise, torch.randn(clean.shape, generator=generator))
    noise = noise.to(clean.device)
    if mask is not None:
        noise = noise * mask
    noise = noise * (relative_noise * clean.norm() / noise.norm())
    return clean + noise
