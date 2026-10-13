"""Shared reconstruction panels for D-Flow, P-Flow, and DLR D-Flow.

Pass one image and its forward measurement from the same checkpoint per method.
Measurements must use the same acquisition mask, including for the clean GT.
Native complex tensors are supported; metrics are computed before magnitude or
log display transforms. Missing methods keep their column, marked unavailable.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cmocean
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import torch

METHODS = ("D-Flow", "P-Flow", "DLR D-Flow")
RESIDUAL_VMAX = {"ct": 1.0, "mri": 25.0}
MEASUREMENT_VMAX = {"linear": 1.0, "log_magnitude": 5.0}
PlotArray = NDArray[np.float64] | NDArray[np.complex128]
History = Sequence[Mapping[str, float | int | None]]


def _as_array(value: torch.Tensor) -> PlotArray:
    dtype = torch.complex128 if value.is_complex() else torch.float64
    array = value.detach().to(device="cpu", dtype=dtype).squeeze().numpy()
    if array.ndim not in (1, 2) or not array.size or not np.isfinite(array).all():
        raise ValueError("Each panel must contain one finite 1D or 2D sample.")
    return np.atleast_2d(array)


def plot_reconstruction_comparison(
    truth: torch.Tensor,
    clean_measurement: torch.Tensor,
    reconstructions: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    observation: torch.Tensor | None = None,
    histories: Mapping[str, History] | None = None,
    measurement_log: bool = False,
    residual_vmax: float = RESIDUAL_VMAX["ct"],
    title: str = "",
) -> Figure:
    """Plot images and clean-reference measurements for all methods.

    Images use the fixed training range 0..1 with saturated endpoint colors.
    Measurements use fixed display limits: 0..1 for linear CT, 0..5 for
    log(1 + magnitude) MRI. Out-of-range values use endpoint colors.
    Metrics use original arrays before any display transform. Observation,
    histories and residual_vmax remain accepted for existing callers, but their
    diagnostic rows are no longer displayed. Callers save and close the figure.
    """
    if set(reconstructions) - set(METHODS):
        raise ValueError(f"Method labels must be drawn from {METHODS}.")
    image_gt, measurement_gt = map(_as_array, (truth, clean_measurement))
    images: list[PlotArray | None] = [image_gt]
    measurements: list[PlotArray | None] = [measurement_gt]
    for name in METHODS:
        image, measurement = (None, None)
        if name in reconstructions:
            image, measurement = map(_as_array, reconstructions[name])
            if image.shape != image_gt.shape or measurement.shape != measurement_gt.shape:
                raise ValueError(f"{name} arrays must match the GT shapes.")
        images.append(image)
        measurements.append(measurement)

    image_errors = [None if x is None else np.abs(x - image_gt) for x in images]
    measurement_errors = [None if y is None else np.abs(y - measurement_gt)
                          for y in measurements]
    rows = (
        ("Image", images, image_errors, image_gt),
        ("Measurement (clean GT reference)", measurements, measurement_errors, measurement_gt),
    )
    fig, grid = plt.subplots(2, 5, figsize=(16, 8), layout="constrained",
                            gridspec_kw={"width_ratios": [1, 1, 1, 1, .04]})
    axes = grid[:, :4]
    for axis in axes.flat:
        axis.set_box_aspect(1)
    for row_index, (label, values, errors, reference) in enumerate(rows):
        displayed = [None if value is None else
                     np.abs(value) if np.iscomplexobj(value) else value for value in values]
        if row_index == 1 and measurement_log:
            displayed = [None if value is None else np.log1p(np.abs(value))
                         for value in displayed]
        vmin = 0.0
        vmax = 1.0 if row_index == 0 else MEASUREMENT_VMAX[
            "log_magnitude" if measurement_log else "linear"]
        reference_norm = float(np.linalg.norm(reference))
        for column, (name, value, error) in enumerate(zip(("GT", *METHODS), displayed, errors)):
            axis = axes[row_index, column]
            percentage = "n/a"
            if error is not None and reference_norm > 0:
                percentage = f"{100 * np.linalg.norm(error) / reference_norm:.2f}%"
            axis.set_title(f"{name}\nRelative L2 = {percentage}", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            if value is None:
                axis.text(0.5, 0.5, "Not available", ha="center", va="center", transform=axis.transAxes)
                continue
            artist = axis.imshow(
                value, cmap=cmocean.cm.dense_r,
                vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest",
            )
        axes[row_index, 0].set_ylabel(label, fontsize=10)
        grid[row_index, 4].set_box_aspect(25)
        colorbar = fig.colorbar(artist, cax=grid[row_index, 4], extend="both")
        colorbar.set_ticks(np.linspace(vmin, vmax, 6))
    fig.suptitle(title or "Reconstruction comparison", fontsize=14)
    if measurement_log:
        axes[1, 0].set_ylabel("Measurement (clean GT; log(1 + magnitude))", fontsize=10)
    return fig
