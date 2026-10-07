"""Shared reconstruction panels for D-Flow, P-Flow, and DLR D-Flow.

Pass one image and its forward measurement from the same checkpoint per method.
Measurements must use the same acquisition mask, including for the clean GT.
Native complex tensors are supported; metrics are computed before magnitude or
log display transforms. Missing methods keep their column, marked unavailable.
"""
from __future__ import annotations

from collections.abc import Mapping

import cmocean
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import torch

METHODS = ("D-Flow", "P-Flow", "DLR D-Flow")
PlotArray = NDArray[np.float64] | NDArray[np.complex128]


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
    measurement_log: bool = False,
    title: str = "",
) -> Figure:
    """Plot images, clean measurements, image errors, and measurement residuals.

    The last row uses the noisy observation when supplied, otherwise clean GT.
    Its GT panel therefore displays the noise magnitude when noise is present.
    Each row shares a color scale. No clipping or per-panel normalization occurs.
    Callers save and close the returned figure.
    """
    if set(reconstructions) - set(METHODS):
        raise ValueError(f"Method labels must be drawn from {METHODS}.")
    image_gt, measurement_gt = map(_as_array, (truth, clean_measurement))
    measured = measurement_gt if observation is None else _as_array(observation)
    if measured.shape != measurement_gt.shape:
        raise ValueError("Clean and observed measurements must have matching shapes.")
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
    residuals = [None if y is None else np.abs(y - measured) for y in measurements]
    reference_label = "observed y" if observation is not None else "clean GT"
    rows = (
        ("Image", images, image_errors, image_gt),
        ("Measurement (clean GT reference)", measurements, measurement_errors, measurement_gt),
        ("Absolute image error", image_errors, image_errors, image_gt),
        (f"Absolute measurement residual ({reference_label})", residuals, residuals, measured),
    )
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), layout="constrained")
    for row_index, (label, values, errors, reference) in enumerate(rows):
        displayed = [None if value is None else
                     np.abs(value) if np.iscomplexobj(value) else value for value in values]
        if row_index == 1 and measurement_log:
            displayed = [None if value is None else np.log1p(np.abs(value))
                         for value in displayed]
        present = [value for value in displayed if value is not None]
        vmin = 0.0 if row_index >= 2 else min(float(value.min()) for value in present)
        vmax = max(float(value.max()) for value in present)
        if vmax == vmin:
            vmax = vmin + 1.0
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
                value, cmap="magma" if row_index >= 2 else cmocean.cm.dense_r,
                vmin=vmin, vmax=vmax, aspect="equal" if row_index in (0, 2) else "auto",
            )
        axes[row_index, 0].set_ylabel(label, fontsize=10)
        fig.colorbar(artist, ax=axes[row_index, :].tolist(), shrink=0.8)
    fig.suptitle(title or "Reconstruction comparison", fontsize=14)
    display_note = "Measurement display: log(1 + magnitude). " if measurement_log else ""
    fig.supxlabel(
        display_note + "Metrics use original arrays; complex panels display magnitude. "
        "Each row uses a shared color scale.", fontsize=9,
    )
    return fig
