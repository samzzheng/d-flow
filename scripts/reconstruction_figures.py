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
    """Plot four image/measurement rows and a fifth row of error histories.

    The residual row uses the observation when supplied, otherwise clean GT.
    Its GT panel therefore displays the noise magnitude when noise is present.
    Histories contain step, image_relative_l2, and measurement_relative_l2_noisy
    (the last key denotes residual against observed y, also when y is noiseless).
    Absolute measurement residuals use a fixed 0..residual_vmax amp scale.
    Pass RESIDUAL_VMAX["mri"] for MRI. Values above the scale are indicated by
    the colorbar extension; arrays and metrics remain unchanged. Other map rows
    share per-figure color scales; curves share axes within the figure.
    Callers save and close the returned figure.
    """
    if set(reconstructions) - set(METHODS):
        raise ValueError(f"Method labels must be drawn from {METHODS}.")
    if not np.isfinite(residual_vmax) or residual_vmax <= 0:
        raise ValueError("The fixed residual color limit must be finite and positive.")
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
    reference_label = "clean GT" if np.array_equal(measured, measurement_gt) else "observed y"
    rows = (
        ("Image", images, image_errors, image_gt),
        ("Measurement (clean GT reference)", measurements, measurement_errors, measurement_gt),
        ("Absolute image error", image_errors, image_errors, image_gt),
        (f"Absolute measurement residual ({reference_label})", residuals, residuals, measured),
    )
    fig, grid = plt.subplots(5, 5, figsize=(16, 20), layout="constrained",
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
        present = [value for value in displayed if value is not None]
        vmin = 0.0 if row_index >= 2 else min(float(value.min()) for value in present)
        vmax = max(float(value.max()) for value in present)
        overflow = row_index == 3 and vmax > residual_vmax
        if row_index == 3:
            vmax = residual_vmax
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
                value, cmap=cmocean.cm.amp if row_index >= 2 else cmocean.cm.dense_r,
                vmin=vmin, vmax=vmax, aspect="auto",
            )
        axes[row_index, 0].set_ylabel(label, fontsize=10)
        grid[row_index, 4].set_box_aspect(25)
        colorbar = fig.colorbar(artist, cax=grid[row_index, 4],
                               extend="max" if overflow else "neither")
        if row_index == 3:
            colorbar.set_ticks(np.linspace(0, residual_vmax, 6))
    grid[4, 4].set_axis_off()

    histories = {} if histories is None else histories
    last_step = max((float(point["step"]) for history in histories.values() for point in history), default=1)
    curve_max = max((100 * float(point[key]) for history in histories.values() for point in history
                     for key in ("image_relative_l2", "measurement_relative_l2_noisy")), default=1)
    for column, name in enumerate(("GT", *METHODS)):
        axis = axes[4, column]
        for errors, reference, label, color, style, key in (
            (image_errors, image_gt, "Image error", cmocean.cm.amp(.9), "-", "image_relative_l2"),
            (residuals, measured, "Measurement residual", cmocean.cm.amp(.5), "--", "measurement_relative_l2_noisy"),
        ):
            if column == 0:
                norm = float(np.linalg.norm(reference))
                if norm > 0:
                    axis.axhline(100 * np.linalg.norm(errors[0]) / norm, color=color, linestyle=style, label=label)
            elif name in reconstructions and histories.get(name):
                history = histories[name]
                axis.plot([point["step"] for point in history],
                          [100 * float(point[key]) for point in history],
                          color=color, linestyle=style, label=label)
        percentages = ["n/a", "n/a"]
        for index, (errors, reference) in enumerate(((image_errors, image_gt), (residuals, measured))):
            norm = float(np.linalg.norm(reference))
            if errors[column] is not None and norm > 0:
                percentages[index] = f"{100 * np.linalg.norm(errors[column]) / norm:.2f}%"
        axis.set_title(f"{name}\nImage L2 = {percentages[0]}\nResidual L2 = {percentages[1]}", fontsize=10)
        if not axis.lines:
            axis.text(.5, .5, "History not available", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.legend(loc="upper right", fontsize=7)
        axis.set(xlabel="Optimization step", xlim=(0, max(last_step, 1)), ylim=(0, max(curve_max * 1.05, 1)))
        axis.tick_params(labelsize=8)
        axis.grid(alpha=.2)
    axes[4, 0].set_ylabel("Relative L2 error (%)", fontsize=10)
    fig.suptitle(title or "Reconstruction comparison", fontsize=14)
    display_note = "Measurement display: log(1 + magnitude). " if measurement_log else ""
    fig.supxlabel(
        display_note + "Metrics use original arrays; complex panels display magnitude. "
        f"Absolute measurement residual scale: 0–{residual_vmax:g} (fixed); "
        "upper colorbar triangle marks values above the scale.", fontsize=9,
    )
    return fig
