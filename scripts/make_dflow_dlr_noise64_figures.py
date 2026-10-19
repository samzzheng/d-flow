"""Comparison figures for the 64x64 D-Flow vs DLR-D-Flow noise benchmark.

For each (noise level, sample) pair, makes a 2x3 figure:
  row 1: ground truth image | D-Flow reconstruction | DLR+D-Flow reconstruction
  row 2: observed (noisy) measurement | D-Flow measurement | DLR+D-Flow measurement
Each panel is titled with its relative L2 error as a percentage (image error
in row 1, measurement error against the noisy observation in row 2).

Usage: python scripts/make_dflow_dlr_noise64_figures.py
"""
from __future__ import annotations

from pathlib import Path

import cmocean
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "outputs/dflow_vs_dlr_noise64"
OUT_DIR = IN_DIR / "figures"


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(a - b)
            / (torch.linalg.vector_norm(b) + 1e-30)).item()


def panel(fig, axis, array, title, vmin, vmax):
    artist = axis.imshow(array, cmap=cmocean.cm.dense_r, vmin=vmin, vmax=vmax,
                          aspect="auto")
    axis.set_title(title, fontsize=11)
    axis.set_xticks([])
    axis.set_yticks([])
    return artist


def make_figure(noise_level: float, sample_index: int) -> None:
    tag = f"noise{int(round(noise_level * 100)):02d}_{sample_index}"
    dflow_path = IN_DIR / f"sample_{tag}_d_flow.pt"
    dlr_path = IN_DIR / f"sample_{tag}_dlr_d_flow.pt"
    if not (dflow_path.exists() and dlr_path.exists()):
        print(f"Skipping {tag}: missing data")
        return

    dflow = torch.load(dflow_path, map_location="cpu", weights_only=False)
    dlr = torch.load(dlr_path, map_location="cpu", weights_only=False)

    truth = dflow["truth"][0, 0].numpy()
    noisy_meas = dflow["noisy_measurement"][0, 0].numpy()

    dflow_recon = dflow["reconstruction"][0, 0].numpy()
    dlr_recon = dlr["reconstruction"][0, 0].numpy()
    dflow_meas = dflow["measurement"][0, 0].numpy()
    dlr_meas = dlr["measurement"][0, 0].numpy()

    dflow_img_err = 100 * dflow["summary"]["best_image_relative_l2"]
    dlr_img_err = 100 * dlr["summary"]["best_image_relative_l2"]
    dflow_meas_err = 100 * relative_l2(dflow["measurement"], dflow["noisy_measurement"])
    dlr_meas_err = 100 * relative_l2(dlr["measurement"], dlr["noisy_measurement"])

    meas_min = min(noisy_meas.min(), dflow_meas.min(), dlr_meas.min())
    meas_max = max(noisy_meas.max(), dflow_meas.max(), dlr_meas.max())

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.0))

    panel(fig, axes[0, 0], truth, "Ground truth (0.00%)", 0, 1)
    panel(fig, axes[0, 1], dflow_recon, f"D-Flow (image rel L2 = {dflow_img_err:.2f}%)", 0, 1)
    panel(fig, axes[0, 2], dlr_recon,
          f"DLR+D-Flow (image rel L2 = {dlr_img_err:.2f}%)", 0, 1)

    panel(fig, axes[1, 0], noisy_meas,
          f"Observed ({100 * noise_level:.0f}% noise)", meas_min, meas_max)
    panel(fig, axes[1, 1], dflow_meas,
          f"D-Flow (meas. rel L2 = {dflow_meas_err:.2f}%)", meas_min, meas_max)
    panel(fig, axes[1, 2], dlr_meas,
          f"DLR+D-Flow (meas. rel L2 = {dlr_meas_err:.2f}%)", meas_min, meas_max)

    fig.suptitle(
        f"64x64 sparse CT, {100 * noise_level:.0f}% noise, sample {sample_index}, "
        f"100-step flow, no source prior",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{tag}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {tag}.png")


def main() -> None:
    import csv
    summary_path = IN_DIR / "samples.csv"
    if not summary_path.exists():
        print("No samples.csv found yet.")
        return
    pairs = set()
    with summary_path.open() as handle:
        for row in csv.DictReader(handle):
            pairs.add((float(row["noise_level"]), int(row["sample_index"])))
    for noise_level, sample_index in sorted(pairs):
        make_figure(noise_level, sample_index)


if __name__ == "__main__":
    main()
