"""Render saved first-crossing reconstructions without evaluating the model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "outputs/rk4_noise_floor_2026_09_27")
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--tau", type=float, default=1.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dataset = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu",
                         weights_only=True, mmap=True)["test"]
    truth = bench.normalize_images(dataset[args.sample:args.sample+1])[0, 0]
    records = []
    for method, directory in (("Euler", "euler"), ("RK4", "rk4")):
        result = json.loads((args.root / f"sample{args.sample}_{directory}" / "results.json").read_text())
        row = result["first_crossings"].get(str(args.tau))
        if row is None:
            raise ValueError(f"{method} has not reached the requested raw discrepancy threshold")
        path = Path(row["saved_state"])
        if not path.is_absolute():
            path = ROOT / path
        state = torch.load(path, map_location="cpu", weights_only=True)
        reconstruction = state["raw_reconstruction"].clamp(0, 1)[0, 0]
        records.append((method, row, reconstruction, (reconstruction-truth).abs()))

    maximum = max(float(error.max()) for _, _, _, error in records)
    fig = plt.figure(figsize=(10.6, 6.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    truth_axis = fig.add_subplot(grid[0, 0])
    truth_axis.imshow(truth.numpy(), cmap="gray", vmin=0, vmax=1)
    truth_axis.set_title("Ground truth")
    truth_axis.set_axis_off()
    notes_axis = fig.add_subplot(grid[1, 0])
    notes_axis.text(.02, .65,
                    "Actual saved accepted states\n\n"
                    "Images clamped to [0, 1]\n"
                    "Stopping uses raw images\n\n"
                    "Same image scale and error color limits\n"
                    "One selected pilot; no population claim",
                    va="center", fontsize=10, transform=notes_axis.transAxes)
    notes_axis.set_axis_off()
    error_axes = []
    for index, (method, row, reconstruction, error) in enumerate(records):
        reconstruction_axis = fig.add_subplot(grid[index, 1])
        reconstruction_axis.imshow(reconstruction.numpy(), cmap="gray", vmin=0, vmax=1)
        reconstruction_axis.set_title(
            f"{method}, accepted step {row['accepted_index']}\n"
            f"Image error {100*row['image_relative_l2']:.4f}%; PSNR {row['psnr_db']:.2f} dB",
            fontsize=10)
        reconstruction_axis.set_axis_off()
        error_axis = fig.add_subplot(grid[index, 2])
        displayed = error_axis.imshow(error.numpy(), cmap="magma", vmin=0, vmax=maximum)
        error_axis.set_title(f"Absolute error; residual/noise {row['residual_over_noise']:.6f}", fontsize=9)
        error_axis.set_axis_off()
        error_axes.append(error_axis)
    fig.colorbar(displayed, ax=error_axes, label="Absolute intensity error", shrink=.8)
    fig.suptitle(f"250 × 250 sample {args.sample}: first accepted raw residual ≤ {args.tau:g} × noise\n"
                 "100 Euler steps / 100 RK4 steps; same initial source and 1% measurement noise",
                 fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(args.root / f"noise_floor_images_tau{args.tau:g}.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
