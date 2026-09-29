"""Write an inspectable report and figures from a completed CPU pilot."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path, default=Path("p_flow/results.md"))
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    summary = report["summary"]
    directory = args.results.parent
    images = torch.load(directory / "images.pt", map_location="cpu", weights_only=True)
    names = {"pflow": "P-Flow", "dflow": "D-Flow (projected GD)", "fast_dlr": "Fast DLR (projected GD)"}
    figure, axes = plt.subplots(len(images), 6, figsize=(13, 2.3 * len(images)), squeeze=False)
    for row, image_set in enumerate(images):
        for column, (key, title) in enumerate(zip(
                ("truth", "measurement", "initial", *names),
                ("Ground truth", "Observation", "Initial", "P-Flow", "D-Flow", "Fast DLR"))):
            axes[row, column].imshow(image_set[key][0, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(f"Sample {row}")
    figure.tight_layout()
    figure.savefig(directory / "reconstructions.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    for method, name in names.items():
        runs = [run for run in report["runs"] if run["method"] == method]
        curves = [[row["psnr"] for row in run["history"]] + [run["final"]["psnr"]] for run in runs]
        average = [statistics.mean(values) for values in zip(*curves)]
        axis.plot(range(len(average)), average, label=name)
    axis.set_xlabel("Source updates")
    axis.set_ylabel("Mean raw-image PSNR (dB)")
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "convergence.png", dpi=180)
    plt.close(figure)
    settings = report["settings"]
    lines = ["# P-Flow CPU pilot", "", f"Run: {report['created_at']}", "",
             f"Non-attention circles-{settings['resolution']}, epoch {settings['epoch']}; "
             f"{settings['steps']} Euler steps, {settings['iterations']} projected-gradient updates, "
             f"{settings['samples']} paired test images, {settings['threads']} CPU threads. "
             f"Blur sigma 1.5 pixels, measurement-noise standard deviation {settings['noise_std']}.", "",
             "Learning rates were selected independently from a common grid using only validation image 0 "
             f"after {settings['calibration_iterations']} updates. Reported accuracy is the final iterate, "
             "without choosing the best test iteration. PSNR uses data range 1; raw and clipped metrics are separate.", "",
             "| Method | Selected learning rate | Mean iteration (s) | Raw MSE | Raw PSNR (dB) | Clipped PSNR (dB) | Measurement MSE |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, name in names.items():
        row = summary[method]
        lines.append(f"| {name} | {report['selected_learning_rates'][method]:g} | {row['mean_iteration_seconds']:.3f} | "
                     f"{row['mse']:.6g} | {row['psnr']:.3f} | {row['clipped_psnr']:.3f} | {row['measurement_mse']:.6g} |")
    lines += ["", "Iteration timings exclude diagnostics and the extra final-reporting forward solve. "
              f"DLR refreshes rank-{settings['rank']} factors every {settings['refresh_every']} updates; its timings "
              "include those refreshes. There is no Armijo in any of these runs.", "",
              "## Per-image results", "", "| Sample | Initial PSNR | P-Flow PSNR | D-Flow PSNR | Fast DLR PSNR |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for initial in report["initial_metrics"]:
        values = [next(run["final"]["psnr"] for run in report["runs"] if run["sample"] == initial["sample"] and run["method"] == method) for method in names]
        lines.append(f"| {initial['sample']} | {initial['psnr']:.3f} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "## Shared initial-source gradients", "",
              "| Sample | Approximation | Cosine with exact | Relative gradient error | Norm / exact norm | Tangent cosine | True derivative along projected update |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["gradient_diagnostics"]:
        lines.append(f"| {row['sample']} | {names[row['method']]} | {row['cosine']:.4f} | {row['relative_error']:.4f} | "
                     f"{row['norm_ratio']:.4f} | {row['projected_cosine']:.4f} | {row['projected_directional_derivative']:.5g} |")
    relative = directory.relative_to(args.report.parent)
    lines += ["", "Negative directional derivative means local descent on the sphere. Positive cosine "
              "does not mean an accurate Jacobian approximation, nor does it guarantee descent for a finite learning rate.", "",
              "## Figures and evidence", "",
              f"![Reconstructions]({relative}/reconstructions.png)", "",
              f"![Convergence]({relative}/convergence.png)", "",
              f"[Full settings and histories]({relative}/results.json)", "",
              "## Limits", "",
              "This is a small single-model, single-task pilot. Three images and one short validation run "
              "cannot establish a general ranking. The grid may not contain the best learning rate. "
              "D-Flow here uses matched projected gradient descent, not the project's tuned LBFGS + Armijo optimizer. "
              "P-Flow includes spherical projection; this is more than just replacing a gradient in an otherwise "
              "unchanged production D-Flow solve. GPU speed and 250×250 performance are untested.", ""]
    args.report.write_text("\n".join(lines))
    print(args.report)


if __name__ == "__main__":
    main()
