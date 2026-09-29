"""Report the shared LBFGS + Armijo comparison and fixed-step baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path, default=Path("p_flow/lbfgs_results.md"))
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    baseline = json.loads((Path(report["baseline"]) / "results.json").read_text())
    summary = report["summary"]
    settings = report["settings"]
    names = {"pflow": "P-Flow", "dflow": "D-Flow", "fast_dlr": "Fast DLR"}
    lines = ["# P-Flow with projected LBFGS + Armijo", "",
             f"Run: {report['created_at']}. Same three images, measurements, normalized initial sources, "
             f"non-attention circles-{settings['resolution']} epoch-{settings['epoch']} model and "
             f"{settings['steps']}-step Euler integrator as the fixed-step pilot.", "",
             f"All methods use learning rate {settings['lr']:g}, history 100, Armijo c1=1e-4, "
             f"at most {settings['max_ls']} trials, and displacement cap {settings['step_cap']:g} sqrt(d). "
             f"Budget: {settings['iterations']} steps; stop early after two failed searches. "
             "Initial L1 scaling occurs only on the first attempted search. Directions are projected "
             "onto the source sphere's tangent plane, trials are projected to the sphere BEFORE "
             "Armijo checks the true objective, and curvature pairs use actual displacements. "
             "This is projected ambient LBFGS, not a full Riemannian BFGS implementation.", "",
             f"Fast DLR refreshes rank-{settings['rank']} factors every {settings['refresh_every']} steps, "
             "clearing curvature history when the approximation changes while preserving lifetime initial-step scaling.", "",
             "| Method | Mean relative image error | Raw PSNR | Mean step incl. Armijo | Mean step excluding extra trial flows | Mean total solve time |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, name in names.items():
        row = summary[method]
        lines.append(f"| {name} | {100 * row['relative_l2']:.2f}% | {row['psnr']:.2f} dB | "
                     f"{row['mean_iteration_seconds']:.3f} s | {row['mean_seconds_excluding_extra_trial_flows']:.3f} s | {row['total_seconds']:.2f} s |")
    lines += ["", "Relative error is ||reconstruction − truth||₂ / ||truth||₂ on unclipped images, averaged "
              "over test images. Accepted-trial forward solves reused for P-Flow/DLR gradients remain counted "
              "in the column excluding extra trial flows; only additional search flows are removed. "
              "Gradient evaluation, factor refreshes, LBFGS arithmetic and projection remain counted. "
              "Metrics and plotting are excluded. The last accepted state is reported.", "",
              "## Per-image errors and search behavior", "",
              "| Sample | Method | Fixed-step GD error | LBFGS + Armijo error | Steps | Trial evaluations | Failed searches |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for run in report["runs"]:
        previous = next(row for row in baseline["runs"] if row["sample"] == run["sample"] and row["method"] == run["method"])
        lines.append(f"| {run['sample']} | {names[run['method']]} | {100 * previous['final']['relative_l2']:.2f}% | "
                     f"{100 * run['final']['relative_l2']:.2f}% | {run['steps_run']} | {run['trial_evaluations']} | {run['failed_searches']} |")
    images = torch.load(args.results.parent / "images.pt", map_location="cpu", weights_only=True)
    figure, axes = plt.subplots(len(images), 4, figsize=(9, 2.4 * len(images)), squeeze=False)
    for index, image in enumerate(images):
        for column, key in enumerate(("truth", *names)):
            axes[index, column].imshow(image[key][0, 0], cmap="gray", vmin=0, vmax=1)
            axes[index, column].set_title(("Truth" if key == "truth" else names[key]) if index == 0 else "")
            axes[index, column].set_xticks([])
            axes[index, column].set_yticks([])
        axes[index, 0].set_ylabel(f"Sample {index}")
    figure.tight_layout()
    figure.savefig(args.results.parent / "reconstructions.png", dpi=180)
    plt.close(figure)
    relative = args.results.parent.relative_to(args.report.parent)
    lines += ["", f"![Reconstructions]({relative}/reconstructions.png)", "",
              f"[Full metrics and settings]({relative}/results.json)", "",
              "This pilot is limited to three 64×64 noisy-blur images and a short common iteration budget. "
              "All methods retain the sphere constraint for a controlled comparison; this differs from "
              "an unconstrained production D-Flow solve. The positive-definite LBFGS model uses approximate "
              "gradients for P-Flow/DLR; Armijo enforces true loss decrease but does not prove that those "
              "gradients approximate the exact derivative. No GPU benchmark or 250×250 conclusion follows.", ""]
    args.report.write_text("\n".join(lines))
    print(args.report)


if __name__ == "__main__":
    main()
