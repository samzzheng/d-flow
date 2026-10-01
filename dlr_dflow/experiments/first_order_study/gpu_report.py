"""Accuracy and convergence reporting for the longer GPU study; no speed tables."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(out: Path) -> None:
    data = json.loads((out / "results.json").read_text())
    settings = data["settings"]
    lines = ["# GPU first-order convergence study", "",
             f"Status: **{data['status']}**. Last checkpoint: {data['updated_at']}.", "",
             "64×64 non-attention epoch-100 model; 100 Euler steps; 15-angle sparse CT; 1% relative noise. "
             "Paired inputs match the CPU pilot. Analytical alpha=1−t, zero trace probes, zero enrichment, "
             "initialization oversampling 8, fresh DLR factors at every gradient, exact zero terminal alpha. "
             "The objective and terminal gradient use the native Euler endpoint for both estimators. "
             "GPU sketches use device-local seeded randomness, so their draws differ from the CPU pilot.", "",
             f"Validation budget: {settings['tuning_updates']} updates per learning rate. "
             f"Reconstruction budget: {settings['min_updates']}–{settings['max_updates']} updates unless line search stalls or an evaluation fails. "
             f"After the minimum, stop if the best objective improves by less than {100 * settings['plateau_relative']:.2f}% "
             f"in each of two consecutive {settings['window']}-update windows. "
             "A plateau is not a proof of stationarity. Oscillatory plateaus have a recent objective range above 5% of the recent minimum. "
             "Runs reaching the step limit are not automatically converged. Noise-floor crossings and image error do not stop a run.", "",
             "All proposed source displacements are capped at 0.64. Initial L1 scaling is applied only on the first attempt. "
             "Armijo uses the actual projected candidate, c1=1e-4 and at most 12 backtracks. "
             "LBFGS gets one retry with cleared history after a failed search. "
             "Failed searches are reported separately from objective plateaus.", "",
             "## Validation selection", "",
             "Settings are selected by final validation image error at the fixed budget; stalled trajectories retain their last iterate. "
             "Best-image checkpoints are retrospective diagnostics, not the stopping rule.", "",
             "| Constraint | Estimator | Optimizer | LR | Updates | Status | Validation error |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in data.get("selected", []):
        lines.append(f"| {'Sphere' if row['projected'] else 'Free'} | {row['estimator']} | {row['optimizer']} | "
                     f"{row['lr']} | {row['attempts']} | {row['status']} | {row['validation_error']:.4%} |")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for run in data.get("tuning", []):
        ax = axes[int(run["projected"]), int(run["estimator"] == "dlr")]
        ax.plot([h["iteration"] for h in run["history"]], [h["relative_image_error"] for h in run["history"]],
                label=f"{run['optimizer']} lr={run['lr']}")
    for row in range(2):
        for col in range(2):
            axes[row, col].set(title=f"{'Sphere' if row else 'Free'} / {'DLR' if col else 'D-Flow'}",
                              xlabel="Updates", ylabel="Validation relative image error")
            if axes[row, col].lines:
                axes[row, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=140)
    plt.close(fig)
    lines += ["", "![Validation](validation.png)", "", "## Held-out reconstructions", "",
              "Means below summarize the current terminal iterates of three test images. "
              "Budgets/statuses are shown because plateau and stalled runs can finish at different counts. "
              "Read per-image curves and terminal statuses before interpreting means.", "",
              "| Constraint | Optimizer | Estimator | Images | Update range | Mean final error | Mean best-image error | Mean residual/noise | Status counts |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    runs = [r for r in data.get("comparisons", []) if r["history"]]
    for projected in (False, True):
        for optimizer in ("gd", "adam", "lbfgs"):
            for estimator in ("exact", "dlr"):
                group = [r for r in runs if (r["projected"], r["optimizer"], r["estimator"]) == (projected, optimizer, estimator)]
                if not group:
                    continue
                error = statistics.mean(r["history"][-1]["relative_image_error"] for r in group)
                best = statistics.mean(min(h["relative_image_error"] for h in r["history"]) for r in group)
                residual = statistics.mean(r["history"][-1]["residual_over_noise"] for r in group)
                counts = ", ".join(f"{key}: {value}" for key, value in Counter(r["status"] for r in group).items())
                lines.append(f"| {'Sphere' if projected else 'Free'} | {optimizer} | {estimator} | {len(group)} | "
                             f"{min(r['attempts'] for r in group)}–{max(r['attempts'] for r in group)} | "
                             f"{error:.4%} | {best:.4%} | {residual:.3f} | {counts} |")
    for metric, label, filename in (("relative_image_error", "Relative image error", "convergence.png"),
                                    ("residual_over_noise", "Noisy residual / noise norm", "residuals.png")):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
        for run in runs:
            ax = axes[int(run["projected"]), ("gd", "adam", "lbfgs").index(run["optimizer"])]
            ax.plot([h["iteration"] for h in run["history"]], [h[metric] for h in run["history"]],
                    color="C0" if run["estimator"] == "exact" else "C1",
                    linestyle=("-", "--", ":")[run["sample"]],
                    label=f"{run['estimator']} image {run['sample']} ({run['status']})")
        for row in range(2):
            for col, optimizer in enumerate(("gd", "adam", "lbfgs")):
                ax = axes[row, col]
                ax.set(title=f"{'Sphere' if row else 'Free'} / {optimizer}", xlabel="Updates", ylabel=label)
                if metric == "residual_over_noise":
                    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
                    ax.set_yscale("log")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=140)
        plt.close(fig)
        lines += ["", f"![{label}]({filename})"]
    lines += ["", "## Individual terminal states", "",
              "Exact tangent norms are evaluated at final sources after optimization. A nonzero exact gradient can remain when DLR or a line search stalls.", "",
              "| Run | Updates | Status | Final image error | Noisy residual | Clean residual | Residual/noise | Exact tangent norm |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in runs + [r for r in data.get("ranks", []) if r["history"]]:
        row = run["history"][-1]
        norm = data.get("terminal_checks", {}).get(run["label"], {}).get("exact_tangent_gradient_norm")
        norm_text = f"{norm:.6g}" if norm is not None else "pending"
        lines.append(f"| {run['label']} | {run['attempts']} | {run['status']} | {row['relative_image_error']:.4%} | "
                     f"{row['measurement_residual']:.6g} | {row['clean_residual']:.6g} | {row['residual_over_noise']:.3f} | {norm_text} |")
    lines += ["", "## Rank sweep", "", "Test image 0; optimizer family and LR fixed from DLR validation for each constraint, including the exact control. "
              "Crosses mark retrospective best-image checkpoints, paired with their own residuals.", ""]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for projected in (False, True):
        group = [r for r in data.get("ranks", []) if r["projected"] == projected and r["history"]]
        for run in group:
            final = run["history"][-1]
            best = min(run["history"], key=lambda h: h["relative_image_error"])
            label = str(run["rank"]) if run["rank"] else "exact"
            color = f"C{(1, 2, 4, 8, 16, 32).index(run['rank'])}" if run["rank"] else "black"
            axes[int(projected), 0].plot([h["iteration"] for h in run["history"]],
                [h["relative_image_error"] for h in run["history"]], color=color, label=label)
            axes[int(projected), 2].scatter(final["measurement_residual"], final["relative_image_error"], color=color, label=label)
            axes[int(projected), 2].scatter(best["measurement_residual"], best["relative_image_error"], color=color, marker="x")
            if not run["rank"]:
                axes[int(projected), 1].axhline(final["relative_image_error"], color="black", linestyle=":", label="Exact final")
        finite = [r for r in group if r["rank"]]
        axes[int(projected), 1].plot([r["rank"] for r in finite],
            [r["history"][-1]["relative_image_error"] for r in finite], "o-", label="Final")
        axes[int(projected), 1].plot([r["rank"] for r in finite],
            [min(h["relative_image_error"] for h in r["history"]) for r in finite], "x--", label="Best image")
        axes[int(projected), 0].set(title="Sphere" if projected else "Free", xlabel="Updates", ylabel="Relative image error")
        axes[int(projected), 1].set(xlabel="Rank", ylabel="Relative image error")
        axes[int(projected), 1].set_xscale("log", base=2)
        axes[int(projected), 1].set_xticks((1, 2, 4, 8, 16, 32), labels=("1", "2", "4", "8", "16", "32"))
        axes[int(projected), 2].set(xlabel="Noisy measurement residual", ylabel="Relative image error")
        for ax in axes[int(projected)]:
            if group:
                ax.legend(title="Rank", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ranks.png", dpi=140)
    plt.close(fig)
    lines += ["![Rank trajectories and residuals](ranks.png)", "", "## Shared-source gradient diagnostics", "",
              "| Constraint | Update | DLR relative error | Subspace error | Truncated reference error | Spectral status | Clean-gradient R | R < 1 cases |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in data.get("diagnostics", []):
        if row["status"] != "complete":
            continue
        spectral = row["spectral"]
        clean = row["clean_gradient"]
        values = [row["dlr"]["relative_error"], row["subspace"]["relative_error"],
                  spectral.get("gradient", {}).get("relative_error")]
        texts = [f"{value:.4%}" if value is not None else "unavailable" for value in values]
        ratio = clean[0]["R"]
        ratio_text = f"{ratio:.4g}" if ratio is not None else "unavailable"
        lines.append(f"| {'Sphere' if row['projected'] else 'Free'} | {row['iteration']} | " + " | ".join(texts) +
                     f" | {spectral['status']} | {ratio_text} | {sum(e['closer_to_clean'] is True for e in clean)}/{len(clean)} |")
    completed = [row for row in data.get("diagnostics", []) if row["status"] == "complete"]
    if completed:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for row in completed:
            defined = [point for point in row["clean_gradient"] if point["R"] is not None]
            axes[int(row["projected"])].plot([point["noise_index"] for point in defined],
                [point["R"] for point in defined], "o-", label=f"Update {row['iteration']}")
        for projected, ax in enumerate(axes):
            ax.axhline(1, color="black", linestyle=":")
            ax.set(yscale="log", xlabel="Noise realization (0 = trajectory)", ylabel="Clean-gradient ratio R",
                   title="Sphere" if projected else "Free")
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        fig.tight_layout()
        fig.savefig(out / "clean_gradient_ratios.png", dpi=140)
        plt.close(fig)
        lines += ["", "![Clean-gradient ratios](clean_gradient_ratios.png)"]
    lines += ["", "Unresolved spectral cutoffs remain labeled; noise cases at fixed sources are not independent reconstruction trials.", "",
              "All detailed gradients, cosines, coefficient errors, and stopping evidence are in [results.json](results.json). "
              "Source and optimizer checkpoints are in `states/`. No production algorithm modules were modified.", ""]
    if data.get("failures"):
        lines += ["## Failures", "", *[f"- {row['label']}: {row['error']}" for row in data["failures"]]]
    path = out / "report.md"
    path.with_suffix(".tmp").write_text("\n".join(lines))
    path.with_suffix(".tmp").replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    render(parser.parse_args().out)
