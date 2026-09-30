"""Re-render tables and figures from saved results, without model evaluation."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Record = dict[str, Any]


def number(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "unavailable"


def render(out: Path) -> None:
    data = json.loads((out / "results.json").read_text())
    refinement_file = out / "spectral_refinement_gpu.json"
    if refinement_file.exists():
        refinement = json.loads(refinement_file.read_text())
        if refinement["checkpoint_sha256"] != data["settings"]["checkpoint_sha256"]:
            raise ValueError("Spectral refinement uses a different model checkpoint")
        for point in refinement["points"]:
            spectral = point["spectral"]
            if point["status"] != "complete" or spectral.get("status") != "certified_numerical":
                continue
            if spectral.get("cpu_verified_triplet_residual", float("inf")) >= 1e-3:
                continue
            for entry in data.get("diagnostics", []):
                if (entry["projected"], entry["iteration"]) == (point["projected"], point["iteration"]):
                    entry["spectral_cpu_initial"] = entry["spectral"]
                    entry["spectral"] = spectral
                    entry["spectral_reference_backend"] = "GPU 0 products, CPU singular-residual verification"
                    entry["cpu_gpu_parity"] = point["cpu_gpu_parity"]
        data["supplementary_spectral_refinement"] = refinement
        if "elapsed_seconds" in data and refinement["points"]:
            last = max(datetime.fromisoformat(point["finished_at"]).timestamp() for point in refinement["points"])
            data["primary_elapsed_seconds"] = data["elapsed_seconds"]
            data["elapsed_seconds"] += max(0, last - datetime.fromisoformat(data["updated_at"]).timestamp())
        (out / "combined_results.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    lines = ["# First-order study at 64×64", "",
             "CPU pilot: 100 Euler steps, non-attention epoch-100 model, 15-angle sparse CT, 1% relative noise. "
             "Analytical alpha = 1 − t, zero trace probes, fresh DLR factors for every gradient.", "",
             "All image errors are raw relative L2 errors (no clipping). Ground-truth best checkpoints are retrospective diagnostics.", "",
             "## Optimizer comparison", "",
             "Common-prefix errors compare the same number of attempted updates across both estimators and all three test images. "
             "Stalled runs remain in the comparison; final errors may use different completed budgets. Timing excludes metric collection and file I/O.", "",
             "| Constraint | Optimizer | Estimator | Common updates | Mean common error | Mean final error | Gradient s | Iteration excluding trials s | Total iteration s | Stalled/failed |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    summary = []
    runs = data.get("comparisons", [])
    for projected in (False, True):
        for optimizer in ("gd", "adam", "lbfgs"):
            both = [r for r in runs if r["projected"] == projected and r["optimizer"] == optimizer and r["history"]]
            if not both:
                continue
            common = min(r["attempts"] for r in both)
            for estimator in ("exact", "dlr"):
                group = [r for r in both if r["estimator"] == estimator]
                steps = [h for r in group for h in r["history"][1:]]
                if not group or not steps:
                    continue
                row = {"projected": projected, "optimizer": optimizer, "estimator": estimator,
                       "common_updates": common, "samples": len(group),
                       "common_error": statistics.mean(r["history"][common]["relative_image_error"] for r in group),
                       "final_error": statistics.mean(r["history"][-1]["relative_image_error"] for r in group),
                       "stalled_or_failed": sum(r["status"] in ("stalled", "failed") for r in group)}
                row.update({key: statistics.mean(h[key] for h in steps) for key in ("gradient_seconds", "seconds_excluding_trials", "seconds")})
                summary.append(row)
                lines.append(f"| {'Sphere' if projected else 'Free'} | {optimizer} | {estimator} | {common} | "
                             f"{row['common_error']:.4f} | {row['final_error']:.4f} | {row['gradient_seconds']:.3f} | "
                             f"{row['seconds_excluding_trials']:.3f} | {row['seconds']:.3f} | {row['stalled_or_failed']}/{len(group)} |")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), squeeze=False)
    for projected in (False, True):
        group = [row for row in summary if row["projected"] == projected]
        ax = axes[int(projected), 0]
        positions = list(range(len(group)))
        base = [row["seconds_excluding_trials"] for row in group]
        trial = [row["seconds"] - row["seconds_excluding_trials"] for row in group]
        colors = ["C0" if row["estimator"] == "exact" else "C1" for row in group]
        ax.bar(positions, base, color=colors, label="Gradient and update")
        ax.bar(positions, trial, bottom=base, color=colors, alpha=.3, label="Trial forwards")
        ax.set_xticks(positions, labels=[f"{row['optimizer'].upper()}\n{'D-Flow' if row['estimator'] == 'exact' else 'DLR'}" for row in group])
        ax.set(ylabel="CPU seconds / iteration", title="Sphere" if projected else "Free sources")
        ax.legend()
    fig.suptitle("Fresh DLR factors; dark bars exclude trial-forward cost")
    fig.tight_layout()
    fig.savefig(out / "iteration_timing.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for projected in (False, True):
        for column, optimizer in enumerate(("gd", "adam", "lbfgs")):
            ax = axes[int(projected), column]
            for r in runs:
                if r["projected"] != projected or r["optimizer"] != optimizer:
                    continue
                ax.plot([h["iteration"] for h in r["history"]], [h["relative_image_error"] for h in r["history"]],
                        color="C0" if r["estimator"] == "exact" else "C1", alpha=.65,
                        label=r["estimator"] if r["sample"] == 0 else None)
            ax.set(title=f"{'Sphere' if projected else 'Free'} / {optimizer}", xlabel="Attempted updates", ylabel="Relative image error")
            if ax.lines:
                ax.legend()
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for i, estimator in enumerate(("exact", "dlr")):
        for j, projected in enumerate((False, True)):
            ax = axes[i, j]
            for optimizer in ("gd", "adam"):
                group = [r for r in data.get("tuning", []) if (r["estimator"], r["projected"], r["optimizer"]) == (estimator, projected, optimizer)]
                if not group or not all(r["history"] for r in group):
                    continue
                common = min(r["attempts"] for r in group)
                ax.plot([r["lr"] for r in group], [r["history"][common]["relative_image_error"] for r in group], "o-", label=f"{optimizer}, {common} updates")
            ax.set(xscale="log", xlabel="Learning rate", ylabel="Validation relative error", title=f"{estimator} / {'sphere' if projected else 'free'}")
            if ax.lines:
                ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_rates.png", dpi=160)
    plt.close(fig)
    lines += ["", "## Rank sweep", "", "Settings are fixed across ranks and the exact control within each constraint.", "",
              "| Constraint | Rank | Updates | Final image error | Best image error | Final noisy residual | Residual at best image |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for projected in (False, True):
        group = [r for r in data.get("ranks", []) if r["projected"] == projected and r["history"]]
        for r in group:
            final = r["history"][-1]
            best = min(r["history"], key=lambda h: h["relative_image_error"])
            label = str(r["rank"]) if r["rank"] else "exact"
            lines.append(f"| {'Sphere' if projected else 'Free'} | {label} | {r['attempts']} | {final['relative_image_error']:.4f} | "
                         f"{best['relative_image_error']:.4f} | {final['measurement_residual']:.5f} | {best['measurement_residual']:.5f} |")
            color = f"C{(1, 2, 4, 8, 16, 32).index(r['rank'])}" if r["rank"] else "black"
            axes[int(projected), 1].scatter(final["measurement_residual"], final["relative_image_error"], c=color, marker="o", label=label)
            axes[int(projected), 1].scatter(best["measurement_residual"], best["relative_image_error"], c=color, marker="x")
        finite = [r for r in group if r["rank"]]
        for selector, style, label in ((lambda hs: hs[-1], "o-", "final"),
                                       (lambda hs: min(hs, key=lambda h: h["relative_image_error"]), "x--", "best seen")):
            axes[int(projected), 0].plot([r["rank"] for r in finite], [selector(r["history"])["relative_image_error"] for r in finite], style, label=label)
        axes[int(projected), 0].set(xlabel="Rank", ylabel="Relative image error", title="Sphere" if projected else "Free")
        axes[int(projected), 0].set_xscale("log", base=2)
        axes[int(projected), 0].set_xticks((1, 2, 4, 8, 16, 32), labels=("1", "2", "4", "8", "16", "32"))
        axes[int(projected), 0].legend()
        axes[int(projected), 1].set(xlabel="Noisy measurement residual", ylabel="Relative image error", title="Circle: final; cross: best seen")
        if group:
            axes[int(projected), 1].legend(title="Rank", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rank_residual.png", dpi=160)
    plt.close(fig)
    lines += ["", "## Gradient diagnostics", "",
              "Spectral references differentiate the full discrete Euler map independently of DLR. Certification is numerical; unresolved references are labeled.", "",
              "| Constraint | Iteration | DLR relative error | DLR cosine | Subspace relative error | Rank-8 reference relative error | Spectral status | Clean ratio R (trajectory noise) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, entry in enumerate(data.get("diagnostics", [])):
        if "dlr" not in entry:
            continue
        spectral = entry.get("spectral", {})
        noise = entry.get("clean_gradient", [])
        trajectory = noise[0]["R"] if noise else None
        spectral_error = spectral.get("gradient", {}).get("relative_error")
        lines.append(f"| {'Sphere' if entry['projected'] else 'Free'} | {entry['iteration']} | {number(entry['dlr']['relative_error'])} | "
                     f"{number(entry['dlr']['cosine'])} | {number(entry['subspace']['relative_error'])} | "
                     f"{number(spectral_error)} | {spectral.get('status', 'not reached')} | {number(trajectory)} |")
        ratios = [r["R"] for r in noise if r["R"] is not None]
        ax.scatter([i] * len(ratios), ratios, alpha=.5)
    ax.axhline(1, color="black", linestyle="--")
    ax.set(yscale="log", xlabel="Saved source (free first, then sphere)", ylabel="DLR-to-clean / noisy-exact-to-clean", title="Below 1 supports the clean-gradient hypothesis")
    fig.tight_layout()
    fig.savefig(out / "clean_gradient_ratios.png", dpi=160)
    plt.close(fig)
    lines += ["", "## Limits and completion", "",
              f"Elapsed experimental time: {data.get('elapsed_seconds', 0) / 60:.1f} minutes. Three held-out images and one validation image constitute a pilot, not a population-level accuracy result.",
              "Fresh-factor DLR timing is not comparable to earlier factor-reuse benchmarks without accounting for the different refresh policy.",
              "The ten extra noise realizations per saved source are conditional diagnostics, not independent reconstruction samples.",
              "Strong Wolfe results are condition checks at selected trial rates using true derivatives, not a tuned Strong Wolfe reconstruction sweep.", ""]
    if refinement_file.exists():
        lines += ["GPU 0 became available after the CPU reconstruction runs. Supplemental spectral references use batched GPU products with TF32 disabled, "
                  "CPU/GPU endpoint and gradient parity checks, and final CPU singular-triplet residual verification. "
                  "Only certified supplemental references replace the initial CPU references in this report; optimizer timings remain CPU-only.", ""]
    lines.extend(f"- {message}" for message in data.get("incomplete", []))
    lines.extend(f"- Failure: {entry}" for entry in data.get("failures", []))
    lines += ["", "## Figures", "", "![Convergence](convergence.png)", "",
              "![Iteration timing](iteration_timing.png)", "",
              "![Learning-rate comparison](learning_rates.png)", "",
              "![Rank and residual comparison](rank_residual.png)", "",
              "![Clean-gradient ratios](clean_gradient_ratios.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    render(parser.parse_args().out)
