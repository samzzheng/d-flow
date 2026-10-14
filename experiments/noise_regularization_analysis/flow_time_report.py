"""Render saved CPU flow-time Jacobian diagnostics without model evaluations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.noise_regularization_analysis.flow_time import OUT


def probe_error(point: dict[str, Any]) -> float:
    return point.get("terminal_frobenius_refinement", {}).get(
        "previous_probe_estimate", point["full_relative_frobenius_error_estimate"])


def main() -> None:
    records = [json.loads(path.read_text()) for path in sorted(OUT.glob("*_steps*.json"))]
    rows = []
    lines = ["# Jacobian evolution along flow time", "",
             "Single-circle64px epoch200 model. Native partial Euler derivatives use the same dt and arithmetic as the sampler. DLR uses alpha=1-t, rank8 residual factors, and the existing fast profile. All diagnostics are CPU only.", "",
             "J_t = derivative of x_t with respect to source z. This is flow time, not optimizer iteration. The approximation includes its identity baseline: Jhat_t=(1-t)I+U_t S_t V_t^T. Its full singular spectrum is calculated exactly using a <=16-dimensional block plus the remaining identity singular values.", "",
             "Stable rank = ||J||_F^2 / ||J||_2^2. Energy rank is the number of singular values needed for99.9% of squared-singular-value energy. Neither is algebraic rank. Native spectra are bounded randomized estimates with independent omitted-energy probes; unresolved energy cutoffs are not assigned a rank. Stable-rank estimates also depend on convergence of the leading singular value.", "",
             "At intermediate times, noise-gradient diagnostics concern hypothetical measurements A(x_t). Only t=1 is the reconstruction objective. The100-step companion uses the same initial source but a different, more finely integrated generator; it is not a rerun of the five-step optimizer.", ""]
    lines += ["Terminal table errors use the low-rank Frobenius identity with native JVPs. Plot lines keep the common probe estimator at every time, and hollow diamonds show refined terminal errors. Leading-norm refinements update stable rank independently of unresolved energy cutoffs.", ""]
    for record in records:
        lines += [f"## {record['case'].upper()}, {record['source_kind']}, {record['flow_steps']} Euler steps", "",
                  f"Basis initialization: {record['basis_initialization']}. Fast/native endpoint discrepancy: {record['fast_native_terminal_relative_difference']:.3g}.", "",
                  "| t | Native stable rank estimate | DLR full stable rank | Native99.9% energy rank | DLR full99.9% energy rank | Relative Frobenius error estimate | Noise-gradient norm ratio |",
                  "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
        for point in record["points"]:
            rank = point["native_energy_rank_estimates"]["0.999"]
            status = point["native_energy_rank_status"]["0.999"]
            rank_text = str(rank) if status != "unresolved at budget" else f"unresolved (estimate {rank})" if rank is not None else "unresolved"
            lines.append(f"| {point['time']:.2f} | {point['native_stable_rank_estimate']:.3f} | {point['approximation_stable_rank']:.3f} | {rank_text} | {point['approximation_energy_ranks']['0.999']} | {100 * point['full_relative_frobenius_error_estimate']:.2f}% | {point['noise_gradient_norm_ratio']:.3f} |")
            rows.append({"case": record["case"], "source": record["source_kind"], "flow_steps": record["flow_steps"],
                         **{key: point[key] for key in ("time", "native_stable_rank_estimate", "approximation_stable_rank",
                             "native_sigma_max_estimate", "approximation_sigma_max", "full_relative_frobenius_error_estimate",
                             "measurement_relative_frobenius_error_estimate", "noise_gradient_norm_ratio", "clean_gradient_R",
                             "noisy_gradient_relative_error", "noisy_gradient_cosine", "right_top3_containment_error",
                             "native_top3_triplet_relative_residual")},
                         "native_energy_rank_999": rank, "native_energy_rank_999_status": status})
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for record in records:
        if record["case"] != "ct" or record["source_kind"] != "initial":
            continue
        points = record["points"]
        t = [p["time"] for p in points]
        color = "#1565c0" if record["flow_steps"] == 5 else "#c65102"
        label = f"{record['flow_steps']} Euler steps"
        for axis, native, approximation in (
            (axes[0, 0], "native_stable_rank_estimate", "approximation_stable_rank"),
            (axes[0, 1], "native_sigma_max_estimate", "approximation_sigma_max"),
        ):
            axis.plot(t, [p[native] for p in points], "o-", color=color, label=f"Native, {label}")
            axis.plot(t, [p[approximation] for p in points], "x--", color=color, label=f"DLR, {label}")
        axes[1, 0].plot(t, [100 * probe_error(p) for p in points], "o-", color=color, label=label)
        axes[1, 0].scatter([points[-1]["time"]], [100 * points[-1]["full_relative_frobenius_error_estimate"]],
                           marker="D", facecolors="none", edgecolors=color, s=90, zorder=4)
        axes[1, 1].plot(t, [p["noise_gradient_norm_ratio"] for p in points], "o-", color=color, label=label)
    axes[0, 0].set_ylabel("Stable rank (native is estimated)")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_ylabel("Largest singular value (native is estimated)")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_ylabel("Estimated relative Frobenius error (%)")
    axes[1, 1].set_ylabel("Noise-gradient norm: DLR / native")
    axes[1, 1].axhline(1, color="black", linestyle=":", linewidth=1)
    for axis in axes.flat:
        axis.set_xlabel("Flow time t")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Jacobian evolution: same initial source, 64² single-circle model\nFull DLR includes (1−t)I; error lines use common probes, hollow diamonds refine terminal errors")
    figure.savefig(OUT / "initial_flow_time.png", dpi=150)
    figure.savefig(OUT / "initial_flow_time.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained", sharex=True)
    colors = {"initial": "#666666", "d_flow_final": "#1565c0", "dlr_final": "#16803d"}
    for record in records:
        if record["flow_steps"] != 5:
            continue
        row = 0 if record["case"] == "ct" else 1
        points = record["points"]
        t = [p["time"] for p in points]
        color, label = colors[record["source_kind"]], record["source_kind"].replace("_", " ")
        axes[row, 0].plot(t, [p["native_stable_rank_estimate"] for p in points], "o-", color=color, label=label)
        axes[row, 0].plot(t, [p["approximation_stable_rank"] for p in points], "x--", color=color)
        axes[row, 1].plot(t, [100 * probe_error(p) for p in points], "o-", color=color, label=label)
        axes[row, 1].scatter([points[-1]["time"]], [100 * points[-1]["full_relative_frobenius_error_estimate"]],
                             marker="D", facecolors="none", edgecolors=color, s=90, zorder=4)
    for row, problem in enumerate(("CT case", "MRI case")):
        axes[row, 0].set_title(f"{problem}: solid native, dashed DLR")
        axes[row, 0].set_ylabel("Stable rank (native is estimated)")
        axes[row, 0].set_yscale("log")
        axes[row, 1].set_title(f"{problem}: full Jacobian approximation error")
        axes[row, 1].set_ylabel("Estimated relative Frobenius error (%)")
        for axis in axes[row]:
            axis.set_xlabel("Flow time t")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    figure.suptitle("Five-step flow trajectories at fixed initial and optimized sources\nIllustrative cases; error lines use common probes, hollow diamonds refine terminal errors")
    figure.savefig(OUT / "optimized_sources_flow_time.png", dpi=150)
    figure.savefig(OUT / "optimized_sources_flow_time.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
